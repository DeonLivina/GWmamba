import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from compact_loader import get_dataloaders
from model import MambaFusionSupConModel
from losses import SupervisedSimCLRLoss

torch.set_float32_matmul_precision("medium")

CLASS_COLOR_MAP = {
    "background": "#1f77b4",  # Blue
    "glitch": "#d62728",      # Red
    "signal": "#2ca02c",      # Green
}


# =====================================================================
# 1. CORNER PLOT GENERATION (standalone function + thin Callback wrapper)
#
# Kept as a plain function so it can be called manually AFTER reloading
# the best checkpoint's weights (see main() below) -- using it only as an
# on_train_end Callback would run it on the LAST epoch's weights, not the
# best one.
# =====================================================================
@torch.no_grad()
def generate_corner_plots(pl_module, data_loader, label_names, detector_names, out_dir="plots"):
    print("\nGenerating corner plots for projection spaces...")
    os.makedirs(out_dir, exist_ok=True)
    pl_module.eval()
    device = pl_module.device

    strain_z_list, witness_z_list = [], []
    types_list, detectors_list = [], []

    for batch in data_loader:
        x, y_type, y_det = batch
        strain = x[:, :, 0:1].to(device)
        witness = x[:, :, 1:].to(device)

        strain_z, witness_z = pl_module.model(strain, witness)

        strain_z_list.append(strain_z.cpu())
        witness_z_list.append(witness_z.cpu())
        types_list.append(y_type.cpu())
        detectors_list.append(y_det.cpu())

    strain_zs = torch.cat(strain_z_list).numpy()
    witness_zs = torch.cat(witness_z_list).numpy()
    y_types = torch.cat(types_list).numpy()
    y_dets = torch.cat(detectors_list).numpy()

    max_samples = 1500
    if len(y_types) > max_samples:
        indices = np.random.choice(len(y_types), max_samples, replace=False)
        strain_zs, witness_zs = strain_zs[indices], witness_zs[indices]
        y_types, y_dets = y_types[indices], y_dets[indices]

    dim_limit = strain_zs.shape[1]

    # --- 1. Strain Projection Corner Plot ---
    strain_df = pd.DataFrame(
        strain_zs[:, :dim_limit],
        columns=[f"S_Proj {i+1}" for i in range(dim_limit)]
    )
    strain_df["Event Type"] = [label_names[int(t)] for t in y_types]

    g_strain = sns.pairplot(
        strain_df,
        hue="Event Type",
        corner=True,
        palette=CLASS_COLOR_MAP,
        plot_kws={"alpha": 0.5, "s": 10}
    )
    g_strain.fig.suptitle("Strain Projection Space (SupCon + CE, by Event Type)", y=1.02, fontsize=14)
    strain_path = os.path.join(out_dir, "strain_projection_corner.png")
    g_strain.savefig(strain_path, dpi=200)
    plt.close()
    print(f"Saved Strain Corner Plot -> {strain_path}")

    # --- 2. Witness Projection Corner Plot ---
    witness_df = pd.DataFrame(
        witness_zs[:, :dim_limit],
        columns=[f"W_Proj {i+1}" for i in range(dim_limit)]
    )
    witness_df["Detector"] = [detector_names[int(d)] for d in y_dets]

    g_witness = sns.pairplot(
        witness_df,
        hue="Detector",
        corner=True,
        palette="Dark2",
        plot_kws={"alpha": 0.5, "s": 10}
    )
    g_witness.fig.suptitle("Witness Projection Space (SupCon + CE, by Detector)", y=1.02, fontsize=14)
    witness_path = os.path.join(out_dir, "witness_projection_corner.png")
    g_witness.savefig(witness_path, dpi=200)
    plt.close()
    print(f"Saved Witness Corner Plot -> {witness_path}")


# =====================================================================
# 2. LIGHTNING MODULE -- SupCon + CE (AutoSciDACT-style joint training)
# =====================================================================
class DualSupConCELitModule(L.LightningModule):
    """Total loss = ce_weight * CE(classifier_logits, y_type)
                   + aux_weight * CE(aux_strain_logits, y_type)
                   + supcon_weight * (SupCon(strain_z, y_type) + SupCon(witness_z, y_detector))

    classifier_logits come from concat(strain_z, witness_z) [or strain_z
    alone if use_witness=False] fed through the model's main classifier.
    aux_strain_logits come from strain_z ALONE fed through a separate
    auxiliary head -- this is the aux-loss design from the original
    (pre-SupCon) model, added back on top of the SupCon terms.
    """
    def __init__(self, model, lr=1e-3, temperature=0.1, supcon_weight=1.0, ce_weight=1.0,
                 aux_weight=0.3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.supcon_weight = supcon_weight
        self.ce_weight = ce_weight
        self.aux_weight = aux_weight
        self.supcon_strain = SupervisedSimCLRLoss(temperature=temperature)
        self.supcon_witness = SupervisedSimCLRLoss(temperature=temperature)

    def _step(self, batch, stage):
        x, y_type, y_det = batch
        strain, witness = x[:, :, 0:1], x[:, :, 1:]

        # classify() runs the shared trunk ONCE and returns both the
        # classifier logits (from the independent cls_temporal_pool
        # embedding) and strain_z/witness_z (from the separate SupCon-side
        # pooling), so nothing here needs to reconstruct an "embed" tensor
        # or call the classifier/aux head directly anymore.
        logits, aux_logits, strain_z, witness_z = self.model.classify(strain, witness)

        ce_loss = F.cross_entropy(logits, y_type)
        aux_ce_loss = F.cross_entropy(aux_logits, y_type)

        # SupervisedSimCLRLoss expects features [bsz, n_views, ...], L2-normalized
        strain_feats = F.normalize(strain_z, dim=1).unsqueeze(1)
        supcon_strain_loss = self.supcon_strain(strain_feats, y_type)

        if self.model.use_witness:
            witness_feats = F.normalize(witness_z, dim=1).unsqueeze(1)
            supcon_witness_loss = self.supcon_witness(witness_feats, y_det)
        else:
            supcon_witness_loss = torch.zeros((), device=strain_z.device)

        supcon_loss = supcon_strain_loss + supcon_witness_loss

        loss = (
            self.ce_weight * ce_loss
            + self.aux_weight * aux_ce_loss
            + self.supcon_weight * supcon_loss
        )

        acc = (logits.argmax(dim=1) == y_type).float().mean()
        aux_acc = (aux_logits.argmax(dim=1) == y_type).float().mean()

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}_ce_loss", ce_loss, on_epoch=True)
        self.log(f"{stage}_aux_ce_loss", aux_ce_loss, on_epoch=True)
        self.log(f"{stage}_supcon_strain_loss", supcon_strain_loss, on_epoch=True)
        self.log(f"{stage}_supcon_witness_loss", supcon_witness_loss, on_epoch=True)
        self.log(f"{stage}_acc", acc, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_aux_acc", aux_acc, on_epoch=True)

        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def test_step(self, batch, _):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }


# =====================================================================
# 3. MAIN SCRIPT
# =====================================================================
def main():
    train_loader, val_loader, test_loader, meta = get_dataloaders(batch_size=128)

    # Determine n_wit dynamically from the data batch shape (x shape: [B, T, N_channels], channel 0 is strain)
    sample_batch = next(iter(train_loader))
    x_sample, _, _ = sample_batch
    n_wit = x_sample.shape[-1] - 1

    print(f"\n--- INITIALIZING MAMBA FUSION SUPCON+CE MODEL (n_wit={n_wit}) ---")
    model = MambaFusionSupConModel(
        d_model=32,
        mamba_layers=4,
        proj_dim=8,
        pool_method="mean",
        n_wit=n_wit,
        n_classes=meta["num_classes"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,}")

    lit_model = DualSupConCELitModule(
        model, lr=1e-3, temperature=0.1, supcon_weight=1.0, ce_weight=1.0, aux_weight=0.3
    )

    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints_mamba_supcon_ce", monitor="val_acc", mode="max",
        save_top_k=1, filename="{epoch:02d}-{val_acc:.4f}",
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor="val_acc", mode="max", patience=12),
    ]

    trainer = L.Trainer(
        max_epochs=30,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=WandbLogger(project="ligo-mamba-supcon-ce", name="Mamba-Fusion-SupCon-CE"),
    )

    trainer.fit(lit_model, train_loader, val_loader)

    print(f"\nLoading best checkpoint (by val_acc): {ckpt_cb.best_model_path}")
    best_lit_model = DualSupConCELitModule.load_from_checkpoint(
        ckpt_cb.best_model_path, model=model
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    trainer.test(best_lit_model, test_loader)
    generate_corner_plots(best_lit_model, test_loader, meta["label_names"], meta["detector_names"])
    print("\nTraining, testing, and corner plot generation complete!")


if __name__ == "__main__":
    main()