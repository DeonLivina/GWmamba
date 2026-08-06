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

from special_loader import get_dataloaders
from cmamba_model import CMambaModel
from losses import SupervisedSimCLRLoss

torch.set_float32_matmul_precision("medium")

CLASS_COLOR_MAP = {
    "background": "#1f77b4",  # Blue
    "glitch": "#d62728",      # Red
    "signal": "#2ca02c",      # Green
    "blip": "#9467bd",        # Purple
}

DETECTOR_COLOR_MAP = {
    "H1": "#ff7f0e",  # Orange
    "L1": "#2ca02c",  # Green
}


# =====================================================================
# 1. CORNER PLOT GENERATION
# =====================================================================
@torch.no_grad()
def generate_corner_plots(pl_module, data_loader, label_names, detector_names, out_dir="plots_cmamba_model"):
    """Generates corner pairplots (sns.pairplot with corner=True) for both

    the Strain Embedding/Projection space and the Witness Embedding/Projection space.
    """
    print("\nGenerating corner plots for projection/embedding spaces...")
    os.makedirs(out_dir, exist_ok=True)
    pl_module.eval()
    
    # Extract the device directly from the model parameters to avoid PL device resolution bugs
    device = next(pl_module.parameters()).device

    strain_z_list, witness_z_list = [], []
    types_list, detectors_list = [], []

    for batch in data_loader:
        x, y_type, y_det, _is_special = batch
        strain = x[:, :, 0:1].to(device)
        witness = x[:, :, 1:].to(device)

        _logits, strain_z, witness_z = pl_module.model(strain, witness)

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

    # --- 1. Strain Embedding Space Corner Plot ---
    strain_df = pd.DataFrame(
        strain_zs[:, :dim_limit],
        columns=[f"Strain Dim {i+1}" for i in range(dim_limit)]
    )
    strain_df["Event Type"] = [label_names[int(t)] for t in y_types]

    g_strain = sns.pairplot(
        strain_df,
        hue="Event Type",
        corner=True,
        palette=CLASS_COLOR_MAP,
        plot_kws={"alpha": 0.5, "s": 12},
        diag_kws={"fill": True, "common_norm": False}
    )
    g_strain.fig.suptitle("Strain Embedding Space (Corner Plot by Event Type)", y=1.02, fontsize=14)
    strain_path = os.path.join(out_dir, "strain_embedding_corner.png")
    g_strain.savefig(strain_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Strain Corner Plot -> {strain_path}")

    # --- 2. Witness Embedding Space Corner Plot ---
    witness_df = pd.DataFrame(
        witness_zs[:, :dim_limit],
        columns=[f"Witness Dim {i+1}" for i in range(dim_limit)]
    )
    witness_df["Detector"] = [detector_names[int(d)] for d in y_dets]

    g_witness = sns.pairplot(
        witness_df,
        hue="Detector",
        corner=True,
        palette=DETECTOR_COLOR_MAP,
        plot_kws={"alpha": 0.5, "s": 12},
        diag_kws={"fill": True, "common_norm": False}
    )
    g_witness.fig.suptitle("Witness Embedding Space (Corner Plot by Detector)", y=1.02, fontsize=14)
    witness_path = os.path.join(out_dir, "witness_embedding_corner.png")
    g_witness.savefig(witness_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Witness Corner Plot -> {witness_path}")

# =====================================================================
# 2. LIGHTNING MODULE WITH SPECIAL EVENT TRACKING
# =====================================================================
class CMambaLitModule(L.LightningModule):
    def __init__(self, model, lr=1e-3, temperature=0.1,
                 supcon_s_weight=1.0, ce_weight=1.0, supcon_w_weight=1.0,
                 zero_witness=False):
        """zero_witness: if True, witness is replaced with zeros before every
        forward pass (train/val/test alike), so the model can only use the
        strain channel. Used to build a "strain-only" ablation baseline
        without changing the architecture (witness_enc/GDD-MLP/witness_proj
        still run, just on constant input)."""
        super().__init__()
        self.model = model
        self.lr = lr
        self.supcon_s_weight = supcon_s_weight
        self.ce_weight = ce_weight
        self.supcon_w_weight = supcon_w_weight
        self.zero_witness = zero_witness
        self.supcon_strain = SupervisedSimCLRLoss(temperature=temperature)
        self.supcon_witness = SupervisedSimCLRLoss(temperature=temperature)

    def _step(self, batch, stage):
        x, y_type, y_det, is_special = batch
        strain, witness = x[:, :, 0:1], x[:, :, 1:]
        if self.zero_witness:
            witness = torch.zeros_like(witness)

        logits, strain_z, witness_z = self.model(strain, witness)

        ce_loss = F.cross_entropy(logits, y_type)

        strain_feats = F.normalize(strain_z, dim=1).unsqueeze(1)
        supcon_s_loss = self.supcon_strain(strain_feats, y_type)

        witness_feats = F.normalize(witness_z, dim=1).unsqueeze(1)
        supcon_w_loss = self.supcon_witness(witness_feats, y_det)

        loss = (
            self.supcon_s_weight * supcon_s_loss
            + self.ce_weight * ce_loss
            + self.supcon_w_weight * supcon_w_loss
        )

        preds = logits.argmax(dim=1)
        acc = (preds == y_type).float().mean()

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}_ce_loss", ce_loss, on_epoch=True)
        self.log(f"{stage}_supcon_s_loss", supcon_s_loss, on_epoch=True)
        self.log(f"{stage}_supcon_w_loss", supcon_w_loss, on_epoch=True)
        self.log(f"{stage}_acc", acc, on_epoch=True, prog_bar=True)

        if stage == "test" and bool(is_special.any()):
            self._track_special_candidate_event(preds, y_type, is_special, logits)

        return loss

    def _track_special_candidate_event(self, preds, y_type, is_special, logits):
        """Track candidate event UTC 2019-05-05 15:10:38 (loudest non-signal/glitch on record).

        Explicitly verify that it is NOT misclassified as a 'signal' (class 2).
        """
        label_names = {0: "background", 1: "glitch", 2: "signal", 3: "blip"}
        idx = torch.where(is_special)[0]

        for i in idx.tolist():
            pred_class = int(preds[i])
            true_class = int(y_type[i])
            pred_name = label_names.get(pred_class, str(pred_class))
            true_name = label_names.get(true_class, str(true_class))

            probs = F.softmax(logits[i], dim=0)
            signal_prob = float(probs[2]) * 100.0

            print("\n" + "=" * 65)
            print("[SPECIAL EVENT EVALUATION] UTC 2019-05-05 15:10:38")
            print(f"  * Ground Truth Class : {true_name} (id={true_class})")
            print(f"  * Predicted Class    : {pred_name} (id={pred_class})")
            print(f"  * Signal Probability : {signal_prob:.2f}%")

            # Check if model correctly rejects this loud transient as a non-signal
            if pred_class == 2:
                print("  ? WARNING: Loudest candidate event WAS FALSELY PREDICTED AS A SIGNAL!")
            elif pred_class == 0:
                print("  ? SUCCESS: Loudest candidate event was predicted as BACKGROUND.")
            else:
                print(f"  ? SUCCESS: Loudest candidate event was correctly identified as non-signal ({pred_name.upper()}).")
            print("=" * 65 + "\n")

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
    # Lower batch size to prevent OOM with union witness channels
    BATCH_SIZE = 8
    train_loader, val_loader, test_loader, meta = get_dataloaders(batch_size=BATCH_SIZE)

    sample_batch = next(iter(train_loader))
    x_sample, _, _, _ = sample_batch
    n_wit = x_sample.shape[-1] - 1  # Evaluates to number of witness channels

    print(f"\n--- INITIALIZING cmamba_model (n_wit={n_wit}) ---")
    model = CMambaModel(
        d_model=16,
        e_layers=3,
        d_state=8,
        expand=1,
        reduction=4,
        proj_dim=8,
        pool_method="mean",
        n_wit=n_wit,
        n_classes=meta["num_classes"],
        classify_from_projection=True,
        channel_mixup=False,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,}")

    lit_model = CMambaLitModule(
        model, lr=1e-3, temperature=0.1,
        supcon_s_weight=1.0, ce_weight=1.0, supcon_w_weight=1.0,
    )

    out_dir = "checkpoints_cmamba_model"
    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir, monitor="val_acc", mode="max",
        save_top_k=1, filename="{epoch:02d}-{val_acc:.4f}",
    )
    callbacks = [ckpt_cb, EarlyStopping(monitor="val_acc", mode="max", patience=12)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(
        max_epochs=30,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=4,
        callbacks=callbacks,
        logger=WandbLogger(project="ligo-cmamba-model", name="cmamba_model"),
    )

    trainer.fit(lit_model, train_loader, val_loader)

    print(f"\nLoading best checkpoint (by val_acc): {ckpt_cb.best_model_path}")
    best_lit_model = CMambaLitModule.load_from_checkpoint(
        ckpt_cb.best_model_path, model=model
    ).to(device)

    # Run testing on test set containing the held-out special event
    trainer.test(best_lit_model, test_loader)

    # Generate seaborn corner plots
    generate_corner_plots(best_lit_model, test_loader, meta["label_names"], meta["detector_names"])
    print("\nTraining, special event evaluation, and corner plot generation complete!")


if __name__ == "__main__":
    main()