"""LeJEPA fine-tuning: Stage 2 (freeze the pretrain.py backbone, train a
classifier head on its transient representation) and Stage 3 (evaluation +
8D corner-pairplot visualization). Stage 1 (self-supervised pretraining)
reuses LeJEPAPretrainModel/LeJEPALitModule from pretrain.py rather than
redefining them here -- main() below still runs all three stages
end-to-end, same as before, just without duplicating the pretrain model's
class definitions.
"""

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
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from compact_loader import get_dataloaders
from pretrain import LeJEPAPretrainModel, LeJEPALitModule

torch.set_float32_matmul_precision("medium")

CLASS_COLOR_MAP = {
    "background": "#1f77b4",  # Blue
    "glitch": "#d62728",      # Red
    "signal": "#2ca02c",      # Green
}


# =====================================================================
# STAGE 2: FINE-TUNING CLASSIFIER MODULES
# =====================================================================
class LeJEPAClassifier(nn.Module):
    def __init__(self, pretrain_model, n_classes=3, freeze_backbone=True):
        super().__init__()
        self.strain_enc = pretrain_model.strain_enc
        self.witness_enc = pretrain_model.witness_enc
        self.strain_proj = pretrain_model.strain_proj
        self.context_fusion = pretrain_model.context_fusion
        self.mamba_blocks = pretrain_model.mamba_blocks
        self.mamba_norms = pretrain_model.mamba_norms
        self.n_wit = pretrain_model.n_wit
        self.d_model = pretrain_model.d_model

        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def extract_latent_spaces(self, strain, witness):
        B, T_raw = strain.shape[0], strain.shape[1]

        h_strain = self.strain_enc(strain)
        B, T_down, _ = h_strain.shape

        wit_flat = witness.permute(0, 2, 1).reshape(B * self.n_wit, T_raw, 1)
        h_wit_flat = self.witness_enc(wit_flat)
        h_wit = h_wit_flat.view(B, self.n_wit, T_down, self.d_model)

        # Instrumental Representation
        inst_proj = self.strain_proj(h_strain)
        inst_rep = self.pool(inst_proj.permute(0, 2, 1)).squeeze(-1)

        # Transient Representation
        wit_flat_seq = h_wit.permute(0, 2, 1, 3).reshape(B, T_down, self.n_wit * self.d_model)
        fused = self.context_fusion(torch.cat([h_strain, wit_flat_seq], dim=-1))

        for norm, block in zip(self.mamba_norms, self.mamba_blocks):
            fused = fused + block(norm(fused))

        transient_rep = self.pool(fused.permute(0, 2, 1)).squeeze(-1)
        return inst_rep, transient_rep

    def forward(self, strain, witness):
        _, transient_rep = self.extract_latent_spaces(strain, witness)
        return self.classifier(transient_rep)


class FineTuneLitModule(L.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def _step(self, batch, stage):
        x, y_type, _ = batch
        strain, witness = x[:, :, 0:1], x[:, :, 1:]

        logits = self.model(strain, witness)
        loss = F.cross_entropy(logits, y_type)
        acc = (logits.argmax(dim=1) == y_type).float().mean()

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def test_step(self, batch, _):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


# =====================================================================
# STAGE 3: 8D CORNER PAIRPLOT EVALUATION
# =====================================================================
@torch.no_grad()
def evaluate_8d_pairplot(classifier_model, test_loader, meta, out_dir="plots_lejepa"):
    """Linear projection from 32D to 8D with Seaborn corner pairplot visualization."""
    print("\nGenerating 8D Linear Projection Corner Pairplots...")
    os.makedirs(out_dir, exist_ok=True)
    classifier_model.eval()
    device = next(classifier_model.parameters()).device

    proj_to_8d = nn.Linear(classifier_model.d_model, 8, bias=False).to(device)
    nn.init.orthogonal_(proj_to_8d.weight)

    inst_reps, transient_reps = [], []
    y_types, y_dets = [], []

    for batch in test_loader:
        x, y_type, y_det = batch
        strain, witness = x[:, :, 0:1].to(device), x[:, :, 1:].to(device)

        inst_z, trans_z = classifier_model.extract_latent_spaces(strain, witness)

        inst_reps.append(proj_to_8d(inst_z).cpu())
        transient_reps.append(proj_to_8d(trans_z).cpu())
        y_types.append(y_type.cpu())
        y_dets.append(y_det.cpu())

    inst_8d_all = torch.cat(inst_reps).numpy()
    trans_8d_all = torch.cat(transient_reps).numpy()
    y_types_all = torch.cat(y_types).numpy()
    y_dets_all = torch.cat(y_dets).numpy()

    cols = [f"z_{i+1}" for i in range(8)]

    # 1. Instrumental Space (Detector: H1 vs L1)
    df_inst = pd.DataFrame(inst_8d_all, columns=cols)
    df_inst["Detector"] = [meta["detector_names"][i] for i in y_dets_all]

    g1 = sns.pairplot(
        df_inst, hue="Detector", corner=True, diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 12, "edgecolor": "none"}
    )
    g1.fig.suptitle("8D Linear Projection: Instrumental Space (Detector Separation)", y=1.02)
    g1.savefig(os.path.join(out_dir, "instrumental_space_8d_pairplot.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Transient Space (Event Class: Signal / Glitch / Background)
    df_trans = pd.DataFrame(trans_8d_all, columns=cols)
    df_trans["Event Type"] = [meta["label_names"][i] for i in y_types_all]

    g2 = sns.pairplot(
        df_trans, hue="Event Type", corner=True, palette=CLASS_COLOR_MAP, diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 12, "edgecolor": "none"}
    )
    g2.fig.suptitle("8D Linear Projection: Transient Space (Morphological Classes)", y=1.02)
    g2.savefig(os.path.join(out_dir, "transient_space_8d_pairplot.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Corner pairplots successfully saved to '{out_dir}/'!")


# =====================================================================
# MAIN PIPELINE: Stage 1 (pretrain, via pretrain.py's classes) -> Stage 2
# (freeze + fine-tune classifier) -> Stage 3 (eval + pairplots)
# =====================================================================
def main():
    train_loader, val_loader, test_loader, meta = get_dataloaders(batch_size=32)
    sample_batch = next(iter(train_loader))
    n_wit = sample_batch[0].shape[-1] - 1

    # Stage 1: Pre-training
    print("\n=== STAGE 1: Starting Self-Supervised LeJEPA Pre-training ===")
    pretrain_base = LeJEPAPretrainModel(d_model=32, mamba_layers=4, proj_dim=32, n_wit=n_wit)
    pretrain_lit = LeJEPALitModule(pretrain_base, lr=1e-3, lambda_reg=0.1, aux_weight=0.5)

    ckpt_pretrain = ModelCheckpoint(
        dirpath="checkpoints_lejepa_pretrain", monitor="val_loss", mode="min", save_top_k=1,
        filename="pretrain-{epoch:02d}-{val_loss:.4f}"
    )

    trainer_pretrain = L.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=2,
        callbacks=[ckpt_pretrain, EarlyStopping(monitor="val_loss", mode="min", patience=10)],
        logger=WandbLogger(project="ligo-lejepa", name="stage1_pretrain"),
    )
    trainer_pretrain.fit(pretrain_lit, train_loader, val_loader)

    # Stage 2: Fine-Tuning
    print(f"\n=== STAGE 2: Loading Pre-trained Weights from {ckpt_pretrain.best_model_path} ===")
    best_pretrain_lit = LeJEPALitModule.load_from_checkpoint(ckpt_pretrain.best_model_path, model=pretrain_base)

    classifier_model = LeJEPAClassifier(best_pretrain_lit.model, n_classes=meta["num_classes"], freeze_backbone=True)
    finetune_lit = FineTuneLitModule(classifier_model, lr=1e-3)

    ckpt_finetune = ModelCheckpoint(
        dirpath="checkpoints_lejepa_finetune", monitor="val_acc", mode="max", save_top_k=1,
        filename="finetune-{epoch:02d}-{val_acc:.4f}"
    )

    trainer_finetune = L.Trainer(
        max_epochs=30,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        callbacks=[ckpt_finetune, EarlyStopping(monitor="val_acc", mode="max", patience=8)],
        logger=WandbLogger(project="ligo-lejepa", name="stage2_finetune"),
    )
    trainer_finetune.fit(finetune_lit, train_loader, val_loader)

    # Stage 3: Testing & Visualization
    print("\n=== STAGE 3: Final Evaluation & 8D Pairplot Visualizations ===")
    best_finetune_lit = FineTuneLitModule.load_from_checkpoint(
        ckpt_finetune.best_model_path, model=classifier_model
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    trainer_finetune.test(best_finetune_lit, test_loader)
    evaluate_8d_pairplot(best_finetune_lit.model, test_loader, meta)


if __name__ == "__main__":
    main()
