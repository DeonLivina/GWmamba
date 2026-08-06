import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from compact_loader import get_dataloaders
from model import BaselineModel

torch.set_float32_matmul_precision("medium")


# =====================================================================
# LIGHTNING MODULE -- cross-entropy only, no SupCon/auxiliary heads
# =====================================================================
class BaselineLitModule(L.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def _step(self, batch, stage):
        x, y_type, _y_det = batch
        strain, witness = x[:, :, 0:1], x[:, :, 1:]

        logits = self.model(strain, witness if self.model.use_witness else None)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }


# =====================================================================
# MAIN SCRIPT
# =====================================================================
def main():
    train_loader, val_loader, test_loader, meta = get_dataloaders(batch_size=128)

    sample_batch = next(iter(train_loader))
    x_sample, _, _ = sample_batch
    n_wit = x_sample.shape[-1] - 1

    print(f"\n--- INITIALIZING baseline (n_wit={n_wit}) ---")
    model = BaselineModel(
        d_model=32,
        mamba_layers=2,
        n_wit=n_wit,
        n_classes=meta["num_classes"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,}")

    lit_model = BaselineLitModule(model, lr=1e-3)

    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints_baseline", monitor="val_acc", mode="max",
        save_top_k=1, filename="{epoch:02d}-{val_acc:.4f}",
    )
    callbacks = [ckpt_cb, EarlyStopping(monitor="val_acc", mode="max", patience=12)]

    trainer = L.Trainer(
        max_epochs=30,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=WandbLogger(project="ligo-baseline", name="baseline"),
    )

    trainer.fit(lit_model, train_loader, val_loader)

    print(f"\nLoading best checkpoint (by val_acc): {ckpt_cb.best_model_path}")
    best_lit_model = BaselineLitModule.load_from_checkpoint(
        ckpt_cb.best_model_path, model=model
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    trainer.test(best_lit_model, test_loader)
    print("\nTraining and testing complete!")


if __name__ == "__main__":
    main()
