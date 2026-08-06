"""Witness x bidirectional ablation on the current SupCon+CE model.

Trains four MambaFusionSupConModel variants -- identical data splits and
hyperparameters, differing ONLY in (use_witness, bidirectional):
  - with_witness_unidir
  - with_witness_bidir
  - without_witness_unidir
  - without_witness_bidir

For each variant, the checkpoint with the BEST val_acc (not the last epoch)
is reloaded before evaluation, so the reported accuracy/confusion matrix
reflect the best epoch rather than wherever training happened to end up.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from compact_loader import get_dataloaders
from model import MambaFusionSupConModel
from train import DualSupConCELitModule

torch.set_float32_matmul_precision("medium")

EPOCHS = 30
LR = 1e-3
TEMPERATURE = 0.1
SUPCON_WEIGHT = 0
CE_WEIGHT = 1.0
AUX_WEIGHT = 0.3

OUT_DIR = "ablation_witness_results"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_confusion_matrix(cm, label_names, out_path, title):
    norm = np.divide(
        cm,
        np.where(cm.sum(axis=1, keepdims=True) == 0, 1, cm.sum(axis=1, keepdims=True)),
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    labels_list = list(label_names.values())
    ax.set_xticks(range(len(label_names)), labels_list, fontsize=11)
    ax.set_yticks(range(len(label_names)), labels_list, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title(title, fontsize=13, pad=12)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Row-normalized fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Confusion Matrix -> {out_path}")


@torch.no_grad()
def evaluate(lit_model, loader, device, n_classes):
    lit_model.eval()
    model = lit_model.model
    all_true, all_pred = [], []

    for batch in loader:
        x, y_type, _y_det = batch
        x = x.to(device)
        strain, witness = x[:, :, 0:1], x[:, :, 1:]
        logits, _aux_logits, _embed = model.classify(strain, witness if model.use_witness else None)
        preds = logits.argmax(dim=1).cpu()
        all_true.append(y_type)
        all_pred.append(preds)

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    acc = float((y_true == y_pred).mean() * 100)
    return acc, cm


def run_variant(use_witness, bidirectional, meta, train_loader, val_loader, test_loader, n_wit):
    witness_tag = "with_witness" if use_witness else "without_witness"
    dir_tag = "bidir" if bidirectional else "unidir"
    tag = f"{witness_tag}_{dir_tag}"
    print(f"\n{'='*60}\nTRAINING VARIANT: {tag.upper()}\n{'='*60}")

    model = MambaFusionSupConModel(
        d_model=32,
        mamba_layers=4,
        proj_dim=8,
        pool_method="mean",
        n_wit=n_wit,
        n_classes=meta["num_classes"],
        use_witness=use_witness,
        bidirectional=bidirectional,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] Total trainable parameters: {total_params:,}")

    lit_model = DualSupConCELitModule(
        model, lr=LR, temperature=TEMPERATURE,
        supcon_weight=SUPCON_WEIGHT, ce_weight=CE_WEIGHT, aux_weight=AUX_WEIGHT,
    )

    ckpt_dir = os.path.join(OUT_DIR, f"ckpt_{tag}")
    # Best-accuracy checkpoint, not best/lowest loss -- this is what gets
    # reloaded below before evaluation, so the reported numbers reflect the
    # best epoch rather than the last one.
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, monitor="val_acc", mode="max", save_top_k=1,
        filename="{epoch:02d}-{val_acc:.4f}",
    )

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[EarlyStopping(monitor="val_acc", mode="max", patience=12), ckpt_cb],
        logger=WandbLogger(project="ligo-witness-ablation", name=tag),
    )

    trainer.fit(lit_model, train_loader, val_loader)

    print(f"\n[{tag}] Loading best checkpoint (by val_acc): {ckpt_cb.best_model_path}")
    best = DualSupConCELitModule.load_from_checkpoint(
        ckpt_cb.best_model_path, model=model
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    acc, cm = evaluate(best, test_loader, best.device, meta["num_classes"])
    print(f"[{tag}] Test accuracy (best epoch): {acc:.2f}%")

    plot_confusion_matrix(
        cm, meta["label_names"],
        os.path.join(OUT_DIR, f"confusion_{tag}.png"),
        f"Confusion Matrix ({tag.replace('_', ' ').title()})",
    )

    return {"tag": tag, "use_witness": use_witness, "bidirectional": bidirectional,
             "acc": acc, "cm": cm, "params": total_params}


def main():
    train_loader, val_loader, test_loader, meta = get_dataloaders(batch_size=128)

    sample_batch = next(iter(train_loader))
    x_sample, _, _ = sample_batch
    n_wit = x_sample.shape[-1] - 1

    results = {}
    for use_witness, bidirectional in itertools.product([True, False], [False, True]):
        res = run_variant(use_witness, bidirectional, meta, train_loader, val_loader, test_loader, n_wit)
        results[res["tag"]] = res

    print("\n" + "=" * 60)
    print("WITNESS x DIRECTIONALITY ABLATION SUMMARY")
    print("=" * 60)
    for tag, res in results.items():
        print(f"  {tag:28s} acc={res['acc']:6.2f}%  params={res['params']:,}")

    best_tag = max(results, key=lambda t: results[t]["acc"])
    print(f"\nBest variant: {best_tag} ({results[best_tag]['acc']:.2f}%)")


if __name__ == "__main__":
    main()