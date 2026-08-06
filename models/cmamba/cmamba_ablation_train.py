"""cmamba_ablation_train: channel-mixing ablation for the mamba_ssm-backed
CMamba variant (cmamba_mambassm_model.py). Three cells, same architecture
family throughout, same train/val/test split for all of them (get_dataloaders
is called once in main() and the same three loaders are reused everywhere,
so any accuracy difference is attributable to the model, not the data):

  1. strain_only               -- plain per-channel Mamba (no GDD-MLP, no
                                   channel mixup), witness zeroed out before
                                   every forward pass. Tests whether the
                                   witness channels carry any signal at all.
  2. plain_mamba_with_witness  -- same architecture, real witness channels.
                                   Tests whether witness helps even without
                                   any cross-channel mixing mechanism.
  3. cmamba_full_with_witness  -- + GDD-MLP + channel mixup, with mixup
                                   extended to include strain
                                   (channel_mixup_include_strain=True) rather
                                   than the witness-only default in
                                   cmamba_mambassm_model.py. Tests the full
                                   channel-mixing mechanism.

To run the same ablation on the faithful hand-rolled backend instead, change
the import below from cmamba_mambassm_model to cmamba_model (constructor
args are compatible; the extra dt_rank/use_pscan/etc. knobs there just take
their defaults).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from special_loader import get_dataloaders
from cmamba_mambassm_model import CMambaModel
from cmamba_train import CMambaLitModule

torch.set_float32_matmul_precision("medium")

CKPT_DIR = Path("ckpt_ablation_cmamba_channel_mixing")


# =====================================================================
# EVALUATION & PLOTTING
# =====================================================================
def plot_confusion_matrix(cm, label_names, out_path, title):
    norm = np.divide(
        cm, np.where(cm.sum(axis=1, keepdims=True) == 0, 1, cm.sum(axis=1, keepdims=True))
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    labels_list = list(label_names.values())
    ax.set_xticks(range(len(label_names)), labels_list, fontsize=12)
    ax.set_yticks(range(len(label_names)), labels_list, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title(title, fontsize=14, pad=15)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color=color, fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Row-normalized fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Confusion Matrix -> {out_path}")


@torch.no_grad()
def evaluate_test_accuracy(pl_module, test_loader, label_names, zero_witness, device):
    pl_module.eval()
    n_classes = len(label_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for batch in test_loader:
        x, y_type, _y_det, _is_special = batch
        strain = x[:, :, 0:1].to(device)
        witness = x[:, :, 1:].to(device)
        if zero_witness:
            witness = torch.zeros_like(witness)

        logits, _strain_z, _witness_z = pl_module.model(strain, witness)
        preds = logits.argmax(dim=1).cpu().numpy()

        for t, p in zip(y_type.numpy(), preds):
            cm[t, p] += 1

    acc_pct = float(np.trace(cm)) / cm.sum() * 100.0
    return acc_pct, cm


def plot_accuracy_comparison(results, out_path="channel_mixing_ablation_accuracy.png"):
    tags = list(results.keys())
    accs = [results[t]["acc"] for t in tags]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar([t.replace("_", "\n") for t in tags], accs, color=colors[: len(tags)])
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{acc:.2f}%", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Channel-Mixing Ablation: Test Accuracy", fontsize=14, pad=15)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved Accuracy Comparison -> {out_path}")


# =====================================================================
# TRAINING RUNNER
# =====================================================================
def run_variant(run_tag, model_kwargs, zero_witness, epochs, meta,
                 train_loader, val_loader, test_loader):
    print(f"\n{'='*60}\nRUNNING VARIANT: {run_tag} (zero_witness={zero_witness})\n"
          f"model_kwargs={model_kwargs}\n{'='*60}")

    model = CMambaModel(n_wit=meta["n_witness"], n_classes=meta["num_classes"], **model_kwargs)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{run_tag}] Total Model Parameters: {total_params:,}")

    lit_model = CMambaLitModule(
        model, lr=1e-3, temperature=0.1,
        supcon_s_weight=1.0, ce_weight=1.0, supcon_w_weight=1.0,
        zero_witness=zero_witness,
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=CKPT_DIR, monitor="val_acc", mode="max",
        save_top_k=1, filename=f"{run_tag}-{{epoch:02d}}-{{val_acc:.4f}}",
    )
    callbacks = [ckpt_cb, EarlyStopping(monitor="val_acc", mode="max", patience=12)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        precision="16-mixed" if device == "cuda" else 32,
        accumulate_grad_batches=4,
        callbacks=callbacks,
        logger=WandbLogger(project="ligo-cmamba-channel-mixing-ablation", name=run_tag),
    )

    trainer.fit(lit_model, train_loader, val_loader)

    print(f"[{run_tag}] Loading best checkpoint (by val_acc): {ckpt_cb.best_model_path}")
    best_lit_model = CMambaLitModule.load_from_checkpoint(
        ckpt_cb.best_model_path, model=model, zero_witness=zero_witness,
    ).to(device)

    trainer.test(best_lit_model, test_loader)

    # mamba_ssm's CUDA kernel has no CPU fallback, and Trainer.test() can
    # silently move the module back to CPU once its loop finishes (same
    # issue before_abi.py works around) -- force it back onto the training
    # device before running our own eval loop below.
    best_lit_model = best_lit_model.to(device)

    acc_pct, cm = evaluate_test_accuracy(best_lit_model, test_loader, meta["label_names"], zero_witness, device)
    print(f"[{run_tag}] Test Accuracy (type): {acc_pct:.2f}%")

    plot_confusion_matrix(
        cm, meta["label_names"], f"confusion_{run_tag}.png",
        f"Confusion Matrix ({run_tag.replace('_', ' ').title()})",
    )

    return {"acc": acc_pct, "cm": cm, "best_ckpt": ckpt_cb.best_model_path}


# =====================================================================
# MAIN ABLATION LOOP
# =====================================================================
def main():
    epochs = 30
    BATCH_SIZE = 32
    train_loader, val_loader, test_loader, meta = get_dataloaders(batch_size=BATCH_SIZE)

    common = dict(
        d_model=32, e_layers=3, d_state=16, d_conv=4, expand=2,
        reduction=4, proj_dim=8, pool_method="mean",
        classify_from_projection=True,
    )

    variants = {
        "strain_only": dict(
            model_kwargs=dict(**common, use_gdd_mlp=False, channel_mixup=False),
            zero_witness=True,
        ),
        "plain_mamba_with_witness": dict(
            model_kwargs=dict(**common, use_gdd_mlp=False, channel_mixup=False),
            zero_witness=False,
        ),
        "cmamba_full_with_witness": dict(
            model_kwargs=dict(
                **common, use_gdd_mlp=True, channel_mixup=True,
                channel_mixup_sigma=1.0, channel_mixup_include_strain=True,
            ),
            zero_witness=False,
        ),
    }

    results = {}
    for run_tag, cfg in variants.items():
        results[run_tag] = run_variant(
            run_tag=run_tag,
            model_kwargs=cfg["model_kwargs"],
            zero_witness=cfg["zero_witness"],
            epochs=epochs,
            meta=meta,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )
        torch.cuda.empty_cache()

    plot_accuracy_comparison(results)

    print("\n" + "=" * 60)
    print("CHANNEL-MIXING ABLATION SUMMARY")
    print("=" * 60)
    for run_tag, res in results.items():
        print(f" -> {run_tag}: Test Accuracy = {res['acc']:.2f}%")


if __name__ == "__main__":
    main()
