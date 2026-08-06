"""Same interface as before, pointed at big_model's layout: DATA_DIR is
full_data/<DETECTOR> (extract_background.py / extract_glitches.py /
inject_signal.py's output directory), and MASTER_CHANNEL_ORDER is derived
from the same per-detector witness channel list used by the rest of the
pipeline (9 channels for H1, 1 so far for L1 -- extend L1_WITNESS_CHANNELS
as more L1 witness data is downloaded).

Same defensive check as before: background samples that share a GPS window
with anything in the signal file are excluded. inject_signal.py already draws
from a disjoint candidate pool, so this should normally drop nothing -- it's
a guard against that guarantee silently breaking (e.g. re-running one script
without the other).

Set DETECTOR below and rerun per detector.
"""

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# =====================================================
# Config
# =====================================================
DETECTOR = "H1"  # or "L1"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "full_data" / DETECTOR

FILES = {
    0: str(DATA_DIR / "whitened_background_full.h5"),
    1: str(DATA_DIR / "whitened_glitches.h5"),
    2: str(DATA_DIR / "whitened_signals_full.h5"),
}

H1_WITNESS_CHANNELS = [
    "ASC-CHARD_P_OUT_DQ",
    "ASC-Y_TR_B_PIT_OUT_DQ",
    "ISI-HAM4_BLND_GS13Z_IN1_DQ",
    "LSC-POP_A_LF_OUT_DQ",
    "LSC-REFL_A_LF_OUT_DQ",
    "LSC-REFL_A_RF45_I_ERR_DQ",
    "LSC-REFL_A_RF9_Q_ERR_DQ",
    "PEM-CS_ACC_LVEAFLOOR_XCRYO_Z_DQ",
    "SUS-SR3_M3_OPLEV_PIT_OUT_DQ",
]

# Only 1 L1 aux channel has been downloaded so far -- add more here as they
# arrive under data/witness_data/.
L1_WITNESS_CHANNELS = [
    "LSC-POP_A_LF_OUT_DQ",
]

WITNESS_CHANNELS = {"H1": H1_WITNESS_CHANNELS, "L1": L1_WITNESS_CHANNELS}[DETECTOR]

# The physical order enforced for the model. Matches extract_glitches.py /
# extract_background.py / inject_signal.py's channel_names order.
MASTER_CHANNEL_ORDER = [f"{DETECTOR}:{channel}" for channel in WITNESS_CHANNELS]

LABEL_NAMES = {0: "background", 1: "glitch", 2: "signal"}
N_PER_CLASS, N_TRAIN, N_VAL, N_TEST = 1500, 1150, 50, 300
BATCH_SIZE = 256
SEED = 42
CHANNELS_LAST = True  # (N, L, C)
WINDOW = 1.0  # seconds -- must match the extraction/injection scripts


def load_file(path, label, master_order=MASTER_CHANNEL_ORDER):
    """Loads and permutes witness channels to match master_order."""
    with h5py.File(path, "r") as f:
        strain = f["strain"][:].astype(np.float32)
        witness = f["witness"][:].astype(np.float32)  # Expects (N, len(master_order), L)
        gps = f["gps"][:] if "gps" in f else None

        raw_names = [n.decode("utf-8") if isinstance(n, bytes) else str(n)
                     for n in f["channel_names"][:]]
        name_to_idx = {name: i for i, name in enumerate(raw_names)}

        try:
            reorder_indices = [name_to_idx[name] for name in master_order]
        except KeyError as e:
            raise KeyError(f"Missing channel {e} in file {path}. Available: {raw_names}")

    witness = witness[:, reorder_indices, :]

    if CHANNELS_LAST:
        # Result: (N, L, 1 + len(master_order))
        x = np.concatenate([strain[:, :, None], np.transpose(witness, (0, 2, 1))], axis=2)
    else:
        # Result: (N, 1 + len(master_order), L)
        x = np.concatenate([strain[:, None, :], witness], axis=1)

    y = np.full(x.shape[0], label, dtype=np.int64)
    return x, y, gps


def _exclude_used_for_signal(x, y, gps, signal_gps):
    """Drop background rows whose window center is within WINDOW/2 of a GPS
    used to build the signal set."""
    if gps is None or signal_gps is None or len(signal_gps) == 0:
        return x, y

    signal_gps = np.sort(signal_gps)
    idx = np.searchsorted(signal_gps, gps)
    idx_hi = np.clip(idx, 0, len(signal_gps) - 1)
    idx_lo = np.clip(idx - 1, 0, len(signal_gps) - 1)
    close = (
        (np.abs(signal_gps[idx_hi] - gps) < (WINDOW / 2.0))
        | (np.abs(signal_gps[idx_lo] - gps) < (WINDOW / 2.0))
    )

    n_dropped = int(close.sum())
    if n_dropped > 0:
        print(f"Excluding {n_dropped} background samples also used for signal injection")

    return x[~close], y[~close]


def get_dataloaders(batch_size=BATCH_SIZE, seed=SEED):
    rng = np.random.default_rng(seed)
    train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []

    signal_gps = None
    if os.path.exists(FILES[2]):
        with h5py.File(FILES[2], "r") as f:
            if "gps" in f:
                signal_gps = f["gps"][:]

    for label, path in FILES.items():
        x, y, gps = load_file(path, label)

        if label == 0:
            x, y = _exclude_used_for_signal(x, y, gps, signal_gps)

        if len(x) < N_PER_CLASS:
            raise RuntimeError(
                f"{LABEL_NAMES[label]}: only {len(x)} samples available in {FILES[label]}, "
                f"need N_PER_CLASS={N_PER_CLASS}"
            )

        sel = rng.permutation(len(x))[:N_PER_CLASS]
        x_sel, y_sel = x[sel], y[sel]

        train_x.append(x_sel[:N_TRAIN]); train_y.append(y_sel[:N_TRAIN])
        val_x.append(x_sel[N_TRAIN:N_TRAIN+N_VAL]); val_y.append(y_sel[N_TRAIN:N_TRAIN+N_VAL])
        test_x.append(x_sel[N_TRAIN+N_VAL:]); test_y.append(y_sel[N_TRAIN+N_VAL:])

    def _ds(xs, ys):
        X = np.concatenate(xs, axis=0)
        Y = np.concatenate(ys, axis=0)
        perm = rng.permutation(len(X))
        return TensorDataset(torch.from_numpy(X[perm]), torch.from_numpy(Y[perm]))

    meta = {"num_classes": len(LABEL_NAMES), "channel_names": MASTER_CHANNEL_ORDER}

    return (
        DataLoader(_ds(train_x, train_y), batch_size=batch_size, shuffle=True, pin_memory=True),
        DataLoader(_ds(val_x, val_y), batch_size=batch_size, shuffle=False, pin_memory=True),
        DataLoader(_ds(test_x, test_y), batch_size=batch_size, shuffle=False, pin_memory=True),
        meta
    )
