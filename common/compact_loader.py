"""
Every sample carries two labels:
  y_type:     0=background, 1=glitch, 2=signal
  y_detector: 0=H1, 1=L1

Sampling strategy (L1 has fewer usable examples than H1 for every class,
since fewer witness channels/strain chunks have been downloaded for it so far):
  - background: exactly N_BG_PER_DETECTOR samples from EACH detector
                (H1 + L1 -> N_BG_PER_DETECTOR * 2 total)
  - glitch:     ALL available L1 glitches, backfilled with H1 glitches up to
                N_GLITCH_TOTAL
  - signal:     ALL available L1 signals, backfilled with H1 signals up to
                N_SIGNAL_TOTAL

Only witness channels present in BOTH detectors' files are kept, so H1 and L1
samples can be concatenated into one witness tensor.
"""

import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

# =====================================================
# Config
# =====================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "full_data"

DETECTORS = ["H1", "L1"]

FILES = {
    0: {d: str(DATA_DIR / d / "whitened_background_full.h5") for d in DETECTORS},
    1: {d: str(DATA_DIR / d / "whitened_glitches_full.h5") for d in DETECTORS},
    2: {d: str(DATA_DIR / d / "whitened_signals_full.h5") for d in DETECTORS},
}

LEAK_CSV = {d: str(DATA_DIR / d / "background_witness_leakage.csv") for d in DETECTORS}

LABEL_NAMES = {0: "background", 1: "glitch", 2: "signal"}
DETECTOR_LABELS = {d: i for i, d in enumerate(DETECTORS)}
DETECTOR_NAMES = {i: d for d, i in DETECTOR_LABELS.items()}

H1_WITNESS_CHANNELS = [
    "ISI-HAM4_BLND_GS13Z_IN1_DQ",
    "LSC-POP_A_LF_OUT_DQ",
    "LSC-REFL_A_LF_OUT_DQ",
    "LSC-REFL_A_RF45_I_ERR_DQ",
    "LSC-REFL_A_RF9_Q_ERR_DQ",
]

L1_WITNESS_CHANNELS = [
    "LSC-POP_A_LF_OUT_DQ",
    "PEM-EY_VMON_ETMY_ESDPOWER24_DQ",
    "ISI-HAM6_BLND_GS13RZ_IN1_DQ",
    "PEM-EY_ACC_BEAMTUBE_MAN_Y_DQ",
    "ASC-CHARD_P_OUT_DQ",
]

ACTIVE_WITNESS_ORDER = [c for c in H1_WITNESS_CHANNELS if c in L1_WITNESS_CHANNELS]
N_WITNESS = len(ACTIVE_WITNESS_ORDER)

N_BG_PER_DETECTOR = 1500
N_GLITCH_TOTAL = 3000
N_SIGNAL_TOTAL = 3000

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 2200/ 3000, 300 / 3000, 500 / 3000
BATCH_SIZE = 256
SEED = 512
CHANNELS_LAST = True
WINDOW = 1.0
NORMALIZE = False


def _strip_detector_prefix(name):
    return name.split(":", 1)[1] if ":" in name else name


def load_file(path, detector, active_order=ACTIVE_WITNESS_ORDER):
    with h5py.File(path, "r") as f:
        strain = f["strain"][:].astype(np.float32)
        witness = f["witness"][:].astype(np.float32)
        gps = f["gps"][:] if "gps" in f else None
        raw_names = [n.decode("utf-8") if isinstance(n, bytes) else str(n)
                     for n in f["channel_names"][:]]

    base_to_idx = {_strip_detector_prefix(n): i for i, n in enumerate(raw_names)}
    try:
        reorder = [base_to_idx[name] for name in active_order]
    except KeyError as e:
        raise KeyError(f"Missing channel {e} in {path}. Available: {raw_names}")

    witness = witness[:, reorder, :]

    if CHANNELS_LAST:
        x = np.concatenate(
            [strain[:, :, None], np.transpose(witness, (0, 2, 1))], axis=2
        )
    else:
        x = np.concatenate([strain[:, None, :], witness], axis=1)

    invalid_mask = np.isnan(x).any(axis=(1, 2)) | np.isinf(x).any(axis=(1, 2))
    strain_slice = x[:, :, 0] if CHANNELS_LAST else x[:, 0, :]
    invalid_mask |= (strain_slice == 0).all(axis=1)
    invalid_mask |= (np.abs(x) > 10000).any(axis=(1, 2))

    n_dropped = int(invalid_mask.sum())
    if n_dropped > 0:
        print(f"[Data Quality] {detector} {os.path.basename(path)}: dropping {n_dropped} corrupted rows")

    valid = ~invalid_mask
    x = x[valid]
    if gps is not None:
        gps = gps[valid]

    return x, gps


def _apply_leakage_filter(x, gps, detector):
    path = LEAK_CSV.get(detector)
    if path is None or not os.path.exists(path) or gps is None:
        return x, gps
    leak_df = pd.read_csv(path)
    contaminated = set(leak_df.loc[leak_df["witness_match_count"] > 0, "gps"])
    if not contaminated:
        return x, gps
    mask = np.array([g in contaminated for g in gps])
    n_leaks = int(mask.sum())
    if n_leaks > 0:
        print(f"[Witness Leakage] {detector}: dropping {n_leaks} contaminated background samples")
    keep = ~mask
    return x[keep], gps[keep]


def _exclude_used_for_signal(x, gps, signal_gps):
    if gps is None or signal_gps is None or len(signal_gps) == 0:
        return x, gps
    signal_gps = np.sort(signal_gps)
    idx = np.searchsorted(signal_gps, gps)
    idx_hi = np.clip(idx, 0, len(signal_gps) - 1)
    idx_lo = np.clip(idx - 1, 0, len(signal_gps) - 1)
    close = (
        (np.abs(signal_gps[idx_hi] - gps) < (WINDOW / 2.0))
        | (np.abs(signal_gps[idx_lo] - gps) < (WINDOW / 2.0))
    )
    keep = ~close
    return x[keep], (gps[keep] if gps is not None else None)


def _split_counts(n_total):
    n_train = int(round(n_total * TRAIN_FRAC))
    n_val = int(round(n_total * VAL_FRAC))
    n_test = n_total - n_train - n_val
    return n_train, n_val, n_test


def _gather_background(rng):
    xs, ys, ds = [], [], []
    for detector in DETECTORS:
        x, gps = load_file(FILES[0][detector], detector)
        x, gps = _apply_leakage_filter(x, gps, detector)

        signal_gps = None
        if os.path.exists(FILES[2][detector]):
            with h5py.File(FILES[2][detector], "r") as f:
                if "gps" in f:
                    signal_gps = f["gps"][:]
        x, gps = _exclude_used_for_signal(x, gps, signal_gps)

        if len(x) < N_BG_PER_DETECTOR:
            raise RuntimeError(
                f"background/{detector}: only {len(x)} valid samples remain, "
                f"need {N_BG_PER_DETECTOR}"
            )
        sel = rng.permutation(len(x))[:N_BG_PER_DETECTOR]
        xs.append(x[sel])
        ys.append(np.full(N_BG_PER_DETECTOR, 0, dtype=np.int64))
        ds.append(np.full(N_BG_PER_DETECTOR, DETECTOR_LABELS[detector], dtype=np.int64))

    print(f"background: {N_BG_PER_DETECTOR} from H1 + {N_BG_PER_DETECTOR} from L1 "
          f"= {N_BG_PER_DETECTOR * len(DETECTORS)}")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ds)


def _gather_all_then_backfill(label, target_total, rng):
    x_l1, _ = load_file(FILES[label]["L1"], "L1")
    x_h1, _ = load_file(FILES[label]["H1"], "H1")

    n_l1 = min(len(x_l1), target_total)
    n_h1_needed = target_total - n_l1

    if len(x_h1) < n_h1_needed:
        raise RuntimeError(
            f"{LABEL_NAMES[label]}: L1 provides {n_l1}, need {n_h1_needed} more from H1, "
            f"but H1 only has {len(x_h1)} valid samples"
        )

    sel_l1 = rng.permutation(len(x_l1))[:n_l1]
    sel_h1 = rng.permutation(len(x_h1))[:n_h1_needed]

    x = np.concatenate([x_l1[sel_l1], x_h1[sel_h1]], axis=0)
    y = np.full(target_total, label, dtype=np.int64)
    d = np.concatenate([
        np.full(n_l1, DETECTOR_LABELS["L1"], dtype=np.int64),
        np.full(n_h1_needed, DETECTOR_LABELS["H1"], dtype=np.int64),
    ])
    print(f"{LABEL_NAMES[label]}: {n_l1} from L1 + {n_h1_needed} from H1 = {target_total}")
    return x, y, d


def get_dataloaders(batch_size=BATCH_SIZE, seed=SEED, normalize=NORMALIZE):
    rng = np.random.default_rng(seed)

    x_bg, y_bg, d_bg = _gather_background(rng)
    x_gl, y_gl, d_gl = _gather_all_then_backfill(1, N_GLITCH_TOTAL, rng)
    x_sg, y_sg, d_sg = _gather_all_then_backfill(2, N_SIGNAL_TOTAL, rng)

    train_x, train_yt, train_yd = [], [], []
    val_x, val_yt, val_yd = [], [], []
    test_x, test_yt, test_yd = [], [], []

    for x, y_type, y_det in [(x_bg, y_bg, d_bg), (x_gl, y_gl, d_gl), (x_sg, y_sg, d_sg)]:
        n_total = len(x)
        perm = rng.permutation(n_total)
        x, y_type, y_det = x[perm], y_type[perm], y_det[perm]

        n_train, n_val, n_test = _split_counts(n_total)

        train_x.append(x[:n_train])
        train_yt.append(y_type[:n_train])
        train_yd.append(y_det[:n_train])

        val_x.append(x[n_train:n_train + n_val])
        val_yt.append(y_type[n_train:n_train + n_val])
        val_yd.append(y_det[n_train:n_train + n_val])

        test_x.append(x[n_train + n_val:n_train + n_val + n_test])
        test_yt.append(y_type[n_train + n_val:n_train + n_val + n_test])
        test_yd.append(y_det[n_train + n_val:n_train + n_val + n_test])

    Xtr = np.concatenate(train_x); Ytr_type = np.concatenate(train_yt); Ytr_det = np.concatenate(train_yd)
    Xva = np.concatenate(val_x); Yva_type = np.concatenate(val_yt); Yva_det = np.concatenate(val_yd)
    Xte = np.concatenate(test_x); Yte_type = np.concatenate(test_yt); Yte_det = np.concatenate(test_yd)

    if normalize:
        ch_axis = 2 if CHANNELS_LAST else 1
        reduce_axes = tuple(a for a in range(Xtr.ndim) if a != ch_axis)
        mean = Xtr.mean(axis=reduce_axes, keepdims=True)
        std = Xtr.std(axis=reduce_axes, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        Xtr = (Xtr - mean) / std
        Xva = (Xva - mean) / std
        Xte = (Xte - mean) / std

    def _ds(X, Yt, Yd):
        return TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X)),
            torch.from_numpy(np.ascontiguousarray(Yt)),
            torch.from_numpy(np.ascontiguousarray(Yd)),
        )

    meta = {
        "num_classes": len(LABEL_NAMES),
        "channel_names": ACTIVE_WITNESS_ORDER,
        "n_witness": N_WITNESS,
        "channels_last": CHANNELS_LAST,
        "label_names": LABEL_NAMES,
        "detector_names": DETECTOR_NAMES,
    }

    # ---- Balanced Sampling for Training Set ----
    class_counts = np.bincount(Ytr_type)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[Ytr_type]
    
    train_sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).type(torch.FloatTensor),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        _ds(Xtr, Ytr_type, Ytr_det), 
        batch_size=batch_size,
        sampler=train_sampler, 
        pin_memory=True
    )
    val_loader = DataLoader(
        _ds(Xva, Yva_type, Yva_det), 
        batch_size=batch_size,
        shuffle=False, 
        pin_memory=True
    )
    test_loader = DataLoader(
        _ds(Xte, Yte_type, Yte_det), 
        batch_size=batch_size,
        shuffle=False, 
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, meta
