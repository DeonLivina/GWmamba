"""Dataloader for the big_model 4-class (background/glitch/signal/blip) GW
model, combining H1 and L1 data with detector as a second label.

Every sample carries three labels/flags:
  y_type:     0=background, 1=glitch, 2=signal, 3=blip
  y_detector: 0=H1, 1=L1
  is_special: True iff this window matches the specially-tracked event below

Witness channels from both detectors are unified into a single fixed list of 
9 unique witness channels (10 total channels when combined with strain). 
Channels absent in a given detector's file are zero-padded (0.0).
"""

import os
from datetime import datetime, timezone
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
    3: {d: str(DATA_DIR / d / "whitened_gaussians.h5") for d in DETECTORS},
}

LEAK_CSV = {d: str(DATA_DIR / d / "background_witness_leakage.csv") for d in DETECTORS}

LABEL_NAMES = {0: "background", 1: "glitch", 2: "signal", 3: "blip"}
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

# Deduplicated union of all channels across both detectors (9 unique witness channels)
ALL_WITNESS_CHANNELS = list(dict.fromkeys(H1_WITNESS_CHANNELS + L1_WITNESS_CHANNELS))
N_WITNESS = len(ALL_WITNESS_CHANNELS)

N_BG_PER_DETECTOR = 1500
N_GLITCH_TOTAL = 3000
N_SIGNAL_TOTAL = 3000
N_BLIP_TOTAL = 3000

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 2200 / 3000, 300 / 3000, 500 / 3000
BATCH_SIZE = 256
SEED = 512
CHANNELS_LAST = True
WINDOW = 1.0
NORMALIZE = False


# -----------------------
# Temporal test holdout + special-event tracking
# -----------------------
def utc_to_gps(year, month, day, hour, minute, second, leap_offset=18):
    """Convert a UTC timestamp to GPS time. `leap_offset` is 18s for dates from 2017-01-01 onward."""
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    return (dt - gps_epoch).total_seconds() + leap_offset


TEST_TIME_CUTOFF_GPS = 1241092818

SPECIAL_EVENT_UTC = "2019-05-05T15:10:38Z"
SPECIAL_EVENT_GPS = utc_to_gps(2019, 5, 5, 15, 10, 38)
SPECIAL_EVENT_DESCRIPTION = "loudest non-signal (glitch) event on record"
SPECIAL_EVENT_TOL = WINDOW / 2.0


def _strip_detector_prefix(name):
    return name.split(":", 1)[1] if ":" in name else name


def load_file(path, detector, target_channels=ALL_WITNESS_CHANNELS):
    with h5py.File(path, "r") as f:
        strain = f["strain"][:].astype(np.float32)          # (N, T)
        raw_witness = f["witness"][:].astype(np.float32)    # (N, C_file, T)
        gps = f["gps"][:] if "gps" in f else None
        raw_names = [
            _strip_detector_prefix(n.decode("utf-8") if isinstance(n, bytes) else str(n))
            for n in f["channel_names"][:]
        ]

    base_to_idx = {name: i for i, name in enumerate(raw_names)}
    n_samples, _, t_steps = raw_witness.shape

    # Allocate zeroed matrix for all 9 union witness channels
    witness = np.zeros((n_samples, len(target_channels), t_steps), dtype=np.float32)

    # Populate channels present in file; missing channels naturally stay 0.0
    for idx, ch_name in enumerate(target_channels):
        if ch_name in base_to_idx:
            witness[:, idx, :] = raw_witness[:, base_to_idx[ch_name], :]

    # Concatenate Strain (1) + Union Witness (9) -> 10 Total Input Channels
    if CHANNELS_LAST:
        x = np.concatenate(
            [strain[:, :, None], np.transpose(witness, (0, 2, 1))], axis=2
        )  # (N, T, 10)
    else:
        x = np.concatenate([strain[:, None, :], witness], axis=1)  # (N, 10, T)

    # Data Quality Masking
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


def _load_file_if_exists(label, detector):
    path = FILES[label][detector]
    if not os.path.exists(path):
        print(f"[Warning] {LABEL_NAMES[label]} file missing for {detector} ({path}) "
              f"-- treating as 0 samples for this detector")
        return None, None
    return load_file(path, detector)


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


def _exclude_used_times(x, gps, other_gps, label):
    if gps is None or other_gps is None or len(other_gps) == 0:
        return x, gps
    other_gps = np.sort(other_gps)
    idx = np.searchsorted(other_gps, gps)
    idx_hi = np.clip(idx, 0, len(other_gps) - 1)
    idx_lo = np.clip(idx - 1, 0, len(other_gps) - 1)
    close = (
        (np.abs(other_gps[idx_hi] - gps) < (WINDOW / 2.0))
        | (np.abs(other_gps[idx_lo] - gps) < (WINDOW / 2.0))
    )
    n_dropped = int(close.sum())
    if n_dropped > 0:
        print(f"[Leakage vs {label}] dropping {n_dropped} background samples")
    keep = ~close
    return x[keep], (gps[keep] if gps is not None else None)


def _split_counts(n_total):
    n_train = int(round(n_total * TRAIN_FRAC))
    n_val = int(round(n_total * VAL_FRAC))
    n_test = n_total - n_train - n_val
    return n_train, n_val, n_test


def _gather_background(rng):
    xs, ys, ds, gs = [], [], [], []
    for detector in DETECTORS:
        x, gps = load_file(FILES[0][detector], detector)
        x, gps = _apply_leakage_filter(x, gps, detector)

        for other_label in (2, 3):
            other_path = FILES[other_label][detector]
            other_gps = None
            if os.path.exists(other_path):
                with h5py.File(other_path, "r") as f:
                    if "gps" in f:
                        other_gps = f["gps"][:]
            x, gps = _exclude_used_times(x, gps, other_gps, LABEL_NAMES[other_label])

        if len(x) < N_BG_PER_DETECTOR:
            raise RuntimeError(
                f"background/{detector}: only {len(x)} valid samples remain, "
                f"need {N_BG_PER_DETECTOR}"
            )
        sel = rng.permutation(len(x))[:N_BG_PER_DETECTOR]
        xs.append(x[sel])
        ys.append(np.full(N_BG_PER_DETECTOR, 0, dtype=np.int64))
        ds.append(np.full(N_BG_PER_DETECTOR, DETECTOR_LABELS[detector], dtype=np.int64))
        gs.append(gps[sel] if gps is not None else np.full(N_BG_PER_DETECTOR, np.nan))

    print(f"background: {N_BG_PER_DETECTOR} from H1 + {N_BG_PER_DETECTOR} from L1 "
          f"= {N_BG_PER_DETECTOR * len(DETECTORS)}")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ds), np.concatenate(gs)


def _gather_all_then_backfill(label, target_total, rng):
    x_l1, g_l1 = _load_file_if_exists(label, "L1")
    x_h1, g_h1 = _load_file_if_exists(label, "H1")

    if x_l1 is None and x_h1 is None:
        raise RuntimeError(f"No {LABEL_NAMES[label]} files found for either detector")
    if x_l1 is None:
        x_l1 = np.zeros((0,) + x_h1.shape[1:], dtype=x_h1.dtype)
        g_l1 = np.zeros((0,), dtype=np.float64)
    if x_h1 is None:
        x_h1 = np.zeros((0,) + x_l1.shape[1:], dtype=x_l1.dtype)
        g_h1 = np.zeros((0,), dtype=np.float64)

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
    g_l1_sel = g_l1[sel_l1] if g_l1 is not None else np.full(n_l1, np.nan)
    g_h1_sel = g_h1[sel_h1] if g_h1 is not None else np.full(n_h1_needed, np.nan)
    gps = np.concatenate([g_l1_sel, g_h1_sel])

    print(f"{LABEL_NAMES[label]}: {n_l1} from L1 + {n_h1_needed} from H1 = {target_total}")
    return x, y, d, gps


def _split_class_with_time_holdout(x, y_type, y_det, gps, rng):
    n_total = len(x)
    _, _, n_test = _split_counts(n_total)

    has_gps = gps is not None and not np.all(np.isnan(gps))
    if not has_gps:
        special_mask = np.zeros(n_total, dtype=bool)
        forced_mask = np.zeros(n_total, dtype=bool)
    else:
        special_mask = np.abs(gps - SPECIAL_EVENT_GPS) <= SPECIAL_EVENT_TOL
        forced_mask = (gps >= TEST_TIME_CUTOFF_GPS) | special_mask

    forced_idx = np.where(forced_mask)[0]
    free_idx = np.where(~forced_mask)[0]
    rng.shuffle(forced_idx)
    rng.shuffle(free_idx)

    n_forced = len(forced_idx)
    if n_forced >= n_test:
        test_idx = forced_idx
        remaining_pool = free_idx
    else:
        n_fill = n_test - n_forced
        fill_idx = free_idx[:n_fill]
        test_idx = np.concatenate([forced_idx, fill_idx])
        remaining_pool = free_idx[n_fill:]

    n_remaining = len(remaining_pool)
    train_ratio = TRAIN_FRAC / (TRAIN_FRAC + VAL_FRAC)
    n_train_actual = int(round(n_remaining * train_ratio))

    train_idx = remaining_pool[:n_train_actual]
    val_idx = remaining_pool[n_train_actual:]

    def _take(idx):
        return x[idx], y_type[idx], y_det[idx], special_mask[idx], (gps[idx] if has_gps else None)

    return _take(train_idx), _take(val_idx), _take(test_idx), n_test, n_forced


def get_dataloaders(batch_size=BATCH_SIZE, seed=SEED, normalize=NORMALIZE):
    rng = np.random.default_rng(seed)

    pools = [
        _gather_background(rng),
        _gather_all_then_backfill(1, N_GLITCH_TOTAL, rng),
        _gather_all_then_backfill(2, N_SIGNAL_TOTAL, rng),
        _gather_all_then_backfill(3, N_BLIP_TOTAL, rng),
    ]

    train_parts, val_parts, test_parts = [], [], []
    special_report = []

    for class_label, (x, y_type, y_det, gps) in enumerate(pools):
        train_pack, val_pack, test_pack, n_test_nominal, n_forced = \
            _split_class_with_time_holdout(x, y_type, y_det, gps, rng)

        if n_forced > n_test_nominal:
            print(f"[{LABEL_NAMES[class_label]}] {n_forced} samples at/after cutoff/special "
                  f"forced into test -- test set grown from {n_test_nominal} to {n_forced}")
        elif n_forced > 0:
            print(f"[{LABEL_NAMES[class_label]}] {n_forced} samples forced into test, "
                  f"backfilled with {n_test_nominal - n_forced} earlier samples")

        train_parts.append(train_pack)
        val_parts.append(val_pack)
        test_parts.append(test_pack)

        for pack, split_name in ((train_pack, "train"), (val_pack, "val"), (test_pack, "test")):
            _, p_yt, p_yd, p_special, p_gps = pack
            for i in np.where(p_special)[0]:
                special_report.append((split_name, class_label, int(p_yd[i]),
                                       None if p_gps is None else float(p_gps[i])))

    def _stack(parts, i):
        return np.concatenate([p[i] for p in parts])

    Xtr, Ytr_type, Ytr_det = _stack(train_parts, 0), _stack(train_parts, 1), _stack(train_parts, 2)
    Str_special = _stack(train_parts, 3)
    Xva, Yva_type, Yva_det = _stack(val_parts, 0), _stack(val_parts, 1), _stack(val_parts, 2)
    Sva_special = _stack(val_parts, 3)
    Xte, Yte_type, Yte_det = _stack(test_parts, 0), _stack(test_parts, 1), _stack(test_parts, 2)
    Ste_special = _stack(test_parts, 3)

    if special_report:
        print(f"\n[Special Event] '{SPECIAL_EVENT_DESCRIPTION}' "
              f"(UTC {SPECIAL_EVENT_UTC}, GPS {SPECIAL_EVENT_GPS:.0f}) matched:")
        for split_name, class_label, det_label, gps_val in special_report:
            gps_str = f"gps={gps_val:.3f}" if gps_val is not None else "gps=unknown"
            print(f"    split={split_name:5s}  class={LABEL_NAMES[class_label]:10s} "
                  f"detector={DETECTOR_NAMES[det_label]}  {gps_str}")
    else:
        print(f"\n[Special Event] '{SPECIAL_EVENT_DESCRIPTION}' "
              f"(UTC {SPECIAL_EVENT_UTC}, GPS {SPECIAL_EVENT_GPS:.0f}) was not found in any file")

    if normalize:
        ch_axis = 2 if CHANNELS_LAST else 1
        reduce_axes = tuple(a for a in range(Xtr.ndim) if a != ch_axis)
        mean = Xtr.mean(axis=reduce_axes, keepdims=True)
        std = Xtr.std(axis=reduce_axes, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        Xtr = (Xtr - mean) / std
        Xva = (Xva - mean) / std
        Xte = (Xte - mean) / std

    def _ds(X, Yt, Yd, Sp):
        return TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X)),
            torch.from_numpy(np.ascontiguousarray(Yt)),
            torch.from_numpy(np.ascontiguousarray(Yd)),
            torch.from_numpy(np.ascontiguousarray(Sp)),
        )

    meta = {
        "num_classes": len(LABEL_NAMES),
        "channel_names": ALL_WITNESS_CHANNELS,
        "n_witness": N_WITNESS,
        "channels_last": CHANNELS_LAST,
        "label_names": LABEL_NAMES,
        "detector_names": DETECTOR_NAMES,
        "test_time_cutoff_gps": TEST_TIME_CUTOFF_GPS,
        "special_event_utc": SPECIAL_EVENT_UTC,
        "special_event_gps": SPECIAL_EVENT_GPS,
        "special_event_description": SPECIAL_EVENT_DESCRIPTION,
    }

    # Class-Balanced Sampling for Training Set
    class_counts = np.bincount(Ytr_type, minlength=len(LABEL_NAMES))
    class_weights = 1.0 / np.where(class_counts == 0, 1, class_counts)
    sample_weights = class_weights[Ytr_type]

    train_sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).type(torch.FloatTensor),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        _ds(Xtr, Ytr_type, Ytr_det, Str_special),
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True
    )
    val_loader = DataLoader(
        _ds(Xva, Yva_type, Yva_det, Sva_special),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    test_loader = DataLoader(
        _ds(Xte, Yte_type, Yte_det, Ste_special),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, meta