#!/usr/bin/env python3
"""
Sanity/leakage check for the "background" class: extract_background.py and
bg_triggers.py only ever guard background windows against STRAIN triggers
(bg_triggers.py's BUFFER_SECONDS gap-shrinking, extract_background.py's own
long-glitch PSD-footprint exclusion) -- neither ever checks whether a
background window's actual 1s kernel overlaps a WITNESS-only trigger (an
environmental disturbance that never got flagged in strain, so nothing
upstream would have excluded it).

This loads every saved window from full_data/<DETECTOR>/whitened_background_full.h5
(using its `gps` dataset -- window CENTERS, per extract_background.py's
`gps_out.append(win_start + window / 2)`) and checks each one's
[gps-0.5, gps+0.5] kernel for interval overlap against every witness
channel's raw Omicron trigger file ([tstart, tend] spans, not just peak
`time`), so contamination can be measured directly instead of assumed away.

Unlike the old O3_data pipeline, trigger CSVs here span the whole downloaded
range in one file, so there is no per-day grouping.

Writes one row per background sample to OUT_CSV with witness_match_count
(out of len(WITNESS_CHANNELS)) and which channels matched.

Set DETECTOR below and rerun per detector.
"""

import glob
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

WINDOW = 1.0  # seconds -- must match extract_background.py

DETECTOR = "H1"  # or "L1"

ROOT = Path(__file__).resolve().parent.parent

TRIGGER_DIR = {
    "H1": ROOT / "triggers_H1",
    "L1": ROOT / "may_triggers_L1",
}[DETECTOR]

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
# arrive under may_triggers_L1/.
L1_WITNESS_CHANNELS = [
    "LSC-POP_A_LF_OUT_DQ",
]

WITNESS_CHANNELS = {"H1": H1_WITNESS_CHANNELS, "L1": L1_WITNESS_CHANNELS}[DETECTOR]

DATA_DIR = ROOT / "full_data" / DETECTOR
BACKGROUND_FILE = DATA_DIR / "whitened_background_full.h5"
OUT_CSV = DATA_DIR / "background_witness_leakage.csv"


def find_trigger_csv(trigger_dir, detector, channel):
    pattern = str(trigger_dir / f"{detector}_{channel}_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No trigger CSV found matching {pattern}")
    if len(matches) > 1:
        print(f"WARNING: multiple trigger CSVs match {pattern}, using {matches[0]}")
    return matches[0]


def load_trigger_spans(path):
    df = pd.read_csv(path, usecols=["tstart", "tend"])
    return df["tstart"].to_numpy(), df["tend"].to_numpy()


def main():
    print(f"Loading background samples from {BACKGROUND_FILE}")
    with h5py.File(BACKGROUND_FILE, "r") as f:
        gps = f["gps"][:]
    print(f"Background samples: {len(gps)}")

    channel_spans = {}
    for channel in WITNESS_CHANNELS:
        path = find_trigger_csv(TRIGGER_DIR, DETECTOR, channel)
        channel_spans[channel] = load_trigger_spans(path)

    results = []
    for g in gps:
        win_start = g - WINDOW / 2.0
        win_end = g + WINDOW / 2.0

        matched_channels = []
        for channel in WITNESS_CHANNELS:
            tstart, tend = channel_spans[channel]
            overlap = (tstart < win_end) & (tend > win_start)
            if overlap.any():
                matched_channels.append(channel)

        results.append({
            "gps": g,
            "witness_match_count": len(matched_channels),
            "witness_total": len(WITNESS_CHANNELS),
            "witness_matched_channels": ";".join(matched_channels),
        })

    out = pd.DataFrame(results).sort_values("gps")
    os.makedirs(DATA_DIR, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    n_total = len(out)
    n_clean = int((out["witness_match_count"] == 0).sum())
    n_contaminated = n_total - n_clean

    print(f"\nWrote {n_total} rows to {OUT_CSV}")
    print(f"Clean (no witness trigger overlap): {n_clean}/{n_total} ({n_clean/n_total:.1%})")
    print(f"Contaminated (>=1 witness trigger in window): {n_contaminated}/{n_total} "
          f"({n_contaminated/n_total:.1%})")
    print("\nwitness_match_count distribution (channels overlapping, out of "
          f"{len(WITNESS_CHANNELS)}):")
    print(out["witness_match_count"].value_counts().sort_index())


if __name__ == "__main__":
    main()
