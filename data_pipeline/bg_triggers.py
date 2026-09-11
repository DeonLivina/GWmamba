#!/usr/bin/env python3
"""Adapted for big_model's layout: background candidate windows are tiled into
the gaps between raw Omicron strain triggers for one detector, then filtered
so that a surviving 1s window overlaps NO trigger at all


Set DETECTOR below and rerun per detector.
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


# Config

DETECTOR = "L1"  # or "H1"

ROOT = Path(__file__).resolve().parent.parent

#
TRIGGER_DIR = {
    "H1": ROOT / "triggers_H1",
    "L1": ROOT / "triggers_L1",
}[DETECTOR]

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

WITNESS_CHANNELS = {"H1": H1_WITNESS_CHANNELS, "L1": L1_WITNESS_CHANNELS}[DETECTOR]

window = 1.0                # seconds, background window length
BUFFER_SECONDS = 1.0        # seconds excluded on each side of every glitch's [tstart, tend]
MAX_GLITCH_DURATION = 1.0   # glitches longer than this block background extraction entirely

OUTPUT_DIR = ROOT / "full_data" / DETECTOR
output_csv = OUTPUT_DIR / "background_triggers.csv"


def find_trigger_csv(trigger_dir, detector, channel):
    pattern = str(trigger_dir / f"{detector}_{channel}_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No trigger CSV found matching {pattern}")
    return matches[0]


def build_overlap_index(tstart, tend):
    """Sorted-by-tstart trigger starts + a running max of tend"""
    order = np.argsort(tstart)
    tstart_sorted = tstart[order]
    running_max_tend = np.maximum.accumulate(tend[order])
    return tstart_sorted, running_max_tend


def overlaps_any(win_start, win_end, tstart_sorted, running_max_tend):
    """True where [win_start, win_end) overlaps at least one
    trigger interval. Correct because among all triggers with
    tstart < win_end, the largest tend is running_max_tend at the
    corresponding index -- if even that doesn't reach past win_start, none
    of them do."""
    hi = np.searchsorted(tstart_sorted, win_end, side="left")
    result = np.zeros(len(win_start), dtype=bool)
    has_candidate = hi > 0
    idx = np.clip(hi - 1, 0, len(tstart_sorted) - 1)
    result[has_candidate] = running_max_tend[idx[has_candidate]] > win_start[has_candidate]
    return result



# tile 1s windows into the gaps between consecutive strain triggers

strain_csv = find_trigger_csv(TRIGGER_DIR, DETECTOR, "GDS-CALIB_STRAIN")
print(f"{DETECTOR}: loading strain triggers from {strain_csv}")

triggers = pd.read_csv(strain_csv)
triggers = triggers.sort_values("tstart").reset_index(drop=True)

tstart = triggers["tstart"].to_numpy()
tend = triggers["tend"].to_numpy()
durations = tend - tstart

print(f"{DETECTOR}: {len(tstart)} strain triggers")

backgrounds = []
for i in range(len(tstart) - 1):
    if durations[i] > MAX_GLITCH_DURATION or durations[i + 1] > MAX_GLITCH_DURATION:
        continue

    gap_start = tend[i] + BUFFER_SECONDS
    gap_end = tstart[i + 1] - BUFFER_SECONDS
    gap_duration = gap_end - gap_start

    if gap_duration < window:
        continue

    n_windows = int(np.floor(gap_duration / window))
    for j in range(n_windows):
        gps_start = gap_start + j * window
        gps_end = gps_start + window
        gps_center = gps_start + window / 2.0
        backgrounds.append([gps_start, gps_end, gps_center])

backgrounds = pd.DataFrame(backgrounds, columns=["gps_start", "gps_end", "gps_center"])
print(f"\n{DETECTOR}: candidate windows after gap tiling: {len(backgrounds)}")


#  drop any window overlapping ANY strain trigger, or any auxiliary/witness channel's trigger

gps_start_arr = backgrounds["gps_start"].to_numpy()
gps_end_arr = backgrounds["gps_end"].to_numpy()
keep = np.ones(len(backgrounds), dtype=bool)

strain_index = build_overlap_index(tstart, tend)
strain_overlap = overlaps_any(gps_start_arr, gps_end_arr, *strain_index)
print(f"  dropped (overlaps a strain trigger not caught by gap tiling): {int(strain_overlap.sum())}")
keep &= ~strain_overlap

for channel in WITNESS_CHANNELS:
    w_path = find_trigger_csv(TRIGGER_DIR, DETECTOR, channel)
    w_trig = pd.read_csv(w_path, usecols=["tstart", "tend"])
    w_index = build_overlap_index(w_trig["tstart"].to_numpy(), w_trig["tend"].to_numpy())
    w_overlap = overlaps_any(gps_start_arr, gps_end_arr, *w_index)
    print(f"  dropped (overlaps a {channel} trigger): {int(w_overlap.sum())}")
    keep &= ~w_overlap

backgrounds = backgrounds[keep].reset_index(drop=True)

print(f"\n{DETECTOR}: total background windows clean in strain + all "
      f"{len(WITNESS_CHANNELS)} witness channels: {len(backgrounds)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
backgrounds.to_csv(output_csv, index=False)
print("\nSaved:", output_csv)
