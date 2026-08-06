#!/usr/bin/env python3
"""
Walks every raw Omicron strain trigger for one detector and, for each one,
checks whether at least one witness channel has a trigger within its
HVeto-optimized `twin` time window of its peak `time` (bidirectional).

Restricted strictly to your specified base witness channels for H1 and L1,
while pulling the corresponding HVeto 'twin' window values from the tuning records.
"""

import glob
import os
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIG
# ============================================================

DETECTOR = "L1"  # or "L1"

# SNR cut configuration (set to None or 0 to disable)
STRAIN_SNR_THRESHOLD = 7  # e.g., 6.0
WITNESS_SNR_THRESHOLD = 7 # e.g., 6.0

ROOT = Path(__file__).resolve().parent.parent

TRIGGER_DIR = {
    "H1": ROOT / "triggers_H1",
    "L1": ROOT / "triggers_L1",
}[DETECTOR]

# HVeto 'twin' time windows (in seconds) for your explicitly chosen channels:
WITNESS_TOLERANCES = {
    # H1 Channels
    "ISI-HAM4_BLND_GS13Z_IN1_DQ": 0.80,  # from H1 May 2 records
    "LSC-POP_A_LF_OUT_DQ": 0.10,         # consistent across H1/L1 logs
    "LSC-REFL_A_LF_OUT_DQ": 0.20,        # from H1 records
    "LSC-REFL_A_RF45_I_ERR_DQ": 0.80,    # from H1 May 4 records
    "LSC-REFL_A_RF9_Q_ERR_DQ": 0.10,     # from H1 May 2 records

    # L1 Channels
    "PEM-EY_VMON_ETMY_ESDPOWER24_DQ": 0.10, # from L1 records
    "ISI-HAM6_BLND_GS13RZ_IN1_DQ": 1.00,    # from L1 May 4 records
    "PEM-EY_ACC_BEAMTUBE_MAN_Y_DQ": 1.00,   # from L1 May 4 records
    "ASC-CHARD_P_OUT_DQ": 0.20,             # from L1 May 3 records
}

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

OUTPUT_DIR = ROOT / "full_data" / DETECTOR
OUT_CSV = OUTPUT_DIR / "strain_witness_coincidence.csv"


# ============================================================
# HELPERS
# ============================================================

def find_trigger_csv(trigger_dir, detector, channel):
    pattern = str(trigger_dir / f"{detector}_{channel}_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No trigger CSV found matching {pattern}")
    if len(matches) > 1:
        print(f"WARNING: multiple trigger CSVs match {pattern}, using {matches[0]}")
    return matches[0]


# ============================================================
# MAIN
# ============================================================

strain_csv = find_trigger_csv(TRIGGER_DIR, DETECTOR, "GDS-CALIB_STRAIN")
print(f"{DETECTOR}: loading strain triggers from {strain_csv}")

strain = pd.read_csv(strain_csv)

if STRAIN_SNR_THRESHOLD is not None:
    initial_len = len(strain)
    strain = strain[strain["snr"] >= STRAIN_SNR_THRESHOLD].reset_index(drop=True)
    print(f"{DETECTOR}: Applied strain SNR threshold >= {STRAIN_SNR_THRESHOLD}. Kept {len(strain)} / {initial_len} triggers.")
else:
    strain = strain.reset_index(drop=True)

strain["_row_id"] = strain.index
print(f"{DETECTOR}: {len(strain)} active strain triggers to evaluate")

matched_cols = []
left_sorted = strain[["_row_id", "time"]].sort_values("time")

for channel in WITNESS_CHANNELS:
    wtrig_path = find_trigger_csv(TRIGGER_DIR, DETECTOR, channel)
    
    # Load witness triggers, checking if SNR is available
    wcols = ["time", "snr"] if "snr" in pd.read_csv(wtrig_path, nrows=1).columns else ["time"]
    wtrig = pd.read_csv(wtrig_path, usecols=wcols)
    
    if WITNESS_SNR_THRESHOLD is not None and "snr" in wtrig.columns:
        wtrig = wtrig[wtrig["snr"] >= WITNESS_SNR_THRESHOLD]
    
    wtrig = wtrig.sort_values("time")
    wtrig["_witness_hit"] = wtrig["time"]

    # Pull the specific HVeto 'twin' window for this channel, default to 0.10s if missing
    tolerance = WITNESS_TOLERANCES.get(channel, 0.10)

    merged = pd.merge_asof(
        left_sorted, wtrig, on="time", direction="nearest",
        tolerance=tolerance,
    ).set_index("_row_id")

    col = f"witness_{channel}_matched"
    strain[col] = strain["_row_id"].map(merged["_witness_hit"].notna())
    matched_cols.append(col)

strain["witness_match_count"] = strain[matched_cols].sum(axis=1)
strain["witness_matched_channels"] = strain[matched_cols].apply(
    lambda row: ";".join(ch for ch, col in zip(WITNESS_CHANNELS, matched_cols) if row[col]),
    axis=1,
)

coincident = strain[strain["witness_match_count"] > 0].drop(columns=["_row_id"] + matched_cols)
print(f"{len(coincident)} / {len(strain)} have >=1 witness coincidence")

base_cols = ["time", "frequency", "tstart", "tend", "fstart", "fend", "snr", "q", "amplitude", "phase"]
out = coincident.sort_values("time")[base_cols + ["witness_match_count", "witness_matched_channels"]]

os.makedirs(OUTPUT_DIR, exist_ok=True)
out.to_csv(OUT_CSV, index=False)

print(f"\nWrote {len(out)} coincident strain triggers to {OUT_CSV}")
print("witness match count distribution (channels hit, out of "
      f"{len(WITNESS_CHANNELS)}):")
print(out["witness_match_count"].value_counts().sort_index())