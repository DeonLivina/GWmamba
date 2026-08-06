#!/usr/bin/env python3
"""
For each glitch event in a GravitySpy-style catalog, check whether it has a
matching Omicron trigger in the detector's strain trigger file and in each
witness channel's trigger file.

*** NOT YET WIRED UP: big_model has no GravitySpy catalog CSV (the old
pipeline's H1_O3b.csv) for either detector. Point CATALOG_CSV at one with the
same columns (peak_time, peak_time_ns, ml_label, ml_confidence, duration,
snr) before running this -- everything else below (trigger paths, channel
list, detector) is already adapted to big_model's layout. If you don't have
a catalog and just want witness-coincident glitches regardless of
ml_label/ml_confidence, use strain_witness_coincidence.py instead -- it only
needs the Omicron trigger CSVs that are already downloaded. ***

Strain triggers were Omicron'd from the exact same frames the catalog's
peak_time came from, so a real match should land at the same timestamp --
STRAIN_TOLERANCE is 0 (exact match, no lag allowed). Witness channels
couple the transient with some delay/jitter, so WITNESS_TOLERANCE allows
some slack.

Writes one row per glitch to OUT_CSV with the strain match (+ its trigger
SNR and trigger tstart/tend), a per-channel witness match flag (+ its trigger
SNR and trigger tstart/tend), and a witness_match_count out of len(WITNESS_CHANNELS).

Set DETECTOR below and rerun per detector.
"""

import glob
import os
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIG
# ============================================================

DETECTOR = "H1"  # or "L1"

ROOT = Path(__file__).resolve().parent.parent

# Not present yet -- see module docstring.
CATALOG_CSV = ROOT / f"{DETECTOR}_O3.csv"

COL_TIME = "peak_time"
COL_LABEL = "ml_label"
COL_CONF = "ml_confidence"
COL_DURATION = "duration"
COL_SNR = "snr"

CONF_THRESHOLD = 0.60
MAX_DURATION = 1.0
STRAIN_TOLERANCE = 0.0     # seconds -- exact match only, no lag
WITNESS_TOLERANCE = 0.5    # seconds, match window

IGNORE_CLASSES = {
    "No_Glitch",
    "None_of_the_Above",
    "Blip",               # generally no witness information
    "Blip_Low_Frequency", # generally no witness information
    "Chirp",              # astrophysical
}

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

OUTPUT_DIR = ROOT / "full_data" / DETECTOR
OUT_CSV = OUTPUT_DIR / "glitch_witness_crosscheck.csv"

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


def load_trigger_table(path):
    return pd.read_csv(path, usecols=["time", "snr", "tstart", "tend"]).sort_values("time")


def match_nearest(events_df, trig_df, tolerance):
    """Nearest trigger row within `tolerance` of each events_df[COL_TIME].

    Returns (trigger_time, dt_ms, trigger_snr, trigger_tstart, trigger_tend)
    as Series index-aligned to events_df.
    """
    left = events_df[[COL_TIME]].reset_index().rename(columns={"index": "_row_id"})
    left = left.sort_values(COL_TIME)
    merged = pd.merge_asof(
        left, trig_df, left_on=COL_TIME, right_on="time",
        direction="nearest", tolerance=tolerance,
    )
    merged = merged.set_index("_row_id")
    dt_ms = (merged[COL_TIME] - merged["time"]).abs() * 1000.0
    return (
        merged["time"],
        dt_ms,
        merged["snr"],
        merged["tstart"],
        merged["tend"]
    )

# ============================================================
# LOAD + FILTER CATALOG
# ============================================================

if not CATALOG_CSV.exists():
    raise FileNotFoundError(
        f"{CATALOG_CSV} not found -- this script needs a GravitySpy-style catalog "
        "(columns: peak_time, peak_time_ns, ml_label, ml_confidence, duration, snr). "
        "See the module docstring; strain_witness_coincidence.py works without one."
    )

print("Loading catalog...")
df = pd.read_csv(CATALOG_CSV)

df = df[
    (df[COL_CONF] >= CONF_THRESHOLD) &
    (df[COL_DURATION] <= MAX_DURATION)
].copy()
df = df[~df[COL_LABEL].isin(IGNORE_CLASSES)]

# peak_time is whole GPS seconds; peak_time_ns is the sub-second remainder.
df[COL_TIME] = df[COL_TIME].astype(float) + df["peak_time_ns"].astype(float) * 1e-9

print(f"{len(df)} candidate glitches after filtering")

strain_trig = load_trigger_table(find_trigger_csv(TRIGGER_DIR, DETECTOR, "GDS-CALIB_STRAIN"))
trigger_time, dt_ms, trigger_snr, tstart, tend = match_nearest(df, strain_trig, STRAIN_TOLERANCE)

df["strain_trigger_time"] = trigger_time
df["strain_dt_ms"] = dt_ms
df["strain_trigger_snr"] = trigger_snr
df["strain_trigger_tstart"] = tstart
df["strain_trigger_tend"] = tend
df["strain_matched"] = df["strain_trigger_time"].notna()

matched_cols = []
for channel in WITNESS_CHANNELS:
    wtrig = load_trigger_table(find_trigger_csv(TRIGGER_DIR, DETECTOR, channel))
    w_trigger_time, _, w_trigger_snr, w_tstart, w_tend = match_nearest(df, wtrig, WITNESS_TOLERANCE)

    col = f"witness_{channel}_matched"
    df[col] = w_trigger_time.notna()
    df[f"witness_{channel}_snr"] = w_trigger_snr
    df[f"witness_{channel}_tstart"] = w_tstart
    df[f"witness_{channel}_tend"] = w_tend
    matched_cols.append(col)

df["witness_match_count"] = df[matched_cols].sum(axis=1)
df["witness_total"] = len(WITNESS_CHANNELS)
df["witness_matched_channels"] = df[matched_cols].apply(
    lambda row: ";".join(ch for ch, col in zip(WITNESS_CHANNELS, matched_cols) if row[col]),
    axis=1,
)

keep_cols = (
    [COL_TIME, COL_LABEL, COL_CONF, COL_DURATION, COL_SNR,
     "strain_matched", "strain_trigger_snr", "strain_trigger_tstart", "strain_trigger_tend",
     "witness_match_count", "witness_total", "witness_matched_channels"]
)
for channel in WITNESS_CHANNELS:
    keep_cols += [
        f"witness_{channel}_matched",
        f"witness_{channel}_snr",
        f"witness_{channel}_tstart",
        f"witness_{channel}_tend"
    ]

out = df.sort_values(COL_TIME)[keep_cols]

os.makedirs(OUTPUT_DIR, exist_ok=True)
out.to_csv(OUT_CSV, index=False)

print(f"\nWrote {len(out)} rows to {OUT_CSV}")
print(f"strain match rate: {out['strain_matched'].mean():.1%}")
print("witness match count distribution (channels hit, out of "
      f"{len(WITNESS_CHANNELS)}):")
print(out["witness_match_count"].value_counts().sort_index())
