#!/usr/bin/env python3
"""Extract & whiten real background windows from big_model's layout.

Background candidates come from full_data/<DETECTOR>/background_triggers.csv
(built by bg_triggers.py: 1s windows tiled into the gaps between raw Omicron
strain triggers). This script adds a second, wider guard on top of that: a
candidate is also dropped if its full whitening footprint (16s PSD context +
pad + kernel + pad) overlaps any glitch longer than MAX_GLITCH_DURATION from
the detector's raw Omicron strain triggers -- a long glitch sitting inside
the 16s PSD-estimation lookback would bias that window's PSD even if it's
nowhere near the 1s kernel itself.

NOTE: this guard is intentionally loose (MAX_GLITCH_DURATION is large). It
exists only to catch the worst case (e.g. a lock-loss transient sitting in
the PSD lookback) -- since background windows aren't selected or vetted using
the auxiliary witness channels here, moderate PSD contamination from an
ordinary glitch is an acceptable tradeoff against discarding a large chunk of
an otherwise-fine candidate pool. Tighten MAX_GLITCH_DURATION back down if
that assumption changes.

Unlike the old O3_data pipeline, this data isn't split into fixed calendar
days that line up across channels -- strain comes in one manifest of
4096s chunks, and each witness channel has its own, independently-sized
chunking (H1's 9 channels are chunked into 3 ~86400s pieces; L1's 1 channel
so far is chunked into 9 ~28800s "reduced" pieces). So instead of opening
one shared "day" of files, every channel (strain included) gets its own
manifest of (gps_start, gps_end, path) built by scanning data/, and each
candidate window looks up whichever chunk file actually covers it,
independently per channel. Selection runs over the entire surviving
candidate pool -- no day-based stratification (there's no day structure to
stratify over anymore) and no upfront cap (most candidates get discarded
downstream in whiten_block as bad strain/witness/bounds, so capping before
extraction just throws away good candidates before they get a chance).

Set DETECTOR below and rerun per detector.
"""

import glob
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from gwpy.timeseries import TimeSeries

from ml4gw.transforms import SpectralDensity, Whiten


# -----------------------
# Config
# -----------------------
DETECTOR = "H1"  # or "L1"

ROOT = Path(__file__).resolve().parent.parent

STRAIN_DIR = ROOT / "data" / "og"
WITNESS_DIR = ROOT / "data" / "witness_data"

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

OUTPUT_DIR = ROOT / "full_data" / DETECTOR
trigger_csv = OUTPUT_DIR / "background_triggers.csv"
output_file = OUTPUT_DIR / "whitened_background_full.h5"


# -----------------------
# Parameters
# -----------------------
window = 1.0

target_sample_rate = 4096

# --- whitening (matches dataset/configs/config_<DETECTOR>.yaml's `whiten:` block) ---
WHITEN_FFTLENGTH = 2.0
WHITEN_OVERLAP = None
WHITEN_AVERAGE = "median"
WHITEN_PSD_LENGTH = 16.0
WHITEN_FDURATION = 2.0
HIGHPASS = 20.0

pad_seconds = WHITEN_FDURATION / 2.0

# Glitches longer than this also guard the wider PSD-lookback footprint (see
# module docstring). Loosened on purpose -- only very long glitches (e.g. a
# lock-loss) get excluded now, since we don't select/vet backgrounds using the
# witness channels here and don't need a pristine PSD to that degree.
MAX_GLITCH_DURATION = 8.0   # seconds

# Cap on number of background samples to extract. None = run every surviving
# candidate (after the long-glitch guard) through extraction.
n_samples = None

random_seed = 42


# -----------------------
# Utilities
# -----------------------
def clean_non_numerical(data):
    data = np.asarray(data, dtype=np.float64)
    bad = ~np.isfinite(data)
    if np.any(bad):
        if np.all(bad):
            return np.zeros_like(data)
        x = np.arange(len(data))
        data[bad] = np.interp(x[bad], x[~bad], data[~bad])
    return data


def is_degenerate(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return True
    if not np.all(np.isfinite(x)):
        return True
    if np.ptp(x) == 0:
        return True
    if np.std(x) == 0:
        return True
    return False


def build_transforms(rate):
    spectral_density = SpectralDensity(
        sample_rate=rate,
        fftlength=WHITEN_FFTLENGTH,
        overlap=WHITEN_OVERLAP,
        average=WHITEN_AVERAGE,
    )
    whiten = Whiten(fduration=WHITEN_FDURATION, sample_rate=rate, highpass=HIGHPASS)
    return spectral_density, whiten


def whiten_block(block, rate, target_rate, transforms, psd_len_native, length):
    """Returns (result, reason). result is None on failure, and reason is a
    short string categorizing why -- used to break down bad_strain/bad_witness
    counts by cause instead of collapsing everything into one number."""
    block = np.asarray(block, dtype=np.float64)

    if len(block) == 0:
        return None, "empty"
    if not np.all(np.isfinite(block)):
        return None, "non_finite_input"
    if np.ptp(block) == 0:
        return None, "flat_ptp"
    if np.std(block) == 0:
        return None, "flat_std"

    try:
        spectral_density, whiten = transforms[rate]

        x = torch.from_numpy(block).double()
        psd_part = x[:psd_len_native].view(1, 1, -1)
        kernel_part = x[psd_len_native:].view(1, 1, -1)

        psd = spectral_density(psd_part)
        whitened = whiten(kernel_part, psd)
        whitened_np = whitened.view(-1).numpy()

        ts = TimeSeries(whitened_np, sample_rate=rate, dtype=np.float64)
        resampled = ts.resample(target_rate)
        y = np.asarray(resampled.value, dtype=np.float64)

        if len(y) != length:
            return None, f"wrong_length(got {len(y)}, want {length})"
        if not np.all(np.isfinite(y)):
            return None, "non_finite_output"

        return y, None

    except Exception as e:
        return None, f"exception:{type(e).__name__}:{e}"


def load_strain_manifest(strain_dir, detector):
    """(gps_start, gps_end, rate, path) for every strain chunk belonging to
    `detector`, independently inspecting each file's own Xspacing attribute."""
    manifest = []
    seen_starts = set()
    prefix = detector[0]  # 'H' or 'L'
    pattern = os.path.join(str(strain_dir), f"{prefix}-{detector}_*.hdf5")
    for path in sorted(glob.glob(pattern)):
        with h5py.File(path, "r") as f:
            gps_start = int(f["meta"]["GPSstart"][()])
            duration = int(f["meta"]["Duration"][()])
            rate = round(1.0 / f["strain"]["Strain"].attrs["Xspacing"])
        if gps_start in seen_starts:
            print(f"  WARNING: duplicate strain chunk at GPS {gps_start} ({path}), skipping")
            continue
        seen_starts.add(gps_start)
        manifest.append((gps_start, gps_start + duration, rate, path))
    return sorted(manifest)


def build_witness_manifest(witness_dir, detector, channel):
    """(gps_start, gps_end, rate, path) for every chunk of one witness
    channel, discovered and read independently from each file's own x0/dx attrs."""
    manifest = []
    pattern = os.path.join(str(witness_dir), f"{detector}_{channel}_*.hdf5")
    for path in sorted(glob.glob(pattern)):
        with h5py.File(path, "r") as f:
            key = f"{detector}:{channel}"
            dset = f[key]
            rate = round(1.0 / dset.attrs["dx"])
            x0 = float(dset.attrs["x0"])
            n = dset.shape[0]
        manifest.append((x0, x0 + n / rate, rate, path))
    if not manifest:
        raise RuntimeError(f"No witness chunks found for {detector}:{channel} (pattern {pattern})")
    return sorted(manifest)


def find_chunk(manifest, seg_start, seg_end):
    for gps_start, gps_end, rate, path in manifest:
        if gps_start <= seg_start and seg_end <= gps_end:
            return gps_start, rate, path
    return None


def diagnose_bounds_failure(manifest, seg_start, seg_end):
    """Only called when find_chunk already failed. Categorizes why: does the
    footprint span two (or more) adjacent manifest chunks (data exists, just
    split across files -- a stitching limitation, not a real gap), does it
    fall entirely before/after all chunks, or does it sit in a genuine gap
    between chunks with no coverage at all."""
    starts_in = [c for c in manifest if c[0] <= seg_start < c[1]]
    ends_in = [c for c in manifest if c[0] < seg_end <= c[1]]

    if starts_in and ends_in and starts_in[0][0] != ends_in[0][0]:
        return "spans_two_chunks"
    if not manifest:
        return "no_chunks_at_all"
    if seg_end <= manifest[0][0]:
        return "before_first_chunk"
    if seg_start >= manifest[-1][1]:
        return "after_last_chunk"
    if not starts_in and not ends_in:
        return "in_gap_between_chunks"
    return "other"


# -----------------------
# Load background candidates
# -----------------------
print(f"Loading background candidates for {DETECTOR}")
triggers = pd.read_csv(trigger_csv)
candidates = triggers["gps_start"].to_numpy()
print("Total candidate windows:", len(candidates))


# -----------------------
# Strain trigger table, for the long-glitch PSD-footprint exclusion
# -----------------------
def find_strain_trigger_csv(trigger_dir, detector):
    pattern = str(trigger_dir / f"{detector}_GDS-CALIB_STRAIN_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No strain trigger CSV found matching {pattern}")
    return matches[0]


strain_triggers = pd.read_csv(find_strain_trigger_csv(TRIGGER_DIR, DETECTOR))
glitch_tstart = strain_triggers["tstart"].to_numpy()
glitch_tend = strain_triggers["tend"].to_numpy()
glitch_duration = glitch_tend - glitch_tstart

long_mask = glitch_duration > MAX_GLITCH_DURATION
long_tstart = glitch_tstart[long_mask]
long_tend = glitch_tend[long_mask]
print(f"long glitches (> {MAX_GLITCH_DURATION}s):", len(long_tstart))

footprint_start = candidates - WHITEN_PSD_LENGTH - pad_seconds
footprint_end = candidates + window + pad_seconds

if len(long_tstart) > 0:
    overlaps_long_glitch = (
        (footprint_start[:, None] < long_tend[None, :])
        & (footprint_end[:, None] > long_tstart[None, :])
    ).any(axis=1)
else:
    overlaps_long_glitch = np.zeros(len(candidates), dtype=bool)

n_dropped_long = int(np.count_nonzero(overlaps_long_glitch))
candidates = candidates[~overlaps_long_glitch]
print(f"dropped (near a glitch > {MAX_GLITCH_DURATION}s):", n_dropped_long)

rng = np.random.default_rng(random_seed)
rng.shuffle(candidates)
if n_samples is not None:
    n_select = min(n_samples, len(candidates))
    candidates = candidates[:n_select]
    print(f"selected for extraction: {n_select} (target {n_samples})")
else:
    print(f"selected for extraction: {len(candidates)} (all surviving candidates)")


# -----------------------
# Manifests
# -----------------------
print("\nIndexing strain chunks")
strain_manifest = load_strain_manifest(STRAIN_DIR, DETECTOR)
print(f"Found {len(strain_manifest)} strain chunks")

print("Indexing witness chunks")
witness_manifests = {}
for channel in WITNESS_CHANNELS:
    witness_manifests[channel] = build_witness_manifest(WITNESS_DIR, DETECTOR, channel)
    print(f"  {channel}: {len(witness_manifests[channel])} chunks")


# -----------------------
# Dimensions & Rate Inspection Printout
# -----------------------
seq_len = int(window * target_sample_rate)

print("\n========================================")
print(f"INDIVIDUALLY READ NATIVE SAMPLE RATES ({DETECTOR})")
print("========================================")
# Print rates dynamically for each unique file/chunk found
for idx, (g_start, g_end, rate, path) in enumerate(strain_manifest):
    print(f"Strain Chunk {idx} ({Path(path).name}): {rate} Hz")
for channel in WITNESS_CHANNELS:
    for idx, (g_start, g_end, rate, path) in enumerate(witness_manifests[channel]):
        print(f"Witness [{channel}] Chunk {idx} ({Path(path).name}): {rate} Hz")
print("========================================\n")


# -----------------------
# Storage
# -----------------------
from collections import Counter

strain_out, witness_out, gps_out = [], [], []
bad_strain = 0
bad_witness = 0
bad_bounds = 0

bad_strain_reasons = Counter()
bad_witness_reasons = Counter()
bad_bounds_reasons = Counter()

channel_names = [f"{DETECTOR}:{channel}" for channel in WITNESS_CHANNELS]
transforms = {}
strain_file_cache = {}
witness_file_cache = {}

all_rates = {rate for _, _, rate, _ in strain_manifest}
for manifest in witness_manifests.values():
    all_rates |= {rate for _, _, rate, _ in manifest}
for rate in all_rates:
    transforms[rate] = build_transforms(rate)


# -----------------------
# Extraction
# -----------------------
for win_start in tqdm(candidates, desc=DETECTOR):
    segment_start = win_start - WHITEN_PSD_LENGTH - pad_seconds
    segment_end = win_start + window + pad_seconds

    chunk = find_chunk(strain_manifest, segment_start, segment_end)
    if chunk is None:
        bad_bounds += 1
        bad_bounds_reasons[diagnose_bounds_failure(strain_manifest, segment_start, segment_end)] += 1
        continue
    chunk_gps_start, s_rate, s_path = chunk

    if s_path not in strain_file_cache:
        strain_file_cache[s_path] = h5py.File(s_path, "r")
    s_dset = strain_file_cache[s_path]["strain"]["Strain"]

    s0 = int(round((segment_start - chunk_gps_start) * s_rate))
    s1 = int(round((segment_end - chunk_gps_start) * s_rate))

    raw = clean_non_numerical(s_dset[s0:s1])
    psd_len_native = int(WHITEN_PSD_LENGTH * s_rate)

    strain_sample, strain_reason = whiten_block(
        raw, s_rate, target_sample_rate, transforms, psd_len_native, seq_len
    )

    if strain_sample is None:
        bad_strain += 1
        bad_strain_reasons[strain_reason] += 1
        continue

    # -----------------------
    # Witness
    # -----------------------
    w_samples = []
    ok = True

    for channel in WITNESS_CHANNELS:
        w_chunk = find_chunk(witness_manifests[channel], segment_start, segment_end)
        if w_chunk is None:
            ok = False
            break
        w_gps_start, w_rate, w_path = w_chunk

        if w_path not in witness_file_cache:
            witness_file_cache[w_path] = h5py.File(w_path, "r")
        w_dset = witness_file_cache[w_path][f"{DETECTOR}:{channel}"]

        w0 = int(round((segment_start - w_gps_start) * w_rate))
        w1 = int(round((segment_end - w_gps_start) * w_rate))

        if w0 < 0 or w1 > w_dset.shape[0]:
            ok = False
            break

        raw_w = clean_non_numerical(w_dset[w0:w1])
        psd_len_native_w = int(WHITEN_PSD_LENGTH * w_rate)

        ws, w_reason = whiten_block(
            raw_w, w_rate, target_sample_rate, transforms, psd_len_native_w, seq_len
        )

        if ws is None:
            ok = False
            bad_witness_reasons[f"{channel}:{w_reason}"] += 1
            break

        w_samples.append(ws)

    if not ok:
        bad_witness += 1
        continue

    strain_out.append(strain_sample)
    witness_out.append(np.stack(w_samples))
    gps_out.append(win_start + window / 2)

for f in strain_file_cache.values():
    f.close()
for f in witness_file_cache.values():
    f.close()


# -----------------------
# Report
# -----------------------
print("\nFinished")
print("Valid samples:", len(gps_out))
print("Bad strain:", bad_strain)
print("Bad witness:", bad_witness)
print("Bad bounds:", bad_bounds)

if bad_strain_reasons:
    print("\nBad strain breakdown:")
    for reason, count in bad_strain_reasons.most_common():
        print(f"  {reason}: {count}")

if bad_witness_reasons:
    print("\nBad witness breakdown:")
    for reason, count in bad_witness_reasons.most_common():
        print(f"  {reason}: {count}")

if bad_bounds_reasons:
    print("\nBad bounds breakdown:")
    for reason, count in bad_bounds_reasons.most_common():
        print(f"  {reason}: {count}")

if n_samples is not None and len(gps_out) < n_samples:
    print("Warning: only", len(gps_out), "valid samples available (requested", n_samples, ")")

if len(gps_out) == 0:
    raise RuntimeError("No samples produced. Check GPS alignment or triggers.")


# -----------------------
# Save
# -----------------------
strain_out = np.asarray(strain_out, dtype=np.float32)
witness_out = np.asarray(witness_out, dtype=np.float32)
gps_out = np.asarray(gps_out, dtype=np.float64)

os.makedirs(OUTPUT_DIR, exist_ok=True)

with h5py.File(output_file, "w") as f:
    f.create_dataset("strain", data=strain_out, compression="gzip")
    f.create_dataset("witness", data=witness_out, compression="gzip")
    f.create_dataset("gps", data=gps_out)
    f.create_dataset("channel_names", data=np.array(channel_names, dtype="S"))
    f.create_dataset("detector", data=np.array([DETECTOR] * len(gps_out), dtype="S"))

print("Saved:", output_file)
print("strain shape:", strain_out.shape)
print("witness shape:", witness_out.shape)
