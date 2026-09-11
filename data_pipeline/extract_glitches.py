#!/usr/bin/env python3
"""Extract & whiten real glitch windows from big_model's layout, prioritizing
low-frequency glitches sorted by witness match count.

Glitch candidates come from full_data/<DETECTOR>/strain_witness_coincidence.csv
(strain_witness_coincidence.py's output). Unlike the old O3_data pipeline,
there's no calendar-day structure to loop over here.

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


# Config

DETECTOR = "L1"  # or "L1"

ROOT = Path(__file__).resolve().parent.parent

STRAIN_DIR = ROOT / "data" / "og"
WITNESS_DIR = ROOT / "data" / "witness_data"

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
glitch_csv = OUTPUT_DIR / "strain_witness_coincidence.csv"
output_file_low = OUTPUT_DIR / "whitened_glitches.h5"
output_file_high = OUTPUT_DIR / "whitened_high_glitches.h5"


# Parameters

window = 1.0
target_sample_rate = 4096

WHITEN_FFTLENGTH = 2.0
WHITEN_OVERLAP = None
WHITEN_AVERAGE = "median"
WHITEN_PSD_LENGTH = 16.0
WHITEN_FDURATION = 2.0
HIGHPASS = 20.0

pad_seconds = WHITEN_FDURATION / 2.0
nyquist = target_sample_rate / 2.0

JITTER_SEED = 44
JITTER_MARGIN = 0.02

MAX_LOW_FREQ_SAMPLES = 10000   # cap for the number of glich samples required for computatioanl efficiencey

# Only keep glitches at or above this SNR. This value comes from Hveto logs
MIN_SNR = 7 


# Utilities

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


def whiten_block_native(block, rate, transforms, psd_len_native, length):
    """Whitens block at its native rate without resampling."""
    block = np.asarray(block, dtype=np.float64)

    if is_degenerate(block):
        return None

    try:
        spectral_density, whiten = transforms[rate]

        x = torch.from_numpy(block).double()
        psd_part = x[:psd_len_native].view(1, 1, -1)
        kernel_part = x[psd_len_native:].view(1, 1, -1)

        psd = spectral_density(psd_part)
        whitened = whiten(kernel_part, psd)
        y = whitened.view(-1).numpy()

        if len(y) != length:
            return None
        if not np.all(np.isfinite(y)):
            return None

        return y

    except Exception:
        return None


def whiten_block_resampled(block, rate, target_rate, transforms, psd_len_native, length):
    """Whitens block and resamples to target rate (used for auxiliary witness channels)."""
    block = np.asarray(block, dtype=np.float64)

    if is_degenerate(block):
        return None

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
            return None
        if not np.all(np.isfinite(y)):
            return None

        return y

    except Exception:
        return None


def load_strain_manifest(strain_dir, detector):
    """(gps_start, gps_end, rate, path) for every strain chunk belonging to
    `detector`. data/strain_data/ holds both H1 and L1 files together, so the
    glob is filtered by the GWOSC `<prefix>-<detector>_...` naming."""
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
    channel, discovered directly from each file's own x0/dx attrs rather than
    parsed from the filename (chunk sizes differ per detector/channel)."""
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


def get_strain_segment(manifest, file_cache, seg_start, seg_end, strain_rate):
    """Stitches a segment across strain chunk boundaries (zero-filling any gap)."""
    total_samples = int(round((seg_end - seg_start) * strain_rate))
    output = np.zeros(total_samples, dtype=np.float64)

    for gps_start, gps_end, rate, path in manifest:
        if gps_end <= seg_start:
            continue
        if gps_start >= seg_end:
            break

        overlap_start = max(seg_start, gps_start)
        overlap_end = min(seg_end, gps_end)

        if overlap_start < overlap_end:
            if path not in file_cache:
                file_cache[path] = h5py.File(path, "r")
            s_dset = file_cache[path]["strain"]["Strain"]

            s0 = int(round((overlap_start - gps_start) * rate))
            s1 = int(round((overlap_end - gps_start) * rate))

            out_s0 = int(round((overlap_start - seg_start) * rate))
            out_s1 = out_s0 + (s1 - s0)

            chunk_data = clean_non_numerical(s_dset[s0:s1])
            output[out_s0:out_s1] = chunk_data[:out_s1 - out_s0]

    return output, strain_rate


def save_h5(path, strain_list, witness_list, gps_list, freq_list, snr_list, channel_names):
    if len(gps_list) == 0:
        print("No samples for", path, "- skipping")
        return

    strain_arr = np.asarray(strain_list, dtype=np.float32)
    witness_arr = np.asarray(witness_list, dtype=np.float32)
    gps_arr = np.asarray(gps_list, dtype=np.float64)
    freq_arr = np.asarray(freq_list, dtype=np.float64)
    snr_arr = np.asarray(snr_list, dtype=np.float32)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset("strain", data=strain_arr, compression="gzip")
        f.create_dataset("witness", data=witness_arr, compression="gzip")
        f.create_dataset("gps", data=gps_arr)
        f.create_dataset("frequency", data=freq_arr)
        f.create_dataset("snr", data=snr_arr)
        f.create_dataset("channel_names", data=np.array(channel_names, dtype="S"))
        f.create_dataset("detector", data=np.array([DETECTOR] * len(gps_list), dtype="S"))

    print("Saved:", path)
    print("  strain shape:", strain_arr.shape)
    print("  witness shape:", witness_arr.shape)



# Load & Filter Glitches

print(f"Loading glitches for {DETECTOR}")
glitches = pd.read_csv(glitch_csv)
print("Total glitches loaded:", len(glitches))

if "snr" in glitches.columns:
    n_before = len(glitches)
    glitches = glitches[glitches["snr"] >= MIN_SNR]
    print(f"Dropped {n_before - len(glitches)} glitches below SNR {MIN_SNR}; {len(glitches)} remain")
else:
    print(f"  WARNING: no 'snr' column found -- skipping SNR >= {MIN_SNR} filter")

glitches["peak_time"] = (glitches["tend"] + glitches["tstart"]) / 2.0

freq_col = "frequency" if "frequency" in glitches.columns else "snr"

if freq_col in glitches.columns and "witness_match_count" in glitches.columns:
    low_freq_mask = glitches[freq_col] <= nyquist
    glitches_low_candidate = glitches[low_freq_mask].sort_values(by="witness_match_count", ascending=False).head(MAX_LOW_FREQ_SAMPLES)
    glitches_high_candidate = glitches[~low_freq_mask].sort_values(by="witness_match_count", ascending=False)
    glitches = pd.concat([glitches_low_candidate, glitches_high_candidate])
    print(f"Targeted low-frequency subset: {len(glitches_low_candidate)} rows prioritized.")
else:
    if "witness_match_count" in glitches.columns:
        glitches = glitches.sort_values(by="witness_match_count", ascending=False)

print("\nIndexing strain chunks")
strain_manifest = load_strain_manifest(STRAIN_DIR, DETECTOR)
print(f"Found {len(strain_manifest)} strain chunks")

print("Indexing witness chunks")
witness_manifests = {}
for channel in WITNESS_CHANNELS:
    witness_manifests[channel] = build_witness_manifest(WITNESS_DIR, DETECTOR, channel)
    print(f"  {channel}: {len(witness_manifests[channel])} chunks")

rng = np.random.default_rng(JITTER_SEED)

strain_low, witness_low, gps_low, freq_low, snr_low = [], [], [], [], []
strain_high, witness_high, gps_high, freq_high, snr_high = [], [], [], [], []

bad_strain = 0
bad_witness = 0

channel_names = [f"{DETECTOR}:{channel}" for channel in WITNESS_CHANNELS]
transforms = {target_sample_rate: build_transforms(target_sample_rate)}
strain_file_cache = {}
witness_file_cache = {}

all_witness_rates = set()
for manifest in witness_manifests.values():
    all_witness_rates |= {rate for _, _, rate, _ in manifest}
for rate in all_witness_rates:
    if rate not in transforms:
        transforms[rate] = build_transforms(rate)



# Candidate windows

gtime = glitches["peak_time"].values
gfreq = glitches[freq_col].values if freq_col in glitches.columns else np.zeros(len(glitches))
gsnr = glitches["snr"].values if "snr" in glitches.columns else np.zeros(len(glitches))
gtstart = glitches["tstart"].values
gtend = glitches["tend"].values

half = window / 2.0
gdur = gtend - gtstart
room = window - gdur - 2 * JITTER_MARGIN
can_jitter = (gdur < 0.6) & (room > 0)

win_starts = np.zeros_like(gtime, dtype=np.float64)

if np.any(can_jitter):
    lo = gtend[can_jitter] + JITTER_MARGIN - window
    hi = gtstart[can_jitter] - JITTER_MARGIN
    win_starts[can_jitter] = rng.uniform(lo, hi)

win_starts[~can_jitter] = gtime[~can_jitter] - half

for t, fr, s_val, win_start in tqdm(zip(gtime, gfreq, gsnr, win_starts), total=len(gtime), desc=DETECTOR):
    if len(gps_low) >= MAX_LOW_FREQ_SAMPLES and fr <= nyquist:
        continue

    segment_start = win_start - WHITEN_PSD_LENGTH - pad_seconds
    segment_end = win_start + window + pad_seconds

    # Determine native rate of the specific strain chunk for this time window
    strain_native_rate = target_sample_rate
    for gps_s, gps_e, s_rate, _ in strain_manifest:
        if gps_s <= win_start < gps_e:
            strain_native_rate = s_rate
            break

    strain_seq_len = int(window * strain_native_rate)

    raw, s_rate = get_strain_segment(strain_manifest, strain_file_cache, segment_start, segment_end, strain_native_rate)
    psd_len_native = int(WHITEN_PSD_LENGTH * s_rate)

    # Whiten strain natively without resampling
    strain_sample = whiten_block_native(
        raw, s_rate, transforms, psd_len_native, strain_seq_len
    )

    if strain_sample is None:
        bad_strain += 1
        continue

    w_samples = []
    ok = True

    witness_seq_len = int(window * target_sample_rate)

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

        ws = whiten_block_resampled(
            raw_w, w_rate, target_sample_rate, transforms, psd_len_native_w, witness_seq_len
        )

        if ws is None:
            ok = False
            break

        w_samples.append(ws)

    if not ok:
        bad_witness += 1
        continue

    witness_stack = np.stack(w_samples)

    if fr <= nyquist:
        if len(gps_low) < MAX_LOW_FREQ_SAMPLES:
            strain_low.append(strain_sample)
            witness_low.append(witness_stack)
            gps_low.append(t)
            freq_low.append(fr)
            snr_low.append(s_val)
    else:
        strain_high.append(strain_sample)
        witness_high.append(witness_stack)
        gps_high.append(t)
        freq_high.append(fr)
        snr_high.append(s_val)

for f in strain_file_cache.values():
    f.close()
for f in witness_file_cache.values():
    f.close()

print("\nFinished")
print("Low-freq samples:", len(gps_low))
print("High-freq samples:", len(gps_high))
print("Bad strain:", bad_strain)
print("Bad witness:", bad_witness)

save_h5(output_file_low, strain_low, witness_low, gps_low, freq_low, snr_low, channel_names)
save_h5(output_file_high, strain_high, witness_high, gps_high, freq_high, snr_high, channel_names)
