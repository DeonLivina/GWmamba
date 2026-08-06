#!/usr/bin/env python3
"""Build the "signal" class for big_model's layout: inject synthetic CBC
waveforms into real background strain, with witness channels drawn from the
same real data and independently whitened.

Candidate windows are now drawn directly from full_data/<DETECTOR>/
whitened_background_full.h5 rather than from background_triggers.csv. That
background file was built from the *entire* surviving candidate pool (see
extract_background.py), so there's no separate "unused" pool left to draw
fresh injection candidates from -- every viable window is already sitting in
the background file. Instead, this script:

  1. Shuffles the background file's GPS list and walks it in that order,
     pulling the *raw* (pre-whitened) strain/witness data for each window
     straight from the strain/witness manifests (the background file itself
     only stores the final whitened kernel, not the raw PSD-context data
     needed to redo whitening after injection).
  2. Injects a synthetic waveform into that raw strain, whitens the result,
     and keeps going until N_TARGET signal examples have been built or the
     background pool is exhausted.
  3. Removes exactly the background-file rows that were successfully
     converted into signal examples, so the same physical window doesn't
     end up counted as both "background" and "signal".

Set DETECTOR below and rerun per detector.
"""

import os
import glob
import copy
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from gwpy.timeseries import TimeSeries

from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.gw import compute_network_snr, reweight_snrs

# big_model/utils/inject_signal.py -> big_model/ -> dataset/
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
sys.path.insert(0, str(DATASET_DIR))

from utils import load_config
from waveforms import generate_signals

from injections import (
    _inject_center,
    _time_jitter,
    _interp_psd,
)


# -----------------------
# Config
# -----------------------
DETECTOR = "L1"  # or "L1"

config_path = str(DATASET_DIR / "configs" / f"config_{DETECTOR}.yaml")

STRAIN_DIR = ROOT / "data" / "strain_data"
WITNESS_DIR = ROOT / "data" / "witness_data"

OUTPUT_DIR = ROOT / "full_data" / DETECTOR
glitch_h5_path = OUTPUT_DIR / "whitened_glitches_full.h5"

background_v2_file = OUTPUT_DIR / "whitened_background_full.h5"
output_file = OUTPUT_DIR / "whitened_signals_full.h5"

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

TARGET_RATE = 4096

# Number of background windows to convert into signal examples.
N_TARGET = 2000

CANDIDATE_SEED = 43
SNR_BOOTSTRAP_SEED = 7

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing pipeline on device: {device}")

torch.set_default_dtype(torch.float64)


# -----------------------
# Config & transforms
# -----------------------
config = load_config(config_path)

f_min = config.general.f_min
fftlength = config.whiten.fftlength
overlap = config.whiten.overlap
average = config.whiten.average
PSD_LENGTH = config.whiten.psd_length
FDURATION = config.whiten.fduration
WAVE_DURATION = config.general.waveform_duration

pad_seconds = FDURATION / 2.0

psd_size = int(PSD_LENGTH * TARGET_RATE)
kernel_size = int(WAVE_DURATION * TARGET_RATE)
pad = int(pad_seconds * TARGET_RATE)
window_size = psd_size + kernel_size + 2 * pad
num_freqs = kernel_size // 2 + 1


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
    if len(x) == 0 or not np.all(np.isfinite(x)):
        return True
    if np.ptp(x) == 0 or np.std(x) == 0:
        return True
    return False


def build_transforms(rate):
    spectral_density = SpectralDensity(
        sample_rate=rate, fftlength=fftlength, overlap=overlap, average=average,
    ).to(device)
    whiten = Whiten(fduration=FDURATION, sample_rate=rate, highpass=f_min).to(device)
    return spectral_density, whiten


def whiten_witness_block(block, rate, target_rate, transforms, psd_len_native, length):
    block = np.asarray(block, dtype=np.float64)
    if is_degenerate(block):
        return None
    try:
        spectral_density, whiten_t = transforms[rate]
        x = torch.from_numpy(block).double().to(device)
        psd_part = x[:psd_len_native].view(1, 1, -1)
        kernel_part = x[psd_len_native:].view(1, 1, -1)

        psd = spectral_density(psd_part)
        whitened = whiten_t(kernel_part, psd)
        whitened_np = whitened.view(-1).cpu().numpy()

        if rate == target_rate:
            y = whitened_np
        else:
            ts = TimeSeries(whitened_np, sample_rate=rate, dtype=np.float64)
            resampled = ts.resample(target_rate)
            y = np.asarray(resampled.value, dtype=np.float64)

        if len(y) != length or not np.all(np.isfinite(y)):
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


def find_chunk(manifest, seg_start, seg_end):
    for gps_start, gps_end, rate, path in manifest:
        if gps_start <= seg_start and seg_end <= gps_end:
            return gps_start, rate, path
    return None


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


def load_glitch_snr_pool(path):
    with h5py.File(path, "r") as f:
        if "snr" in f:
            pool = f["snr"][:].astype(np.float64)
        else:
            raise RuntimeError(f"Dataset 'snr' not found in {path}")
    if len(pool) == 0:
        raise RuntimeError(f"No usable snr values in {path}")
    return pool


def sample_bootstrap_snr(pool, batch_size, rng, device):
    draw = rng.choice(pool, size=batch_size, replace=True)
    return torch.as_tensor(draw, dtype=torch.get_default_dtype(), device=device)


# -----------------------
# Initialization
# -----------------------
print("\nIndexing strain chunks")
strain_manifest = load_strain_manifest(STRAIN_DIR, DETECTOR)
print(f"Found {len(strain_manifest)} strain chunks")

print("Indexing witness chunks")
witness_manifests = {}
for channel in WITNESS_CHANNELS:
    witness_manifests[channel] = build_witness_manifest(WITNESS_DIR, DETECTOR, channel)
    print(f"  {channel}: {len(witness_manifests[channel])} chunks")

transforms = {TARGET_RATE: build_transforms(TARGET_RATE)}
all_witness_rates = set()
for manifest in witness_manifests.values():
    all_witness_rates |= {rate for _, _, rate, _ in manifest}
for rate in all_witness_rates:
    if rate not in transforms:
        transforms[rate] = build_transforms(rate)

print("\nLoading glitch SNR distribution from HDF5")
glitch_snr_pool = load_glitch_snr_pool(glitch_h5_path)
print(f"Glitch SNR pool: {len(glitch_snr_pool)} values")
snr_rng = np.random.default_rng(SNR_BOOTSTRAP_SEED)

# -----------------------
# Candidate windows: drawn from the background file itself
# -----------------------
if not os.path.exists(background_v2_file):
    raise FileNotFoundError(
        f"{background_v2_file} not found -- this script now sources its "
        f"candidate windows from that file instead of background_triggers.csv"
    )

with h5py.File(background_v2_file, "r") as f:
    bg_gps_all = f["gps"][:]
print(f"\nTotal existing background windows: {len(bg_gps_all)}")

rng = np.random.default_rng(CANDIDATE_SEED)
shuffled_gps = bg_gps_all.copy()
rng.shuffle(shuffled_gps)
candidates = shuffled_gps - WAVE_DURATION / 2.0  # gps_out was win_start + WAVE_DURATION/2
print(f"usable candidates for injection (drawn from background pool): {len(candidates)} (target {N_TARGET})")

strain_out = np.zeros((N_TARGET, kernel_size), dtype=np.float32)
witness_out = np.zeros((N_TARGET, len(WITNESS_CHANNELS), kernel_size), dtype=np.float32)
gps_out = np.zeros(N_TARGET, dtype=np.float64)
snr_out = np.zeros(N_TARGET, dtype=np.float64)

used_bg_gps = []  # exact original background-file gps values that got consumed

count = 0
skipped_bounds = 0
skipped_resample = 0
skipped_whiten = 0
skipped_witness = 0
skipped_waveform = 0

channel_names = [f"{DETECTOR}:{channel}" for channel in WITNESS_CHANNELS]
strain_file_cache = {}
witness_file_cache = {}
debug_printed = False

pbar = tqdm(total=N_TARGET, desc="Injecting signals")


def _live_status():
    pbar.set_postfix(bounds=skipped_bounds, resample=skipped_resample,
                     whiten=skipped_whiten, witness=skipped_witness, wave=skipped_waveform)


# -----------------------
# Extraction loop
# -----------------------
for win_start, orig_gps in zip(candidates, shuffled_gps):
    if count >= N_TARGET:
        break

    segment_start = win_start - PSD_LENGTH - pad_seconds
    segment_end = win_start + WAVE_DURATION + pad_seconds

    chunk = find_chunk(strain_manifest, segment_start, segment_end)
    if chunk is None:
        skipped_bounds += 1
        _live_status()
        continue
    chunk_gps_start, s_rate, s_path = chunk

    if s_path not in strain_file_cache:
        strain_file_cache[s_path] = h5py.File(s_path, "r")
    s_dset = strain_file_cache[s_path]["strain"]["Strain"]

    s0 = int(round((segment_start - chunk_gps_start) * s_rate))
    s1 = int(round((segment_end - chunk_gps_start) * s_rate))

    raw_native = clean_non_numerical(s_dset[s0:s1])
    if s_rate == TARGET_RATE:
        raw_np_full = raw_native
    else:
        strain_ts = TimeSeries(raw_native, sample_rate=s_rate, t0=segment_start)
        raw_np_full = strain_ts.resample(TARGET_RATE).value.copy()

    raw_np = raw_np_full[:window_size]
    if raw_np.shape[0] < window_size:
        skipped_resample += 1
        _live_status()
        continue

    raw = torch.tensor(raw_np, dtype=torch.float64, device=device)[None, None, :]

    try:
        spectral_density, whiten_t = transforms[TARGET_RATE]
        psd = spectral_density(raw[..., :psd_size])
        psd_i = _interp_psd(psd, num_freqs)

        sig_cfg = copy.deepcopy(config)
        sig_cfg.general.sample_rate = TARGET_RATE
        sig_cfg.general.batch_size = 1
        sig_cfg.general.waveform_duration = WAVE_DURATION

        waveform, params = generate_signals(sig_cfg, device)
        waveform = waveform.double()
        waveform, _ = _time_jitter(waveform, config, TARGET_RATE, device)

        target = sample_bootstrap_snr(glitch_snr_pool, 1, snr_rng, device)
        waveform = reweight_snrs(waveform, target, psd_i, TARGET_RATE, highpass=f_min)
        snr = compute_network_snr(waveform, psd_i, TARGET_RATE, highpass=f_min)
    except Exception as e:
        if not debug_printed:
            import traceback
            with open("debug_error.txt", "w") as err_f:
                err_f.write(f"Exception: {e}\n")
                traceback.print_exc(file=err_f)
            debug_printed = True
        skipped_waveform += 1
        _live_status()
        continue

    raw_psd = raw[..., :psd_size]
    raw_inj = raw[..., psd_size:]
    injected_segment = _inject_center(raw_inj.clone(), waveform, kernel_size, pad)

    try:
        whitened = whiten_t(injected_segment, psd)
        white_kernel_np = whitened.view(-1).cpu().numpy()
    except Exception:
        skipped_whiten += 1
        _live_status()
        continue

    if white_kernel_np.shape[0] != kernel_size or not np.all(np.isfinite(white_kernel_np)):
        skipped_whiten += 1
        _live_status()
        continue

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
        psd_len_native_w = int(PSD_LENGTH * w_rate)
        ws = whiten_witness_block(raw_w, w_rate, TARGET_RATE, transforms, psd_len_native_w, kernel_size)
        if ws is None:
            ok = False
            break
        w_samples.append(ws)

    if not ok:
        skipped_witness += 1
        _live_status()
        continue

    strain_out[count] = white_kernel_np.astype(np.float32)
    witness_out[count] = np.stack(w_samples).astype(np.float32)
    gps_out[count] = orig_gps
    snr_out[count] = float(snr.item())
    used_bg_gps.append(orig_gps)

    count += 1
    _live_status()
    pbar.update(1)

pbar.close()
for f in strain_file_cache.values():
    f.close()
for f in witness_file_cache.values():
    f.close()

print("\nFinished")
print("Signal examples built:", count)
print("Skipped (bounds):", skipped_bounds)
print("Skipped (resample length):", skipped_resample)
print("Skipped (whiten NaN/shape):", skipped_whiten)
print("Skipped (witness):", skipped_witness)
print("Skipped (waveform generation):", skipped_waveform)

if count < N_TARGET:
    print(f"WARNING: only {count} examples produced ({N_TARGET} requested, "
          f"{len(bg_gps_all)} background windows available to draw from).")
if count == 0:
    raise RuntimeError("No signal examples produced.")

strain_out = strain_out[:count]
witness_out = witness_out[:count]
gps_out = gps_out[:count]
snr_out = snr_out[:count]

os.makedirs(OUTPUT_DIR, exist_ok=True)
with h5py.File(output_file, "w") as f:
    f.create_dataset("strain", data=strain_out, compression="gzip")
    f.create_dataset("witness", data=witness_out, compression="gzip")
    f.create_dataset("gps", data=gps_out)
    f.create_dataset("snr", data=snr_out)
    f.create_dataset("channel_names", data=np.array(channel_names, dtype="S"))
    f.create_dataset("detector", data=np.array([DETECTOR] * count, dtype="S"))

print("\nSaved:", output_file)


# -----------------------
# Remove the consumed windows from the background file
# -----------------------
print(f"\nRemoving {len(used_bg_gps)} used windows from {background_v2_file}")
with h5py.File(background_v2_file, "r") as f:
    bg_strain = f["strain"][:]
    bg_witness = f["witness"][:]
    bg_gps = f["gps"][:]
    bg_channel_names = f["channel_names"][:]
    bg_detector = f["detector"][:]

used_bg_gps_arr = np.asarray(used_bg_gps, dtype=np.float64)
keep_mask = ~np.isin(bg_gps, used_bg_gps_arr)
n_removed = int(np.count_nonzero(~keep_mask))
n_remaining = int(np.count_nonzero(keep_mask))
print(f"  matched and removed: {n_removed}")
print(f"  remaining background windows: {n_remaining}")

if n_removed != len(used_bg_gps):
    print(
        f"  WARNING: expected to remove {len(used_bg_gps)} rows but matched "
        f"{n_removed} -- check for duplicate/rounded gps values before trusting this file"
    )

tmp_path = str(background_v2_file) + ".tmp"
with h5py.File(tmp_path, "w") as f:
    f.create_dataset("strain", data=bg_strain[keep_mask], compression="gzip")
    f.create_dataset("witness", data=bg_witness[keep_mask], compression="gzip")
    f.create_dataset("gps", data=bg_gps[keep_mask])
    f.create_dataset("channel_names", data=bg_channel_names)
    f.create_dataset("detector", data=bg_detector[keep_mask])
os.replace(tmp_path, background_v2_file)
print(f"  Updated: {background_v2_file}")