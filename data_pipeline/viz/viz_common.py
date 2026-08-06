"""Shared loading helpers for the visualization scripts, pointed at
big_model's full_data/<DETECTOR> output layout.

Set DETECTOR below and rerun per detector.
"""

from pathlib import Path

import h5py
import numpy as np

TARGET_SAMPLE_RATE = 4096

DETECTOR = "H1"  # or "L1"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "full_data" / DETECTOR

# Output of extract_background.py, extract_glitches.py, and inject_signal.py.
FILES = {
    "background": str(DATA_DIR / "whitened_background_full.h5"),
    "glitch_low": str(DATA_DIR / "whitened_glitches.h5"),
    "glitch_high": str(DATA_DIR / "whitened_high_glitches.h5"),
    "signal": str(DATA_DIR / "whitened_signals_full.h5"),
}

# Which classes get an "important event" marker, and what to call it. Neither
# extraction script stores a per-sample event timestamp (glitch peak times
# live in strain_witness_coincidence.csv keyed by GPS, not in
# whitened_glitches.h5; signal merger times are jittered by
# inject_signal.py's _time_jitter and never saved), so both are recovered
# from the whitened strain itself via find_event_time -- the peak-power
# proxy. Background has no expected event, so it gets no marker.
EVENT_LABEL_BY_CLASS = {
    "glitch_low": "Peak power",
    "glitch_high": "Peak power",
    "signal": "Coalescence (peak-power proxy)",
}


def find_event_time(strain_1d, sample_rate=TARGET_SAMPLE_RATE, smooth_window=64):
    """Index/time of peak smoothed instantaneous power in a raw strain
    timeseries. smooth_window=64 samples (~15.6ms @ 4096Hz) avoids locking
    onto a single noisy sample while staying short relative to both glitch
    transients and a CBC merger envelope. For a CBC waveform this lands very
    close to the true coalescence time (amplitude peaks at merger); for a
    glitch it's a direct, honest measure of "peak power"."""
    power = np.asarray(strain_1d, dtype=np.float64) ** 2
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        power = np.convolve(power, kernel, mode="same")
    idx = int(np.argmax(power))
    return idx, idx / sample_rate


def load_h5(path):
    """Returns dict with strain (N,T), witness (N,C,T), channel_names (list[str]),
    and gps/frequency/snr (N,) where present in the file."""
    with h5py.File(path, "r") as f:
        strain = f["strain"][:]
        witness = f["witness"][:]
        gps = f["gps"][:] if "gps" in f else None
        freq = f["frequency"][:] if "frequency" in f else None
        snr = f["snr"][:] if "snr" in f else None
        names = [
            n.decode("utf-8") if isinstance(n, bytes) else str(n)
            for n in f["channel_names"][:]
        ]
    return {
        "strain": strain,
        "witness": witness,
        "gps": gps,
        "frequency": freq,
        "snr": snr,
        "channel_names": names,
    }
