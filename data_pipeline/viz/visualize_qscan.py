# -*- coding: utf-8 -*-
"""Plot Q-transform spectrograms (Q-scans) for samples with SNR > 35,
retaining the original colorbar setup, clean witness names, 20pt titles, 
and 15pt text formatting for your research poster.
"""

import os
import numpy as np
from gwpy.timeseries import TimeSeries

from viz_common import FILES, TARGET_SAMPLE_RATE, load_h5, find_event_time, EVENT_LABEL_BY_CLASS, DETECTOR

SAVE_DIR = os.path.join("new_utils_plots", "qscan_high_snr", DETECTOR)
N_SAMPLES_PER_CLASS = 3
MIN_SNR = 35.0
SEED = 0

QRANGE = (4, 64)
FRANGE = (20, TARGET_SAMPLE_RATE / 2)  # matches the 20Hz highpass used in extraction

# Plain-English display names for witness channels, keyed by the
# "<DETECTOR>:<CHANNEL>" names extract_background.py / extract_glitches.py /
# inject_signal.py write into channel_names. L1 currently only has 1 aux
# channel downloaded -- add its display name here as more arrive.
CHANNEL_DISPLAY_NAMES = {
    "H1:ASC-CHARD_P_OUT_DQ": "Common Hard Pitch (arm cavity alignment)",
    "H1:ASC-Y_TR_B_PIT_OUT_DQ": "Y-Arm Transmon Pitch",
    "H1:ISI-HAM4_BLND_GS13Z_IN1_DQ": "HAM4 Seismic Isolation (vertical)",
    "H1:LSC-POP_A_LF_OUT_DQ": "Power Recycling Cavity Power",
    "H1:LSC-REFL_A_LF_OUT_DQ": "Reflected Power",
    "H1:LSC-REFL_A_RF45_I_ERR_DQ": "Reflected RF45 Error (I)",
    "H1:LSC-REFL_A_RF9_Q_ERR_DQ": "Reflected RF9 Error (Q)",
    "H1:PEM-CS_ACC_LVEAFLOOR_XCRYO_Z_DQ": "Corner Station Floor Accelerometer",
    "H1:SUS-SR3_M3_OPLEV_PIT_OUT_DQ": "SR3 Optical Lever Pitch",
    "L1:LSC-POP_A_LF_OUT_DQ": "Power Recycling Cavity Power",
}


def q_scan_channel(data_1d, save_path, title, event_time=None, event_label=None):
    ts = TimeSeries(np.asarray(data_1d, dtype=np.float64), sample_rate=TARGET_SAMPLE_RATE)

    try:
        qgram = ts.q_transform(qrange=QRANGE, frange=FRANGE, whiten=False)
    except Exception as e:
        print(f"  Skipping {title}: q_transform failed ({e})")
        return

    plot = qgram.plot(figsize=(9, 5))
    ax = plot.gca()
    ax.set_yscale("log")
    ax.set_ylim(FRANGE)
    
    # 20pt bold title
    ax.set_title(title, fontsize=20, fontweight="bold", pad=12)
    
    # 15pt formatting for axes, labels, and ticks
    ax.set_ylabel("Frequency [Hz]", fontsize=15, fontweight="bold")
    ax.set_xlabel("Time [s]", fontsize=15, fontweight="bold")
    ax.tick_params(labelsize=15)

    # Original script colorbar implementation
    plot.colorbar(label="Normalized energy")

    if event_time is not None:
        ax.axvline(event_time, color='white', lw=1.5, ls='--')
        if event_label is not None:
            ax.annotate(event_label, xy=(event_time, FRANGE[1]), xytext=(3, -8),
                        textcoords="offset points", color='white', va='top', fontsize=15,
                        fontweight='bold', ha='center')

    plot.savefig(save_path, dpi=300, bbox_inches="tight")
    plot.close()


def run():
    rng = np.random.default_rng(SEED)

    for class_name, path in FILES.items():
        if not os.path.exists(path):
            print(f"Skipping {class_name}: {path} not found")
            continue

        data = load_h5(path)
        n = data["strain"].shape[0]

        # Calculate or extract SNR for all samples to filter
        snrs = []
        for i in range(n):
            s = data["snr"][i] if "snr" in data and data["snr"] is not None else None
            if s is None or np.isnan(s):
                s = float(np.max(np.abs(data["strain"][i])))
            snrs.append(s)
        snrs = np.array(snrs)

        # Filter indices where SNR > MIN_SNR
        valid_idxs = np.where(snrs > MIN_SNR)[0]

        if len(valid_idxs) == 0:
            print(f"Warning: No samples found with SNR > {MIN_SNR} for {class_name}")
            continue

        n_pick = min(N_SAMPLES_PER_CLASS, len(valid_idxs))
        idxs = rng.choice(valid_idxs, size=n_pick, replace=False)

        save_dir = os.path.join(SAVE_DIR, class_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"{class_name}: Q-scanning {n_pick} of {len(valid_idxs)} samples with SNR > {MIN_SNR}")

        event_label = EVENT_LABEL_BY_CLASS.get(class_name)

        for i in idxs:
            i = int(i)
            gps = data["gps"][i] if "gps" in data and data["gps"] is not None else None
            gps_str = f" | GPS {gps:.3f}" if gps is not None and not np.isnan(gps) else ""
            snr = snrs[i]
            snr_str = f" | SNR {snr:.1f}" if snr is not None and not np.isnan(snr) else ""

            event_time = None
            if event_label is not None:
                _, event_time = find_event_time(data["strain"][i], TARGET_SAMPLE_RATE)

            q_scan_channel(
                data["strain"][i],
                os.path.join(save_dir, f"{i}_strain.png"),
                f"{class_name.replace('_', ' ').title()} - Sample {i} (Strain){gps_str}{snr_str}",
                event_time=event_time, event_label=event_label,
            )

            for c, name in enumerate(data["channel_names"]):
                clean_name = CHANNEL_DISPLAY_NAMES.get(name, f"Witness {c + 1}")
                q_scan_channel(
                    data["witness"][i, c],
                    os.path.join(save_dir, f"{i}_wit{c}.png"),
                    f"{class_name.replace('_', ' ').title()} - Sample {i} ({clean_name}){gps_str}{snr_str}",
                    event_time=event_time, event_label=event_label,
                )

    print("Done. High-SNR Q-scan plots saved under", SAVE_DIR)


if __name__ == "__main__":
    run()