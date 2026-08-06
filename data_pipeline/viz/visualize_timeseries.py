"""Plot whitened strain + witness timeseries for samples with SNR > 35,
optimized with 15pt panel titles (18pt for peak power), robust filtering, and poster styling.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from viz_common import FILES, TARGET_SAMPLE_RATE, load_h5, find_event_time, EVENT_LABEL_BY_CLASS, DETECTOR

SAVE_DIR = os.path.join("new_utils_plots", "timeseries_high_snr", DETECTOR)
N_SAMPLES_PER_CLASS = 3
MIN_SNR = 35.0
SEED = 0

# Plain-English labels for a general scientific audience, keyed by the
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


def plot_sample(save_dir, class_name, idx, strain, witness, channel_names,
                gps=None, freq=None, snr=None):
    C = witness.shape[0]
    t = np.arange(strain.shape[0]) / TARGET_SAMPLE_RATE

    event_name = EVENT_LABEL_BY_CLASS.get(class_name)
    event_time = None
    if event_name is not None:
        _, event_time = find_event_time(strain, TARGET_SAMPLE_RATE)

    # Generous vertical spacing to accommodate panel titles
    fig, axes = plt.subplots(C + 1, 1, figsize=(12, 3 * (C + 1)), sharex=True)

    # --- Strain Panel ---
    axes[0].plot(t, strain, color="black", lw=1.2)
    axes[0].set_title("Gravitational Wave Strain", fontsize=22, fontweight="bold", loc="left", pad=8)
    axes[0].set_ylabel("Amplitude", fontsize=12)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].tick_params(labelsize=12)

    # Metadata string on the top right
    meta_str = f"Sample {idx}"
    if gps is not None and not np.isnan(gps):
        meta_str += f" | GPS {gps:.3f}"
    if freq is not None and not np.isnan(freq):
        meta_str += f" | {freq:.1f} Hz"
    if snr is not None and not np.isnan(snr):
        meta_str += f" | SNR {snr:.1f}"
    axes[0].set_title(meta_str, fontsize=12, loc="right", color="gray", pad=8)

    # --- Witness Panels ---
    for c in range(C):
        raw_name = channel_names[c]
        clean_name = CHANNEL_DISPLAY_NAMES.get(raw_name, f"Witness Channel {c + 1}")
        
        # 18pt for peak power (Arm Laser Light Power), 15pt for all other titles
        if raw_name == "H1:ASC-X_TR_A_NSUM_OUT_DQ" or "Power" in clean_name:
            title_fontsize = 22
        else:
            title_fontsize = 22
        
        ax = axes[c + 1]
        ax.plot(t, witness[c], color="tab:blue", lw=1.1)
        
        ax.set_title(clean_name, fontsize=title_fontsize, fontweight="bold", loc="left", pad=8)
        ax.set_ylabel("Amplitude", fontsize=20)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=12)

    # --- Subtle, Professional Event / Merger Marker ---
    if event_time is not None:
        for ax in axes:
            ax.axvline(event_time, color='crimson', lw=1.2, ls='--', alpha=0.7)
        axes[0].annotate(
            event_name, 
            xy=(event_time, axes[0].get_ylim()[1]), 
            xytext=(6, -8),
            textcoords="offset points", 
            color='crimson', 
            va='top', 
            fontsize=11, 
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="crimson", alpha=0.85, lw=0.8)
        )

    axes[-1].set_xlabel("Time [s]", fontsize=14, fontweight="bold")
    
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{class_name}_{idx}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


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
        print(f"{class_name}: plotting {n_pick} of {len(valid_idxs)} samples with SNR > {MIN_SNR}")

        for i in idxs:
            i = int(i)
            gps = data["gps"][i] if "gps" in data and data["gps"] is not None else None
            freq = data["frequency"][i] if "frequency" in data and data["frequency"] is not None else None
            snr = snrs[i]
            
            plot_sample(
                save_dir, class_name, i,
                data["strain"][i], data["witness"][i], data["channel_names"],
                gps=gps, freq=freq, snr=snr,
            )

    print("Done. High-SNR plots saved under", SAVE_DIR)


if __name__ == "__main__":
    run()