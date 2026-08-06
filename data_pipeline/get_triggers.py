from gwpy.table import EventTable
from astropy.table import vstack
import glob, os

TRIG_ROOT = "/home/detchar/triggers/H1"
OUTDIR = "/home/deon.fernando/may_triggers"
os.makedirs(OUTDIR, exist_ok=True)

channels = [
    "H1:GDS-CALIB_STRAIN",
    "H1:LSC-POP_A_LF_OUT_DQ",
    "H1:LSC-REFL_A_LF_OUT_DQ",
    "H1:LSC-REFL_A_RF45_I_ERR_DQ",
    "H1:ISI-HAM4_BLND_GS13Z_IN1_DQ",
    "H1:LSC-REFL_A_RF9_Q_ERR_DQ",
    "H1:ASC-CHARD_P_OUT_DQ",
    "H1:PEM-CS_ACC_LVEAFLOOR_XCRYO_Z_DQ",
    "H1:ASC-Y_TR_B_PIT_OUT_DQ",
    "H1:SUS-SR3_M3_OPLEV_PIT_OUT_DQ",
]

spans = [
    (1240704018, 1241136018),
]

def chan_to_dir(ch):
    return ch.split(":", 1)[1].replace("-", "_") + "_OMICRON"

def triggers_for_span(chan_dir, start, end, snr_min=5.0):
    prefixes = range(start // 100000, end // 100000 + 1)
    files = []
    for p in prefixes:
        files.extend(glob.glob(f"{TRIG_ROOT}/{chan_dir}/{p}/*.h5"))
    files = sorted(set(files))
    if not files:
        return None
    tables = [EventTable.read(f, format="hdf5", path="triggers") for f in files]
    tab = vstack(tables)
    mask = (tab["time"] >= start) & (tab["time"] < end) & (tab["snr"] >= snr_min)
    tab = tab[mask]
    tab.sort("time")
    return tab

# --- Execution and Saving Loop ---
for start, end in spans:
    for ch in channels:
        chan_dir = chan_to_dir(ch)
        print(f"Processing {ch} for span {start}-{end}...")
        
        trig_table = triggers_for_span(chan_dir, start, end)
        
        if trig_table is not None and len(trig_table) > 0:
            # Format filename safely (replacing colons for file systems)
            safe_chan_name = ch.replace(":", "_")
            out_file = os.path.join(OUTDIR, f"{safe_chan_name}_{start}_{end}.csv")
            
            # Save to CSV format
            trig_table.write(out_file, format="ascii.csv", overwrite=True)
            print(f"Saved {len(trig_table)} triggers to {out_file}")
        else:
            print(f"No triggers found for {ch} in this span.")
