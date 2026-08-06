from gwpy.timeseries import TimeSeries
import os

ch = "L1:GDS-CALIB_STRAIN"
CHUNK = 28800    # 1 hour per read

def download_day(start, end, outdir):
    os.makedirs(outdir, exist_ok=True)
    for t0 in range(start, end, CHUNK):
        t1 = min(t0 + CHUNK, end)
        ts = TimeSeries.get(ch, t0, t1, frametype="L1_HOFT_C00", nproc=4, verbose=True)
                         # 16384 -> 4096, clean /4
        ts.write(os.path.join(outdir, f"L1_strain_4096_{t0}_{t1}.hdf5"),
                 format="hdf5")
        del ts
        print(f"  {t0}-{t1} done")

download_day(1241107218, 1241136018, "/home/deon.fernando/may_four")
