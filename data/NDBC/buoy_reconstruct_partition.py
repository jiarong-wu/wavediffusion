'''
Reconstruct wave partitions from NDBC buoy WW3 spectra using MEM2 (from roguewavespectrum package), 
then compute partition stats hs, tp (tm01), dm (from wavespectra package) and save to a new NetCDF file 
together with the original WW3 data, both in descending energy order between two partitions.
  - NDBC (time, part): hs, tp, dm
  - WW3  (time, part): hs_m, tp_m, dm_m

Run with:
    module load python/2.6.0
    python extract_ww3_at_buoy.py
    python reconstruct_partition.py
'''

import numpy as np
import xarray as xr
from roguewavespectrum import Spectrum
import wavespectra

INPUT = "/pscratch/sd/j/jiarongw/wave_data/NDBC/ww3_46042_2022.nc"
OUTPUT = "/pscratch/sd/j/jiarongw/wave_data/NDBC/ww3_buoy_partitioned_46042_2022.nc"
N_PARTS = 2

ds = xr.open_dataset(INPUT)

# Convert WW3 polar form (r, alpha) to Fourier coefficients (a, b)
# a1 = r1*cos(alpha1), b1 = r1*sin(alpha1)
# a2 = r2*cos(2*alpha2), b2 = r2*sin(2*alpha2)
alpha1_rad = np.deg2rad(ds.alpha1)
alpha2_rad = np.deg2rad(ds.alpha2)

ds_1d = xr.Dataset({
    "variance_density": ds.spectral_density,
    "a1": ds.r1 * np.cos(alpha1_rad),
    "b1": ds.r1 * np.sin(alpha1_rad),
    "a2": ds.r2 * np.cos(2 * alpha2_rad),
    "b2": ds.r2 * np.sin(2 * alpha2_rad),
})

# Reconstruct 2D frequency-direction spectrum via MEM2
spec1d = Spectrum.from_dataset(ds_1d)
spec2d = spec1d.as_frequency_direction_spectrum(
    number_of_directions=36, method="mem2", solution_method="scipy"
)

# Convert to wavespectra format (efth, freq, dir)
E = spec2d.directional_variance_density  # (time, frequency, direction)
ds_ws = xr.Dataset({"efth": E.rename({"frequency": "freq", "direction": "dir"})})

# Partition into N_PARTS using PTM3 (watershed, no wind/depth needed)
da_part = ds_ws.spec.partition.ptm3(parts=N_PARTS)  # (part, time, freq, dir)

# Compute hs, tp, dm per partition
stats = da_part.spec.stats(["hs", "tm01", "dm"]).compute()  # Dataset (part, time)

# Assemble output dataset with (time, part) dimensions
# --- buoy-reconstructed partitions ---
ds_out = xr.Dataset(
    {
        "hs": stats.hs.transpose("time", "part"),
        "tp": stats.tm01.transpose("time", "part"), # tm01 computation was more robust than tp (sometimes give nan)
        "dm": stats.dm.transpose("time", "part"),
    }
)
ds_out.hs.attrs = {"long_name": "Significant wave height (buoy reconstructed)", "units": "m"}
ds_out.tp.attrs = {"long_name": "Peak period (buoy reconstructed)", "units": "s"}
ds_out.dm.attrs = {"long_name": "Mean wave direction (buoy reconstructed)", "units": "deg"}

# --- WW3 modeled partitions stacked along a new part dimension ---
hs_m = xr.concat([ds.hs_p1, ds.hs_p2], dim="part").assign_coords(part=[0, 1]).T
tp_m = xr.concat([ds.tp_p1, ds.tp_p2], dim="part").assign_coords(part=[0, 1]).T
dm_m = xr.concat([ds.dir_p1, ds.dir_p2], dim="part").assign_coords(part=[0, 1]).T

ds_out["hs_m"] = hs_m
ds_out["tp_m"] = tp_m
ds_out["dm_m"] = dm_m
ds_out.hs_m.attrs = {"long_name": "Significant wave height (WW3 modeled)", "units": "m"}
ds_out.tp_m.attrs = {"long_name": "Peak period (WW3 modeled)", "units": "s"}
ds_out.dm_m.attrs = {"long_name": "Mean wave direction (WW3 modeled)", "units": "deg"}

ds_out.to_netcdf(OUTPUT)
print(f"Saved to {OUTPUT}")
print(ds_out)
