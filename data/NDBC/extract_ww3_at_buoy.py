"""
Extract WW3 model output at buoy 46042 (36.787 N, 122.408 W) for 2022,
and co-locate NDBC directional spectra to the same WW3 timestamps.

Memory-efficient: the buoy grid point is selected immediately after opening
each monthly file so the full global grid (~230 MB/var/month) is never kept
in memory. Peak usage is ~17 MB (NDBC) + negligible WW3 time series.

The WW3 3-hourly grid defines the time axis. NDBC observations are mapped
to the nearest WW3 timestamp. Both are saved in one NetCDF file:
  - WW3  (time):            hs, t0m1, dir, spr, hs_p1, tp_p1, dir_p1,
                            hs_p2, tp_p2, dir_p2
  - NDBC (time, frequency): r1, r2, alpha1, alpha2, spectral_density

Run with:
    module load pytorch/2.6.0
    python extract_ww3_at_buoy.py
"""

import sys
import os
import xarray as xr
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ifremer'))
from utils import get_top2_indices

RAW_DIR   = '/global/homes/j/jiarongw/scratch_folder/wave_data/raw'
NDBC_FILE = '/global/homes/j/jiarongw/scratch_folder/wave_data/NDBC/ndbc_directional_46042_2022.nc'
OUT_FILE  = '/global/homes/j/jiarongw/scratch_folder/wave_data/NDBC/ww3_46042_2022.nc'

BUOY_LAT =  36.787
BUOY_LON = -122.408
YEAR     = 2022

MEAN_VARS    = ['hs', 't0m1', 'dir', 'spr']
PART_VARS    = ['phs0', 'phs1', 'phs2', 'ptp0', 'ptp1', 'ptp2', 'pdir0', 'pdir1', 'pdir2']
OUT_WW3_VARS = MEAN_VARS + ['hs_p1', 'tp_p1', 'dir_p1', 'hs_p2', 'tp_p2', 'dir_p2']
NDBC_VARS    = ['r1', 'r2', 'alpha1', 'alpha2', 'spectral_density']


def process_month(fname):
    """Open one monthly WW3 file, select the buoy point, compute partitions.

    The spatial selection happens before any data is loaded into memory, so
    the global grid (~230 MB/var) is never materialised.
    """
    with xr.open_dataset(fname) as ds:
        # Select buoy point while data is still lazy (no global load)
        ds_pt = ds[MEAN_VARS + PART_VARS].sel(
            latitude=BUOY_LAT, longitude=BUOY_LON, method='nearest'
        )
        # Load only the tiny time-series into memory
        ds_pt = ds_pt.load()

    for i in range(3):
        ds_pt[f'phs{i}'] = ds_pt[f'phs{i}'].fillna(0)
    ds_pt, _, _ = get_top2_indices(ds_pt)
    return ds_pt[OUT_WW3_VARS]


def main():
    # --- Process one month at a time to keep memory low ---
    monthly = []
    for month in range(1, 13):
        fname = os.path.join(RAW_DIR, f'LOPS_WW3-GLOB-30M_{YEAR:04d}{month:02d}.nc')
        print(f'Processing {os.path.basename(fname)} ...')
        monthly.append(process_month(fname))

    ds_ww3 = xr.concat(monthly, dim='time')

    # --- Load NDBC data and map to nearest WW3 timestamps ---
    print('Aligning NDBC spectra to WW3 timestamps ...')
    ds_ndbc = xr.open_dataset(NDBC_FILE)[NDBC_VARS]
    ds_ndbc_aligned = ds_ndbc.sel(time=ds_ww3.time, method='nearest')
    # Replace NDBC time coord with WW3 time so both share the same axis
    ds_ndbc_aligned['time'] = ds_ww3.time

    # --- Merge and save ---
    ds_out = xr.merge([ds_ww3, ds_ndbc_aligned])
    ds_out.attrs['description'] = (
        f'WW3 LOPS-GLOB-30M hindcast at buoy 46042 (lat={BUOY_LAT}, lon={BUOY_LON}) '
        f'for {YEAR}. NDBC directional spectra nearest-neighbor mapped to WW3 timestamps.'
    )

    ds_out.to_netcdf(OUT_FILE)
    print(f'Saved {OUT_FILE}')
    print(ds_out)


if __name__ == '__main__':
    main()
