"""
Gather diffusion model sampled sea states from a given output directory,
concatenate with timestamps, and interpolate to a buoy location.

Each file index N maps to: base_time + N * 3h, where base_time is
{year}-01-01T00:00:00, aligned with the WW3 model time axis.
Files are named sample/mean/std/truth_{N}.npy and step by 8 (daily cadence).

Output: diffusion_{station}_{year}.nc in scratch_folder/wave_data/NDBC/.

Usage:
    module load pytorch/2.6.0
    python gather_diffusion_at_buoy.py /path/to/OPTION3_sigma100_epoch10_2022/
    python gather_diffusion_at_buoy.py /path/to/dir/ --station 46042 --lat 36.787 --lon -122.408
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from wavediffusion.waveana import add_lat_lon

OUT_DIR = '/global/homes/j/jiarongw/scratch_folder/wave_data/NDBC'

# Variable names per OPTION (matches sample.py var_names)
OPTION_VARS = {
    1: ['hs', 'tp', 'dir'],
    2: ['hs', 'uss', 'vss', 'mssd', 'mssc'],
    3: ['hs_p1', 'tp_p1', 'dir_p1', 'hs_p2', 'tp_p2', 'dir_p2', 'ci'],
}

# For OPTION3: channel pairs to stack along a 'part' dimension (part=[0,1])
# Each entry is (output_name, [channel_idx_p1, channel_idx_p2])
OPTION3_PART_GROUPS = [
    ('hs', [0, 3]),
    ('tp', [1, 4]),
    ('dm', [2, 5]),
]
OPTION3_SCALAR = [('ci', 6)]

PREFIXES = ['sample', 'mean', 'std', 'truth']


def parse_dir_name(dirpath):
    """Extract OPTION and year from directory name, e.g. OPTION3_sigma100_epoch10_2022."""
    name = os.path.basename(dirpath.rstrip('/'))
    m = re.search(r'OPTION(\d+)', name)
    option = int(m.group(1)) if m else None
    # Match trailing _YYYY or _YYYYMM
    m = re.search(r'_(\d{4})(\d{2})?$', name)
    if m:
        year = int(m.group(1))
        month = int(m.group(2)) if m.group(2) else None
    else:
        year, month = None, None
    return option, year, month


def collect_indices(dirpath):
    """Return sorted list of integer indices for sample_*.npy files present."""
    indices = []
    for fname in os.listdir(dirpath):
        m = re.fullmatch(r'sample_(\d+)\.npy', fname)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def main():
    parser = argparse.ArgumentParser(description='Extract diffusion output at a buoy location.')
    parser.add_argument('dirpath', help='Directory containing sample_*.npy etc.')
    parser.add_argument('--station', default='46042', help='Buoy station ID (used in output filename)')
    parser.add_argument('--lat', type=float, default=36.787, help='Buoy latitude')
    parser.add_argument('--lon', type=float, default=-122.408, help='Buoy longitude')
    args = parser.parse_args()

    dirpath = args.dirpath.rstrip('/')
    option, year, month = parse_dir_name(dirpath)

    if option not in OPTION_VARS:
        raise ValueError(f'Could not determine OPTION from directory name, or unsupported. '
                         f'Known options: {list(OPTION_VARS)}')
    if year is None:
        raise ValueError('Could not parse year from directory name.')

    var_names = OPTION_VARS[option]

    # Base timestamp: year (or year-month) start, aligned with WW3 time axis
    # There should be 10 days of off-set because of wind history
    # CORRECTION: it should be 5 days off...
    if month is not None:
        base_time = pd.Timestamp(f'{year}-{month:02d}-06T00:00:00')
    else:
        base_time = pd.Timestamp(f'{year}-01-06T00:00:00')

    indices = collect_indices(dirpath)
    if not indices:
        raise FileNotFoundError(f'No sample_*.npy files found in {dirpath}')
    print(f'Found {len(indices)} time steps (indices {indices[0]}..{indices[-1]})')

    times = [base_time + pd.Timedelta(hours=idx * 3) for idx in indices]

    # Accumulate per-prefix point values using add_lat_lon + sel
    arrays = {pfx: [] for pfx in PREFIXES}
    sel_lat = sel_lon = None
    for idx in indices:
        for pfx in PREFIXES:
            arr = np.load(os.path.join(dirpath, f'{pfx}_{idx}.npy'))  # (n_ch, 320, 720)
            ds_snap = add_lat_lon(arr, var_names)
            ds_pt = ds_snap.sel(lat=args.lat, lon=args.lon, method='nearest')
            if sel_lat is None:
                sel_lat = float(ds_pt.lat)
                sel_lon = float(ds_pt.lon)
                print(f'Buoy {args.station}: nearest grid point lat={sel_lat:.3f}, lon={sel_lon:.3f}')
            arrays[pfx].append([float(ds_pt[v]) for v in var_names])

    # Build xarray dataset
    data_vars = {}
    if option == 3:
        for pfx in PREFIXES:
            stacked = np.array(arrays[pfx])   # (n_time, 7)
            for base_name, ch_idx in OPTION3_PART_GROUPS:
                data_vars[f'{base_name}_{pfx}'] = (['time', 'part'], stacked[:, ch_idx])
            for base_name, ch_idx in OPTION3_SCALAR:
                data_vars[f'{base_name}_{pfx}'] = ('time', stacked[:, ch_idx])
        coords = {'time': pd.DatetimeIndex(times), 'part': [0, 1]}
    else:
        for pfx in PREFIXES:
            stacked = np.array(arrays[pfx])   # (n_time, n_channels)
            for i, vname in enumerate(var_names):
                data_vars[f'{vname}_{pfx}'] = ('time', stacked[:, i])
        coords = {'time': pd.DatetimeIndex(times)}

    ds_out = xr.Dataset(data_vars, coords=coords)
    ds_out.attrs['source_dir'] = os.path.basename(dirpath)
    ds_out.attrs['description'] = (
        f'OPTION{option} diffusion model output at buoy {args.station} '
        f'(lat={sel_lat:.3f}, lon={sel_lon:.3f}). '
        f'Index N -> {base_time.isoformat()} + N*3h.'
    )

    out_path = os.path.join(OUT_DIR, f'diffusion_{args.station}_{year}.nc')
    ds_out.to_netcdf(out_path)
    print(f'Saved {out_path}')
    print(ds_out)


if __name__ == '__main__':
    main()
