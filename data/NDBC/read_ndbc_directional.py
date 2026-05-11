#!/usr/bin/env python3
"""
Read NDBC directional spectral data and merge r1, r2, alpha1, alpha2, and spectral density.
Save to file ndbc_directional_{station}_{year}.nc. The output dataset has dimensions (time, frequency) and variables
r1, r2, alpha1, alpha2, spectral_density.

This script expects files for the same station and year with suffixes:
- j: r1 direction
- k: r2 direction
- d: alpha1 direction
- i: alpha2 direction
- w: spectral density

Example:
  python read_ndbc_directional.py --station 46042 --year 2022 --input-dir /path/to/buoy
  
NOTE: It seems that the 2004 data files have different convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SUFFIX_TO_VAR = {
    "j": "r1",
    "k": "r2",
    "d": "alpha1",
    "i": "alpha2",
    "w": "spectral_density",
}

NA_VALUES = ["MM", "99", "99.0", "999", "999.0", "9999", "9999.0"]


def parse_ndbc_file(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Read one NDBC spectral/directional file and return time index, frequency vector, and values."""
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()

    if not header:
        raise ValueError(f"Empty file: {path}")

    col_names = [c.lstrip("#") for c in header.split()]
    if len(col_names) < 6:
        raise ValueError(f"Unexpected header in {path}: {header}")

    df = pd.read_csv(
        path,
        sep=r'\s+',
        header=None,
        names=col_names,
        skiprows=1,
        na_values=NA_VALUES,
        comment=None,
        dtype=float,
    )

    df = df.rename(columns={"YY": "year", "MM": "month", "DD": "day", "hh": "hour", "mm": "minute"})
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    df["hour"] = df["hour"].astype(int)
    df["minute"] = df["minute"].astype(int)

    times = pd.to_datetime(
        df[["year", "month", "day", "hour", "minute"]], format="%Y %m %d %H %M"
    )

    freq_cols = [c for c in col_names[5:] if c not in {"year", "MM", "DD", "hh", "mm"}]
    frequencies = np.array([float(c) for c in freq_cols], dtype=float)
    values = df[freq_cols].to_numpy(dtype=float)
    return times, frequencies, values


def build_dataset(input_dir: Path, station: str, year: int) -> xr.Dataset:
    """Build an xarray dataset containing r1, r2, alpha1, alpha2, and spectral density."""
    data_vars = {}
    time_index = None
    frequency_coord = None

    for suffix, var_name in SUFFIX_TO_VAR.items():
        filename = f"{station}{suffix}{year}.txt"
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing expected file: {path}")

        times, frequencies, values = parse_ndbc_file(path)

        if time_index is None:
            time_index = times
        elif not np.array_equal(time_index.values, times.values):
            raise ValueError(f"Time coordinates do not match in file {path}")

        if frequency_coord is None:
            frequency_coord = frequencies
        elif not np.allclose(frequency_coord, frequencies):
            raise ValueError(f"Frequency coordinates do not match in file {path}")

        data_vars[var_name] = (("time", "frequency"), values)

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": time_index,
            "frequency": frequency_coord,
        },
    )
    ds["frequency"].attrs["units"] = "Hz"
    ds["time"].attrs["standard_name"] = "time"
    ds["r1"].attrs["description"] = "spectral wave r1 direction"
    ds["r2"].attrs["description"] = "spectral wave r2 direction"
    ds["alpha1"].attrs["description"] = "spectral wave alpha1 direction"
    ds["alpha2"].attrs["description"] = "spectral wave alpha2 direction"
    ds["spectral_density"].attrs["description"] = "spectral wave density"
    ds["spectral_density"].attrs["units"] = "m^2/Hz"

    return ds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read NDBC directional spectral data and merge into one xarray dataset."
    )
    parser.add_argument("--station", default="46042", help="NDBC station ID")
    parser.add_argument("--year", type=int, default=2022, help="Year to read")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing the NDBC text files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output NetCDF file path",
    )
    args = parser.parse_args()

    ds = build_dataset(args.input_dir, args.station, args.year)
    if not args.output:
        ds.to_netcdf(f"ndbc_directional_{args.station}_{args.year}.nc")
        print(f"Wrote dataset with variables {list(ds.data_vars)} to ndbc_directional_{args.station}_{args.year}.nc")
    else:
        ds.to_netcdf(args.output)
        print(f"Wrote dataset with variables {list(ds.data_vars)} to {args.output}")

if __name__ == "__main__":
    main()
