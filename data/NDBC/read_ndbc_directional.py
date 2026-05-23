#!/usr/bin/env python3
"""
Read NDBC directional spectral data and merge r1, r2, alpha1, alpha2, and spectral density.
Save to ndbc_directional_{station}_{year}.nc with dimensions (time, frequency).

Input files
-----------
Five per-station per-year text files are required, identified by their single-letter suffix:
  j  -> r1               (mean direction first Fourier coefficient)
  k  -> r2               (mean direction second Fourier coefficient)
  d  -> alpha1           (principal direction, first Fourier coefficient)
  i  -> alpha2           (principal direction, second Fourier coefficient)
  w  -> spectral_density (variance density, m^2/Hz)

File format compatibility
-------------------------
Two header formats are supported:

  New (2005+):  #YY  MM DD hh mm  <freq1>  <freq2> ...
                5 time columns; year prefixed with '#'.

  Old (<=2004): YYYY MM DD hh  <freq1>  <freq2> ...
                4 time columns (no minutes); year column named 'YYYY'.

The parser detects the number of time columns automatically by finding the first
column whose name parses as a float in (0, 1) Hz, so it is robust to either format.
Missing minute values (old format) are filled with 0.

Usage
-----
  module load pytorch/2.6.0
  python read_ndbc_directional.py --station 46042 --year 2022 --input-dir /path/to/data/
  python read_ndbc_directional.py --station 46042 --year 2004 --input-dir /path/to/data/
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


def _is_frequency(s: str) -> bool:
    """Return True if s looks like a frequency bin label (float in Hz range)."""
    try:
        return 0.0 < float(s) < 1.0
    except ValueError:
        return False


def parse_ndbc_file(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Read one NDBC spectral/directional file.

    Returns
    -------
    times : pd.DatetimeIndex
    frequencies : np.ndarray, shape (n_freq,)  — frequency bins in Hz
    values : np.ndarray, shape (n_time, n_freq) — spectral/directional values
    """
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

    df = df.rename(columns={"YY": "year", "YYYY": "year",
                             "MM": "month", "DD": "day", "hh": "hour", "mm": "minute"})
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    df["hour"] = df["hour"].astype(int)
    # Older files (e.g. 2004) have no minutes column; default to 0
    if "minute" not in df.columns:
        df["minute"] = 0
    else:
        df["minute"] = df["minute"].astype(int)

    times = pd.to_datetime(
        df[["year", "month", "day", "hour", "minute"]], format="%Y %m %d %H %M"
    )

    # Detect where frequency columns start: first col_name parseable as a float in (0, 1) Hz
    n_time_cols = next(
        i for i, c in enumerate(col_names)
        if _is_frequency(c)
    )
    freq_cols = col_names[n_time_cols:]
    frequencies = np.array([float(c) for c in freq_cols], dtype=float)
    values = df[freq_cols].to_numpy(dtype=float)
    return times, frequencies, values


def build_dataset(input_dir: Path, station: str, year: int) -> xr.Dataset:
    """Read all five suffix files and merge into one xarray Dataset.

    Files for different variables may have different frequency grids (notably
    the old <=2004 format where the spectral density file covers a wider range
    than the directional files). All variables are trimmed to their common
    frequency intersection before merging.

    Parameters
    ----------
    input_dir : Path  — directory containing the NDBC text files
    station   : str   — NDBC station ID (e.g. '46042')
    year      : int   — year to read

    Returns
    -------
    xr.Dataset with dims (time, frequency) and variables r1, r2, alpha1, alpha2,
    spectral_density.
    """
    # Collect all parsed data first, then align to common frequency grid
    parsed = {}
    time_index = None

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

        parsed[var_name] = (frequencies, values)

    # Find common frequency grid (intersection), rounding to avoid float precision issues
    all_freqs = [np.round(freqs, 6) for freqs, _ in parsed.values()]
    common_freqs = all_freqs[0]
    for f in all_freqs[1:]:
        common_freqs = np.intersect1d(common_freqs, f)

    data_vars = {}
    for var_name, (frequencies, values) in parsed.items():
        mask = np.isin(np.round(frequencies, 6), common_freqs)
        data_vars[var_name] = (("time", "frequency"), values[:, mask])

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": time_index,
            "frequency": common_freqs,
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
