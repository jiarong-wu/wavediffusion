import numpy as np
import xarray as xr

def get_top2_indices(ds):
    """
    For each grid point and time step, find the two most energetic swell 
    partitions and assign their hs, tp, dir fields to ds.
    NOTE: should fill nan with 0 for phs{i} before calling this function.

    Parameters
    ----------
    ds : xr.Dataset
         WW3 output with variables like phs0..phs2, ptp0..ptp2, pdir0..pdir2

    Returns
    -------
    ds : xr.Dataset with new fields hs_p1, hs_p2, tp_p1, tp_p2, dir_p1, dir_p2
    idx_first  : xr.DataArray — partition index of the most energetic swell
    idx_second : xr.DataArray — partition index of the 2nd most energetic swell
    """

    partitions = range(0, 3)

    # --- Find indices of top 2 most energetic partitions ---
    hs_swells = xr.concat(
        [ds[f"phs{i}"].fillna(0) for i in partitions],
        dim=xr.DataArray(list(partitions), dims="partition", name="partition")
    ).rename("phs")

    sorted_positions = (-hs_swells).argsort(axis=hs_swells.get_axis_num("partition"))

    idx_first  = sorted_positions.isel(partition=0)
    idx_second = sorted_positions.isel(partition=1)

    idx_first.name  = "most_energetic_partition"
    idx_second.name = "second_energetic_partition"

    # --- Vectorized lookup of hs, tp, dir for top 2 partitions ---
    for ww3_prefix, out_prefix in [("phs", "hs"), ("ptp", "tp"), ("pdir", "dir")]:

        stacked = xr.concat(
            [ds[f"{ww3_prefix}{i}"].fillna(0) for i in partitions],
            dim=xr.DataArray(list(partitions), dims="partition", name="partition")
        )

        ds[f"{out_prefix}_p1"] = stacked.isel(partition=idx_first).drop_vars("partition")
        ds[f"{out_prefix}_p2"] = stacked.isel(partition=idx_second).drop_vars("partition")

    return ds, idx_first, idx_second