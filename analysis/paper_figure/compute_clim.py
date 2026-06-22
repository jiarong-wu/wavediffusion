''' Compute monthly climatology of CI and save as a netcdf file. '''
import gc
import numpy as np
import os
import xarray as xr
from tqdm import tqdm
from wavediffusion.wavedata import npyDataResized
from wavediffusion.waveana import add_lat_lon

OPTION = 2

###############################
# Saving the mean variables climatology for ACC computation
if OPTION == 1:
    train_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
    forcing_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
    # years = list(range(1993, 2004)) + list(range(2005, 2021))
    years = list(range(2005, 2021))
    stats_file = os.path.join(train_file_path, f'stats_OPTION{OPTION}.npz')
    stats = np.load(stats_file)
    meanx, stdx = stats['meanx'], stats['stdx']
    meanf, stdf = stats['meanf'], stats['stdf']

    clims = []
    for m in tqdm(range(1, 13), desc='Monthly climatology'):
        train_file_names = [*( (f'wavemean_{y:04d}{m:02d}', f'forcing_{y:04d}{m:02d}') for y in years)]
        train_file_list = [(os.path.join(train_file_path, f'{x}.npy'),
                            os.path.join(forcing_file_path, f'{f}.npy')) for x, f in train_file_names]
        data = npyDataResized(
            train_file_list,
            resize_x=(320,320), resize_f=(320,320),
            landmaskname=os.path.join(train_file_path, 'mask.npy'),
            use_icymask=True, compute_stats=False,
            meanx=meanx, stdx=stdx, meanf=meanf, stdf=stdf,
            OPTION=OPTION
        )
        clim_meanx, clim_stdx, clim_meanf, clim_stdf = data.compute_clim_map()
        ds_clim = add_lat_lon(clim_meanx[:], ['hs','tm','dir']) 
        ds_clim.to_netcdf(f'~/scratch_folder/final/processed/mean_clim_{m:02d}.nc')  # Save each month's climatology separately to avoid memory issues
        clims.append(ds_clim)
        del data, clim_meanx, clim_stdx, clim_meanf, clim_stdf
        gc.collect()

    # Stack the monthly climatology into a single xarray dataset
    clims = xr.concat(clims, dim='month')
    clims.to_netcdf(f'~/scratch_folder/final/processed/mean_clim_monthly.nc')

###############################
# Saving the mean variables climatology for OPTION=2
if OPTION == 2:
    train_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
    forcing_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
    years = list(range(2005, 2021))
    stats_file = os.path.join(train_file_path, f'stats_OPTION{OPTION}.npz')
    stats = np.load(stats_file)
    meanx, stdx = stats['meanx'], stats['stdx']
    meanf, stdf = stats['meanf'], stats['stdf']

    clims = []
    for m in tqdm(range(1, 13), desc='Monthly climatology'):
        train_file_names = [*( (f'wavemean_{y:04d}{m:02d}', f'forcing_{y:04d}{m:02d}') for y in years)]
        train_file_list = [(os.path.join(train_file_path, f'{x}.npy'),
                            os.path.join(forcing_file_path, f'{f}.npy')) for x, f in train_file_names]
        data = npyDataResized(
            train_file_list,
            resize_x=(320,320), resize_f=(320,320),
            landmaskname=os.path.join(train_file_path, 'mask.npy'),
            use_icymask=True, compute_stats=False,
            meanx=meanx, stdx=stdx, meanf=meanf, stdf=stdf,
            OPTION=OPTION
        )
        clim_meanx, clim_stdx, clim_meanf, clim_stdf = data.compute_clim_map()
        ds_clim = add_lat_lon(clim_meanx[[0,4,5,6,7]], ['hs', 'uss', 'vss', 'mssd', 'mssc'])
        ds_clim.to_netcdf(f'~/scratch_folder/final/processed/derived_clim_{m:02d}.nc')
        clims.append(ds_clim)
        del data, clim_meanx, clim_stdx, clim_meanf, clim_stdf
        gc.collect()

    # Stack the monthly climatology into a single xarray dataset
    clims = xr.concat(clims, dim='month')
    clims.to_netcdf(f'~/scratch_folder/final/processed/derived_clim_monthly.nc')
    
################################
# Saving the mean variables climatology for OPTION=3 (the partition variables)
if OPTION == 3:
    train_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/partition_global/'
    forcing_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
    years = list(range(1993, 2004)) + list(range(2005, 2021))

    stats_file = os.path.join(train_file_path, f'stats_OPTION{OPTION}.npz')
    stats = np.load(stats_file)
    meanx, stdx = stats['meanx'], stats['stdx']
    meanf, stdf = stats['meanf'], stats['stdf']

    clims = []
    for m in tqdm(range(1, 13), desc='Monthly climatology'):
        train_file_names = [*( (f'waveparts_{y:04d}{m:02d}', f'forcing_{y:04d}{m:02d}') for y in years)]
        train_file_list = [(os.path.join(train_file_path, f'{x}.npy'),
                            os.path.join(forcing_file_path, f'{f}.npy')) for x, f in train_file_names]
        data = npyDataResized(
            train_file_list,
            resize_x=(320,320), resize_f=(320,320),
            landmaskname=os.path.join(train_file_path, 'mask.npy'),
            use_icymask=True, compute_stats=False,
            meanx=meanx, stdx=stdx, meanf=meanf, stdf=stdf,
            OPTION=OPTION
        )
        clim_meanx, clim_stdx, clim_meanf, clim_stdf = data.compute_clim_map()
        ds_clim = add_lat_lon(clim_meanx, ['hs_p1', 'tp_p1', 'thetap_p1', 'hs_p2', 'tp_p2', 'thetap_p2', 'ci'])  # Only save the first three variables, and add lat/lon coordinates for saving as netcdf
        ds_clim.to_netcdf(f'~/scratch_folder/final/processed/partition_clim_{m:02d}.nc')  # Save each month's climatology separately to avoid memory issues
        clims.append(ds_clim)
        del data, clim_meanx, clim_stdx, clim_meanf, clim_stdf
        gc.collect()

    # Stack the monthly climatology into a single xarray dataset
    clims = xr.concat(clims, dim='month')
    clims.to_netcdf(f'~/scratch_folder/final/processed/partition_clim_monthly.nc')
