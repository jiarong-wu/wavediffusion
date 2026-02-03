import xarray as xr
import numpy as np

def read_save_mean (readpath='/scratch/jw8736/wavecnn/data/', savepath='../datasets/',
                    year=2008, month=1):
    ### Generate mean data
    ds = xr.open_dataset(readpath + f'LOPS_WW3-GLOB-30M_{year:04d}{month:02d}.nc')
    hs_mean = ds.hs.values
    lp_mean = ds.lm.values
    dir_mean = ds.dir.values
    spr_mean = ds.spr.values
    uuss = ds.uuss.values
    vuss = ds.vuss.values
    mssu = ds.mssu.values
    mssc = ds.mssc.values
    mssd = ds.mssd.values
    mean = np.stack([hs_mean, lp_mean, dir_mean, spr_mean, uuss, vuss, mssu, mssc, mssd], axis=0)
    mean = np.swapaxes(mean, 0, 1)
    # np.save(savepath + 'wave_mean.npy', mean)
    
    ### Generate forcing data
    # Icy mask
    criterion = ds.dpt.isnull()
    mask = xr.where(criterion == 0, 1, 0)
    icymask = xr.where((mask == 0) | (ds.ice > 0), 0, 1)
    forcing = np.stack([ds.uwnd.values, ds.vwnd.values, icymask.values, ds.dpt.values, ds.ice.values], axis=0)
    forcing = np.swapaxes(forcing, 0, 1)
    # np.save(savepath + 'forcing.npy', np.swapaxes(forcing, 0, 1))
    return mean, forcing

### This works with the half-processed data 
### e.g. 201101.nc files with hs, lp, dm, dspr variables 
def read_save_maxe (readpath='/scratch/jw8736/wavecnn/data/', savepath='../datasets/',
                    year=2008, month=1):
    ds = xr.open_dataset(readpath + f'{year:04d}{month:02d}.nc')   
    hs = np.swapaxes(ds.hs.values, 0, 1)
    lp = np.swapaxes(ds.lp.values, 0, 1)
    dm = np.swapaxes(ds.dm.values, 0, 1)
    dspr = np.swapaxes(ds.dspr.values, 0, 1)
    hs_fill = np.nan_to_num(hs, nan=0.0)   

    wave_max = np.zeros((hs.shape[0], 4, hs.shape[-2], hs.shape[-1]))
    for i in range(hs.shape[0]):
        max_hs = np.max(hs_fill[i], axis=0)
        max_idx  = np.argmax(hs_fill[i], axis=0)
        I = np.arange(max_idx.shape[0])[:, None]
        J = np.arange(max_idx.shape[1])[None, :]
        max_lp = lp[i][max_idx, I, J]
        max_dm = dm[i][max_idx, I, J]
        max_dspr = dspr[i][max_idx, I, J]
        wave_max[i] = np.stack([max_hs, max_lp, max_dm, max_dspr], axis=0)
    # np.save(savepath + 'wave_maxe.npy', wave_max)
    return wave_max
    

''' For now only do the mean.
    Try working on the maxe data later. 
    Training: 2011-01 to 2011-12
    Testing: 2008-04, 2008-12
'''
    
if __name__=='__main__':
    readpath = '/global/homes/j/jiarongw/scratch_folder/wave_data/raw/'
    savepath = '/global/homes/j/jiarongw/scratch_folder/wave_data/stokes_global/'
    for year in range(2001,2003):
        for month in range(1,13):
            print('Processing year: {}, month: {} ...'.format(year, month))
            wave, forcing = read_save_mean(readpath, savepath, year, month)
            # Global but remove three latitude rows
            np.save(savepath + f'wave_{year:04d}{month:02d}.npy', wave[:401, :, 1:-2])
            np.save(savepath + f'forcing_{year:04d}{month:02d}.npy', forcing[:401, :, 1:-2])
            # Mask already saved
            # mask = np.load('../datasets/mask.npy')
            # np.save('../datasets/train_global/mask.npy', mask[1:-2])
            # np.save('../datasets/test_global/mask.npy', mask[1:-2])    
    
    
    # Pick a region
    # wave = np.load('../datasets/wave_mean.npy')
    # forcing = np.load('../datasets/forcing.npy')    
    # np.save('../datasets/train/wave.npy', wave[:401,:,50:114,0:64])
    # np.save('../datasets/train/forcing.npy', forcing[:401,:,50:114,0:64])
    # np.save('../datasets/test/wave.npy', wave[402:,:,50:114,0:64])
    # np.save('../datasets/test/forcing.npy', forcing[402:,:,50:114,0:64])

    # mask = np.load('../datasets/mask.npy')
    # np.save('../datasets/train/mask.npy', mask[50:114,0:64])
    # np.save('../datasets/test/mask.npy', mask[50:114,0:64])