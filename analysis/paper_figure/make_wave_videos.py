''' Render side-by-side truth vs. diffusion-sample animations from the OSN netCDF files.
    One video per variable (wave height, wave period, wave direction).
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
import os

NC_PATHS = [
    '/global/homes/j/jiarongw/scratch_folder/final/temp/OPTION1_moredata_sigma100_epoch20_200404/sample_data_200404.nc',
    '/global/homes/j/jiarongw/scratch_folder/final/temp/OPTION1_moredata_sigma100_epoch20_200409/sample_data_200409.nc',
]
OUT_DIR = '/global/u1/j/jiarongw/wavediffusion/analysis/paper_figure/videos/'

VARIABLES = {
    'hs':  dict(long_name='Significant wave height', units='m',   cmap='Blues',    vmin=0, vmax=10,  ticks=[0, 2, 4, 6, 8, 10]),
    'tm':  dict(long_name='Mean wave period',         units='s',   cmap='Reds',     vmin=0, vmax=15,  ticks=[0, 5, 10, 15]),
    'dir': dict(long_name='Mean wave direction',      units='deg', cmap='twilight', vmin=0, vmax=360, ticks=[0, 90, 180, 270, 360]),
}

FPS = 10
HOLD_SECONDS = 0.2  # how long each date is held on screen


def load_dataset(paths):
    ds = xr.concat([xr.open_dataset(p) for p in paths], dim='time')
    return ds.sortby('time')


def make_video(ds, var, out_path):
    meta = VARIABLES[var]
    truth = ds[f'{var}_truth']
    sample = ds[f'{var}_sample']
    times = ds.time.values
    extent = [float(ds.lon.min()), float(ds.lon.max()), float(ds.lat.min()), float(ds.lat.max())]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), dpi=130,
                              subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(f"{meta['long_name']} ({meta['units']})", fontsize=14)

    ims = []
    for ax, title, da in zip(axes, ['Truth', 'Diffusion sample'], [truth, sample]):
        ax.set_title(title)
        ax.add_feature(cfeature.LAND, facecolor='0.85', zorder=2)
        ax.coastlines(resolution='110m', linewidth=0.5, zorder=3)
        im = ax.imshow(da.isel(time=0).values, origin='lower', extent=extent,
                        transform=ccrs.PlateCarree(),
                        cmap=meta['cmap'], vmin=meta['vmin'], vmax=meta['vmax'])
        ims.append(im)
    cbar = fig.colorbar(ims[0], ax=axes, orientation='horizontal', fraction=0.05, pad=0.08, ticks=meta['ticks'])
    cbar.set_label(meta['units'])

    date_text = fig.text(0.5, 0.90, '', ha='center', fontsize=11)

    n_hold = max(1, int(round(HOLD_SECONDS * FPS)))
    frame_order = [t for t in range(len(times)) for _ in range(n_hold)]

    def update(i):
        t = frame_order[i]
        ims[0].set_data(truth.isel(time=t).values)
        ims[1].set_data(sample.isel(time=t).values)
        date_text.set_text(np.datetime_as_string(times[t], unit='D'))
        return ims + [date_text]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_order), blit=False)
    ani.save(out_path, writer=animation.PillowWriter(fps=FPS))
    plt.close(fig)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_dataset(NC_PATHS)
    print(f"Frames (dates): {np.datetime_as_string(ds.time.values, unit='D')}")
    for var in VARIABLES:
        make_video(ds, var, os.path.join(OUT_DIR, f'{var}_truth_vs_sample.gif'))
