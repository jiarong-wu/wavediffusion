''' Paper figure: climatology of the truth (WW3 hindcast) over the evaluation
    period, 2020-2025 every 5 days (00 UTC, 439 dates; the same dates as
    compare_checkpoints_period.sh): temporal mean and standard deviation maps.
      OPTION1: rows mean / std, columns Hs, Tm, theta_m; with the region boxes of
               regional_variability.py (REGIONS).
      OPTION3: rows partition 1 mean / std, partition 2 mean / std, columns Hs,
               Tp, theta_p of that partition.

      python climatology_maps.py compute [--option 1|3]    (CPU, ~1 min; caches the stats)
      python climatology_maps.py plot [--option 1|3] [--season all|DJF|MAM|JJA|SON]

    `compute` reads the native 0.5 deg truth (wavemean_YYYYMM.npy / waveparts_YYYYMM.npy)
    and the ice flag of the forcing files (channel 2, > 0.5 = ice-free) and writes,
    for all dates and each season, the per-pixel mean and variance and the
    fraction of dates the pixel is ice-covered, to
    output/climatology_OPTION<n>_<period>.npz. Directions get the circular mean
    and, as their variance, the mean squared angular difference from it (deg^2).
    OPTION3 stores an absent partition as 0: Hs_pk keeps those zeros (its mean is
    the mean partition height, absent = 0), Tp_pk and theta_pk count only dates
    where partition k exists (hs_pk > 0), so their date counts differ per pixel.
    `plot` masks pixels that are ice-covered on more than ICE_FRAC_MAX of the
    dates of that season (default 0: any ice in the period), so every plotted
    pixel averages the same dates; the masked ice zone is left white, land grey.
    The std plotted is the square root of the cached variance (directions: the
    RMS angular difference from the circular mean, deg). Maps are centred on the
    Pacific (180 deg).
    Writes figures/climatology_OPTION<n>_<season>.{png,pdf}.
'''
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR, FIG_DIR = os.path.join(HERE, 'output'), os.path.join(HERE, 'figures')
WAVE_DATA = '/global/homes/j/jiarongw/scratch_folder/wave_data/'
MEAN_PATH = WAVE_DATA + 'mean_global/'    # wavemean_* and forcing_*
DATES = pd.date_range('2020-01-01', '2025-12-31', freq='5D')
PERIOD = f'{DATES[0].year}-{DATES[-1].year}_every5d'
SEASONS = {'DJF': (12, 1, 2), 'MAM': (3, 4, 5), 'JJA': (6, 7, 8), 'SON': (9, 10, 11)}
ICE_FRAC_MAX = 0.0
DIR_TICKS = [0, 90, 180, 270, 360]

# vars: channel in the data file, label, unit, mean / std panels (cmap, vmin, vmax, ticks), and
#   `requires`: a variable that must be > 0 for a date to count (OPTION3: the partition exists).
# rows: (row label, 'mean' | 'std', variable per column); columns are titled by `col_titles`.
# groups (optional): (label, row indices) drawn as a grey background box behind those rows.
OPTIONS = {
    1: dict(path=MEAN_PATH, prefix='wavemean', boxes=True,
            vars=[
                dict(name='hs', ch=0, label='$H_s$', unit='m', circular=False,
                     mean=('Blues', 0, 5, None), std=('magma', 0, 2, None)),
                dict(name='tm', ch=1, label='$T_m$', unit='s', circular=False,
                     mean=('pink_r', 0, 12, None), std=('magma', 0, 2.5, None)),
                dict(name='thetam', ch=2, label=r'$\theta_m$', unit='deg', circular=True,
                     mean=('twilight', 0, 360, DIR_TICKS), std=('magma', 0, 100, None)),
            ],
            col_titles=['$H_s$', '$T_m$', r'$\theta_m$'],
            rows=[('Mean', 'mean', ['hs', 'tm', 'thetam']),
                  ('Std. dev.', 'std', ['hs', 'tm', 'thetam'])]),
    3: dict(path=WAVE_DATA + 'partition_global/', prefix='waveparts', boxes=False,
            vars=[
                dict(name='hs_p1', ch=0, label='$H_{sp1}$', unit='m', circular=False,
                     mean=('Blues', 0, 5, None), std=('magma', 0, 2, None)),
                dict(name='tp_p1', ch=1, label='$T_{p1}$', unit='s', circular=False, requires='hs_p1',
                     mean=('pink_r', 0, 14, None), std=('magma', 0, 5, None)),
                dict(name='thetap_p1', ch=2, label=r'$\theta_{p1}$', unit='deg', circular=True, requires='hs_p1',
                     mean=('twilight', 0, 360, DIR_TICKS), std=('magma', 0, 100, None)),
                dict(name='hs_p2', ch=3, label='$H_{sp2}$', unit='m', circular=False,
                     mean=('Blues', 0, 1.5, None), std=('magma', 0, 1, None)),
                dict(name='tp_p2', ch=4, label='$T_{p2}$', unit='s', circular=False, requires='hs_p2',
                     mean=('pink_r', 0, 14, None), std=('magma', 0, 5, None)),
                dict(name='thetap_p2', ch=5, label=r'$\theta_{p2}$', unit='deg', circular=True, requires='hs_p2',
                     mean=('twilight', 0, 360, DIR_TICKS), std=('magma', 0, 100, None)),
            ],
            col_titles=['$H_s$', '$T$', r'$\theta$'],
            rows=[('Mean', 'mean', ['hs_p1', 'tp_p1', 'thetap_p1']),
                  ('Std. dev.', 'std', ['hs_p1', 'tp_p1', 'thetap_p1']),
                  ('Mean', 'mean', ['hs_p2', 'tp_p2', 'thetap_p2']),
                  ('Std. dev.', 'std', ['hs_p2', 'tp_p2', 'thetap_p2'])],
            groups=[('Partition 1', (0, 1)), ('Partition 2', (2, 3))]),
}


def cache_path(option):
    return os.path.join(OUT_DIR, f'climatology_OPTION{option}_{PERIOD}.npz')


def load_date(cfg, d):
    ''' Native-grid truth (C, H, W) at date d, NaN where ice-covered or land, or
        (for a `requires` variable) where that variable is not > 0. '''
    ym = f'{d.year}{d.month:02d}'
    step = (d - pd.Timestamp(year=d.year, month=d.month, day=1)) // pd.Timedelta(hours=3)
    x = np.load(os.path.join(cfg['path'], f"{cfg['prefix']}_{ym}.npy"), mmap_mode='r')[step]
    x = np.array(x[[v['ch'] for v in cfg['vars']]], dtype=np.float32)
    ice_free = np.load(os.path.join(MEAN_PATH, f'forcing_{ym}.npy'), mmap_mode='r')[step, 2] > 0.5
    names = [v['name'] for v in cfg['vars']]
    for c, v in enumerate(cfg['vars']):
        if 'requires' in v:
            x[c, ~(x[names.index(v['requires'])] > 0)] = np.nan
    x[:, ~ice_free] = np.nan
    return x, ice_free


def mean_var(a, circular):
    ''' Temporal mean and variance (axis 0), ignoring NaN; circular in degrees. '''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)    # all-NaN pixels (land)
        if not circular:
            return np.nanmean(a, 0), np.nanvar(a, 0)
        rad = np.deg2rad(a)
        mean = np.rad2deg(np.arctan2(np.nanmean(np.sin(rad), 0), np.nanmean(np.cos(rad), 0))) % 360
        diff = (a - mean[None] + 180) % 360 - 180
        return mean, np.nanmean(diff ** 2, 0)


def compute(option):
    cfg = OPTIONS[option]
    x, ice_free = zip(*(load_date(cfg, d) for d in tqdm(DATES, desc='truth')))
    x, ice_free = np.stack(x), np.stack(ice_free)    # (T, C, H, W), (T, H, W)
    out = dict(dates=DATES.values.astype('datetime64[s]'), vars=np.array([v['name'] for v in cfg['vars']]))
    for season, months in [('all', range(1, 13))] + list(SEASONS.items()):
        sel = DATES.month.isin(months)
        for c, v in enumerate(cfg['vars']):
            out[f"{v['name']}_mean_{season}"], out[f"{v['name']}_var_{season}"] = \
                (s.astype(np.float32) for s in mean_var(x[sel, c], v['circular']))
        # fraction of the season's dates the pixel is ice-covered (land: 1)
        out[f'ice_frac_{season}'] = (1 - ice_free[sel].mean(0)).astype(np.float32)
        out[f'n_dates_{season}'] = sel.sum()
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(cache_path(option), **out)
    print(f'Wrote {cache_path(option)}')


def draw_boxes(ax, proj, labels):
    ''' The regions of regional_variability.py, in the Pacific-centred frame. '''
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle
    from regional_variability import REGIONS
    for name, (_, lat0, lat1, lon0, lon1) in REGIONS.items():
        a, b = lon0 % 360, lon1 % 360
        b = b + 360 if b < a else b
        ax.add_patch(Rectangle((a - 180, lat0), b - a, lat1 - lat0, transform=proj, fill=False,
                               ec='k', lw=0.9, zorder=3,
                               path_effects=[pe.withStroke(linewidth=2.2, foreground='w')]))    # visible on dark too
        if labels:
            ax.text(a - 180 + 1.5, lat1 - 1.5, name, transform=proj, va='top', fontsize=7, zorder=4,
                    path_effects=[pe.withStroke(linewidth=2, foreground='w')])


def draw_groups(fig, axes, cbar_axes, groups, pad=0.006, label_w=0.035):
    ''' A grey background box behind each group of rows (maps, row labels and
        colorbars), split halfway between neighbouring groups, with the group's
        label written vertically on its left. '''
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Bbox
    fig.canvas.draw()    # final axes positions (cartopy fixes the aspect at draw)
    to_fig = fig.transFigure.inverted()
    boxes = []
    for _, rows in groups:
        arts = [a for i in rows for a in list(axes[i]) + cbar_axes[i]]
        bb = Bbox.union([a.get_tightbbox() for a in arts]).transformed(to_fig)
        boxes.append([bb.x0, bb.y0, bb.x1, bb.y1])
    x0, x1 = min(b[0] for b in boxes) - pad - label_w, max(b[2] for b in boxes) + pad
    for k in range(len(boxes) - 1):    # rows go top to bottom: split the gap between groups k and k + 1
        mid = 0.5 * (boxes[k][1] + boxes[k + 1][3])
        boxes[k][1], boxes[k + 1][3] = mid + pad / 2, mid - pad / 2
    boxes[0][3] += pad
    boxes[-1][1] -= pad
    for (label, _), (_, y0, _, y1) in zip(groups, boxes):
        fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure, facecolor='#e6e6e6',
                                 edgecolor='none', zorder=-1))
        fig.text(x0 + label_w / 2, (y0 + y1) / 2, label, rotation=90, ha='center', va='center',
                 fontsize=18, fontweight='bold')


def plot(option, season):
    from matplotlib import pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from wavediffusion.waveana import lat_deg, lon_deg
    cfg = OPTIONS[option]
    var = {v['name']: v for v in cfg['vars']}
    z = np.load(cache_path(option))
    masked = z[f'ice_frac_{season}'] > ICE_FRAC_MAX
    proj, data_crs = ccrs.PlateCarree(central_longitude=180), ccrs.PlateCarree()    # Pacific-centred
    n_rows, n_cols = len(cfg['rows']), len(cfg['col_titles'])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.1), subplot_kw={'projection': proj},
                             dpi=300, squeeze=False)
    cbar_axes = [[] for _ in range(n_rows)]
    for i, (row_label, kind, names) in enumerate(cfg['rows']):
        for j, name in enumerate(names):
            v = var[name]
            cmap, vmin, vmax, ticks = v[kind]
            field = z[f'{name}_mean_{season}'] if kind == 'mean' else np.sqrt(z[f'{name}_var_{season}'])
            field = np.where(masked, np.nan, field)
            ax = axes[i, j]
            im = ax.pcolormesh(lon_deg, lat_deg, field, transform=data_crs, cmap=cmap, vmin=vmin, vmax=vmax,
                               shading='auto', rasterized=True)
            ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
            ax.set_global()
            if cfg['boxes']:
                draw_boxes(ax, proj, labels=(j == 0))
            if i == 0:
                ax.set_title(cfg['col_titles'][j], fontsize=18, pad=10)
            if j == 0:
                ax.text(-0.08, 0.5, row_label, transform=ax.transAxes, fontsize=16, va='center',
                        ha='center', rotation=90)
            cb = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8, aspect=30,
                              extend='neither' if v['circular'] and kind == 'mean' else 'max')
            cb.set_label(f"{v['label']} {'mean' if kind == 'mean' else 'std. dev.'} [{v['unit']}]", fontsize=12)
            cb.ax.tick_params(labelsize=10)
            if ticks is not None:
                cb.set_ticks(ticks)
            cbar_axes[i].append(cb.ax)
    if cfg.get('groups'):
        draw_groups(fig, axes, cbar_axes, cfg['groups'])
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        out = os.path.join(FIG_DIR, f'climatology_OPTION{option}_{season}.{ext}')
        fig.savefig(out, bbox_inches='tight')
        print(f'Wrote {out}')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['compute', 'plot'])
    p.add_argument('--option', type=int, default=1, choices=sorted(OPTIONS))
    p.add_argument('--season', default='all', choices=['all'] + list(SEASONS))
    a = p.parse_args()
    compute(a.option) if a.stage == 'compute' else plot(a.option, a.season)


if __name__ == '__main__':
    main()
