''' Paper figure: maps of the mean bias, RMSE and CRPS of the OPTION3 (or, with
    --option 1, OPTION1) models
    against the truth over 2020-2025 every 5 days (00 UTC, 439 dates), from the
    per-date map files of `compare_checkpoints_period.sh` (compare_checkpoints_stats.py
    `maps`). One figure per variable: rows = models (MODELS), columns = bias of
    the ensemble mean (model - truth), RMSE of the ensemble mean, CRPS (fair ensemble CRPS; for the
    deterministic model the MAE, its CRPS).

      python skill_maps.py compute [--option 1|3] [--native] [VAR ...]   (CPU, ~10 min; caches the per-pixel sums)
      python skill_maps.py plot [--option 1|3] [--native] [VAR ...]      (default: all of VARS in the cache)

    The truth is the one in the map files: on the model grid (through the
    320x320 resize, like the models) and zero on land / ice. A pixel counts on
    a date where it is ice-free then (the files' `valid`), for every variable:
    the partition filter below is currently commented out in compute(), so
    absent partitions (stored as 0 in truth and model) count too. With it on,
    tp_pk and thetap_pk count only where partition k exists in the truth
    (hs_pk > HS_EXISTS). Not hs_pk > 0:
    the resize smears a partition into a halo of near-zero heights, and how far
    depends on a column's position relative to the coarse grid (the grids line
    up every 9 columns), so that test keeps a slightly different halo per column
    and stripes the tp_p2 bias. Directions use the
    circular difference (deg). As in climatology_maps.py, pixels ice-covered on
    any date are masked (white), land is grey, and maps are centred on the Pacific.
    With --native, the truth is instead the native-grid one (waveparts_*.npy,
    hs_tot from its hs_p1 and hs_p2), so the models are scored on the grid they
    are used on, and the truth carries no resize artefacts (the 720 -> 320 -> 720
    longitude resampling lines up every 9 columns and stripes fields with sharp
    jumps, such as tp_p2). A pixel then counts where it is ice-free on both grids
    (the native forcing flag and the files' `valid`) and, for tp_pk / thetap_pk,
    where partition k exists in the native truth. Results go to *_native files.
    Its CRPS is approximate: the saved fair CRPS is against the resized truth and
    the members aren't saved, so it is the CRPS of a Gaussian with the ensemble
    mean and variance (the deterministic model: the MAE, as before).
    The bias column is relative (% of the truth's mean over the same dates) for
    heights and periods, absolute (deg) for directions.
    --native is OPTION3 only.
    Writes output/skill_maps_OPTION<n>_<period>[_native].npz and
    figures/skill_maps_<var>_OPTION<n>_<period>[_native].{png,pdf}.
'''
import os
import sys
import numpy as np
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR, FIG_DIR = os.path.join(HERE, 'output'), os.path.join(HERE, 'figures')
PERIOD = '2020-2025_every5d'
OPTION = 3
if '--option' in sys.argv:
    _i = sys.argv.index('--option')
    OPTION = int(sys.argv[_i + 1])
    del sys.argv[_i:_i + 2]
MAPS_DIR = f'/global/homes/j/jiarongw/scratch_folder/review/checkpoint_comparison_maps_{PERIOD}/OPTION{OPTION}/'
WAVE_DATA = '/global/homes/j/jiarongw/scratch_folder/wave_data/'
PART_VARS = ['hs_p1', 'tp_p1', 'thetap_p1', 'hs_p2', 'tp_p2', 'thetap_p2', 'ci']    # waveparts_* channels
NATIVE = '--native' in sys.argv
HS_EXISTS = 0.05    # m; partition k exists where the truth's hs_pk exceeds this (see above)
# rows, top to bottom
MODELS = {3: {'highsigma_ckpt5': 'DDPM', 'hist10d_det_ckpt20': 'Deterministic', 'resid_ckpt10': 'Res-DDPM'},
          1: {'highsigma_ckpt6': 'DDPM', 'hist10d_det_ckpt20': 'Deterministic', 'resid_ckpt10': 'Res-DDPM'}}[OPTION]
# Per variable: label, unit, circular, the partition it needs, and colour limits for bias (+-; % for
# heights and periods, deg for directions and `absolute` ones), RMSE and CRPS
VARS3 = {
    'hs_tot':    dict(label=r'$\sqrt{H_{sp1}^2 + H_{sp2}^2}$', unit='m', lim=(10, 0.4, 0.25)),
    'hs_p1':     dict(label='$H_{sp1}$', unit='m', lim=(10, 0.4, 0.25)),
    'tp_p1':     dict(label='$T_{p1}$', unit='s', requires='hs_p1', lim=(5, 2.5, 1.5)),
    'thetap_p1': dict(label=r'$\theta_{p1}$', unit='deg', circular=True, requires='hs_p1', lim=(10, 50, 25)),
    'hs_p2':     dict(label='$H_{sp2}$', unit='m', lim=(20, 0.5, 0.25)),
    'tp_p2':     dict(label='$T_{p2}$', unit='s', requires='hs_p2', lim=(10, 5, 3)),
    'thetap_p2': dict(label=r'$\theta_{p2}$', unit='deg', circular=True, requires='hs_p2', lim=(10, 75, 50)),
    # crossing-sea criterion, 0..1 and mostly 0 or 1: absolute bias (a percentage blows up where it is rare)
    'ci':        dict(label='CI', unit='-', absolute=True, lim=(0.1, 0.4, 0.2)),
}
VARS1 = {
    'hs':     dict(label='$H_s$', unit='m', lim=(10, 0.4, 0.25)),
    'tm':     dict(label='$T_m$', unit='s', lim=(5, 1.5, 1)),
    'thetam': dict(label=r'$\theta_m$', unit='deg', circular=True, lim=(10, 40, 20)),
}
VARS = {1: VARS1, 3: VARS3}[OPTION]


def tag():
    return '_native' if NATIVE else ''


def cache_path():
    return os.path.join(OUT_DIR, f'skill_maps_OPTION{OPTION}_{PERIOD}{tag()}.npz')


def load_native(d):
    ''' Native-grid truth {var: (H, W)} (VARS, hs_tot included) and the ice-free flag at date d (00 UTC). '''
    import pandas as pd
    d = pd.Timestamp(d)
    ym, step = f'{d.year}{d.month:02d}', (d.day - 1) * 8
    p = np.load(os.path.join(WAVE_DATA, 'partition_global', f'waveparts_{ym}.npy'), mmap_mode='r')[step]
    x = {v: np.asarray(p[c], dtype=np.float32) for c, v in enumerate(PART_VARS)}
    x['hs_tot'] = np.hypot(x['hs_p1'], x['hs_p2'])
    ice_free = np.load(os.path.join(WAVE_DATA, 'mean_global', f'forcing_{ym}.npy'), mmap_mode='r')[step, 2] > 0.5
    return x, ice_free


def circular_diff(a, b):
    return (a - b + 180) % 360 - 180


def crps_gaussian(err, sigma):
    ''' CRPS of N(mean, sigma^2) for an observation at mean - err (|err| where sigma = 0). '''
    from scipy.special import erf
    with np.errstate(invalid='ignore', divide='ignore'):
        u = err / sigma
        c = sigma * (u * erf(u / np.sqrt(2)) + 2 * np.exp(-u ** 2 / 2) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))
    return np.where(sigma > 0, c, np.abs(err))


def compute(names):
    names = names or list(VARS)
    dates = sorted(f[:-4] for f in os.listdir(os.path.join(MAPS_DIR, 'truth')) if f.endswith('.npz'))
    for m in MODELS:
        missing = set(dates) - {f[:-4] for f in os.listdir(os.path.join(MAPS_DIR, m))}
        if missing:
            raise RuntimeError(f'{m} has no maps for {len(missing)} dates, e.g. {sorted(missing)[:3]}')
    sums = {}
    n_valid = 0
    for d in tqdm(dates, desc='maps'):
        t = np.load(os.path.join(MAPS_DIR, 'truth', f'{d}.npz'))
        valid = t['valid']
        tv = list(t['vars'])
        truth = {v: t['truth'][tv.index(v)] for v in set(names + ['hs_p1', 'hs_p2']) if v in tv}
        if NATIVE:
            truth, ice_free = load_native(d)
            valid = valid & ice_free
        n_valid = n_valid + valid
        # Partition filter, off: tp_pk / thetap_pk only where partition k exists (truth hs_pk > HS_EXISTS)
        # ok = {v: valid & (truth[VARS[v]['requires']] > HS_EXISTS) if 'requires' in VARS[v] else valid for v in names}
        ok = {v: valid for v in names}    # every ice-free date (absent partitions are 0 in truth and model)
        for v in names:    # the truth's mean over the same dates, for the relative bias
            sums[f'truth/{v}/sum'] = sums.get(f'truth/{v}/sum', 0) + np.where(ok[v], truth[v], 0).astype(np.float64)
        for m in MODELS:
            z = np.load(os.path.join(MAPS_DIR, m, f'{d}.npz'))
            zv = list(z['vars'])
            for v in names:
                cfg = VARS[v]
                pred, true = z['mean'][zv.index(v)], truth[v]
                err = circular_diff(pred, true) if cfg.get('circular') else pred - true
                if 'crps' not in z.files:
                    crps = np.abs(err)    # deterministic
                elif NATIVE:    # the saved CRPS is against the resized truth; only mean and variance are kept
                    crps = crps_gaussian(err, np.sqrt(np.maximum(z['var'][zv.index(v)], 0)))
                else:
                    crps = z['crps'][zv.index(v)]
                for k, a in dict(n=np.ones_like(err), err=err, se=err ** 2, crps=crps).items():
                    key = f'{m}/{v}/{k}'
                    sums[key] = sums.get(key, 0) + np.where(ok[v], a, 0).astype(np.float64)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(cache_path(), n_dates=len(dates), n_valid=n_valid, dates=np.array(dates),
                        **{k: a.astype(np.float32) for k, a in sums.items()})
    print(f'Wrote {cache_path()} ({len(dates)} dates)')


def stats(z, m, v):
    ''' Per-pixel bias, rbias (100 * bias / the truth's mean; NaN for directions), RMSE and CRPS. '''
    n = z[f'{m}/{v}/n']
    with np.errstate(invalid='ignore', divide='ignore'):
        bias = z[f'{m}/{v}/err'] / n
        rbias = np.full_like(bias, np.nan) if VARS[v].get('circular') else 100 * bias / (z[f'truth/{v}/sum'] / n)
        return dict(bias=bias, rbias=rbias, rmse=np.sqrt(z[f'{m}/{v}/se'] / n), crps=z[f'{m}/{v}/crps'] / n)


def plot(names):
    from matplotlib import pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from wavediffusion.waveana import lat_deg, lon_deg
    z = np.load(cache_path())
    ice = z['n_valid'] < z['n_dates']    # ice-covered (or land) on some date
    proj, data_crs = ccrs.PlateCarree(central_longitude=180), ccrs.PlateCarree()    # Pacific-centred
    os.makedirs(FIG_DIR, exist_ok=True)
    for v in names:
        cfg = VARS[v]
        absolute = cfg.get('circular') or cfg.get('absolute')
        bias_col = ('bias', 'Bias', 'RdBu_r') if absolute else ('rbias', 'Relative bias', 'RdBu_r')
        crps_title = 'CRPS (Gaussian)' if NATIVE else 'CRPS'
        cols = [bias_col, ('rmse', 'RMSE', 'viridis'), ('crps', crps_title, 'cividis')]
        fig, axes = plt.subplots(len(MODELS), len(cols), figsize=(len(cols) * 5, len(MODELS) * 3.1),
                                 subplot_kw={'projection': proj}, dpi=300, squeeze=False)
        for i, (m, row_label) in enumerate(MODELS.items()):
            s = stats(z, m, v)
            for j, ((k, title, cmap), lim) in enumerate(zip(cols, cfg['lim'])):
                ax = axes[i, j]
                vmin = -lim if k in ('bias', 'rbias') else 0
                im = ax.pcolormesh(lon_deg, lat_deg, np.where(ice, np.nan, s[k]), transform=data_crs, cmap=cmap,
                                   vmin=vmin, vmax=lim, shading='auto', rasterized=True)
                ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
                ax.set_global()
                if i == 0:
                    ax.set_title(title, fontsize=18, pad=10)
                if j == 0:
                    ax.text(-0.08, 0.5, row_label, transform=ax.transAxes, fontsize=16, va='center',
                            ha='center', rotation=90)
                if i == len(MODELS) - 1:    # one colorbar per column, shared by the models
                    cb = fig.colorbar(im, ax=axes[:, j].tolist(), orientation='horizontal', pad=0.04,
                                      shrink=0.8, aspect=30, extend='both' if k in ('bias', 'rbias') else 'max')
                    unit = '%' if k == 'rbias' else cfg['unit']
                    cb.set_label(f"{cfg['label']} {title} [{unit}]", fontsize=12)
                    cb.ax.tick_params(labelsize=10)
        for ext in ('png', 'pdf'):
            out = os.path.join(FIG_DIR, f'skill_maps_{v}_OPTION{OPTION}_{PERIOD}{tag()}.{ext}')
            fig.savefig(out, bbox_inches='tight')
            print(f'Wrote {out}')
        plt.close(fig)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    names = [a for a in sys.argv[2:] if not a.startswith('--')]
    if NATIVE and OPTION != 3:
        sys.exit('--native is OPTION3 only.')
    if stage == 'compute':
        compute(names)
    elif stage == 'plot':
        cached = [v for v in VARS if f'truth/{v}/sum' in np.load(cache_path()).files]
        plot(names or cached)
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
