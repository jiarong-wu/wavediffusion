''' Seasonal and interannual variability of regional-mean wave fields over the
    test years (2019-2025), for comparing sampled fields against the truth.

    Stages:
      python regional_variability.py dates
      python regional_variability.py compute [SOURCE ...]   (default: truth)
      python regional_variability.py sample SOURCE [SOURCE ...]   (GPU; see MODELS)
      python regional_variability.py plot
      python regional_variability.py maps [SOURCE ...] --maps   (CPU; default MAPS_SOURCES)
      python regional_variability.py pixstd [SOURCE ...] --maps   (CPU; default MAPS_SOURCES)
      python regional_variability.py plot --maps
    CSVs go to output/, plots to figures/.
    With --maps, the regional means come instead from the per-date map files of
    compare_checkpoints_period.sh (MAPS_PERIOD_DIR: 2020-2025 every 5 days, 00 UTC),
    for the truth stored there (on the model grid, through the 320x320 resize)
    and each model's ensemble mean, over the pixels ice-free on the model grid
    (the files' `valid`); tp_pk / thetap_pk only where the truth's hs_pk >
    HS_EXISTS (not > 0: the resize smears a partition into a halo of near-zero
    heights, which differs per column and stripes the means), ci where finite.
    No sampling and no members (only the ensemble mean is saved); u10 is not in
    the files. CSVs and figures get the MAPS_TAG suffix.
    Add `--option 3` to any stage (except `dates`, which both share) for the
    OPTION3 partition fields (hs/tp/thetap of partitions 1 and 2, plus ci) and
    partition models instead of OPTION1's mean fields; its CSVs and figures get
    an _OPTION3 suffix (see OPTION_CONFIGS).
    `dates` writes regional_variability_dates.csv: the dates sample.py would
    sample for the test years under the sampling plan below (one sample.py run
    per year, MONTHLY=False, previous December loaded for history), without
    running any sampling. Once real samples
    exist, point DATES_CSV at their dates.csv files instead.
    `compute` reads the dates in DATES_CSV and writes one CSV per source
    (regional_variability_<source>.csv): cos-lat (wlat) weighted mean over
    ice-free ocean pixels of each region on exactly those dates, so truth and
    sampled cover the same steps; monthly means average those values.
    OPTION3 stores an absent partition as 0, so tp_pk and thetap_pk are averaged
    only where the TRUE partition k exists (hs_pk > 0), and ci only where it is
    defined (finite) -- the same pixels for every source.
    `sample` runs a model from MODELS on the same dates and writes the same
    regional means to regional_variability_<source>.csv, one row per ensemble
    member (column `member`); it appends date by date, so a re-run resumes.
    Before the regional averaging it also saves each date's maps (ensemble mean,
    variance, member 0) to MAPS_DIR/OPTION<n>/<source>/<YYYYmmddHH>.npz.
    `truth_resized` is the truth sent through the models' 320x320 resize and
    back, to separate that smoothing from model error.
    `plot` reads every CSV present for PLOT_SOURCES (members averaged first, i.e.
    the regional mean of the ensemble mean) and draws, per variable:
      - regional_seasonal_<var>.png:    monthly climatology per region (mean over
        years, +-1 interannual std band; thin lines = individual truth years)
      - regional_interannual_<var>.png: monthly mean vs. year for Jan/Apr/Jul/Oct
        (rows = regions, columns = months)
      - regional_seasonal_std_<var>.png (--maps, after the pixstd stage): per
        calendar month, the regional mean of each pixel's std over the dates in
        that month (years pooled); directions: circular std sqrt(-2 ln R)
    plus regional_boxes.png, a map of the region boxes for checking them.

    Regions: REGIONS below holds a subset of the regions of Young (1999),
    "Seasonal variability of the global ocean wind and wave climate", Int. J.
    Climatol. 19, 931-950 (Table I). Bounds are inclusive, so regions sharing an
    edge (e.g. PN / PSTN at 33N) both include that grid row. To add a region, add
    an entry; the lon range may cross the dateline (lon0 > lon1).

    Model outputs and truth are averaged over the same pixels: the native-grid
    ice-free mask of that date. u10 is a model input, not an output, so it is
    only plotted for truth (and only under OPTION1).
'''
import os
import sys
import glob
from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from wavediffusion.waveana import lat_deg, lon_deg
from wavediffusion.metrics import circular_diff
from wavediffusion.timeindex import (hist_window, months_file_list, file_lengths,
                                     build_date_index, select_sample_indices)

DATA_PATH = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'       # wavemean_* and forcing_*
PART_PATH = '/global/homes/j/jiarongw/scratch_folder/wave_data/partition_global/'  # waveparts_*
WLAT_FILE = '/global/homes/j/jiarongw/scratch_folder/wave_data/wlat.npy'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'output')     # CSVs
FIG_DIR = os.path.join(HERE, 'figures')    # plots
# Per-date sampled maps from the `sample` stage (ensemble mean, variance, member 0),
# saved before the regional averaging; ~2-5 GB per source, so on scratch.
MAPS_DIR = '/global/homes/j/jiarongw/scratch_folder/review/regional_variability_maps/'
SAVE_MAPS = True
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
N_WORKERS = int(os.environ.get('N_WORKERS', 8))    # months read in parallel (I/O bound)
OPTION = 1    # set by --option on the command line
if '--option' in sys.argv:
    _i = sys.argv.index('--option')
    OPTION = int(sys.argv[_i + 1])
    del sys.argv[_i:_i + 2]

YEARS = list(range(2019, 2022))

# Sampling plan -- must match the sample.py runs being compared against.
HIST_DURATION_DAYS, HIST_FREQ_DAYS = 10, 1    # of the checkpoint (sets the history padding)
EVERY_N = 40                                  # 5 day stride 
SAMPLE_DAYS, SAMPLE_HOUR = None, 0
# Dates to evaluate: a path or glob, e.g. later the sample runs' own files:
# '/global/homes/j/jiarongw/scratch_folder/final/temp/OPTION1_..._20??/dates.csv'
# The DATES_CSV environment variable overrides it (e.g. for a subset of dates).
DATES_CSV = os.environ.get('DATES_CSV', os.path.join(OUT_DIR, 'regional_variability_dates.csv'))
INTERANNUAL_MONTHS = [1, 4, 7, 10]    # columns of the interannual plots
N_ENSEM = 10       # diffusion ensemble members per date (~7.5 s/date on one A100 at 10 members)
STEPS = 20         # diffusion sampling steps (a model's `steps` overrides it)
GAM, MU = 1, 0.5  # sampler: gam = eps averaging (>= 1), mu in [0, 1) = added noise, 0 = deterministic
                   # (a model's `gam` / `mu` override them)
SIGMA0 = 100
MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Models for the `sample` stage. hist_* must match how each checkpoint was trained
# (OPTION{1,3}_periodic use a 5-day wind history, the *_hist10d runs a 10-day one).
CKPT_ROOT = '/pscratch/sd/j/jiarongw/review/'
OPTION1_MODELS = {
    'hist5d_ckpt12':      dict(kind='diffusion', dir=CKPT_ROOT + 'OPTION1_periodic/', ckpt=12,
                               hist_duration_days=5, hist_freq_days=1, periodic=True),
    'hist10d_ckpt16':     dict(kind='diffusion', dir=CKPT_ROOT + 'OPTION1_periodic_hist10d/', ckpt=16,
                               hist_duration_days=10, hist_freq_days=1, periodic=True),
    'hist10d_det_ckpt16': dict(kind='deterministic', dir=CKPT_ROOT + 'OPTION1_periodic_hist10d_det/', ckpt=16,
                               hist_duration_days=10, hist_freq_days=1, periodic=True),
    'truth_resized':      dict(kind='identity', hist_duration_days=10, hist_freq_days=1),
}
# Partition models train on 1993-2017, so YEARS are out of sample. The hist10d
# runs were still training on 2026-10-19 (latest ckpts 10 / 11); bump as needed.
OPTION3_MODELS = {
    'hist5d_ckpt12':      dict(kind='diffusion', dir=CKPT_ROOT + 'OPTION3_periodic/', ckpt=12,
                               hist_duration_days=5, hist_freq_days=1, periodic=True),
    'hist10d_ckpt17':     dict(kind='diffusion', dir=CKPT_ROOT + 'OPTION3_periodic_hist10d/', ckpt=17,
                               hist_duration_days=10, hist_freq_days=1, periodic=True),
    'hist10d_det_ckpt18': dict(kind='deterministic', dir=CKPT_ROOT + 'OPTION3_periodic_hist10d_det/', ckpt=18,
                               hist_duration_days=10, hist_freq_days=1, periodic=True),
    'truth_resized':      dict(kind='identity', hist_duration_days=10, hist_freq_days=1),
}

# Fixed per source (Okabe-Ito, colorblind-safe; diffusion blue / deterministic
# orange as in plot_checkpoint_comparison.py, hist5d green as in
# partition_energy.py); markers and line styles differ too, so identity isn't
# color-alone. Sources are plotted in this order, if their CSV exists.
_TRUTH_STYLES = {
    'truth':         dict(color='#222222', ls='-', marker='o', label='truth'),
    'truth_resized': dict(color='#999999', ls=':', marker='.', label='truth, 320x320 resized'),
}
_DET, _HIST5D, _HIST10D = (dict(color='#D55E00', ls='-.', marker='^'),
                           dict(color='#009E73', ls=(0, (5, 1, 1, 1)), marker='D'),
                           dict(color='#0072B2', ls='--', marker='s'))
OPTION1_STYLES = {
    **_TRUTH_STYLES,
    'hist10d_det_ckpt16': dict(_DET, label='deterministic hist10d, ckpt 16'),
    'hist5d_ckpt12':      dict(_HIST5D, label=f'diffusion hist5d, ckpt 12 ({N_ENSEM}-member mean)'),
    'hist10d_ckpt16':     dict(_HIST10D, label=f'diffusion hist10d, ckpt 16 ({N_ENSEM}-member mean)'),
}
OPTION3_STYLES = {
    **_TRUTH_STYLES,
    'hist10d_det_ckpt18': dict(_DET, label='deterministic hist10d, ckpt 18'),
    'hist5d_ckpt12':      dict(_HIST5D, label=f'diffusion hist5d, ckpt 12 ({N_ENSEM}-member mean)'),
    'hist10d_ckpt17':     dict(_HIST10D, label=f'diffusion hist10d, ckpt 17 ({N_ENSEM}-member mean)'),
}

VAR_LABELS = {'hs': 'Wave height $H_s$ (m)', 'tm': 'Mean period $T_m$ (s)',
              'thetam': 'Mean direction $\\theta_m$ (deg, from, cw from N)',
              'u10': 'Wind speed $U_{10}$ (m/s)',
              'ci': 'Crossing-sea criterion $c_i$ (fraction of pixels)'}
for _k in (1, 2):
    VAR_LABELS.update({f'hs_p{_k}': f'Partition {_k} wave height $H_{{s,{_k}}}$ (m)',
                       f'tp_p{_k}': f'Partition {_k} peak period $T_{{p,{_k}}}$ (s)',
                       f'thetap_p{_k}': f'Partition {_k} peak direction $\\theta_{{p,{_k}}}$ (deg, from, cw from N)'})

# var_names: what is averaged and plotted; model_vars: the model's output
# channels, in order (for OPTION3 also the waveparts_* channels). Angles in
# `circular` (degrees) are averaged as unit vectors (the per-date sin/cos means
# are kept in the CSV so monthly means are circular too), and plotted unwrapped
# around each region's mean direction so lines don't jump at 0/360.
OPTION_CONFIGS = {
    1: dict(var_names=['hs', 'tm', 'thetam', 'u10'], model_vars=['hs', 'tm', 'thetam'],
            circular={'thetam'}, x_dir=DATA_PATH, x_prefix='wavemean', suffix='',
            models=OPTION1_MODELS, styles=OPTION1_STYLES),
    3: dict(var_names=['hs_p1', 'tp_p1', 'thetap_p1', 'hs_p2', 'tp_p2', 'thetap_p2', 'ci'],
            model_vars=['hs_p1', 'tp_p1', 'thetap_p1', 'hs_p2', 'tp_p2', 'thetap_p2', 'ci'],
            circular={'thetap_p1', 'thetap_p2'}, x_dir=PART_PATH, x_prefix='waveparts', suffix='_OPTION3',
            models=OPTION3_MODELS, styles=OPTION3_STYLES),
}
_cfg = OPTION_CONFIGS[OPTION]
VAR_NAMES, MODEL_VARS, CIRCULAR = _cfg['var_names'], _cfg['model_vars'], _cfg['circular']
X_DIR, X_PREFIX, SUFFIX = _cfg['x_dir'], _cfg['x_prefix'], _cfg['suffix']    # forcing always from DATA_PATH
MODELS, SOURCE_STYLE = _cfg['models'], _cfg['styles']
PLOT_SOURCES = list(SOURCE_STYLE)

# --maps: regional means from compare_checkpoints_period.sh's saved maps (see the docstring)
MAPS = '--maps' in sys.argv
if MAPS:
    sys.argv.remove('--maps')
MAPS_PERIOD_DIR = f'/global/homes/j/jiarongw/scratch_folder/review/checkpoint_comparison_maps_2020-2025_every5d/OPTION{OPTION}/'
MAPS_TAG = '_2020-2025_every5d'
HS_EXISTS = 0.05    # m
# rows of the legend / plotting order, and their styles (paper names); DDPM is the high-noise fine-tune
_DDPM = {1: 'highsigma_ckpt6', 3: 'highsigma_ckpt5'}[OPTION]
MAPS_STYLES = {
    'truth':              dict(color='#222222', ls='-', marker='o', label='truth'),
    _DDPM:                dict(_HIST10D, label='DDPM (10-member mean)'),
    'hist10d_det_ckpt20': dict(_DET, label='Deterministic'),
    'resid_ckpt10':       dict(_HIST5D, label='Res-DDPM (10-member mean)'),
}
# '<model>_member0': that model's member 0 alone (the files' `member0`), dotted, thinner, no band
for _m, _lab in [(_DDPM, 'DDPM'), ('resid_ckpt10', 'Res-DDPM')]:
    MAPS_STYLES[f'{_m}_member0'] = dict(MAPS_STYLES[_m], ls=':', lw_scale=0.6, band=False, label=f'{_lab} (member 0)')
MAPS_SOURCES = [s for s in MAPS_STYLES
                if os.path.isdir(os.path.join(MAPS_PERIOD_DIR, s.removesuffix('_member0')))]
if MAPS:
    YEARS = list(range(2020, 2026))
    SUFFIX = SUFFIX + MAPS_TAG
    SOURCE_STYLE, PLOT_SOURCES = dict(MAPS_STYLES), MAPS_SOURCES


def csv_path(source):
    return os.path.join(OUT_DIR, f'regional_variability{SUFFIX}_{source}.csv')

# name: (description, lat0, lat1, lon0, lon1); lon in [-180, 180), degrees.
# From Young (1999) Table I (deg E there, converted here); comments give the
# table's longitude extent.
REGIONS = {
    'PN':   ('North Pacific',               33,  47,  160, -136),    # 160-224E
    'PSTN': ('North Subtropical Pacific',   20,  33,  160, -136),    # 160-224E
    'PEN':  ('Equatorial Pacific (north)',   0,  20,  173, -114),    # 173-246E
    'PE':   ('Eastern Pacific',            -20,   0, -110,  -92),    # 250-268E
    'SP':   ('Southern Ocean (Pacific)',   -60, -40, -170,  -90),    # 190-270E
    'SI':   ('Southern Ocean (Indian)',    -60, -40,   54,  106),    # 54-106E
}


############### Data loading ###############
def load_truth_month(year, month, steps):
    """Truth fields for one month on the native grid at the given within-month
    steps. Returns ({var: (T, H, W)}, valid (T, H, W) bool)."""
    x = np.load(os.path.join(DATA_PATH, f'wavemean_{year}{month:02d}.npy'), mmap_mode='r')
    f = np.load(os.path.join(DATA_PATH, f'forcing_{year}{month:02d}.npy'), mmap_mode='r')
    hs = np.asarray(x[steps, 0])
    fields = {
        'hs': hs,
        'tm': np.asarray(x[steps, 1]),
        'thetam': np.asarray(x[steps, 2]),
        'u10': np.hypot(f[steps, 0], f[steps, 1]),
    }
    valid = (np.asarray(f[steps, 2]) > 0.5) & np.isfinite(hs)    # forcing ch 2 = ice-free ocean mask
    return fields, valid


def load_truth_parts_month(year, month, steps):
    """OPTION3 truth partitions for one month on the native grid at the given
    within-month steps. Returns ({var: (T, H, W)}, {var: valid (T, H, W) bool}):
    the ice-free ocean mask of load_truth_month, further restricted for tp/thetap
    of partition k to where it exists (hs_pk > 0) and for ci to where it is finite."""
    _, valid = load_truth_month(year, month, steps)
    p = np.asarray(np.load(os.path.join(PART_PATH, f'waveparts_{year}{month:02d}.npy'), mmap_mode='r')[steps])
    fields = {v: p[:, c] for c, v in enumerate(MODEL_VARS)}
    masks = {'ci': valid & np.isfinite(fields['ci'])}
    for k in (1, 2):
        exists = valid & (fields[f'hs_p{k}'] > 0)
        masks.update({f'hs_p{k}': valid, f'tp_p{k}': exists, f'thetap_p{k}': exists})
    return fields, masks


SOURCES = {
    'truth': load_truth_month if OPTION == 1 else load_truth_parts_month,
}


############### Compute ###############
def region_index(lat0, lat1, lon0, lon1):
    rows = np.where((lat_deg >= lat0) & (lat_deg <= lat1))[0]
    if lon0 <= lon1:
        cols = np.where((lon_deg >= lon0) & (lon_deg <= lon1))[0]
    else:    # crosses the dateline
        cols = np.where((lon_deg >= lon0) | (lon_deg <= lon1))[0]
    return rows, cols


def plan_dates():
    """Dates sample.py would sample: one run per test year, each loading that
    year's months plus the previous December (for the wind history)."""
    _, padding = hist_window(HIST_DURATION_DAYS, HIST_FREQ_DAYS)
    frames = []
    for y in YEARS:
        test_months = [(y, m) for m in range(1, 13)]
        file_list, ym_list = months_file_list(test_months, DATA_PATH)
        dates, _ = build_date_index(ym_list, file_lengths(file_list))
        raw = select_sample_indices(dates, padding, EVERY_N, SAMPLE_DAYS, SAMPLE_HOUR, keep_months=test_months)
        frames.append(pd.DataFrame({'run': y, 'date': [dates[r] for r in raw]}))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(DATES_CSV, index=False)
    print(f'Wrote {DATES_CSV}: {len(df)} dates, {df.date.min()} to {df.date.max()}')


def load_dates():
    """Sorted unique dates from every file matching DATES_CSV."""
    paths = sorted(glob.glob(DATES_CSV))
    if not paths:
        raise RuntimeError(f'No dates file matches {DATES_CSV}; run the dates stage first.')
    dates = pd.concat([pd.read_csv(p, usecols=['date'], parse_dates=['date']) for p in paths])
    print(f'Evaluating {dates.date.nunique()} dates from {len(paths)} file(s)')
    return pd.DatetimeIndex(dates.date.unique()).sort_values()


def region_means(fields, valid, idx, wlat, times):
    """cos-lat weighted means over valid pixels of each region, one row per time.
    `valid` is one (T, H, W) mask for all variables or a {var: mask} dict.
    Circular variables get the weighted mean of sin/cos and the angle of that."""
    frames = []
    for name, (rows, cols) in idx.items():
        wl = wlat[np.ix_(rows, cols)][None]
        rec = {'time': times, 'region': name}
        with np.errstate(invalid='ignore'):
            for v in VAR_NAMES:
                if v not in fields:
                    rec[v] = np.nan
                    continue
                w = wl * (valid[v] if isinstance(valid, dict) else valid)[:, rows][:, :, cols]
                wmean = lambda a: (np.nan_to_num(a) * w).sum(axis=(1, 2)) / w.sum(axis=(1, 2))
                a = fields[v][:, rows][:, :, cols]
                if v in CIRCULAR:
                    rad = np.deg2rad(a)
                    rec[f'{v}_sin'], rec[f'{v}_cos'] = wmean(np.sin(rad)), wmean(np.cos(rad))
                    rec[v] = np.rad2deg(np.arctan2(rec[f'{v}_sin'], rec[f'{v}_cos'])) % 360
                else:
                    rec[v] = wmean(a)
        frames.append(pd.DataFrame(rec))
    return pd.concat(frames, ignore_index=True)


def compute_month(ym_dates, source, idx, wlat):
    (y, m), times = ym_dates
    steps = ((times - pd.Timestamp(year=y, month=m, day=1)) // pd.Timedelta(hours=3)).to_numpy()
    try:
        fields, valid = SOURCES[source](y, m, steps)
    except FileNotFoundError as e:
        print(f'[skip] {y}-{m:02d}: {e}')
        return None
    return region_means(fields, valid, idx, wlat, times)


def compute(source):
    wlat = np.load(WLAT_FILE)
    idx = {name: region_index(*spec[1:]) for name, spec in REGIONS.items()}
    dates = load_dates()
    by_month = [((y, m), dates[(dates.year == y) & (dates.month == m)])
                for y, m in sorted({(d.year, d.month) for d in dates})]
    with Pool(N_WORKERS) as pool:
        frames = list(tqdm(pool.imap(partial(compute_month, source=source, idx=idx, wlat=wlat), by_month),
                           total=len(by_month), desc=source))
    df = pd.concat([f for f in frames if f is not None], ignore_index=True)
    out_csv = csv_path(source)
    df.to_csv(out_csv, index=False, float_format='%.5g')
    print(f'Wrote {out_csv} ({len(df)} rows)')


def compute_from_maps(source):
    """Regional means of `source` ('truth', a model's ensemble mean, or
    '<model>_member0' for its member 0) from the saved maps, one
    row per date and region, to csv_path(source)."""
    wlat = np.load(WLAT_FILE)
    idx = {name: region_index(*spec[1:]) for name, spec in REGIONS.items()}
    names = sorted(f for f in os.listdir(os.path.join(MAPS_PERIOD_DIR, 'truth')) if f.endswith('.npz'))
    frames = []
    for f in tqdm(names, desc=source):
        t = np.load(os.path.join(MAPS_PERIOD_DIR, 'truth', f))
        model = source.removesuffix('_member0')
        z = t if source == 'truth' else np.load(os.path.join(MAPS_PERIOD_DIR, model, f))
        tv, zv = list(t['vars']), list(z['vars'])
        x = z['truth'] if source == 'truth' else z['member0' if source.endswith('_member0') else 'mean']
        fields = {v: x[zv.index(v)][None] for v in VAR_NAMES if v in zv}
        valid = {}
        for v in fields:
            m = t['valid']
            if OPTION == 3 and v[:-1] in ('tp_p', 'thetap_p'):
                m = m & (t['truth'][tv.index(f'hs_p{v[-1]}')] > HS_EXISTS)
            elif v == 'ci':
                m = m & np.isfinite(t['truth'][tv.index('ci')])
            valid[v] = m[None]
        frames.append(region_means(fields, valid, idx, wlat, pd.DatetimeIndex([pd.Timestamp(f[:-4])])))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(csv_path(source), index=False, float_format='%.5g')
    print(f'Wrote {csv_path(source)} ({len(df)} rows)')


############### Models (GPU) ###############
def build_model(cfg, test):
    """Diffusion (Scaled(myUnet)), deterministic (DetUnet) or residual
    (ResidualDiffusion: a Scaled(myUnet) on the residual of a frozen DetUnet, as
    trained by example/experiments/partdiff_periodic_hist10d_resid.py) model with
    EMA weights. A residual checkpoint holds its DetUnet's weights and rstd too;
    its EMA covers only the diffusion net."""
    import torch
    from torch_ema import ExponentialMovingAverage as EMA
    from wavediffusion.model_unet import myUnet, DetUnet
    from wavediffusion.model import Scaled, ResidualDiffusion
    n_ch, n_forcing = len(MODEL_VARS), 4 + 2 * test.n_hist
    unet = lambda cls, precond_ch: cls(in_dim=320, in_ch=n_ch, out_ch=n_ch, ch=256, precond_ch=precond_ch,
                                       scale=(test.meanx, test.stdx, test.meanf, test.stdf),
                                       ch_mult=(1, 2, 2), attn_resolutions=(16,), periodic=cfg['periodic'])
    if cfg['kind'] == 'residual':
        model = ResidualDiffusion(unet(Scaled(myUnet), n_forcing + n_ch), unet(DetUnet, n_forcing),
                                  rstd=np.ones(n_ch), stdx=torch.as_tensor(test.stdx).flatten(),
                                  circular=[MODEL_VARS.index(v) for v in sorted(CIRCULAR)])
        ema = EMA(model.net.parameters(), decay=0.999)
    else:
        model = unet(DetUnet if cfg['kind'] == 'deterministic' else Scaled(myUnet), n_forcing)
        ema = EMA(model.parameters(), decay=0.999)
    ckpt = torch.load(os.path.join(cfg['dir'], f"ckpt_{cfg['ckpt']}.pt"), map_location='cpu')
    model.load_state_dict(ckpt['model'])
    ema.load_state_dict(ckpt['ema'])
    return model, ema


def predict_members(cfg, model, ema, accelerator, schedule, test, x, f, icymask, start=None):
    """Members (n, C, 320, 720) in physical units, not multiplied by any mask
    (the binary native mask is applied when averaging). For a diffusion cfg with
    `start_from`, `start` is that deterministic (model, ema): sampling then begins
    from its forecast plus noise at the first sampler sigma <= cfg['sigma_start']
    (SDEdit-style) and runs only the remaining steps."""
    import torch
    from wavediffusion.diffusion import samples
    with torch.no_grad():
        if cfg['kind'] == 'identity':
            out = [test.invert_x(x)]
        elif cfg['kind'] == 'residual':
            mask = icymask[None].to(accelerator.device)
            with ema.average_parameters():
                x0, _ = accelerator.unwrap_model(model).sample(
                    f[None].repeat(N_ENSEM, 1, 1, 1).to(accelerator.device), mask,
                    schedule.sample_sigmas(cfg.get('steps', STEPS)),
                    gam=cfg.get('gam', GAM), mu=cfg.get('mu', MU), accelerator=accelerator)
            out = [test.invert_x(x0[i]) for i in range(N_ENSEM)]
        elif cfg['kind'] == 'deterministic':
            model.eval()
            with ema.average_parameters():
                out = [test.invert_x(model(f[None].to(accelerator.device))[0])]
        else:
            mask = icymask[None].to(accelerator.device)
            sigmas = schedule.sample_sigmas(cfg.get('steps', STEPS))
            xt = None
            if start is not None:
                det_model, det_ema = start
                sigmas = sigmas[sigmas <= cfg['sigma_start']]
                det_model.eval()
                with det_ema.average_parameters():
                    x_det = det_model(f[None].to(accelerator.device))
                xt = (x_det + sigmas[0] * torch.randn((N_ENSEM,) + x_det.shape[1:], device=x_det.device)) * mask
            with ema.average_parameters():
                *_, x0 = samples(model, sigmas,
                                 gam=cfg.get('gam', GAM), mu=cfg.get('mu', MU), batchsize=N_ENSEM,
                                 accelerator=accelerator, cond=f[None].repeat(N_ENSEM, 1, 1, 1),
                                 mask=mask, xt=xt)
            out = [test.invert_x(x0[i]) for i in range(N_ENSEM)]
    return np.stack([o.float().cpu().numpy() for o in out])


def maps_path(source, d):
    return os.path.join(MAPS_DIR, f'OPTION{OPTION}', source, f'{d:%Y%m%d%H}.npz')


def save_maps(source, d, members):
    """Ensemble mean, variance (ddof=1; 0 for a single member) and member 0 of
    one date, (C, H, W) float32 each, channels in MODEL_VARS order. Circular
    channels get the circular mean (deg) and circular variance 1 - R, in [0, 1]."""
    n = len(members)
    with np.errstate(invalid='ignore'):
        mean = members.mean(axis=0)
        var = members.var(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
        for c, v in enumerate(MODEL_VARS):
            if v in CIRCULAR:
                rad = np.deg2rad(members[:, c])
                s, co = np.sin(rad).mean(axis=0), np.cos(rad).mean(axis=0)
                mean[c] = np.rad2deg(np.arctan2(s, co)) % 360
                var[c] = 1 - np.hypot(s, co)
    path = maps_path(source, d)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mean=mean.astype(np.float32), var=var.astype(np.float32),
             member0=members[0].astype(np.float32), vars=np.array(MODEL_VARS),
             circular=np.array(sorted(CIRCULAR)), n_members=n, time=str(d))


def run_model(source):
    import torch
    from accelerate import Accelerator
    from wavediffusion.wavedata import npyDataWndHist
    from wavediffusion.diffusion import ScheduleLogLinear
    from wavediffusion.timeindex import dates_to_raw_indices, to_dataset_idx
    cfg = MODELS[source]
    out_csv = csv_path(source)
    done = set(pd.read_csv(out_csv, usecols=['time'], parse_dates=['time']).time) if os.path.exists(out_csv) else set()
    dates = load_dates()
    print(f'[{source}] {len(done)} dates already done')

    stats = np.load(os.path.join(X_DIR, f'stats_OPTION{OPTION}.npz'))
    wlat = np.load(WLAT_FILE)
    idx = {name: region_index(*spec[1:]) for name, spec in REGIONS.items()}
    accelerator = Accelerator()
    schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=SIGMA0, N=80)
    model = ema = start = None
    columns = None

    # One dataset per test year (that year's months plus the previous one for the
    # history), as in sample.py; rows are appended date by date.
    for y in sorted({d.year for d in dates}):
        want = [d for d in dates if d.year == y and d not in done]
        if not want:
            continue
        test_months = sorted({(d.year, d.month) for d in dates if d.year == y})
        file_list, ym_list = months_file_list(test_months, X_DIR, DATA_PATH, x_prefix=X_PREFIX)
        test = npyDataWndHist(file_list, resize_x=(320, 320), resize_f=(320, 320),
                              landmaskname=os.path.join(X_DIR, 'mask.npy'), use_icymask=True,
                              compute_stats=False, meanx=stats['meanx'], stdx=stats['stdx'],
                              meanf=stats['meanf'], stdf=stats['stdf'], OPTION=OPTION,
                              hist_duration_days=cfg['hist_duration_days'], hist_freq_days=cfg['hist_freq_days'])
        if cfg['kind'] != 'identity' and model is None:
            model, ema = build_model(cfg, test)
            model = accelerator.prepare(model)
            ema.to(accelerator.device)
            if cfg.get('start_from'):
                det_model, det_ema = build_model(MODELS[cfg['start_from']], test)
                det_ema.to(accelerator.device)
                start = (accelerator.prepare(det_model), det_ema)
        date_index, _ = build_date_index(ym_list, test.file_lengths)
        for r in tqdm(dates_to_raw_indices(want, date_index, test.padding), desc=f'{source} {y}'):
            d = date_index[r]
            x, f, icymask = test[to_dataset_idx(r, test.padding)]
            members = predict_members(cfg, model, ema, accelerator, schedule, test, x, f, icymask, start)
            if SAVE_MAPS:
                save_maps(source, d, members)
            step = np.array([(d - pd.Timestamp(year=d.year, month=d.month, day=1)) // pd.Timedelta(hours=3)])
            _, valid = SOURCES['truth'](d.year, d.month, step)    # the truth's pixels (see compute)
            rows = []
            for k, mem in enumerate(members):
                fields = {v: mem[c][None] for c, v in enumerate(MODEL_VARS)}
                rows.append(region_means(fields, valid, idx, wlat, pd.DatetimeIndex([d])).assign(member=k))
            df = pd.concat(rows, ignore_index=True)
            if columns is None:
                columns = (pd.read_csv(out_csv, nrows=0).columns.tolist() if os.path.exists(out_csv)
                           else df.columns.tolist())
                if not os.path.exists(out_csv):
                    df.iloc[:0].to_csv(out_csv, index=False)
            df[columns].to_csv(out_csv, mode='a', header=False, index=False, float_format='%.5g')
        del test
        torch.cuda.empty_cache()
    print(f'[{source}] done: {out_csv}')


############### Plot ###############
def load_monthly():
    """{source: DataFrame of monthly means indexed by (region, year, month)}.
    Ensemble sources also get a '<source>_member0' entry (member 0 alone), styled
    as its source but dotted and thinner, without a band."""
    out = {}
    cols = [c for v in VAR_NAMES for c in ([f'{v}_sin', f'{v}_cos'] if v in CIRCULAR else [v])]

    def monthly(df):
        df = df.assign(year=df.time.dt.year, month=df.time.dt.month)
        mdf = df.groupby(['region', 'year', 'month'])[cols].mean()
        for v in CIRCULAR:
            mdf[v] = np.rad2deg(np.arctan2(mdf[f'{v}_sin'], mdf[f'{v}_cos'])) % 360
        return mdf

    for source in PLOT_SOURCES:
        path = csv_path(source)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=['time'])
        missing = sorted(set(cols) - set(df.columns))
        if missing:
            raise RuntimeError(f'{path} lacks {missing}; re-run the compute stage.')
        if 'member' not in df.columns:
            out[source] = monthly(df)
            continue
        # ensemble: average members first (= regional mean of the ensemble mean)
        out[source] = monthly(df.groupby(['time', 'region'], as_index=False)[cols].mean())
        if df.member.nunique() > 1:
            st = SOURCE_STYLE[source]
            SOURCE_STYLE[f'{source}_member0'] = dict(
                st, ls=':', lw_scale=0.6, band=False,
                label=st['label'].replace(f'({N_ENSEM}-member mean)', '(member 0)'))
            out[f'{source}_member0'] = monthly(df[df.member == 0])
    if not out:
        raise RuntimeError(f'No regional_variability{SUFFIX}_*.csv found; run the compute stage first.')
    return unwrap_circular(out)


def pixel_std_path(source):
    return os.path.join(OUT_DIR, f'regional_pixelstd{SUFFIX}_{source}.csv')


def compute_pixel_std(source):
    """Per region and calendar month, the cos-lat weighted regional mean of each
    pixel's std over that month's dates (years pooled), from the saved maps
    (--maps), on the pixels and dates compute_from_maps uses. Directions: the
    circular std sqrt(-2 ln R) (deg) of the pixel's angles. Writes pixel_std_path."""
    wlat = np.load(WLAT_FILE)
    idx = {name: region_index(*spec[1:]) for name, spec in REGIONS.items()}
    model = source.removesuffix('_member0')
    names = sorted(f for f in os.listdir(os.path.join(MAPS_PERIOD_DIR, 'truth')) if f.endswith('.npz'))
    acc = {}    # (month, var) -> [n, sum a, sum a^2] or [n, sum sin, sum cos]
    for f in tqdm(names, desc=f'{source} pixel std'):
        month = int(f[5:7])
        t = np.load(os.path.join(MAPS_PERIOD_DIR, 'truth', f))
        z = t if source == 'truth' else np.load(os.path.join(MAPS_PERIOD_DIR, model, f))
        tv, zv = list(t['vars']), list(z['vars'])
        x = z['truth'] if source == 'truth' else z['member0' if source.endswith('_member0') else 'mean']
        for v in VAR_NAMES:
            if v not in zv:
                continue
            m = t['valid']
            if OPTION == 3 and v[:-1] in ('tp_p', 'thetap_p'):
                m = m & (t['truth'][tv.index(f'hs_p{v[-1]}')] > HS_EXISTS)
            elif v == 'ci':
                m = m & np.isfinite(t['truth'][tv.index('ci')])
            a = np.where(m, x[zv.index(v)], 0).astype(np.float64)
            if v in CIRCULAR:
                parts = [m, np.where(m, np.sin(np.deg2rad(a)), 0), np.where(m, np.cos(np.deg2rad(a)), 0)]
            else:
                parts = [m, a, a ** 2]
            sums = acc.setdefault((month, v), [0, 0, 0])
            for k in range(3):
                sums[k] = sums[k] + parts[k]
    rows = []
    for (month, v), (n, s1, s2) in acc.items():
        with np.errstate(invalid='ignore', divide='ignore'):
            if v in CIRCULAR:
                r = np.clip(np.hypot(s1, s2) / n, 1e-12, 1)
                std = np.rad2deg(np.sqrt(-2 * np.log(r)))
            else:
                std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0) * n / (n - 1))
        std = np.where(n >= 10, std, np.nan)    # too few dates for a std
        for name, (r_, c_) in idx.items():
            sub, w = std[np.ix_(r_, c_)], wlat[np.ix_(r_, c_)]
            ok = np.isfinite(sub)
            rows.append(dict(region=name, month=month, var=v, std=(sub[ok] * w[ok]).sum() / w[ok].sum()))
    df = pd.DataFrame(rows).pivot_table(index=['region', 'month'], columns='var', values='std').reset_index()
    df.to_csv(pixel_std_path(source), index=False, float_format='%.5g')
    print(f'Wrote {pixel_std_path(source)}')


def plot_seasonal_std(v):
    """Seasonal cycle of the regional mean of the pixel std (compute_pixel_std)."""
    std = {s: pd.read_csv(pixel_std_path(s)).set_index(['region', 'month'])
           for s in SOURCE_STYLE if os.path.exists(pixel_std_path(s))}
    std = {s: d[v] for s, d in std.items() if v in d}
    if not std:
        return
    fig, axes = region_grid(len(REGIONS))
    months = np.arange(1, 13)
    for ax, (name, spec) in zip(axes, REGIONS.items()):
        for source, sd in std.items():
            st = SOURCE_STYLE[source]
            ax.plot(months, sd.loc[name].reindex(months), color=st['color'], lw=2 * st.get('lw_scale', 1),
                    ls=st['ls'], label=st['label'])
        ax.set_title(f'{name}: {spec[0]}', fontsize=10)
        ax.set_xticks(months)
        ax.set_xticklabels([m[0] for m in MONTH_ABBR])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f'std of {VAR_LABELS[v]}')
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f'{VAR_LABELS[v]}: regional mean of the pixel std over the dates of each month, '
                 f'{YEARS[0]}-{YEARS[-1]} pooled', fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f'regional_seasonal_std_{v}{SUFFIX}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def unwrap_circular(monthly):
    """Express each angle as center + wrapped difference from it, where center is
    the region's circular mean over all months of the first source (truth), so
    all sources share one reference and means/stds/trends act on continuous
    values (valid while regional monthly means stay within 180 deg of it)."""
    ref = next(iter(monthly.values()))
    for v in CIRCULAR:
        g = ref.groupby(level='region')[[f'{v}_sin', f'{v}_cos']].mean()
        center = np.rad2deg(np.arctan2(g[f'{v}_sin'], g[f'{v}_cos'])) % 360
        for mdf in monthly.values():
            c = center.reindex(mdf.index.get_level_values('region')).to_numpy()
            mdf[v] = c + circular_diff(mdf[v].to_numpy(), c)
    return monthly


def region_grid(n):
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def plot_seasonal(monthly, v):
    fig, axes = region_grid(len(REGIONS))
    months = np.arange(1, 13)
    for ax, (name, spec) in zip(axes, REGIONS.items()):
        for source, mdf in monthly.items():
            tab = mdf.loc[name, v].unstack('year').reindex(months)    # (month, year)
            if tab.isna().all().all():
                continue    # e.g. u10 for model sources
            st = SOURCE_STYLE[source]
            if source == 'truth':
                for y in tab.columns:
                    ax.plot(months, tab[y], color=st['color'], lw=0.6, alpha=0.25)
            mean, std = tab.mean(axis=1), tab.std(axis=1)
            if st.get('band', True):
                ax.fill_between(months, mean - std, mean + std, color=st['color'], alpha=0.10, lw=0)
            ax.plot(months, mean, color=st['color'], lw=2 * st.get('lw_scale', 1), ls=st['ls'], label=st['label'])
        ax.set_title(f'{name}: {spec[0]}', fontsize=10)
        ax.set_xticks(months)
        ax.set_xticklabels([m[0] for m in MONTH_ABBR])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(VAR_LABELS[v])
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f'{VAR_LABELS[v]}: seasonal cycle {YEARS[0]}-{YEARS[-1]} '
                 '(mean $\\pm$1 interannual std; thin lines = individual truth years)', fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f'regional_seasonal_{v}{SUFFIX}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def plot_interannual(monthly, v):
    nrow, ncol = len(REGIONS), len(INTERANNUAL_MONTHS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.5 * nrow), sharex=True, squeeze=False)
    for i, (name, spec) in enumerate(REGIONS.items()):
        for j, m in enumerate(INTERANNUAL_MONTHS):
            ax = axes[i, j]
            for source, mdf in monthly.items():
                s = mdf.loc[name, v].xs(m, level='month').dropna()
                if s.empty:
                    continue
                st = SOURCE_STYLE[source]
                ax.plot(s.index, s.values, ls=st['ls'], marker=st['marker'], ms=4 * st.get('lw_scale', 1),
                        lw=1.3 * st.get('lw_scale', 1),
                        color=st['color'], label=st['label'])
            if i == 0:
                ax.set_title(MONTH_ABBR[m - 1], fontsize=10)
            if j == 0:
                ax.set_ylabel(f'{name}', fontsize=10)
            ax.set_xticks(YEARS)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f'{VAR_LABELS[v]}: monthly mean by year', fontsize=10, y=1.025)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(FIG_DIR, f'regional_interannual_{v}{SUFFIX}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def plot_boxes():
    """Map of the region boxes, centred on the Pacific (lon in [0, 360))."""
    landmask = np.load(os.path.join(DATA_PATH, 'mask.npy'))
    lon360 = np.mod(lon_deg, 360)
    order = np.argsort(lon360)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.pcolormesh(lon360[order], lat_deg, landmask[:, order], cmap='Greys_r', vmin=-1, vmax=1.5, shading='auto')
    for name, (_, lat0, lat1, lon0, lon1) in REGIONS.items():
        a, b = lon0 % 360, lon1 % 360
        b = b + 360 if b < a else b
        ax.add_patch(plt.Rectangle((a, lat0), b - a, lat1 - lat0, fill=False, ec='#D55E00', lw=1.5))
        ax.text(a + 1, lat1 - 1, name, va='top', fontsize=8, color='#222222')
    ax.set_xlabel('lon (deg E)'); ax.set_ylabel('lat')
    ax.set_aspect('equal')
    out = os.path.join(FIG_DIR, 'regional_boxes.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'plot'
    if stage == 'dates':
        plan_dates()
    elif stage == 'compute':
        for source in (sys.argv[2:] or ['truth']):
            compute(source)
    elif stage == 'sample':
        for source in sys.argv[2:]:
            run_model(source)
    elif stage == 'maps':
        if not MAPS:
            raise SystemExit('The maps stage needs --maps (its CSVs carry MAPS_TAG).')
        for source in (sys.argv[2:] or MAPS_SOURCES):
            compute_from_maps(source)
    elif stage == 'pixstd':
        if not MAPS:
            raise SystemExit('The pixstd stage needs --maps.')
        for source in (sys.argv[2:] or MAPS_SOURCES):
            compute_pixel_std(source)
    elif stage == 'plot':
        plot_boxes()
        monthly = load_monthly()
        for v in VAR_NAMES:
            if MAPS and v == 'u10':
                continue    # not in the map files
            plot_seasonal(monthly, v)
            if MAPS:
                plot_seasonal_std(v)
            plot_interannual(monthly, v)
    else:
        raise SystemExit(f'Unknown stage {stage!r}; use dates, compute, sample, maps, pixstd or plot.')


if __name__ == '__main__':
    main()
