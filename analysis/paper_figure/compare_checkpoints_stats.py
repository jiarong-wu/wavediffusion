''' Compare MSE, SSR (spread-skill ratio), ACC, and CRPS between two OPTION1 checkpoints,
    configured via the CHECKPOINTS list below (currently: OPTION1_periodic_hist10d's
    diffusion ckpt_9/10 vs its deterministic DetUnet baseline's ckpt_9/10/11), over
    multi-year test data (2019-2024), subsampled every STRIDE_DAYS days for
    efficiency. Each evaluation date is tagged with year/month/season so that
    plot_checkpoint_comparison.py can check for seasonal/yearly trends in error.

    MSE and ACC are computed against the N_ENSEM-member ensemble mean (not a
    single draw); CRPS (fair/unbiased ensemble estimator) uses all members.
    Metric definitions live in wavediffusion/metrics.py. SSR isn't a per-date ratio -- following the convention already
    used in stats.ipynb, it's sqrt(mean(ensemble_variance) / mean(mse)) aggregated
    over however many dates you group by (all, per year, per season, ...), since
    a per-date ratio of two noisy scalars is too unstable to be useful. So this
    script only writes the per-date ensemble variance (`{v}_spreadvar`) to the
    CSV; plot_checkpoint_comparison.py does the aggregation into SSR.

    All years 2019-2024 (including 2023, backfilled via data/ifremer/wavedata_process.py)
    are covered by wavemean_*/forcing_* .npy files in mean_global/. The file list
    and evaluation dates below are still built dynamically from whatever months
    are actually present on disk, so this keeps working unmodified if a future
    year turns out to be missing or only partially preprocessed.

    A checkpoint entry with deterministic=True is a DetUnet regressor (trained by
    example/experiments/wave_simple.py): one forward pass per date instead of an
    ensemble, so its CRPS column is the MAE (CRPS of a point forecast) and its
    spreadvar is NaN (SSR is undefined without a spread).

    OPTION3 (partition models: hs/tp/thetap of partitions 1 and 2, plus the
    crossing-sea criterion ci) is selected with `python compare_checkpoints_stats.py 3`
    (default 1). Each option has its own variables, data/climatology files,
    CHECKPOINTS list and output CSV -- see OPTION_CONFIGS below.

    Requires a GPU (diffusion sampling) -- submit via compare_checkpoints.sh.
    Results are appended to OUT_CSV row-by-row, so a re-run resumes where a
    previous one stopped or timed out.

    Stages (second argument, default `stats`):
      python compare_checkpoints_stats.py OPTION [stats]          the per-date CSV above
      python compare_checkpoints_stats.py OPTION maps [NAME ...]   (GPU) per-pixel maps
      python compare_checkpoints_stats.py OPTION plot_maps [SEASON] [NAME ...]   (CPU)
      python compare_checkpoints_stats.py OPTION truth_maps                      (CPU)
    `maps` samples the same dates as `stats` (it does not reuse them: `stats`
    keeps no fields) for the named CHECKPOINTS (default all) and saves, per
    checkpoint and date, the ensemble mean, variance, per-pixel fair CRPS
    (ensemble_crps; MAE for deterministic entries) and member 0, plus all
    members with `maps --all-members [NAME ...]`; see compute_maps. For OPTION3
    the maps also hold hs_tot = sqrt(hs_p1^2 + hs_p2^2), computed per member.
    The truth and ice-free mask are saved once per date. One file per date,
    so it resumes; ~26 MB per date for OPTION3 (~65 MB more with all members).
    `plot_maps` computes from those files, per variable, the mean bias (model - truth) and RMSE of
    the ensemble mean, the ensemble spread (std, on the RMSE's color scale),
    the CRPS, and CRPS minus that of the first deterministic entry
    (negative = better), one row per checkpoint with
    a maps file (or the NAMEs given), for SEASON (DJF/MAM/JJA/SON, default
    all dates), to figures/checkpoint_maps_<var>[_OPTION3]_<season>.png.
    `truth_maps` draws the truth's temporal mean and standard deviation over the same
    evaluation dates, for all dates and per season, on the native grid (not
    the 320x320-resized truth the scores use), to
    figures/truth_maps_<var>[_OPTION3].png. Pixels count only on dates they
    are ice-free; for OPTION3, tp_pk / thetap_pk only where partition k
    exists (hs_pk > 0; absent partitions are stored as 0) and ci only where
    finite. Directions get the circular mean and the RMS circular_diff
    from it (deg).
'''
import os
import sys
import csv
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch_ema import ExponentialMovingAverage as EMA
from torchvision import transforms as tf
from accelerate import Accelerator
from tqdm import tqdm

from wavediffusion.model_unet import myUnet, DetUnet
from wavediffusion.model import Scaled, ResidualDiffusion
from wavediffusion.wavedata import npyDataWndHist
from wavediffusion.diffusion import ScheduleLogLinear, samples
from wavediffusion.metrics import (circular_diff, weighted_mse, weighted_mean, weighted_acc,
                                   ensemble_mean_var, weighted_crps, ensemble_crps)
from wavediffusion.timeindex import (SEASON_OF_MONTH, hist_window, available_file_list, file_lengths,
                                     months_file_list, previous_month,
                                     build_date_index, select_eval_indices, to_dataset_idx)

OPTION = int(sys.argv[1]) if len(sys.argv) > 1 else 1
STAGE = sys.argv[2] if len(sys.argv) > 2 else 'stats'
STAGE_ARGS = sys.argv[3:]

MEAN_PATH = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'     # wavemean_* and forcing_*
PART_PATH = '/global/homes/j/jiarongw/scratch_folder/wave_data/partition_global/'  # waveparts_*
WLAT_FILE = '/global/homes/j/jiarongw/scratch_folder/wave_data/wlat.npy'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
MAPS_DIR = '/global/homes/j/jiarongw/scratch_folder/review/checkpoint_comparison_maps/'    # ~80 MB per checkpoint
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = list(range(2023, 2025))
STRIDE_DAYS = 15
# Another evaluation period, e.g. EVAL_YEARS=2020-2025 EVAL_STRIDE_DAYS=5: its CSV,
# map files and figures get PERIOD_TAG (e.g. _2020-2025_every5d) in their names,
# so they never mix with the default period's.
PERIOD_TAG = ''
if os.environ.get('EVAL_YEARS') or os.environ.get('EVAL_STRIDE_DAYS'):
    _y = os.environ.get('EVAL_YEARS', f'{YEARS[0]}-{YEARS[-1]}').split('-')
    YEARS = list(range(int(_y[0]), int(_y[-1]) + 1))
    STRIDE_DAYS = int(os.environ.get('EVAL_STRIDE_DAYS', STRIDE_DAYS))
    PERIOD_TAG = f'_{YEARS[0]}-{YEARS[-1]}_every{STRIDE_DAYS}d'
    MAPS_DIR = MAPS_DIR.rstrip('/') + PERIOD_TAG + '/'


def sampling_variant(entry, steps, n_ensem):
    """Copy of a diffusion CHECKPOINTS entry sampled with `steps` denoising steps and
    `n_ensem` members instead of STEPS/N_ENSEM. The settings go into the name
    (e.g. hist10d_ckpt16_s40_e20), so its rows sit next to the default-sampling
    ones in the same CSV and resume independently of them."""
    return dict(entry, name=f"{entry['name']}_s{steps}_e{n_ensem}", steps=steps, n_ensem=n_ensem)


OPTION1_CHECKPOINTS = [
    # Same run (OPTION1_periodic_hist10d), two epochs apart -- identical training
    # data/config, so unlike some earlier comparisons run through this script
    # there's no train/test-overlap or periodic-flag mismatch to worry about here.
    # `periodic` still needs to match how a checkpoint was trained if you swap
    # one of these out later: it doesn't change parameter shapes, so a mismatch
    # loads silently and just runs inference with the wrong boundary padding.
    # hist_duration_days / hist_freq_days must match training too: together they
    # set the number of wind-history channels (precond_ch) and the index padding.
    # OPTION1_periodic (example/training/meandiff_periodic.py) uses a 5-day wind history.
    dict(name='hist5d_ckpt11', dir='/pscratch/sd/j/jiarongw/review/OPTION1_periodic/', ckpt=11,
         hist_duration_days=5, hist_freq_days=1, periodic=True),
    dict(name='hist5d_ckpt12', dir='/pscratch/sd/j/jiarongw/review/OPTION1_periodic/', ckpt=12,
         hist_duration_days=5, hist_freq_days=1, periodic=True),
    dict(name='hist10d_ckpt15',  dir='/pscratch/sd/j/jiarongw/review/OPTION1_periodic_hist10d/', ckpt=15,  hist_duration_days=10, hist_freq_days=1, periodic=True),
    dict(name='hist10d_ckpt20', dir='/pscratch/sd/j/jiarongw/review/OPTION1_periodic_hist10d/', ckpt=16, hist_duration_days=10, hist_freq_days=1, periodic=True),
    # Deterministic baseline (same backbone, trained by example/experiments/wave_simple.py):
    dict(name='hist10d_det_ckpt15', dir='/pscratch/sd/j/jiarongw/review/OPTION1_periodic_hist10d_det/', ckpt=15,
         hist_duration_days=10, hist_freq_days=1, periodic=True, deterministic=True),
    dict(name='hist10d_det_ckpt16', dir='/pscratch/sd/j/jiarongw/review/OPTION1_periodic_hist10d_det/', ckpt=16,
         hist_duration_days=10, hist_freq_days=1, periodic=True, deterministic=True),
]
# Same checkpoints, more sampling steps and ensemble members (each costs ~8x a default entry).
OPTION1_CHECKPOINTS += [sampling_variant(c, steps=40, n_ensem=20) for c in OPTION1_CHECKPOINTS
                        if c['name'] in ('hist5d_ckpt12', 'hist10d_ckpt16')]
OPTION3_CHECKPOINTS = [
    # Partition models (example/experiments/partdiff_periodic_hist10d{,_det}.py), trained on
    # 1993-2017, so YEARS above are out of sample. Same caveats on periodic/hist_* as above.
    # OPTION3_periodic (example/training/partdiff_periodic.py) uses a 5-day wind history.
    dict(name='hist5d_ckpt15', dir='/pscratch/sd/j/jiarongw/review/OPTION3_periodic/', ckpt=15,
         hist_duration_days=5, hist_freq_days=1, periodic=True),
    dict(name='hist5d_ckpt20', dir='/pscratch/sd/j/jiarongw/review/OPTION3_periodic/', ckpt=20,
         hist_duration_days=5, hist_freq_days=1, periodic=True),
    dict(name='hist10d_ckpt15', dir='/pscratch/sd/j/jiarongw/review/OPTION3_periodic_hist10d/', ckpt=15,
         hist_duration_days=10, hist_freq_days=1, periodic=True),
    dict(name='hist10d_ckpt20', dir='/pscratch/sd/j/jiarongw/review/OPTION3_periodic_hist10d/', ckpt=20,
         hist_duration_days=10, hist_freq_days=1, periodic=True),
    dict(name='hist10d_det_ckpt15', dir='/pscratch/sd/j/jiarongw/review/OPTION3_periodic_hist10d_det/', ckpt=15,
         hist_duration_days=10, hist_freq_days=1, periodic=True, deterministic=True),
    dict(name='hist10d_det_ckpt20', dir='/pscratch/sd/j/jiarongw/review/OPTION3_periodic_hist10d_det/', ckpt=20,
         hist_duration_days=10, hist_freq_days=1, periodic=True, deterministic=True),
]
OPTION3_CHECKPOINTS += [sampling_variant(c, steps=40, n_ensem=20) for c in OPTION3_CHECKPOINTS
                        if c['name'] in ('hist5d_ckpt20', 'hist10d_ckpt20')]


# Extra entries for the maps / plot_maps stages only (by name; the `stats` CSV
# and a `maps` run without names leave them out): the first epoch of the hist10d
# runs, to see whether the diffusion model's Southern Ocean Hs bias (see
# so_bias/README.md) is there from the start.
OPTION1_MAP_ONLY = [
    dict(name='hist10d_det_ckpt20', dir='/pscratch/sd/j/jiarongw/review/OPTION1_periodic_hist10d_det/', ckpt=20,
         hist_duration_days=10, hist_freq_days=1, periodic=True, deterministic=True),
    # OPTION1 versions of the OPTION3 Southern Ocean fixes (meandiff_periodic_hist10d_{highsigma,resid}.py),
    # sampled like them with 40 steps x 10 members
    dict(name='highsigma_ckpt6', dir='/pscratch/sd/j/jiarongw/review/OPTION1_periodic_hist10d_highsigma/', ckpt=6,
         hist_duration_days=10, hist_freq_days=1, periodic=True, steps=40, n_ensem=10),
    dict(name='resid_ckpt10', dir='/pscratch/sd/j/jiarongw/review/OPTION1_periodic_hist10d_resid/', ckpt=10,
         hist_duration_days=10, hist_freq_days=1, periodic=True, steps=40, n_ensem=10, kind='residual'),
    dict(name='hist10d_ckpt1', dir='/pscratch/sd/j/jiarongw/review/OPTION1_periodic_hist10d/', ckpt=1,
         hist_duration_days=10, hist_freq_days=1, periodic=True),
    dict(name='hist10d_det_ckpt1', dir='/pscratch/sd/j/jiarongw/review/OPTION1_periodic_hist10d_det/', ckpt=1,
         hist_duration_days=10, hist_freq_days=1, periodic=True, deterministic=True),
]
OPTION3_MAP_ONLY = [
    # Candidate fixes for the Southern Ocean Hs bias (so_bias/README.md), sampled with
    # 40 steps x 10 members (so_bias/sampling_test.py: steps matter more than members).
    dict(name='highsigma_ckpt5', dir='/pscratch/sd/j/jiarongw/review/OPTION3_periodic_hist10d_highsigma/', ckpt=5,
         hist_duration_days=10, hist_freq_days=1, periodic=True, steps=40, n_ensem=10),
    dict(name='resid_ckpt10', dir='/pscratch/sd/j/jiarongw/review/OPTION3_periodic_hist10d_resid/', ckpt=10,
         hist_duration_days=10, hist_freq_days=1, periodic=True, steps=40, n_ensem=10, kind='residual'),
    dict(name='hist10d_ckpt1', dir='/pscratch/sd/j/jiarongw/review/OPTION3_periodic_hist10d/', ckpt=1,
         hist_duration_days=10, hist_freq_days=1, periodic=True),
    dict(name='hist10d_det_ckpt1', dir='/pscratch/sd/j/jiarongw/review/OPTION3_periodic_hist10d_det/', ckpt=1,
         hist_duration_days=10, hist_freq_days=1, periodic=True, deterministic=True),
]

OPTION_CONFIGS = {
    1: dict(var_names=['hs', 'tm', 'thetam'],      # this repo's OPTION1 channel names
            clim_names=['hs', 'tm', 'dir'],        # matching variable names inside clim_file
            circular=['thetam'],
            x_dir=MEAN_PATH, x_prefix='wavemean',
            clim_file='~/scratch_folder/final/processed/mean_clim_monthly.nc',
            out_csv='checkpoint_comparison_stats_OPTION1.csv',
            checkpoints=OPTION1_CHECKPOINTS, map_only=OPTION1_MAP_ONLY),
    3: dict(var_names=['hs_p1', 'tp_p1', 'thetap_p1', 'hs_p2', 'tp_p2', 'thetap_p2', 'ci'],
            clim_names=['hs_p1', 'tp_p1', 'thetap_p1', 'hs_p2', 'tp_p2', 'thetap_p2', 'ci'],
            circular=['thetap_p1', 'thetap_p2'],
            x_dir=PART_PATH, x_prefix='waveparts',
            clim_file='~/scratch_folder/final/processed/partition_clim_monthly.nc',
            out_csv='checkpoint_comparison_stats_OPTION3.csv',
            checkpoints=OPTION3_CHECKPOINTS, map_only=OPTION3_MAP_ONLY),
}
_cfg = OPTION_CONFIGS[OPTION]
VAR_NAMES = _cfg['var_names']
CLIM_NAMES = _cfg['clim_names']
CIRCULAR = {v: v in _cfg['circular'] for v in VAR_NAMES}
DATA_PATH, X_PREFIX = _cfg['x_dir'], _cfg['x_prefix']    # forcing always comes from MEAN_PATH
CLIM_FILE = os.path.expanduser(_cfg['clim_file'])
STATS_FILE = os.path.join(DATA_PATH, f'stats_OPTION{OPTION}.npz')
OUT_CSV = os.path.join(OUT_DIR, _cfg['out_csv'].replace('.csv', f'{PERIOD_TAG}.csv'))
CHECKPOINTS = _cfg['checkpoints']
MAP_ONLY_CHECKPOINTS = _cfg['map_only']

# Defaults; a CHECKPOINTS entry can override them with its own `n_ensem` / `steps` (see sampling_variant).
N_ENSEM = 10    # ensemble members per date; drives both the mean (MSE/ACC) and the spread (SSR)
STEPS = 10
SIGMA0 = 100


############### Legacy checkpoint compatibility ###############
def remap_legacy_conv_keys(state_dict):
    """Some checkpoints (e.g. OPTION1_moredata's ckpt_20.pt, saved 2026-05-09)
    predate commit d4cca51 ('Adjust padding for UNet for periodic condition in
    longitude', 2026-08-26), which wrapped conv_in / ResnetBlock's convs /
    Downsample / Upsample's conv / out_layer's conv in new GeoConv2d/GeoDownsample
    classes. For periodic=False that wrapping is numerically a no-op (identical
    padding=1, identical computation) -- only the parameter names changed (an
    extra '.conv' in the path), so a plain key rename recovers the exact original
    model. Attention, GroupNorm, the temb/cond embeddings, and the 1x1 shortcut
    conv were untouched by that refactor and are left alone."""
    def rename(k):
        for suf in ('.weight', '.bias'):
            if k.endswith('.downsample.1' + suf):
                return k[:-len('.downsample.1' + suf)] + '.downsample.conv' + suf
            if k.endswith('.upsample.1' + suf):
                return k[:-len(suf)] + '.conv' + suf
            if k == 'conv_in' + suf:
                return 'conv_in.conv' + suf
            if k == 'out_layer.2' + suf:
                return 'out_layer.2.conv' + suf
            if k.endswith('.layer1.2' + suf) or k.endswith('.layer2.3' + suf):
                return k[:-len(suf)] + '.conv' + suf
        return k
    return {rename(k): v for k, v in state_dict.items()}


def load_model_weights(model, state_dict, label):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f'[{label}] strict load_state_dict failed ({type(e).__name__}); '
              f'retrying assuming a pre-GeoConv2d-refactor checkpoint...')
        model.load_state_dict(remap_legacy_conv_keys(state_dict))


############### Sampling helper (adapted from analysis/paper_figure/sample.py) ###############
@torch.no_grad()
def sampler_sigmas(schedule_infer, steps, spacing='loglinear'):
    """steps + 1 decreasing sigmas: the log-linear grid of schedule_infer, or
    'karras' (EDM, rho = 7) spacing over the same range, ending at sigma_min."""
    if spacing == 'loglinear':
        return schedule_infer.sample_sigmas(steps)
    rho, smin, smax = 7.0, float(schedule_infer.sigmas.min()), float(schedule_infer.sigmas.max())
    i = torch.arange(steps, dtype=torch.float32) / (steps - 1)
    s = (smax ** (1 / rho) + i * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
    return torch.cat([s, s[-1:]])


def sample_ensemble(model, ema, accelerator, schedule_infer, test, f, x, mask, n_ensem=10, steps=10,
                    spacing='loglinear', residual=False, chunk=10):
    """Draws an n_ensem-member ensemble for one date (in chunks of `chunk`
    members, to bound GPU memory) and reduces it to a per-pixel mean (for
    MSE/ACC) and variance (for SSR), variable-by-variable so direction gets the
    circular treatment. The raw members are also returned for CRPS. residual:
    the model is a ResidualDiffusion and samples through its sample()."""
    sigmas = sampler_sigmas(schedule_infer, steps, spacing)
    x0_ensem = []
    with ema.average_parameters():
        for start in range(0, n_ensem, chunk):
            n = min(chunk, n_ensem - start)
            cond = f.repeat(n, 1, 1, 1).to(accelerator.device)
            if residual:
                x0, _ = accelerator.unwrap_model(model).sample(cond, mask, sigmas, gam=1, mu=0.5,
                                                               accelerator=accelerator)
            else:
                *_, x0 = samples(model, sigmas, gam=1, mu=0.5,
                                  batchsize=n, accelerator=accelerator, cond=cond, mask=mask, xt=None)
            for i in range(n):
                x_ = test.invert_x(x0[i]) * tf.Resize((320, 720))(mask[0].to(x0))
                x0_ensem.append(x_.cpu().numpy())
    x0_ensem = np.stack(x0_ensem, axis=0)  # (n_ensem, C, H, W)
    x_truth = (test.invert_x(x[0]) * tf.Resize((320, 720))(mask[0].to(x[0]))).cpu().numpy()

    mean = np.empty(x0_ensem.shape[1:], dtype=x0_ensem.dtype)
    var = np.empty_like(mean)
    for c, v in enumerate(VAR_NAMES):
        mean[c], var[c] = ensemble_mean_var(x0_ensem[:, c], circular=CIRCULAR[v])
    return x_truth, mean, var, x0_ensem


@torch.no_grad()
def predict_deterministic(model, ema, test, f, x, mask):
    """Single forward pass of a DetUnet, returned in the same shape as
    sample_ensemble so the metric code is shared: the prediction is the 'mean',
    a 1-member 'ensemble' (so weighted_crps reduces to MAE), and NaN variance."""
    model.eval()    # samples() does this for the diffusion path; needed here to switch off dropout
    with ema.average_parameters():
        pred = model(f.to(mask.device))
    out_mask = tf.Resize((320, 720))(mask[0].to(pred))
    x_pred = (test.invert_x(pred[0]) * out_mask).cpu().numpy()
    x_truth = (test.invert_x(x[0]) * tf.Resize((320, 720))(mask[0].to(x[0]))).cpu().numpy()
    return x_truth, x_pred, np.full_like(x_pred, np.nan), x_pred[None]


############### Resume support ###############
def csv_columns():
    cols = ['checkpoint', 'date', 'year', 'month', 'season']
    for v in VAR_NAMES:
        cols += [f'{v}_mse', f'{v}_spreadvar', f'{v}_acc', f'{v}_crps']
    return cols


def already_done():
    if not os.path.exists(OUT_CSV):
        return set()
    with open(OUT_CSV) as fh:
        header = fh.readline().strip().split(',')
    if header != csv_columns():
        # Appending rows with a different column layout would silently misalign them.
        raise RuntimeError(f'{OUT_CSV} has columns {header}, expected {csv_columns()} '
                           f'(e.g. written before CRPS was added). Move it aside and re-run.')
    df = pd.read_csv(OUT_CSV, usecols=['checkpoint', 'date'])
    return set(zip(df['checkpoint'], df['date'].astype(str)))


def eval_setup():
    """Monthly files and the evaluation dates shared by every checkpoint."""
    file_list, ym_list = available_file_list(YEARS, DATA_PATH, MEAN_PATH, x_prefix=X_PREFIX)
    if not file_list:
        raise RuntimeError('No preprocessed monthly files found for the requested years.')
    n_history = 0
    if PERIOD_TAG:
        # Override periods also load the month before the first one, for history
        # only, so evaluation starts on the first day (the default period keeps
        # its original dates, which its CSV and maps were computed on).
        prev, _ = months_file_list([ym_list[0]], DATA_PATH, MEAN_PATH, x_prefix=X_PREFIX)
        if len(prev) == 2:
            file_list, ym_list = prev[:1] + file_list, [previous_month(*ym_list[0])] + ym_list
            n_history = file_lengths(prev[:1])[0]
    # File geometry is identical across checkpoints; only the usable `padding` at
    # the front differs, so all checkpoints are scored on the same dates.
    lengths = file_lengths(file_list)
    dates, gap_before = build_date_index(ym_list, lengths)
    paddings = [hist_window(c['hist_duration_days'], c['hist_freq_days'])[1] for c in CHECKPOINTS]
    eval_raw_idx = select_eval_indices(len(dates), paddings, STRIDE_DAYS, gap_before, start=n_history)
    print(f'Evaluating {len(eval_raw_idx)} dates from {dates[eval_raw_idx[0]].date()} '
          f'to {dates[eval_raw_idx[-1]].date()} (every {STRIDE_DAYS} days)')
    return file_list, lengths, dates, eval_raw_idx


def predictions(checkpoints, is_done):
    """For each checkpoint, and each evaluation date for which is_done(name, date)
    is False, yields (name, date, x_truth, x_mean, x_var, x_ensem, valid) on the
    native grid; valid is the ice-free mask (resized icymask > 0.9)."""
    file_list, lengths, dates, eval_raw_idx = eval_setup()
    stats = np.load(STATS_FILE)
    meanx, stdx, meanf, stdf = stats['meanx'], stats['stdx'], stats['meanf'], stats['stdf']
    a = Accelerator()
    schedule_infer = ScheduleLogLinear(sigma_min=0.01, sigma_max=SIGMA0, N=80)

    for ckpt_cfg in checkpoints:
        name = ckpt_cfg['name']
        pending = [r for r in eval_raw_idx if not is_done(name, dates[r])]
        print(f'[{name}] {len(eval_raw_idx) - len(pending)} dates already done, {len(pending)} remaining')
        if not pending:
            continue
        hist_days, hist_freq = ckpt_cfg['hist_duration_days'], ckpt_cfg['hist_freq_days']
        n_hist, padding = hist_window(hist_days, hist_freq)
        precond_ch = 4 + 2 * n_hist    # 4 base forcing channels + (u, v) per wind-history snapshot

        test = npyDataWndHist(file_list, resize_x=(320, 320), resize_f=(320, 320),
                               landmaskname=os.path.join(DATA_PATH, 'mask.npy'),
                               use_icymask=True, compute_stats=False,
                               meanx=meanx, stdx=stdx, meanf=meanf, stdf=stdf, OPTION=OPTION,
                               hist_duration_days=hist_days, hist_freq_days=hist_freq)
        assert test.file_lengths == lengths and test.padding == padding, 'timeindex out of sync with npyDataWndHist'

        deterministic = ckpt_cfg.get('deterministic', False)
        residual = ckpt_cfg.get('kind') == 'residual'
        unet = lambda cls, pc: cls(in_dim=320, in_ch=len(VAR_NAMES), out_ch=len(VAR_NAMES), ch=256, precond_ch=pc,
                                   scale=(test.meanx, test.stdx, test.meanf, test.stdf),
                                   ch_mult=(1, 2, 2), attn_resolutions=(16,), periodic=ckpt_cfg['periodic'])
        if residual:    # diffusion on the residual of a frozen DetUnet (partdiff_periodic_hist10d_resid.py)
            model = ResidualDiffusion(unet(Scaled(myUnet), precond_ch + len(VAR_NAMES)), unet(DetUnet, precond_ch),
                                      rstd=np.ones(len(VAR_NAMES)), stdx=torch.as_tensor(test.stdx).flatten(),
                                      circular=[i for i, v in enumerate(VAR_NAMES) if CIRCULAR[v]])
            ema = EMA(model.net.parameters(), decay=0.999)
        else:
            model = unet(DetUnet if deterministic else Scaled(myUnet), precond_ch)
            ema = EMA(model.parameters(), decay=0.999)
        ckpt = torch.load(os.path.join(ckpt_cfg['dir'], f"ckpt_{ckpt_cfg['ckpt']}.pt"), map_location='cpu')
        load_model_weights(model, ckpt['model'], name)
        ema.load_state_dict(ckpt['ema'])
        model = a.prepare(model)
        ema.to(a.device)

        for raw_idx in tqdm(pending, desc=name):
            date = dates[raw_idx]
            x, f, icymask = test.__getitem__(to_dataset_idx(raw_idx, padding))
            x = x.unsqueeze(0); f = f.unsqueeze(0); icymask = icymask.unsqueeze(0)
            mask = icymask.to(a.device)

            if deterministic:
                x_truth, x_mean, x_var, x_ensem = predict_deterministic(model, ema, test, f, x, mask)
            else:
                x_truth, x_mean, x_var, x_ensem = sample_ensemble(model, ema, a, schedule_infer, test, f, x, mask,
                                                                  n_ensem=ckpt_cfg.get('n_ensem', N_ENSEM),
                                                                  steps=ckpt_cfg.get('steps', STEPS),
                                                                  spacing=ckpt_cfg.get('spacing', 'loglinear'),
                                                                  residual=residual)
            valid = tf.Resize((320, 720))(icymask[0].float()).numpy()[0] > 0.9
            yield name, date, x_truth, x_mean, x_var, x_ensem, valid

        del model, ema, test
        torch.cuda.empty_cache()


def main():
    wlat = np.load(WLAT_FILE)
    clim_ds = xr.open_dataset(CLIM_FILE).load().assign_coords(month=np.arange(1, 13))

    done = already_done()
    write_header = not os.path.exists(OUT_CSV)
    csv_file = open(OUT_CSV, 'a', newline='')
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(csv_columns())

    for name, date, x_truth, x_mean, x_var, x_ensem, icymask_resized in predictions(
            CHECKPOINTS, lambda name, date: (name, str(date.date())) in done):
        row = [name, date.date().isoformat(), date.year, date.month, SEASON_OF_MONTH[date.month]]
        for i, (v, cv) in enumerate(zip(VAR_NAMES, CLIM_NAMES)):
            circular = CIRCULAR[v]
            truth_v, mean_v, var_v = x_truth[i], x_mean[i], x_var[i]
            mse = weighted_mse(truth_v, mean_v, icymask_resized, wlat, circular=circular)
            spreadvar = weighted_mean(var_v, icymask_resized, wlat)
            crps = weighted_crps(x_ensem[:, i], truth_v, icymask_resized, wlat, circular=circular)
            clim = clim_ds[cv].sel(month=date.month).values
            if circular:
                truth_anom = circular_diff(truth_v, clim)
                mean_anom = circular_diff(mean_v, clim)
            else:
                truth_anom = truth_v - clim
                mean_anom = mean_v - clim
            acc = weighted_acc(mean_anom, truth_anom, icymask_resized, wlat)
            row += [mse, spreadvar, acc, crps]
        writer.writerow(row)
        csv_file.flush()

    csv_file.close()
    print(f'Done. Results written to {OUT_CSV}')

    df = pd.read_csv(OUT_CSV)
    print(f'\n=== OPTION{OPTION} summary (aggregated over all evaluated dates) ===')
    for v in VAR_NAMES:
        print(f'\n{v}:')
        agg = df.groupby('checkpoint')[[f'{v}_mse', f'{v}_spreadvar', f'{v}_acc', f'{v}_crps']].mean()
        summary = pd.DataFrame({
            'rmse': agg[f'{v}_mse'] ** 0.5,
            'ssr': (agg[f'{v}_spreadvar'] / agg[f'{v}_mse']) ** 0.5,
            'acc': agg[f'{v}_acc'],
            'crps': agg[f'{v}_crps'],    # linear in the data, so a plain mean over dates is correct
        })
        print(summary)


############### Per-pixel maps ###############
SEASONS = ['DJF', 'MAM', 'JJA', 'SON']
# Variables in the saved maps: the model's channels, plus for OPTION3 the total
# Hs of the two partitions, sqrt(hs_p1^2 + hs_p2^2), computed per member (its
# spread and CRPS can't be recovered from the partitions' statistics).
MAP_VARS = VAR_NAMES + (['hs_tot'] if OPTION == 3 else [])
MAP_CIRCULAR = {v: CIRCULAR.get(v, False) for v in MAP_VARS}


def maps_dir(name):
    return os.path.join(MAPS_DIR, f'OPTION{OPTION}', name)


def maps_file(name, date):
    return os.path.join(maps_dir(name), f'{pd.Timestamp(date).date()}.npz')


def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path + '.tmp.npz', **arrays)
    os.replace(path + '.tmp.npz', path)    # never leave a half-written file behind on timeout


def with_derived(x):
    """(..., C, H, W) -> (..., len(MAP_VARS), H, W): appends hs_tot for OPTION3."""
    if OPTION != 3:
        return x
    h1, h2 = x[..., VAR_NAMES.index('hs_p1'), :, :], x[..., VAR_NAMES.index('hs_p2'), :, :]
    return np.concatenate([x, np.sqrt(h1 ** 2 + h2 ** 2)[..., None, :, :]], axis=-3)


def compute_maps(args):
    """Per checkpoint and date, MAPS_DIR/OPTION<n>/<name>/<date>.npz with the
    ensemble mean, variance, per-pixel fair CRPS and member 0, each
    (len(MAP_VARS), H, W) -- for deterministic entries only the forecast as
    `mean` (map_stats derives the rest); with --all-members also
    `members` (n, C, H, W). The truth and ice-free mask, the same for every
    checkpoint, go to MAPS_DIR/OPTION<n>/truth/<date>.npz. Existing files are
    skipped, so it resumes; with --all-members, files without members are redone."""
    all_members = '--all-members' in args
    names = [a for a in args if a != '--all-members']
    chosen = [c for c in CHECKPOINTS + MAP_ONLY_CHECKPOINTS if c['name'] in names] if names else CHECKPOINTS
    missing = set(names) - {c['name'] for c in chosen}
    if missing:
        raise RuntimeError(f'{sorted(missing)} not in CHECKPOINTS for OPTION{OPTION}')

    def is_done(name, date):
        path = maps_file(name, date)
        if not os.path.exists(path):
            return False
        if not all_members:
            return True
        with np.load(path) as z:
            return 'members' in z.files

    for name, date, x_truth, x_mean, x_var, x_ensem, valid in predictions(chosen, is_done):
        truth = with_derived(x_truth).astype(np.float32)
        if not os.path.exists(maps_file('truth', date)):
            save_npz(maps_file('truth', date), truth=truth, valid=valid, vars=np.array(MAP_VARS))
        ens = with_derived(x_ensem)
        mean, var, crps = (np.empty(ens.shape[1:], np.float32) for _ in range(3))
        for c, v in enumerate(MAP_VARS):
            mean[c], var[c] = ensemble_mean_var(ens[:, c], circular=MAP_CIRCULAR[v])
            crps[c] = ensemble_crps(ens[:, c], truth[c], circular=MAP_CIRCULAR[v])
        if len(ens) == 1:
            var[:] = np.nan    # no spread for a deterministic entry
        if len(ens) == 1:    # deterministic: var is NaN, member0 = mean, crps = |error| (map_stats)
            out = dict(mean=mean, n_members=1, vars=np.array(MAP_VARS))
        else:
            out = dict(mean=mean, var=var, crps=crps, member0=ens[0].astype(np.float32),
                       n_members=len(ens), vars=np.array(MAP_VARS))
        if all_members:
            out['members'] = x_ensem.astype(np.float32)
        save_npz(maps_file(name, date), **out)
    print(f'Maps in {os.path.join(MAPS_DIR, f"OPTION{OPTION}")}')


def map_stats(name, season):
    """Per-pixel bias and RMSE of the ensemble mean, spread (sqrt of the mean
    variance) and CRPS, each (len(MAP_VARS), H, W), averaged over the ice-free
    dates of `season` ('all' or DJF/...) among that checkpoint's saved dates."""
    files = [f for f in sorted(glob.glob(os.path.join(maps_dir(name), '*.npz')))
             if not f.endswith('.tmp.npz')
             and season in ('all', SEASON_OF_MONTH[int(os.path.basename(f)[5:7])])]
    sums, count = {}, 0
    for f in files:
        z, t = np.load(f), np.load(maps_file('truth', os.path.basename(f)[:10]))
        valid = t['valid']
        err = np.stack([circular_diff(z['mean'][c], t['truth'][c]) if MAP_CIRCULAR[v]
                        else z['mean'][c] - t['truth'][c] for c, v in enumerate(MAP_VARS)])
        var = z['var'] if 'var' in z.files else np.full_like(err, np.nan)    # deterministic files: mean only
        crps = z['crps'] if 'crps' in z.files else np.abs(err)
        for k, a in dict(err=err, se=err ** 2, var=var, crps=crps).items():
            sums[k] = sums.get(k, 0) + np.where(valid, a, 0).astype(np.float64)    # NaN var stays NaN
        count = count + valid
    with np.errstate(invalid='ignore', divide='ignore'):
        return dict(bias=sums['err'] / count, rmse=np.sqrt(sums['se'] / count),
                    spread=np.sqrt(sums['var'] / count), crps=sums['crps'] / count, n_dates=len(files))


def plot_maps(args):
    from matplotlib import pyplot as plt
    from wavediffusion.waveana import lat_deg, lon_deg
    season = args[0] if args and args[0] in SEASONS + ['all'] else 'all'
    names = [a for a in args if a not in SEASONS + ['all']]
    entries = [c for c in CHECKPOINTS + MAP_ONLY_CHECKPOINTS if (not names or c['name'] in names)
               and glob.glob(os.path.join(maps_dir(c['name']), '*.npz'))]
    if not entries:
        raise RuntimeError(f'No maps in {os.path.join(MAPS_DIR, f"OPTION{OPTION}")}; run the maps stage first.')
    fields = {c['name']: map_stats(c['name'], season) for c in tqdm(entries, desc='reading maps')}
    ref = next((c['name'] for c in entries if c.get('deterministic')), None)
    lon_order = np.argsort(np.mod(lon_deg, 360))    # Pacific-centred
    lon360 = np.mod(lon_deg, 360)[lon_order]
    os.makedirs(FIG_DIR, exist_ok=True)
    suffix = ('' if OPTION == 1 else f'_OPTION{OPTION}') + PERIOD_TAG
    titles = dict(bias='mean bias (model - truth)', rmse='RMSE', spread='ensemble spread (std)', crps='CRPS',
                  dcrps=f'CRPS - CRPS({ref})')
    for c, v in enumerate(MAP_VARS):
        cols = ['bias', 'rmse', 'spread', 'crps'] + (['dcrps'] if ref else [])
        maps = {(e['name'], k): fields[e['name']][k][c] for e in entries for k in ('bias', 'rmse', 'spread', 'crps')}
        if ref:
            maps.update({(e['name'], 'dcrps'): fields[e['name']]['crps'][c] - fields[ref]['crps'][c]
                         for e in entries if e['name'] != ref})
        has_spread = {e['name']: not e.get('deterministic') for e in entries}

        def p99(keys, absolute=False):
            vals = [np.abs(m) if absolute else m for (n_, k), m in maps.items() if k in keys
                    and (k != 'spread' or has_spread[n_])]
            return np.nanpercentile(np.stack(vals), 99) if vals else 1.0

        # RMSE and spread share a scale, so their ratio (spread-skill) reads directly.
        scale = dict(bias=('RdBu_r', -p99(['bias'], True), p99(['bias'], True)),
                     rmse=('viridis', 0, p99(['rmse', 'spread'])), spread=('viridis', 0, p99(['rmse', 'spread'])),
                     crps=('viridis', 0, p99(['crps'])), dcrps=('RdBu_r', -p99(['dcrps'], True), p99(['dcrps'], True)))
        fig, axes = plt.subplots(len(entries), len(cols), figsize=(4.6 * len(cols), 2.4 * len(entries)),
                                 squeeze=False)
        for i, e in enumerate(entries):
            name = e['name']
            for j, k in enumerate(cols):
                ax = axes[i, j]
                missing = {'dcrps': 'reference', 'spread': 'no ensemble spread'}.get(k) if (
                    (name, k) not in maps or (k == 'spread' and not has_spread[name])) else None
                if missing:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, f'{name}:\n{missing}', ha='center', va='center', fontsize=8,
                            transform=ax.transAxes)
                    continue
                cmap, vmin, vmax = scale[k]
                pc = ax.pcolormesh(lon360, lat_deg, maps[(name, k)][:, lon_order], cmap=cmap, vmin=vmin, vmax=vmax,
                                   shading='auto')
                ax.set_title(f'{name}: {titles[k]}', fontsize=8)
                ax.set_aspect('equal')
                ax.tick_params(labelsize=6)
                fig.colorbar(pc, ax=ax, shrink=0.8)
        n_dates = sorted({fields[e['name']]['n_dates'] for e in entries})
        fig.suptitle(f'{v}: per-pixel bias, RMSE and spread of the ensemble mean, and fair CRPS, {season} '
                     f'({"/".join(map(str, n_dates))} dates, {YEARS[0]}-{YEARS[-1]})', fontsize=10)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f'checkpoint_maps_{v}{suffix}_{season}.png')
        fig.savefig(out, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {out}')


def load_truth(date):
    """Native-grid truth (C, H, W) at `date`, NaN where a variable is undefined
    (ice / land; see truth_maps)."""
    ym = f'{date.year}{date.month:02d}'
    step = (date - pd.Timestamp(year=date.year, month=date.month, day=1)) // pd.Timedelta(hours=3)
    x = np.load(os.path.join(DATA_PATH, f'{X_PREFIX}_{ym}.npy'), mmap_mode='r')[step]
    x = np.array(x[:len(VAR_NAMES)], dtype=np.float32)    # OPTION1 files hold extra channels after hs, tm, thetam
    ice_free = np.load(os.path.join(MEAN_PATH, f'forcing_{ym}.npy'), mmap_mode='r')[step, 2] > 0.5
    x[:, ~ice_free] = np.nan
    if OPTION == 3:
        for k in (1, 2):
            absent = ~(x[VAR_NAMES.index(f'hs_p{k}')] > 0)
            for v in (f'tp_p{k}', f'thetap_p{k}'):
                x[VAR_NAMES.index(v), absent] = np.nan
    return x


def truth_maps():
    from matplotlib import pyplot as plt
    from wavediffusion.waveana import lat_deg, lon_deg
    _, _, dates, eval_raw_idx = eval_setup()
    eval_dates = [dates[r] for r in eval_raw_idx]
    truth = np.stack([load_truth(d) for d in tqdm(eval_dates, desc='truth')])    # (T, C, H, W)
    seasons = np.array([SEASON_OF_MONTH[d.month] for d in eval_dates])
    periods = ['all'] + SEASONS
    lon_order = np.argsort(np.mod(lon_deg, 360))    # Pacific-centred
    lon360 = np.mod(lon_deg, 360)[lon_order]
    os.makedirs(FIG_DIR, exist_ok=True)
    suffix = ('' if OPTION == 1 else f'_OPTION{OPTION}') + PERIOD_TAG
    for c, v in enumerate(VAR_NAMES):
        mean, std, n_dates = {}, {}, {}
        for p in periods:
            a = truth[:, c] if p == 'all' else truth[seasons == p, c]
            n_dates[p] = len(a)
            with np.errstate(invalid='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)    # all-NaN pixels (land, permanent ice)
                if CIRCULAR[v]:
                    rad = np.deg2rad(a)
                    mean[p] = np.rad2deg(np.arctan2(np.nanmean(np.sin(rad), 0), np.nanmean(np.cos(rad), 0))) % 360
                    std[p] = np.sqrt(np.nanmean(circular_diff(a, mean[p][None]) ** 2, 0))
                else:
                    mean[p], std[p] = np.nanmean(a, 0), np.nanstd(a, 0)
        fig, axes = plt.subplots(2, len(periods), figsize=(4.4 * len(periods), 5.2), squeeze=False)
        rows = [('mean', mean, 'twilight' if CIRCULAR[v] else 'viridis',
                 (0, 360) if CIRCULAR[v] else (np.nanpercentile(np.stack(list(mean.values())), 1),
                                               np.nanpercentile(np.stack(list(mean.values())), 99))),
                ('std', std, 'magma', (0, np.nanpercentile(np.stack(list(std.values())), 99)))]
        for i, (label, field, cmap, (vmin, vmax)) in enumerate(rows):
            for j, p in enumerate(periods):
                ax = axes[i, j]
                pc = ax.pcolormesh(lon360, lat_deg, field[p][:, lon_order], cmap=cmap, vmin=vmin, vmax=vmax,
                                   shading='auto')
                ax.set_title(f'{label}, {p} ({n_dates[p]} dates)', fontsize=8)
                ax.set_aspect('equal')
                ax.tick_params(labelsize=6)
            fig.colorbar(pc, ax=list(axes[i]), shrink=0.8, label=f'{v} {label}')
        fig.suptitle(f'Truth {v}: temporal mean and standard deviation over the evaluation dates, '
                     f'{YEARS[0]}-{YEARS[-1]} every {STRIDE_DAYS} days', fontsize=10)
        out = os.path.join(FIG_DIR, f'truth_maps_{v}{suffix}.png')
        fig.savefig(out, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {out}')


if __name__ == '__main__':
    if STAGE == 'stats':
        main()
    elif STAGE == 'maps':
        compute_maps(STAGE_ARGS)
    elif STAGE == 'plot_maps':
        plot_maps(STAGE_ARGS)
    elif STAGE == 'truth_maps':
        truth_maps()
    else:
        sys.exit(f'Unknown stage {STAGE!r}; see the module docstring.')
