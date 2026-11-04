''' PDF of the partition-energy fraction: how much of the true total wave energy
    the first two partitions carry, in the truth and in sampled partitions.

    Energy goes as Hs^2, so per pixel the fractions are
        r1  = Hs_p1^2 / Hs^2,   r2 = Hs_p2^2 / Hs^2,   r12 = r1 + r2,
    with Hs the TRUE total significant wave height (wavemean channel 0) in both
    cases. In the truth r12 <= 1 up to rounding (the rest of the energy is in
    higher partitions / unpartitioned); a sampled r12 > 1 means the model puts
    more energy in the two partitions than the sea state has.

    Stages:
      python partition_energy.py truth
      python partition_energy.py sample SOURCE [SOURCE ...]   (GPU; see MODELS)
      python partition_energy.py plot
      python partition_energy.py maps [SOURCE ...]     (CPU; see MAPS_SOURCES)
      python partition_energy.py plot --maps        (PDF of sqrt(r12), linear, all dates; and the correlation map)
    `truth` and `sample` write output/partition_energy_<source>.npz: per date, the
    cos-lat (wlat) weighted histogram over ice-free ocean pixels with true
    Hs > HS_MIN of r1, r2 and r12 on BINS (array `hist`, (date, 3, nbin)). For
    models these are of the ENSEMBLE MEAN: Hs_p1 and Hs_p2 averaged over members,
    then squared (the field sample.py saves as mean_*.npy); the per-member
    histograms, summed over members, are kept as `hist_members` but not plotted.
    `sample` saves after every date, so a re-run
    resumes. `truth_resized` is the truth partitions sent through the models'
    320x320 resize and back, to separate that smoothing from model error.
    `sample` also accumulates, per pixel over dates, the sums needed for the
    Pearson correlation between the true r12 and the ensemble-mean r12 (as above)
    (array `corr_sums`, (6, H, W): n, sum x, sum y, sum x^2, sum y^2, sum xy;
    x = truth, y = sample; a pixel counts on a date when it is valid then).
    `plot` draws figures/partition_energy_pdf.png (all dates; one panel per fraction)
    and figures/partition_energy_pdf_season.png (r12 by season), and prints summary
    statistics per source; and figures/partition_energy_r12_corr.png, the map of
    that correlation per model (pixels valid on fewer than CORR_N_MIN dates blank).
    `maps` does the same from the per-date map files that compare_checkpoints_stats.py
    `maps` already saved (MAPS_DIR: the 2020-2025 every-5-day run), for the
    dates of MAPS_YEARS, without sampling: the truth from the native files as
    in `truth`, `truth_resized` from the maps' truth (already through the
    320x320 resize), and each model's saved ensemble mean (`mean`). Only member
    0 is saved, so `hist_members` there is member 0's histogram. Its results
    and figures carry the suffix MAPS_TAG; `plot --maps` plots them.
    With --hs-option1 (on both `maps` and `plot --maps`), each source's Hs is
    instead the total Hs of its OPTION1 counterpart (HS_OPTION1: the OPTION1
    truth for truth_resized, the OPTION1 model's ensemble mean for a model; map
    files of the same run in MAPS_DIR_OPTION1), so the ratio compares two
    separately trained models' energies; pixels must also be ice-free on that
    file's grid and have that Hs > HS_MIN. Suffix HS1_TAG; the native `truth`
    curve keeps the true Hs.
'''
import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from wavediffusion.waveana import lat_deg, lon_deg
from wavediffusion.timeindex import months_file_list, build_date_index, SEASON_OF_MONTH

PART_PATH = '/global/homes/j/jiarongw/scratch_folder/wave_data/partition_global/'
MEAN_PATH = '/global/homes/j/jiarongw/scratch_folder/wave_data/mean_global/'
WLAT_FILE = '/global/homes/j/jiarongw/scratch_folder/wave_data/wlat.npy'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'output')     # per-date histograms (.npz)
FIG_DIR = os.path.join(HERE, 'figures')    # plots
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Dates: SAMPLE_DAYS of every month of YEARS at SAMPLE_HOUR (UTC). Partition
# models train on 1993-2017.
YEARS = list(range(2019, 2021))
SAMPLE_DAYS, SAMPLE_HOUR = [1, 8, 15, 22], 0
HS_MIN = 0.1       # m; drop near-calm pixels, where the ratio is ill-conditioned
BINS = np.linspace(0, 1.5, 151)
CORR_N_MIN = 20    # dates a pixel must be valid on to get a correlation
FRACTIONS = ['r1', 'r2', 'r12']
FRAC_LABELS = {'r1': '$H_{s,1}^2 / H_s^2$', 'r2': '$H_{s,2}^2 / H_s^2$',
               'r12': '$(H_{s,1}^2 + H_{s,2}^2) / H_s^2$'}

# Models for the `sample` stage. hist_* must match how each checkpoint was trained.
CKPT_ROOT = '/pscratch/sd/j/jiarongw/review/'
MODELS = {
    'hist5d_ckpt6':  dict(kind='diffusion', dir=CKPT_ROOT + 'OPTION3_periodic/', ckpt=6,
                          hist_duration_days=5, hist_freq_days=1, periodic=True),
    'hist10d_ckpt4': dict(kind='diffusion', dir=CKPT_ROOT + 'OPTION3_periodic_hist10d/', ckpt=4,
                          hist_duration_days=10, hist_freq_days=1, periodic=True),
    # 'hist10d_det_ckpt4': dict(kind='deterministic', dir=CKPT_ROOT + 'OPTION3_periodic_hist10d_det/', ckpt=4,
    #                           hist_duration_days=10, hist_freq_days=1, periodic=True),
    'truth_resized': dict(kind='identity', hist_duration_days=10, hist_freq_days=1),
}
N_ENSEM = 10
STEPS = 20
SIGMA0 = 100
# `maps` stage: sources whose compare_checkpoints_stats.py map files to use.
MAPS_DIR = '/global/homes/j/jiarongw/scratch_folder/review/checkpoint_comparison_maps_2020-2025_every5d/OPTION3/'
MAPS_YEARS = [2020]
MAPS_SOURCES = ['truth_resized', 'hist10d_det_ckpt20', 'highsigma_ckpt5', 'resid_ckpt10']
MAPS_TAG = f'_maps{MAPS_YEARS[0]}-{MAPS_YEARS[-1]}_every5d' if len(MAPS_YEARS) > 1 else f'_maps{MAPS_YEARS[0]}_every5d'
TAG = ''    # MAPS_TAG for the `maps` stage and `plot --maps`
# --hs-option1: Hs from the OPTION1 counterpart of each source (see the docstring)
HS_OPTION1 = {'truth_resized': 'truth', 'hist10d_det_ckpt20': 'hist10d_det_ckpt20',
              'highsigma_ckpt5': 'highsigma_ckpt6', 'resid_ckpt10': 'resid_ckpt10'}
MAPS_DIR_OPTION1 = MAPS_DIR.replace('/OPTION3/', '/OPTION1/')
HS1_TAG = '_hsOPTION1'
HS1 = '--hs-option1' in sys.argv
# Fixed per source (Okabe-Ito, as in regional_variability.py); line styles differ too.
SOURCE_STYLE = {
    'truth':             dict(color='#222222', ls='-',  label='truth'),
    'truth_resized':     dict(color='#999999', ls=':',  label='truth, 320x320 resized'),
    'hist10d_det_ckpt4': dict(color='#D55E00', ls='-.', label='deterministic, ckpt 4'),
    'hist5d_ckpt6':      dict(color='#009E73', ls=(0, (5, 1, 1, 1)), label=f'diffusion hist5d, ckpt 6 ({N_ENSEM}-member mean)'),
    'hist10d_ckpt4':     dict(color='#0072B2', ls='--', label=f'diffusion hist10d, ckpt 4 ({N_ENSEM}-member mean)'),
    'hist10d_det_ckpt20': dict(color='#D55E00', ls='-.', label='deterministic, ckpt 20'),
    'highsigma_ckpt5':   dict(color='#0072B2', ls='--', label='high-noise fine-tune, ckpt 5 (10-member mean)'),
    'resid_ckpt10':      dict(color='#009E73', ls=(0, (5, 1, 1, 1)), label='residual, ckpt 10 (10-member mean)'),
}


def plan_dates():
    """SAMPLE_DAYS of every month of YEARS; days past a month's end are skipped."""
    return pd.DatetimeIndex([pd.Timestamp(year=y, month=m, day=d, hour=SAMPLE_HOUR)
                             for y in YEARS for m in range(1, 13) for d in SAMPLE_DAYS
                             if d <= pd.Timestamp(year=y, month=m, day=1).days_in_month])


def out_path(source):
    return os.path.join(OUT_DIR, f'partition_energy_{source}{TAG}.npz')


def load_result(source):
    """(dates, hist, corr_sums); corr_sums is None for truth, which has none, and
    for files written before `hist` meant the ensemble mean (no `hist_members`)."""
    p = out_path(source)
    if not os.path.exists(p):
        return pd.DatetimeIndex([]), np.zeros((0, len(FRACTIONS), len(BINS) - 1)), None
    z = np.load(p)
    current = 'corr_sums' in z and 'hist_members' in z
    return pd.DatetimeIndex(z['dates']), z['hist'], (z['corr_sums'] if current else None)


def save_result(source, dates, hist, corr_sums=None, hist_members=None):
    extra = {} if corr_sums is None else {'corr_sums': corr_sums, 'hist_members': hist_members}
    np.savez(out_path(source), dates=np.asarray(dates, dtype='datetime64[ns]'), hist=hist, bins=BINS, **extra)


def month_step(d):
    return (d - pd.Timestamp(year=d.year, month=d.month, day=1)) // pd.Timedelta(hours=3)


def load_total(d):
    """True total Hs (H, W) and the valid-pixel mask at date d, native grid."""
    s = month_step(d)
    hs = np.asarray(np.load(os.path.join(MEAN_PATH, f'wavemean_{d.year}{d.month:02d}.npy'), mmap_mode='r')[s, 0])
    f = np.load(os.path.join(MEAN_PATH, f'forcing_{d.year}{d.month:02d}.npy'), mmap_mode='r')
    valid = (np.asarray(f[s, 2]) > 0.5) & np.isfinite(hs) & (hs > HS_MIN)    # forcing ch 2 = ice-free ocean
    return hs, valid


def load_parts(d):
    """True partitions (7, H, W) at date d, native grid."""
    p = np.load(os.path.join(PART_PATH, f'waveparts_{d.year}{d.month:02d}.npy'), mmap_mode='r')
    return np.asarray(p[month_step(d)])


def r12_field(hs1, hs2, hs, valid):
    """r12 (H, W), NaN outside valid; negative sampled Hs clipped to 0."""
    e = np.maximum(np.nan_to_num(hs1), 0) ** 2 + np.maximum(np.nan_to_num(hs2), 0) ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(valid, e / hs ** 2, np.nan)


def add_corr_sums(acc, x, y):
    """Accumulate Pearson sums of x (truth) and y (sample) where both are finite."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(m, x, 0.0), np.where(m, y, 0.0)
    acc += np.stack([m, x, y, x * x, y * y, x * y])


def corr_map(acc):
    n, sx, sy, sxx, syy, sxy = acc
    with np.errstate(invalid='ignore', divide='ignore'):
        cov = sxy / n - sx / n * sy / n
        r = cov / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))
    return np.where(n >= CORR_N_MIN, r, np.nan)


def fraction_hist(hs1, hs2, hs, valid, wlat):
    """(3, nbin) wlat-weighted histograms of r1, r2, r12 over valid pixels.
    Negative sampled Hs is clipped to 0; values beyond BINS land in the end bins."""
    e = hs[valid] ** 2
    r1 = np.maximum(np.nan_to_num(hs1[valid]), 0) ** 2 / e
    r2 = np.maximum(np.nan_to_num(hs2[valid]), 0) ** 2 / e
    w = wlat[valid]
    clip = lambda r: np.clip(r, BINS[0], BINS[-1])
    return np.stack([np.histogram(clip(r), BINS, weights=w)[0] for r in (r1, r2, r1 + r2)])


############### Truth (CPU) ###############
def compute_truth():
    wlat = np.load(WLAT_FILE)
    dates = plan_dates()
    hist, kept = [], []
    for d in tqdm(dates, desc='truth'):
        try:
            x = load_parts(d)
            hs, valid = load_total(d)
        except FileNotFoundError as e:
            print(f'[skip] {d}: {e}')
            continue
        hist.append(fraction_hist(x[0], x[3], hs, valid, wlat))
        kept.append(d)
    save_result('truth', kept, np.stack(hist))
    print(f'Wrote {out_path("truth")} ({len(kept)} dates)')


############### Models (GPU) ###############
def build_model(cfg, test):
    """Diffusion (Scaled(myUnet)) or deterministic (DetUnet) OPTION3 model with EMA weights."""
    import torch
    from torch_ema import ExponentialMovingAverage as EMA
    from wavediffusion.model_unet import myUnet, DetUnet
    from wavediffusion.model import Scaled
    model_cls = DetUnet if cfg['kind'] == 'deterministic' else Scaled(myUnet)
    model = model_cls(in_dim=320, in_ch=7, out_ch=7, ch=256, precond_ch=4 + 2 * test.n_hist,
                      scale=(test.meanx, test.stdx, test.meanf, test.stdf),
                      ch_mult=(1, 2, 2), attn_resolutions=(16,), periodic=cfg['periodic'])
    ema = EMA(model.parameters(), decay=0.999)
    ckpt = torch.load(os.path.join(cfg['dir'], f"ckpt_{cfg['ckpt']}.pt"), map_location='cpu')
    model.load_state_dict(ckpt['model'])
    ema.load_state_dict(ckpt['ema'])
    return model, ema


def predict_members(cfg, model, ema, accelerator, schedule, test, x, f, icymask):
    """Members (n, 7, 320, 720) in physical units."""
    import torch
    from wavediffusion.diffusion import samples
    with torch.no_grad():
        if cfg['kind'] == 'identity':
            out = [test.invert_x(x)]
        elif cfg['kind'] == 'deterministic':
            model.eval()
            with ema.average_parameters():
                out = [test.invert_x(model(f[None].to(accelerator.device))[0])]
        else:
            mask = icymask[None].to(accelerator.device)
            with ema.average_parameters():
                *_, x0 = samples(model, schedule.sample_sigmas(STEPS), gam=1, mu=0.5, batchsize=N_ENSEM,
                                 accelerator=accelerator, cond=f[None].repeat(N_ENSEM, 1, 1, 1),
                                 mask=mask, xt=None)
            out = [test.invert_x(x0[i]) for i in range(N_ENSEM)]
    return np.stack([o.float().cpu().numpy() for o in out])


def run_model(source):
    import torch
    from accelerate import Accelerator
    from wavediffusion.wavedata import npyDataWndHist
    from wavediffusion.diffusion import ScheduleLogLinear
    from wavediffusion.timeindex import dates_to_raw_indices, to_dataset_idx
    cfg = MODELS[source]
    done_dates, done_hist, corr_sums = load_result(source)
    if len(done_dates) and corr_sums is None:
        # Written by an older version (no correlation sums / per-member
        # histograms); keep it aside and sample again.
        old = out_path(source).replace('.npz', '_old.npz')
        os.replace(out_path(source), old)
        print(f'[{source}] {out_path(source)} is from an older version; moved to {old}, resampling')
        done_dates, done_hist = done_dates[:0], done_hist[:0]
    done = set(done_dates)
    kept, hist = list(done_dates), list(done_hist)
    hist_members = list(np.load(out_path(source))['hist_members']) if len(done_dates) else []
    dates = plan_dates()
    print(f'[{source}] {len(done)} dates already done')

    stats = np.load(os.path.join(PART_PATH, 'stats_OPTION3.npz'))
    wlat = np.load(WLAT_FILE)
    accelerator = Accelerator()
    schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=SIGMA0, N=80)
    model = ema = None
    if corr_sums is None:
        corr_sums = np.zeros((6,) + wlat.shape)

    # One dataset per test year (its months plus the previous one for the history).
    for y in sorted({d.year for d in dates}):
        want = [d for d in dates if d.year == y and d not in done]
        if not want:
            continue
        test_months = [(y, m) for m in range(1, 13)]
        try:
            file_list, ym_list = months_file_list(test_months, PART_PATH, MEAN_PATH, x_prefix='waveparts')
        except FileNotFoundError as e:
            print(f'[skip] {y}: {e}')
            continue
        test = npyDataWndHist(file_list, resize_x=(320, 320), resize_f=(320, 320),
                              landmaskname=os.path.join(PART_PATH, 'mask.npy'), use_icymask=True,
                              compute_stats=False, meanx=stats['meanx'], stdx=stats['stdx'],
                              meanf=stats['meanf'], stdf=stats['stdf'], OPTION=3,
                              hist_duration_days=cfg['hist_duration_days'], hist_freq_days=cfg['hist_freq_days'])
        if cfg['kind'] != 'identity' and model is None:
            model, ema = build_model(cfg, test)
            model = accelerator.prepare(model)
            ema.to(accelerator.device)
        date_index, _ = build_date_index(ym_list, test.file_lengths)
        for r in tqdm(dates_to_raw_indices(want, date_index, test.padding), desc=f'{source} {y}'):
            d = date_index[r]
            x, f, icymask = test[to_dataset_idx(r, test.padding)]
            members = predict_members(cfg, model, ema, accelerator, schedule, test, x, f, icymask)
            hs, valid = load_total(d)
            ens_mean = members.mean(axis=0)
            hist.append(fraction_hist(ens_mean[0], ens_mean[3], hs, valid, wlat))
            hist_members.append(sum(fraction_hist(m[0], m[3], hs, valid, wlat) for m in members))
            true_parts = load_parts(d)
            add_corr_sums(corr_sums, r12_field(true_parts[0], true_parts[3], hs, valid),
                          r12_field(ens_mean[0], ens_mean[3], hs, valid))
            kept.append(d)
            save_result(source, kept, np.stack(hist), corr_sums, np.stack(hist_members))
        del test
        torch.cuda.empty_cache()
    print(f'[{source}] done: {out_path(source)}')


############### From saved maps (CPU) ###############
def maps_dates():
    """Dates of MAPS_YEARS with a saved truth map (00 UTC; the files are named by day)."""
    names = sorted(os.listdir(os.path.join(MAPS_DIR, 'truth')))
    dates = pd.DatetimeIndex([pd.Timestamp(n[:-len('.npz')]) for n in names if n.endswith('.npz')])
    return dates[dates.year.isin(MAPS_YEARS)]


def run_maps(sources):
    """Truth plus each of `sources` on the dates every one of them has a map for."""
    wlat = np.load(WLAT_FILE)
    dates = maps_dates()
    for source in sources:
        name = 'truth' if source == 'truth_resized' else source
        have = {pd.Timestamp(n[:-len('.npz')]) for n in os.listdir(os.path.join(MAPS_DIR, name))}
        dates = dates[dates.isin(list(have))]
    print(f'{len(dates)} dates of {MAPS_YEARS} with maps for {sources}')
    out = {s: dict(hist=[], members=[], corr=np.zeros((6,) + wlat.shape)) for s in sources}
    truth_hist = []
    for d in tqdm(dates, desc='maps'):
        hs, valid = load_total(d)
        parts = load_parts(d)
        truth_hist.append(fraction_hist(parts[0], parts[3], hs, valid, wlat))
        r_true = r12_field(parts[0], parts[3], hs, valid)
        for source in sources:
            z = np.load(os.path.join(MAPS_DIR, 'truth' if source == 'truth_resized' else source, f'{d.date()}.npz'))
            v = list(z['vars'])
            i1, i2 = v.index('hs_p1'), v.index('hs_p2')
            mean = z['truth'] if source == 'truth_resized' else z['mean']
            first = mean if source == 'truth_resized' else z.get('member0', mean)
            o = out[source]
            hs_s, valid_s = hs, valid
            if HS1:    # total Hs of the OPTION1 counterpart (its ensemble mean / the OPTION1 truth)
                z1 = np.load(os.path.join(MAPS_DIR_OPTION1, HS_OPTION1[source], f'{d.date()}.npz'))
                t1 = np.load(os.path.join(MAPS_DIR_OPTION1, 'truth', f'{d.date()}.npz'))
                hs_s = (z1['truth'] if HS_OPTION1[source] == 'truth' else z1['mean'])[list(z1['vars']).index('hs')]
                valid_s = valid & t1['valid'] & (hs_s > HS_MIN)
            o['hist'].append(fraction_hist(mean[i1], mean[i2], hs_s, valid_s, wlat))
            o['members'].append(fraction_hist(first[i1], first[i2], hs_s, valid_s, wlat))
            add_corr_sums(o['corr'], r_true, r12_field(mean[i1], mean[i2], hs_s, valid_s))
    save_result('truth', dates, np.stack(truth_hist))
    for source, o in out.items():
        save_result(source, dates, np.stack(o['hist']), o['corr'], np.stack(o['members']))
        print(f'Wrote {out_path(source)}')


############### Plot ###############
def pdf(h):
    """Weighted histogram (..., nbin) -> probability density on BINS."""
    return h / (h.sum(axis=-1, keepdims=True) * np.diff(BINS))


def summarize(source, h):
    """Weighted mean, median and P(r >= 1.01) of each fraction from the histogram
    (1.01, not 1: truth partitions overshoot Hs by up to a few percent from rounding)."""
    centers = 0.5 * (BINS[1:] + BINS[:-1])
    p = h / h.sum(axis=-1, keepdims=True)
    above = BINS[:-1] >= 1.01 - 1e-9
    parts = []
    for k, name in enumerate(FRACTIONS):
        cdf = np.cumsum(p[k])
        med = centers[np.searchsorted(cdf, 0.5)]
        parts.append(f'{name}: mean {np.sum(p[k] * centers):.3f} median {med:.3f} P(>=1.01) {p[k][above].sum():.4f}')
    print(f'  {source:18s} ' + ' | '.join(parts))


def plot():
    results, corrs = {}, {}
    for source in SOURCE_STYLE:
        dates, h, c = load_result(source)
        if len(dates):
            results[source] = (dates, h)
        if c is not None:
            corrs[source] = (len(dates), corr_map(c))
    if 'truth' not in results:
        raise RuntimeError('No partition_energy_truth.npz; run the truth stage first.')
    # Compare on the dates every source has, so missing samples don't bias a PDF.
    common = sorted(set.intersection(*(set(d) for d, _ in results.values())))
    print(f'Comparing on {len(common)} common dates')
    sel = {s: (d[d.isin(common)], h[d.isin(common)]) for s, (d, h) in results.items()}
    centers = 0.5 * (BINS[1:] + BINS[:-1])
    if TAG:    # `plot --maps`: only the r12 PDF over all dates (linear) and the correlation map
        print(f'\n=== energy fractions (Hs > {HS_MIN} m, ice-free ocean, cos-lat weighted) ===')
        for source, (_, h) in sel.items():
            summarize(source, h.sum(axis=0))
        plot_r12_pdf(sel, common)
        if corrs:
            plot_corr(corrs)
        return

    print(f'\n=== energy fractions (Hs > {HS_MIN} m, ice-free ocean, cos-lat weighted) ===')
    fig, axes = plt.subplots(1, len(FRACTIONS), figsize=(4.4 * len(FRACTIONS), 3.4))
    for ax, (k, name) in zip(axes, enumerate(FRACTIONS)):
        for source, (_, h) in sel.items():
            st = SOURCE_STYLE[source]
            ax.plot(centers, pdf(h.sum(axis=0))[k], color=st['color'], ls=st['ls'], lw=1.6, label=st['label'])
        ax.axvline(1.0, color='#999999', lw=0.8)
        ax.set_yscale('log')
        ax.set_xlabel(FRAC_LABELS[name])
        ax.grid(alpha=0.25)
    for source, (_, h) in sel.items():
        summarize(source, h.sum(axis=0))
    axes[0].set_ylabel('PDF')
    axes[0].legend(fontsize=7, frameon=False)
    years = sorted({t.year for t in common})
    fig.suptitle(f'Partition energy relative to true total energy ({len(common)} dates, '
                 f'{years[0]}{"" if len(years) == 1 else f"-{years[-1]}"}; $H_s$ > {HS_MIN} m; models: ensemble mean)', fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f'partition_energy_pdf{TAG}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')

    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    k = FRACTIONS.index('r12')
    for ax, season in zip(axes, seasons):
        for source, (d, h) in sel.items():
            m = np.array([SEASON_OF_MONTH[t.month] == season for t in d])
            if not m.any():
                continue
            st = SOURCE_STYLE[source]
            ax.plot(centers, pdf(h[m].sum(axis=0))[k], color=st['color'], ls=st['ls'], lw=1.6, label=st['label'])
        ax.axvline(1.0, color='#999999', lw=0.8)
        ax.set_yscale('log')
        ax.set_title(season, fontsize=10)
        ax.set_xlabel(FRAC_LABELS['r12'])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('PDF')
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f'partition_energy_pdf_season{TAG}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')

    if corrs:
        plot_corr(corrs)


# Legend labels of plot_r12_pdf (no checkpoint numbers)
PDF_LABELS = {'truth_resized': 'truth', 'hist10d_det_ckpt20': 'deterministic',
              'highsigma_ckpt5': 'DDPM', 'resid_ckpt10': 'res-DDPM'}


def plot_r12_pdf(sel, common):
    """PDF of sqrt(r12) = sqrt(Hs_p1^2 + Hs_p2^2) / Hs over all common dates,
    linear axes, solid lines. The r12 histogram maps onto sqrt(BINS) (bins
    narrower towards 1, so the density divides by each one's width). The truth
    shown is truth_resized (labelled 'truth'): through the same 320x320 resize
    as the models, so only model error separates them. The last bin collects
    every r12 >= BINS[-2] (fraction_hist clips), so it is left out."""
    edges = np.sqrt(BINS)
    centers = 0.5 * (edges[1:] + edges[:-1])
    k = FRACTIONS.index('r12')
    years = sorted({t.year for t in common})
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for source, (_, h) in sel.items():
        if source == 'truth':
            continue
        h = h.sum(axis=0)[k]
        color = '#222222' if source == 'truth_resized' else SOURCE_STYLE[source]['color']
        density = h / (h.sum() * np.diff(edges))
        ax.plot(centers[:-1], density[:-1], color=color, ls='-', lw=1.6,
                label=PDF_LABELS.get(source, SOURCE_STYLE[source]['label']))
    ax.axvline(1.0, color='#999999', lw=0.8)
    ax.set_xlim(0.6, 1.2)
    ax.set_ylim(0, 25)
    ax.set_xlabel(r'$\sqrt{H_{sp1}^2 + H_{sp2}^2}\,/\,H_s$' + (r', $H_s$ from the OPTION1 models' if HS1 else ''))
    ax.set_ylabel('PDF')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.set_title(f'Year {years[0]}' if len(years) == 1 else f'Years {years[0]}-{years[-1]}', fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f'partition_energy_r12_pdf{TAG}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def plot_corr(corrs):
    """Map of the per-pixel correlation over dates of true vs ensemble-mean r12.
    Each source uses all its own dates (the count is in its title)."""
    wlat = np.load(WLAT_FILE)
    lon360 = np.mod(lon_deg, 360)
    order = np.argsort(lon360)    # Pacific-centred
    print(f'\n=== r12 correlation, truth vs ensemble mean (cos-lat weighted mean over pixels) ===')
    fig, axes = plt.subplots(len(corrs), 1, figsize=(9, 4.0 * len(corrs)), squeeze=False)
    for ax, (source, (n_dates, r)) in zip(axes[:, 0], corrs.items()):
        ok = np.isfinite(r)
        print(f'  {source:18s} mean {np.sum(r[ok] * wlat[ok]) / np.sum(wlat[ok]):.3f}  '
              f'median {np.median(r[ok]):.3f}  ({n_dates} dates)')
        ax.set_facecolor('#d9d9d9')    # land / too few valid dates
        im = ax.pcolormesh(lon360[order], lat_deg, r[:, order], cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
        ax.set_title(f"{SOURCE_STYLE[source]['label']} ({n_dates} dates)", fontsize=10)
        ax.set_aspect('equal')
        ax.set_ylabel('lat')
    axes[-1, 0].set_xlabel('lon')
    fig.colorbar(im, ax=axes[:, 0], shrink=0.6, label='corr(true $r_{12}$, ensemble-mean $r_{12}$)')
    out = os.path.join(FIG_DIR, f'partition_energy_r12_corr{TAG}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def main():
    global TAG
    stage = sys.argv[1] if len(sys.argv) > 1 else 'plot'
    if stage == 'maps' or '--maps' in sys.argv:
        TAG = MAPS_TAG + (HS1_TAG if HS1 else '')
    if stage == 'truth':
        compute_truth()
    elif stage == 'sample':
        for source in sys.argv[2:]:
            run_model(source)
    elif stage == 'maps':
        run_maps([a for a in sys.argv[2:] if not a.startswith('--')] or MAPS_SOURCES)
    elif stage == 'plot':
        plot()
    else:
        raise SystemExit(f'Unknown stage {stage!r}; use truth, sample, maps or plot.')


if __name__ == '__main__':
    main()
