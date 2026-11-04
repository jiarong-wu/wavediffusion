''' Plot MSE/SSR/ACC comparisons between the two checkpoints evaluated by
    compare_checkpoints_stats.py, and check for seasonal/yearly trends in
    error magnitude. Reads output/checkpoint_comparison_stats.csv (no GPU needed).

    SSR (spread-skill ratio) is not a per-date column in the CSV: following the
    convention in stats.ipynb, it's sqrt(mean(ensemble_variance) / mean(mse))
    aggregated over whatever dates you group by, so it's computed here, grouped
    by checkpoint x year and checkpoint x season.

    `python plot_checkpoint_comparison.py 3` plots the OPTION3 (partition) CSV
    instead; its figures get an _OPTION3 suffix.
'''
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats as sstats
from matplotlib import pyplot as plt

OPTION = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SEASON_ORDER = ['DJF', 'MAM', 'JJA', 'SON']

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Fixed categorical color assignment, colorblind-safe (Okabe-Ito blue/orange).
# Keys must match the `name` fields in compare_checkpoints_stats.py's CHECKPOINTS lists.
OPTION_CONFIGS = {
    1: dict(
        var_labels={'hs': 'Wave height $H_s$', 'tm': 'Wave period $T_m$', 'thetam': 'Direction $\\theta_m$'},
        in_csv='checkpoint_comparison_stats_OPTION1.csv', fig_suffix='',
        colors={
            'hist5d_ckpt11': '#009E73',
            'hist5d_ckpt12': '#CC79A7',
            'hist10d_ckpt15': '#56B4E9',
            'hist10d_ckpt16': '#0072B2',
            'hist10d_det_ckpt15': '#E69F00',
            'hist10d_det_ckpt16': '#D55E00',
            'hist5d_ckpt12_s40_e20': '#999999',
            'hist10d_ckpt16_s40_e20': '#000000',
        },
        labels={
            'hist5d_ckpt11': 'periodic (hist5d) diffusion, ckpt 11',
            'hist5d_ckpt12': 'periodic (hist5d) diffusion, ckpt 12',
            'hist10d_ckpt15': 'periodic_hist10d diffusion, ckpt 15',
            'hist10d_ckpt16': 'periodic_hist10d diffusion, ckpt 16',
            'hist10d_det_ckpt15': 'periodic_hist10d deterministic, ckpt 15',
            'hist10d_det_ckpt16': 'periodic_hist10d deterministic, ckpt 16',
            'hist5d_ckpt12_s40_e20': 'periodic (hist5d) diffusion, ckpt 12, 40 steps x 20 members',
            'hist10d_ckpt16_s40_e20': 'periodic_hist10d diffusion, ckpt 16, 40 steps x 20 members',
        }),
    3: dict(
        var_labels={'hs_p1': 'Partition 1 $H_s$', 'tp_p1': 'Partition 1 $T_p$', 'thetap_p1': 'Partition 1 $\\theta_p$',
                    'hs_p2': 'Partition 2 $H_s$', 'tp_p2': 'Partition 2 $T_p$', 'thetap_p2': 'Partition 2 $\\theta_p$',
                    'ci': 'Crossing-sea criterion'},
        in_csv='checkpoint_comparison_stats_OPTION3.csv', fig_suffix='_OPTION3',
        colors={
            'hist5d_ckpt11': '#009E73',
            'hist5d_ckpt12': '#CC79A7',
            'hist10d_ckpt8': '#56B4E9',
            'hist10d_ckpt9': '#0072B2',
            'hist10d_det_ckpt8': '#E69F00',
            'hist10d_det_ckpt9': '#D55E00',
            'hist5d_ckpt12_s40_e20': '#999999',
            'hist10d_ckpt9_s40_e20': '#000000',
        },
        labels={
            'hist5d_ckpt11': 'OPTION3 periodic (hist5d) diffusion, ckpt 11',
            'hist5d_ckpt12': 'OPTION3 periodic (hist5d) diffusion, ckpt 12',
            'hist10d_ckpt8': 'OPTION3 periodic_hist10d diffusion, ckpt 8',
            'hist10d_ckpt9': 'OPTION3 periodic_hist10d diffusion, ckpt 9',
            'hist10d_det_ckpt8': 'OPTION3 periodic_hist10d deterministic, ckpt 8',
            'hist10d_det_ckpt9': 'OPTION3 periodic_hist10d deterministic, ckpt 9',
            'hist5d_ckpt12_s40_e20': 'OPTION3 periodic (hist5d) diffusion, ckpt 12, 40 steps x 20 members',
            'hist10d_ckpt9_s40_e20': 'OPTION3 periodic_hist10d diffusion, ckpt 9, 40 steps x 20 members',
        }),
}
_cfg = OPTION_CONFIGS[OPTION]
VAR_LABELS = _cfg['var_labels']
VAR_NAMES = list(VAR_LABELS)
IN_CSV = os.path.join(HERE, 'output', _cfg['in_csv'])
FIG_SUFFIX = _cfg['fig_suffix']
CHECKPOINT_COLORS = _cfg['colors']
CHECKPOINT_LABELS = _cfg['labels']


def fig_path(stem):
    return os.path.join(FIG_DIR, f'{stem}{FIG_SUFFIX}.png')


def var_panels():
    """One panel per variable, wrapped to at most 4 per row (OPTION3 has 7)."""
    ncols = min(len(VAR_NAMES), 4)
    nrows = -(-len(VAR_NAMES) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    for ax in axes.flat[len(VAR_NAMES):]:
        ax.set_visible(False)
    return fig, axes.flat[:len(VAR_NAMES)]


def load():
    df = pd.read_csv(IN_CSV, parse_dates=['date'])
    # checkpoint_comparison_stats.csv is append-only and accumulates across every
    # run against it, so it may hold rows from a checkpoint that isn't part of
    # the pair currently configured in CHECKPOINT_COLORS/LABELS (e.g. a leftover
    # checkpoint slot swapped out in compare_checkpoints_stats.py's CHECKPOINTS
    # list). Keep only what we're actually set up to color/label here.
    other = sorted(set(df['checkpoint']) - set(CHECKPOINT_COLORS))
    if other:
        print(f'Ignoring rows for checkpoint(s) not in CHECKPOINT_COLORS: {other}')
        df = df[df['checkpoint'].isin(CHECKPOINT_COLORS)].copy()
    for v in VAR_NAMES:
        df[f'{v}_rmse'] = df[f'{v}_mse'] ** 0.5
    df['season'] = pd.Categorical(df['season'], categories=SEASON_ORDER, ordered=True)
    return df


def ssr(df, v):
    """sqrt(mean(ensemble variance) / mean(mse)) -- an aggregate over all rows in df."""
    return (df[f'{v}_spreadvar'].mean() / df[f'{v}_mse'].mean()) ** 0.5


def plot_timeseries(df):
    fig, axes = plt.subplots(len(VAR_NAMES), 2, figsize=(11, 3.2 * len(VAR_NAMES)), sharex=True)
    metrics = [('rmse', 'RMSE'), ('acc', 'ACC')]
    for row, v in enumerate(VAR_NAMES):
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            colname = f'{v}_{metric}'
            for ckpt, g in df.groupby('checkpoint'):
                g = g.sort_values('date')
                ax.plot(g['date'], g[colname], lw=1, marker='o', ms=2.5,
                        color=CHECKPOINT_COLORS[ckpt], label=CHECKPOINT_LABELS[ckpt])
            ax.set_title(f'{VAR_LABELS[v]} — {label}', fontsize=10)
            ax.tick_params(axis='x', rotation=30)
            ax.grid(alpha=0.25)
    axes[0, 0].legend(loc='upper left', bbox_to_anchor=(0, 1.35), ncol=2, fontsize=9, frameon=False)
    fig.tight_layout()
    out = fig_path('checkpoint_comparison_timeseries')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def plot_seasonal(df):
    fig, axes = var_panels()
    checkpoints = list(CHECKPOINT_COLORS.keys())
    width = 0.8 / len(checkpoints)
    positions = np.arange(len(SEASON_ORDER))
    for ax, v in zip(axes, VAR_NAMES):
        colname = f'{v}_rmse'
        for i, ckpt in enumerate(checkpoints):
            data = [df.loc[(df.checkpoint == ckpt) & (df.season == s), colname].dropna().values
                    for s in SEASON_ORDER]
            offset = (i - (len(checkpoints) - 1) / 2) * width
            bp = ax.boxplot(data, positions=positions + offset, widths=width * 0.9,
                             patch_artist=True, showfliers=False)
            for patch in bp['boxes']:
                patch.set_facecolor(CHECKPOINT_COLORS[ckpt])
                patch.set_alpha(0.55)
            for median in bp['medians']:
                median.set_color('black')
        ax.set_xticks(positions)
        ax.set_xticklabels(SEASON_ORDER)
        ax.set_title(f'{VAR_LABELS[v]} RMSE by season', fontsize=10)
        ax.grid(alpha=0.25, axis='y')
    handles = [plt.Rectangle((0, 0), 1, 1, color=CHECKPOINT_COLORS[c], alpha=0.55) for c in checkpoints]
    axes[0].legend(handles, [CHECKPOINT_LABELS[c] for c in checkpoints], loc='upper left', fontsize=8)
    fig.tight_layout()
    out = fig_path('checkpoint_comparison_seasonal')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def plot_yearly_trend(df):
    fig, axes = var_panels()
    print('\n=== Year-over-year trend in RMSE (linear fit vs. calendar year) ===')
    for ax, v in zip(axes, VAR_NAMES):
        colname = f'{v}_rmse'
        for ckpt, g in df.groupby('checkpoint'):
            yearly = g.groupby('year')[colname].mean().dropna()
            if len(yearly) < 2:
                continue
            ax.plot(yearly.index, yearly.values, 'o-', color=CHECKPOINT_COLORS[ckpt],
                     label=CHECKPOINT_LABELS[ckpt])
            slope, intercept, r, p, se = sstats.linregress(yearly.index.values, yearly.values)
            fit_x = np.array([yearly.index.min(), yearly.index.max()])
            ax.plot(fit_x, slope * fit_x + intercept, '--', color=CHECKPOINT_COLORS[ckpt], alpha=0.6, lw=1)
            flag = ' <-- significant (p<0.05)' if p < 0.05 else ''
            print(f'  {v:8s} {ckpt:20s} slope={slope:+.4f}/yr  p={p:.3f}{flag}')
        ax.set_title(f'{VAR_LABELS[v]} RMSE by year', fontsize=10)
        ax.set_xlabel('year')
        ax.grid(alpha=0.25)
    axes[0].legend(loc='best', fontsize=8)
    fig.tight_layout()
    out = fig_path('checkpoint_comparison_yearly_trend')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def plot_ssr(df):
    """SSR aggregated by year and by season -- a per-date SSR would be too noisy
    to be meaningful, so each bar/point here summarizes many dates at once."""
    fig, axes = plt.subplots(len(VAR_NAMES), 2, figsize=(11, 3.2 * len(VAR_NAMES)))
    # Deterministic checkpoints have NaN spread (no SSR), so only ensembles get a bar slot
    checkpoints = [c for c in CHECKPOINT_COLORS
                   if df.loc[df.checkpoint == c, [f'{v}_spreadvar' for v in VAR_NAMES]].notna().any().any()]
    width = 0.8 / max(len(checkpoints), 1)
    season_positions = np.arange(len(SEASON_ORDER))

    for row, v in enumerate(VAR_NAMES):
        ax_year, ax_season = axes[row, 0], axes[row, 1]

        for ckpt, g in df.groupby('checkpoint'):
            yearly_ssr = g.groupby('year').apply(lambda gg: ssr(gg, v)).dropna()
            if len(yearly_ssr) == 0:
                continue
            ax_year.plot(yearly_ssr.index, yearly_ssr.values, 'o-',
                         color=CHECKPOINT_COLORS[ckpt], label=CHECKPOINT_LABELS[ckpt])
        ax_year.axhline(1.0, color='#888888', lw=1, ls=':')
        ax_year.set_title(f'{VAR_LABELS[v]} SSR by year', fontsize=10)
        ax_year.set_xlabel('year')
        ax_year.grid(alpha=0.25)

        for i, ckpt in enumerate(checkpoints):
            g = df[df.checkpoint == ckpt]
            season_ssr = [ssr(g[g.season == s], v) if len(g[g.season == s]) else np.nan
                          for s in SEASON_ORDER]
            offset = (i - (len(checkpoints) - 1) / 2) * width
            ax_season.bar(season_positions + offset, season_ssr, width=width * 0.9,
                          color=CHECKPOINT_COLORS[ckpt], alpha=0.7)
        ax_season.axhline(1.0, color='#888888', lw=1, ls=':')
        ax_season.set_xticks(season_positions)
        ax_season.set_xticklabels(SEASON_ORDER)
        ax_season.set_title(f'{VAR_LABELS[v]} SSR by season', fontsize=10)
        ax_season.grid(alpha=0.25, axis='y')

    axes[0, 0].legend(loc='best', fontsize=8)
    fig.tight_layout()
    out = fig_path('checkpoint_comparison_ssr')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def print_seasonal_outliers(df):
    print('\n=== Seasonal mean RMSE (flagging the worst season per checkpoint/variable) ===')
    for v in VAR_NAMES:
        colname = f'{v}_rmse'
        for ckpt, g in df.groupby('checkpoint'):
            by_season = g.groupby('season', observed=True)[colname].mean().dropna()
            if by_season.empty:
                continue
            worst = by_season.idxmax()
            print(f'  {v:8s} {ckpt:20s} ' +
                  ', '.join(f'{s}={by_season[s]:.4f}' for s in by_season.index) +
                  f'  -> worst: {worst}')


def print_ssr_summary(df):
    print('\n=== SSR (spread-skill ratio; 1.0 = ensemble spread matches error) ===')
    for v in VAR_NAMES:
        for ckpt, g in df.groupby('checkpoint'):
            print(f'  {v:8s} {ckpt:20s} overall SSR={ssr(g, v):.4f}')


def main():
    df = load()
    plot_timeseries(df)
    plot_seasonal(df)
    plot_yearly_trend(df)
    plot_ssr(df)
    print_seasonal_outliers(df)
    print_ssr_summary(df)


if __name__ == '__main__':
    main()
