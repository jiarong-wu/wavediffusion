''' Paper tables: global skill of the OPTION1 and OPTION3 models
    over 2020-2025 every 5 days (00 UTC, 439 dates), from the per-date map files of
    `compare_checkpoints_period.sh` (the same inputs as skill_maps.py).

      python skill_table.py            (CPU, ~2 min)

    Each metric is an average over every ice-free pixel and date (the files'
    `valid`), weighted by cos(latitude), against the truth in the map files (on
    the model grid, through the 320x320 resize):
      bias   mean(model mean - truth); also % of the truth's mean for heights/periods
      RMSE   sqrt(mean((model mean - truth)^2))
      CRPS   fair ensemble CRPS (the deterministic model: the MAE)
      CRPSS  1 - CRPS / CRPS(deterministic)
      SSR    sqrt((N+1)/(N-1) * mean(var) / MSE), var the stored ddof=0 ensemble
             variance of N members: 1 for a statistically consistent N-member ensemble
    Directions use the circular difference (deg); ci gets an absolute bias only.
    Writes output/skill_table_OPTION<n>_<period>.npz per option and
    output/skill_table_OPTION1_OPTION3_<period>.tex, a standalone document
    holding both tables (booktabs).
'''
import os
import sys
import numpy as np
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'output')
PERIOD = '2020-2025_every5d'
OPTIONS = [1, 3]
MAPS_ROOT = f'/global/homes/j/jiarongw/scratch_folder/review/checkpoint_comparison_maps_{PERIOD}/'
DET = 'hist10d_det_ckpt20'
# table rows, top to bottom
MODELS = {3: {DET: 'Deterministic', 'highsigma_ckpt5': 'DDPM', 'resid_ckpt10': 'Res-DDPM'},
          1: {DET: 'Deterministic', 'highsigma_ckpt6': 'DDPM', 'resid_ckpt10': 'Res-DDPM'}}
VARS = {1: {'hs': (r'$H_s$', 'm'), 'tm': (r'$T_m$', 's'), 'thetam': (r'$\theta_m$', 'deg')},
        3: {'hs_p1': (r'$H_{sp1}$', 'm'), 'tp_p1': (r'$T_{p1}$', 's'), 'thetap_p1': (r'$\theta_{p1}$', 'deg'),
            'hs_p2': (r'$H_{sp2}$', 'm'), 'tp_p2': (r'$T_{p2}$', 's'), 'thetap_p2': (r'$\theta_{p2}$', 'deg'),
            'ci': ('CI', '--')}}
CIRCULAR = {'thetam', 'thetap_p1', 'thetap_p2'}
ABSOLUTE = CIRCULAR | {'ci'}    # no relative bias (ci: 0..1, mostly 0 or 1)


def circular_diff(a, b):
    return (a - b + 180) % 360 - 180


def compute(option):
    maps_dir, models, names = os.path.join(MAPS_ROOT, f'OPTION{option}'), MODELS[option], VARS[option]
    from wavediffusion.waveana import lat_deg
    coslat = np.cos(np.deg2rad(lat_deg))[:, None]
    dates = sorted(f[:-4] for f in os.listdir(os.path.join(maps_dir, 'truth')) if f.endswith('.npz'))
    s, n_members = {}, {}

    def add(key, a, w):
        s[key] = s.get(key, 0.) + float(np.sum(a * w, dtype=np.float64))

    for d in tqdm(dates, desc=f'OPTION{option}'):
        t = np.load(os.path.join(maps_dir, 'truth', f'{d}.npz'))
        tv = list(t['vars'])
        w = np.where(t['valid'], coslat, 0.)
        add('w', 1., w)
        for v in names:
            add(f'truth/{v}', t['truth'][tv.index(v)], w)
        for m in models:
            z = np.load(os.path.join(maps_dir, m, f'{d}.npz'))
            zv = list(z['vars'])
            n_members[m] = int(z['n_members'])
            for v in names:
                c, true = zv.index(v), t['truth'][tv.index(v)]
                err = circular_diff(z['mean'][c], true) if v in CIRCULAR else z['mean'][c] - true
                add(f'{m}/{v}/err', err, w)
                add(f'{m}/{v}/se', err ** 2, w)
                add(f'{m}/{v}/crps', z['crps'][c] if 'crps' in z.files else np.abs(err), w)
                if 'var' in z.files:
                    add(f'{m}/{v}/var', z['var'][c], w)
    out = {}
    for v in names:
        tmean = s[f'truth/{v}'] / s['w']
        for m in models:
            k = lambda q: s[f'{m}/{v}/{q}'] / s['w']
            mse, n = k('se'), n_members[m]
            out[f'{m}/{v}/bias'] = k('err')
            out[f'{m}/{v}/rbias'] = np.nan if v in ABSOLUTE else 100 * k('err') / tmean
            out[f'{m}/{v}/rmse'] = np.sqrt(mse)
            out[f'{m}/{v}/crps'] = k('crps')
            out[f'{m}/{v}/ssr'] = np.sqrt((n + 1) / (n - 1) * k('var') / mse) if n > 1 else np.nan
        for m in models:
            out[f'{m}/{v}/crpss'] = 1 - out[f'{m}/{v}/crps'] / out[f'{DET}/{v}/crps']
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, f'skill_table_OPTION{option}_{PERIOD}.npz'), n_dates=len(dates),
             **{f'{m}/n_members': n for m, n in n_members.items()}, **out)
    return out, len(dates), n_members


def fmt(x, nd):
    return '--' if not np.isfinite(x) else f'{x:.{nd}f}'.replace('-', '$-$')


def table(option, out, n_dates, n_members):
    ''' The LaTeX table environment for one option. '''
    models, names = MODELS[option], VARS[option]
    # decimals per unit: (bias/RMSE/CRPS)
    nd = {'m': 3, 's': 3, 'deg': 2, '--': 3}
    best = lambda v, q, f: f(models, key=lambda m: out[f'{m}/{v}/{q}'])
    rows = []
    for i, (v, (label, unit)) in enumerate(names.items()):
        b_bias = min(models, key=lambda m: abs(out[f'{m}/{v}/bias']))
        b_rmse = best(v, 'rmse', min)
        b_crps = best(v, 'crps', min)
        ens = [m for m in models if n_members[m] > 1]
        b_ssr = min(ens, key=lambda m: abs(out[f'{m}/{v}/ssr'] - 1))
        for j, (m, name) in enumerate(models.items()):
            bold = lambda txt, on: rf'\textbf{{{txt}}}' if on else txt
            rb = out[f'{m}/{v}/rbias']
            bias = fmt(out[f'{m}/{v}/bias'], nd[unit]) + ('' if not np.isfinite(rb) else f' ({fmt(rb, 1)}\\%)')
            cells = [
                rf'\multirow{{{len(models)}}}{{*}}{{{label}' + ('' if unit == '--' else f' ({unit})') + '}' if j == 0 else '',
                name,
                bold(bias, m == b_bias),
                bold(fmt(out[f'{m}/{v}/rmse'], nd[unit]), m == b_rmse),
                bold(fmt(out[f'{m}/{v}/crps'], nd[unit]), m == b_crps),
                '--' if m == DET else fmt(100 * out[f'{m}/{v}/crpss'], 1) + r'\%',
                bold(fmt(out[f'{m}/{v}/ssr'], 2), m == b_ssr) if m in ens else '--',
            ]
            rows.append(' & '.join(cells) + r' \\')
        if i < len(names) - 1:
            rows.append(r'\midrule')
    n_ens = sorted({n for n in n_members.values() if n > 1})
    caption = (
        rf'Global skill of the OPTION{option} models over 2020--2025 (every 5 days, 00~UTC, {n_dates} dates), '
        r'averaged over ice-free ocean pixels with $\cos(\text{latitude})$ weights. '
        r'Bias and RMSE are those of the ensemble mean (bias as model minus truth, in parentheses as a percentage '
        r'of the mean truth for heights and periods); CRPS is the fair ensemble CRPS (the MAE for the deterministic model); '
        r'CRPSS $= 1 - \mathrm{CRPS}/\mathrm{CRPS}_{\mathrm{det}}$; '
        rf'SSR $= \sqrt{{(N+1)/(N-1)\,\overline{{s^2}}/\mathrm{{MSE}}}}$ with $N={"/".join(map(str, n_ens))}$ '
        r'members and $s^2$ the ensemble variance (1 for a consistent ensemble). '
        r'Smallest absolute bias, best RMSE and CRPS, and SSR closest to 1 in bold.')
    return '\n'.join([
        r'\begin{table}[ht]',
        r'\centering',
        r'\small',
        rf'\caption{{{caption}}}',
        rf'\label{{tab:skill_option{option}}}',
        r'\begin{tabular}{llrrrrr}',
        r'\toprule',
        r'Variable & Model & Bias & RMSE & CRPS & CRPSS & SSR \\',
        r'\midrule',
        *rows,
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])


def write_doc(tables):
    doc = '\n'.join([
        r'\documentclass{article}',
        r'\usepackage{amsmath}',
        r'\usepackage{booktabs}',
        r'\usepackage{multirow}',
        r'\usepackage[margin=1in]{geometry}',
        r'\begin{document}',
        '',
        *[f'% --- OPTION{o} table: copy from here into the paper (needs booktabs, multirow, amsmath) ---\n'
          f'{t}\n% --- end OPTION{o} table ---\n' for o, t in tables.items()],
        r'\end{document}',
        '',
    ])
    path = os.path.join(OUT_DIR, f'skill_table_{"_".join(f"OPTION{o}" for o in tables)}_{PERIOD}.tex')
    with open(path, 'w') as f:
        f.write(doc)
    print(f'Wrote {path}')


if __name__ == '__main__':
    write_doc({o: table(o, *compute(o)) for o in OPTIONS})
