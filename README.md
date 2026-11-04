# wavediffusion

A lightweight diffusion model for sea state estimation from wind forcing, built on
a pruned fork of [smalldiffusion][smalldiffusion-url] (see [Attribution](#attribution)).

## Repository layout

- `src/wavediffusion/` — the model/data/diffusion library (installable package, see below).
- `example/` — training scripts (`example/training/`), architecture experiments
  (`example/experiments/`), and `sample_wind_forced.ipynb`, a walkthrough of sampling
  from a trained checkpoint given wind forcing.
- `data/` — scripts for processing the raw [Ifremer hindcast][ifremer-hindcast] data
  used for training, plus buoy (NDBC) and Sofar spotter observations used for
  validation. Buoy partition reconstruction additionally needs
  [roguewavespectrum] and [wavespectra] (included in `environment.yml`).
- `analysis/` — notebooks and scripts that generate the paper figures and other
  post-hoc analysis of trained models.
- `script/` — data download and cluster (Slurm) job scripts.

## Installation

Create the conda environment from [`environment.yml`](environment.yml), then install
the package itself in editable mode:

```
conda env create -f environment.yml
conda activate wavediffusion
pip install -e .
```

`make install-local` does the `pip install` step with the `dev`/`test` extras included
(see `pyproject.toml`).

This env is self-contained and does **not** require any NERSC `pytorch` module — it
was verified from scratch with `mamba env create -f environment.yml` (no
`module load pytorch/...`) and passes `pytest` on its own.

### Notes for NERSC/Perlmutter users
- Use the system-provided `module load conda/Miniforge3-...` rather than installing
  your own Miniconda, and prefer `mamba` over `conda` for the solve — both are
  [NERSC's own recommendation](https://docs.nersc.gov/development/languages/python/nersc-python/).
- Keep `envs_dirs` (actual installed environments) on a persistent location such as
  `/global/common/software/<project>/<user>/conda/envs` — NERSC recommends this for
  environments used to run jobs, and `$PSCRATCH` is purged after 8 weeks of
  inactivity, which can silently corrupt an environment you haven't touched in a
  while. Keep `pkgs_dirs` (the disposable download cache) on `$PSCRATCH` instead —
  losing it to a purge just means a re-download, and it keeps cache growth off your
  (often small) project quota on `/global/common/software`. This repo's own
  `~/.condarc` is split exactly this way; see `CHANGELOG.md` for the cleanup that
  motivated it (a stale `~/.condarc` pointing both at the same full project
  directory caused a `Disk quota exceeded` failure mid-solve).
- Set `PYTHONNOUSERSITE=1` **and** `unset PYTHONPATH` before running Python in this
  env — both are needed, they block different leaks. `PYTHONNOUSERSITE` stops the
  implicit `~/.local/lib/pythonX.Y/site-packages` user-site mechanism; it does
  *not* stop an explicit `PYTHONPATH` entry, and this account's `~/.bashrc`
  unconditionally exports one pointing at `~/.local`'s site-packages for every
  shell. That's exactly what caused an unrelated `huggingface-hub`/`transformers`
  conflict in one of this project's Slurm jobs (see `script/slurm-57637075.out`),
  and later silently made `pip install`, inside `mamba env create`, skip
  installing `accelerate`/`einops` into this very env because they looked
  "already satisfied" via the leaked path — the job would have failed the moment
  it hit a machine where that stray `~/.local` copy wasn't there. `script/perlmutter-gpu.sh`
  does both; do the same in any new job script or interactive shell using this env.

## Package overview

The core abstractions — `data`, `model`, and `schedule` objects, and how they interact
during training/sampling — come from [smalldiffusion][smalldiffusion-url]. For a
detailed introduction to diffusion models and the notation used below, see the
[accompanying tutorial][blog-url].

### Model
Model objects are `torch.nn.Module` subclasses with:
  - An `input_dims` attribute: the shape of the model's input, excluding batch size.
  - A `rand_input(batchsize)` method returning i.i.d. standard normal noise of shape
    `[batchsize, *input_dims]`. Inherited from `ModelMixin` when `input_dims` is set.

Models are called as `model(x, sigma, cond=None)`:
 - `x` is a batch of shape `[B, *model.input_dims]`.
 - `sigma` is either a singleton (same value applied to every `x`) or shape
   `[B, 1, ..., 1]` (paired per-example).
 - `cond` is optional conditioning, `[B, ...]`.

`wavediffusion.model_unet.myUnet` is the main architecture used in this project: a
U-Net that takes a spatial forcing field (wind, depth, ice fraction, ...) as
preconditioning, with optional longitude-periodic (circular) padding for global grids
(`periodic=True`). `wavediffusion.plain_unet.plainUnet` is a non-diffusion regression
baseline with the same backbone, trained to directly map forcing to wave state.
`wavediffusion.model_unet.DetUnet` is the deterministic baseline for the periodic model:
`myUnet` called as `model(cond)`, with zeroed noisy-input channels and a fixed sigma, so
it has the same architecture and checkpoint layout (trained by
`example/experiments/wave_simple.py`).

`Scaled` and `PredX0` (in `wavediffusion.model`) are decorators that wrap a model class
to rescale its input, or to have it predict `x0` instead of noise `eps`.

### Schedule
A `Schedule` determines the rate at which noise level `sigma` increases during
diffusion, and has:
  - `sample_sigmas(steps)` to subsample the schedule for sampling.
  - `sample_batch(batchsize)` to draw `sigma` values uniformly at random for training.

Provided schedules (`wavediffusion.diffusion`): `ScheduleLogLinear` (simple, works
well on small models), `ScheduleDDPM` (standard pixel-space diffusion),
`ScheduleLDM` (latent diffusion, e.g. Stable Diffusion), and `ScheduleCosine`
([iDDPM][iddpm]).

### Training
`masked_training_loop` (`wavediffusion.diffusion`) trains a conditional model given a
`loader` that yields `(x, forcing, mask)` batches, with gradient accumulation via
`accelerate`. It yields a namespace with local variables each step, e.g.:

```python
for ns in masked_training_loop(loader, model, schedule, cond=True):
    print(ns.loss.item())
```

`wavediffusion.waveutils.evaluate` computes held-out loss the same way, and
`wavediffusion.waveutils.sample_and_save` draws samples from a checkpoint and saves
comparison plots against ground truth.

### Sampling
`samples` (and the threshold-constrained variant `samples_thres`) take a `model` and a
decreasing list of `sigmas` — usually `schedule.sample_sigmas(steps)` — and yield the
sequence of denoised iterates `xt`. The loop generalizes most commonly-used samplers:
 - DDPM [[Ho et al.]](https://arxiv.org/abs/2006.11239): `gam=1, mu=0.5`.
 - DDIM [[Song et al.]](https://arxiv.org/abs/2010.02502): `gam=1, mu=0`.
 - Accelerated sampling [[Permenter and Yuan]][arxiv-url]: `gam=2`.

### Data
`wavediffusion.wavedata.npyDataWndHist` (and its parent `npyDataResized`) load
preprocessed `.npy` forcing/target files, apply land masking and normalization, and
optionally append a short history of past wind snapshots as extra conditioning
channels. See the "Appendix" section of `example/sample_wind_forced.ipynb` for the
exact `.npy` layout expected.

`wavediffusion.waveana` has helpers (`add_lat_lon`, `assemble`) for turning raw model
fields back into georeferenced `xarray.Dataset`s for analysis.

## Evaluation

### Metrics (`wavediffusion.metrics`)
All scores are computed per date on the 320×720 output grid, in physical units (after
`invert_x`), and are shared by `analysis/paper_figure/compare_checkpoints_stats.py`
and `stats.ipynb`. Each spatial mean is **masked** and **latitude-weighted**:
- `mask` — boolean, True where the pixel is valid (ice-free ocean on that date: the
  ice mask resized to 320×720 and thresholded at 0.9). Masked-out pixels are dropped,
  not multiplied by zero, so NaNs on land don't propagate.
- `w` — area weights `wlat = cos(lat) / mean(cos(lat))`, broadcast over longitude,
  computed in `data/ifremer/wavedata.ipynb` and saved to
  `scratch_folder/wave_data/wlat.npy`.
- `circular=True` (default `False`) marks a direction in degrees (`thetam`, `dir`):
  differences are wrapped to [-180, 180) and ensemble means are circular means.

| Function | Score | Notes |
|---|---|---|
| `weighted_mse` | MSE of the ensemble mean | RMSE is reported as `sqrt(mean over dates)` |
| `weighted_mae` | mean absolute error | equals the CRPS of a deterministic forecast |
| `weighted_crps` | CRPS, `E\|X-y\| - ½E\|X-X'\|` | "fair" (unbiased) ensemble estimator by default; for 1 member (or a plain field) it returns `weighted_mae` |
| `weighted_acc` | anomaly correlation coefficient | uncentered; anomalies are forecast/truth minus the monthly climatology (below) |
| `ensemble_mean_var` + `weighted_mean` | ensemble spread (variance) | SSR = `sqrt(mean(spread var) / mean(MSE))`, aggregated over a group of dates rather than per date |

`ensemble_crps` and `ensemble_mean_var` return per-pixel maps; the `weighted_*`
functions reduce them to one number per date. SSR uses the ensemble variance without
the finite-ensemble correction, so a well-calibrated 10-member ensemble scores
≈ `sqrt(10/11)` ≈ 0.95 rather than 1.

`compare_checkpoints_stats.py` writes one CSV row per (checkpoint, date) with
`{v}_mse`, `{v}_spreadvar`, `{v}_acc` and `{v}_crps` for `v` in `hs`, `tm`, `thetam`.
Checkpoints with `deterministic=True` are `DetUnet` regressors scored with a single
forward pass: their `crps` column is the MAE and their `spreadvar` is NaN. The
evaluation dates and the history-window index bookkeeping come from
`wavediffusion.timeindex`.

### Climatology (for ACC)
Monthly climatologies are computed by `analysis/paper_figure/compute_clim.py`
(`sbatch compute_clim.sh <OPTION>` submits it as a CPU job, about 20 minutes for
OPTION 1). For each calendar month it pools that month from **2005–2020** and calls
`npyDataResized.compute_clim_map`, which:
- averages each pixel only over the timesteps where it's ice-free (it divides by the
  pixel's own valid count; pixels never valid in that month are NaN), and
- uses the circular mean for direction channels (`circular_x`).

Outputs go to `~/scratch_folder/final/processed/`, one file per month
(`*_clim_MM.nc`) plus all 12 stacked along `month` (`*_clim_monthly.nc`):

| OPTION | File | Variables | Status |
|---|---|---|---|
| 1 | `mean_clim_monthly.nc` | `hs`, `tm`, `dir` | regenerated with the fixed method on 2026-10-03; the old version is in `old_clim_arith/` |
| 2 | `derived_clim_monthly.nc` | `hs`, `uss`, `vss`, `mssd`, `mssc` | **old method** (arithmetic mean, all-steps denominator); run `compute_clim.sh 2` to update |
| 3 | `partition_clim_monthly.nc` | `hs_p1`, `tp_p1`, `thetap_p1`, `hs_p2`, `tp_p2`, `thetap_p2`, `ci` | **old method**; run `compute_clim.sh 3` to update |

`ci_clim_monthly.nc` (sea-ice concentration, plotted in `stats.ipynb`) is in the
same directory but isn't produced by `compute_clim.py`.

The saved `month` dimension has no coordinate values, so readers relabel it with
`.assign_coords(month=np.arange(1, 13))` and then look up `.sel(month=date.month)`.
The climatology names differ from the model's for OPTION 1: `tm` and `dir` correspond
to the model's `tm` and `thetam` (`CLIM_NAMES` in `compare_checkpoints_stats.py`).

## Attribution

This project started as a fork of [smalldiffusion][smalldiffusion-url]
([paper][arxiv-url], [tutorial][blog-url]), which supplies the general `data`/`model`/
`schedule` diffusion framework described above. Sea-state-specific code
(`wavedata.py`, `waveutils.py`, `waveana.py`, `model_unet.py`'s periodic U-Net and
`plain_unet.py`) and everything under `example/`, `data/`, `analysis/`, and `script/`
is specific to this project. Generic components from smalldiffusion that this project
never used (toy datasets, an unconditional DiT, an ideal denoiser, label-conditioned
MLPs) have been pruned from `src/wavediffusion/`.

[smalldiffusion-url]:https://github.com/yuanchenyang/smalldiffusion
[ifremer-hindcast]:https://data-dataref.ifremer.fr/ww3/GLOBMULTI_ERA5_GLOBCUR_01/GLOB-30M/
[blog-url]:https://www.chenyang.co/diffusion.html
[arxiv-url]:https://arxiv.org/abs/2306.04848
[iddpm]:https://arxiv.org/abs/2102.09672

[roguewavespectrum]: https://sofarocean.github.io/oceanwavespectrum/roguewavespectrum.html
[wavespectra]: https://wavespectra.readthedocs.io/en/latest/
