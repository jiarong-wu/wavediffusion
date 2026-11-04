# Changelog

## 2026-10-07 — Use every step of each monthly file; regional variability analysis

### Data loading: last step of each month no longer dropped
- `MultiFileNpyData.__init__` used `file_len = len(x_mmap) - 1`, so each month's
  last step (21Z on the last day) was never used. This has been there since the
  initial commit. No code reads a following step, and the dropped steps are valid
  data. `file_len` is now `len(x_mmap)`, and `timeindex.file_lengths` matches.
- Effects of the old `- 1`:
  - **Wind history at month boundaries.** History is read by index, so a lag that
    reached into the previous month came out 3 h too long. For example, the
    1-day-ago wind for Feb 1 00Z came from Jan 30 21Z. This affected about 30% of
    training samples for a 10-day window. Lags are now exact.
  - **Sampling stride.** Stepping 8 indices through concatenated months drifted
    3 h per month (Jan 00Z, Feb 03Z, Mar 06Z, …). Daily stepping now stays at 00Z.
  - **Training data.** It skipped 1 step in ~248 (0.4%).
- **Compatibility:**
  - Existing checkpoints load and run unchanged. They were trained with
    slightly off lags on part of the data, so retraining isn't required.
  - For multi-month datasets, the date a given index points to has changed (by
    3 h per month since the first file). Outputs saved by index from earlier
    multi-month `sample.py` runs no longer line up with a newly built dataset.
    Single-month runs are unaffected.
  - Existing rows in `checkpoint_comparison_stats.csv` were scored at the old,
    drifting hours. The resume logic keys on the day only, so new rows would be
    mixed with old ones. Move the CSV aside before re-running for consistent
    results.

### `npyDataWndHist` docstring corrected
- The history channels hold winds from `hist_duration_days - hist_freq_days`
  days ago up to the current step, oldest first. That is 4,3,2,1,0 days ago for
  (5, 1) and 8,6,4,2,0 days ago for (10, 2), not 1..5 / 2..10 as documented.
  The last pair repeats the current wind already in channels 0–1. The code is
  unchanged, since checkpoints depend on this layout. The `timeindex.hist_window`
  docstring was corrected to match.

### `analysis/paper_figure/sample.py`
- `HIST_DURATION_DAYS` / `HIST_FREQ_DAYS` set the wind-history window (default
  5 / 1, as before). It used to be the `npyDataWndHist` default, with
  `precond_ch=14` hard-coded. `precond_ch` is now derived from the window, so a
  hist10d checkpoint only needs `HIST_DURATION_DAYS = 10`.
- With wind history (`HIST = True`), it also loads the month before the first
  test month: the previous December for a year run, the previous month for a
  single-month run. This gives the first test days a full history window.
  Previously Jan 1–5 (5-day window) or Jan 1–10 (10-day) of each year run were
  never sampled. Nothing in the extra month is sampled. If its files are missing
  (e.g. before 1993), it is left out with a note, and those first days are
  skipped as before.
- `SAMPLE_DAYS` / `SAMPLE_HOUR` pick the days of every test month to sample and
  the hour, e.g. `[1, 8, 15, 22]` at 00Z. Days past a month's end are skipped.
  With the default `SAMPLE_DAYS = None` it steps `every_n` from the first test
  day, which stays on the same hour every month.
- Writes `dates.csv` (dataset index → date) into `save_path`. Because of the
  extra month, the dataset index of a given date differs from earlier runs, and
  with `SAMPLE_DAYS` set the saved indices aren't evenly spaced. Notebooks that
  loop `range(0, N, every_n)` over saved files should read the indices from
  `dates.csv`.

### `src/wavediffusion/timeindex.py`
- `months_file_list` builds the file list, including the extra history month.
- `select_sample_indices` holds the date-selection rule shared by `sample.py` and
  `regional_variability.py`. `dates_to_raw_indices` and `write_dates_csv` are
  its helpers.

### New: `analysis/paper_figure/regional_variability.py` (+ `.sh`)
- Seasonal cycle and interannual trend (Jan/Apr/Jul/Oct) of regional-mean `hs`,
  `tm`, `thetam` and `u10` over the test years 2019–2025. It uses
  latitude-weighted means over ice-free ocean pixels.
- `thetam` is averaged circularly: the CSV keeps per-date `thetam_sin` and
  `thetam_cos`, and monthly means are the atan2 of their averages. For plotting
  and trends, each region's directions are unwrapped around its overall truth
  mean, so lines don't jump at 0°/360°.
- Three stages: `dates` → `compute` → `plot`.
  - `dates` writes the dates `sample.py` would sample (one run per year, daily at
    00Z) without running any sampling.
  - `compute` evaluates truth only on the dates in `DATES_CSV`. Once samples
    exist, point it at the sample runs' own `dates.csv` files so truth and
    sampled cover the same dates.
- Pacific regions follow Young (1999, Int. J. Climatol. 19, 931–950). The box
  bounds are provisional and still need checking against the paper's Fig. 2.
- `sample` stage (GPU, `regional_variability_sample.sh`) runs the models in
  `MODELS` on the same dates and writes the same regional means, one row per
  ensemble member. It appends date by date, so a re-run resumes after a timeout.
  - `hist10d_ckpt16`: diffusion, 10 members × 10 steps (~5 h for 2019–2025).
  - `hist10d_det_ckpt16`: deterministic.
  - `truth_resized`: truth sent through the 320×320 resize and back, to
    separate the resize smoothing from model error.
- Model outputs aren't multiplied by the interpolated ice mask (unlike
  `compare_checkpoints_stats.py`). Model and truth are averaged over the same
  binary native-grid ice-free pixels.
- Plots: members are averaged first (the regional mean of the ensemble mean).
  The interannual figure is a region × month grid with all sources overlaid.
  `u10` is a model input, so it's plotted for truth only.

## 2026-10-05 — Shared metrics/time-index modules, CRPS, deterministic baseline, fixed climatology

### New modules in `src/wavediffusion/`
- `metrics.py` — the masked, latitude-weighted metrics, factored out of
  `compare_checkpoints_stats.py`: `weighted_mse`, `weighted_mae`, `weighted_acc`,
  `weighted_crps` (new; fair ensemble estimator, falls back to MAE for a single
  member), plus per-pixel `ensemble_mean_var`/`ensemble_crps` and `circular_diff`.
  Unlike the copies in `stats.ipynb`, these wrap direction differences and use
  circular ensemble means (`circular=True`). See README "Evaluation".
- `timeindex.py` — file discovery, the date index over concatenated monthly files,
  history-window padding (`hist_window`), selection of evaluation dates that skips
  calendar gaps (`select_eval_indices`), and raw-to-dataset index conversion.

### Models and training
- `model_unet.DetUnet` — deterministic regressor with the same backbone as `myUnet`.
  `example/experiments/wave_simple.py` was rewritten to train it on the
  `meandiff_periodic_hist10d.py` setup.
- `plain_unet.masked_training_loop_plain` now uses `accelerator.accumulate()`.
  Previously `gradient_accumulation_steps` had no effect there.
- `model_unet.AttnBlock`: `.contiguous()` after the output `rearrange`, which
  silences DDP's "Grad strides do not match bucket view strides" warning. Outputs and
  checkpoints are unchanged.

### `analysis/paper_figure/compare_checkpoints_stats.py`
- Adds a `{v}_crps` column. Old CSVs without it are rejected instead of being
  appended to with misaligned columns.
- Checkpoints marked `deterministic=True` (`DetUnet`) are scored with one forward
  pass. Their `crps` column is the MAE and their `spreadvar` is NaN.
- `hist_freq_days` per checkpoint. The number of history channels was previously
  `4 + 2*hist_days`, which was wrong for any frequency other than 1 day.

### Renames (OPTION 1 only)
`tp` → `tm` and `thetap` → `thetam` in `compare_checkpoints_stats.py`,
`plot_checkpoint_comparison.py`, `sample.py`, and `data/NDBC/gather_diffusion_at_buoy.py`
(`tp` only). In `waveana.assemble`, `tp` became `tm` and `tp_hat` became `tm_hat`,
with the matching updates in `phys.ipynb`. OPTION 3's `tp_p*`/`thetap_p*` are real
peak quantities and are unchanged. CSV columns changed accordingly.

### Climatology (`npyDataResized.compute_clim_map`, `compute_clim.py`)
- Each pixel is now averaged over its own ice-free steps. Previously it divided by
  the total number of steps, which biased the climatology low at partially iced
  pixels (about 0.5× on a March 2010 test).
- Direction channels (`circular_x`) use a circular mean. The old arithmetic mean
  was more than 30° off on about 24–29% of valid pixels.
- `compute_clim.py` takes `OPTION` as an argument; `compute_clim.sh` submits it as a job.
- `mean_clim_monthly.nc` was regenerated (the old files are in
  `processed/old_clim_arith/`). `derived_clim_monthly.nc` and
  `partition_clim_monthly.nc` have not been regenerated yet.
- Known data issue, judged negligible: 9 whole days in 2005–2020 have an anomalous
  ice mask in `forcing_*.npy`, with far more ice-free ocean than that month's median.
  The worst is 2013-10-23 (0.958 vs 0.853). The others are 2007-03-15/16,
  2009-03-18/19, 2006-04-16, 2006-07-06, 2009-08-05, 2011-08-27 and 2015-08-06.
  These days only affect the climatology near the ice edge.

### Results that need re-running
All ACC values computed before this change, in `checkpoint_comparison_stats.csv` and
`stats.ipynb`, used the old climatology. The CSV also needs regenerating for the
CRPS column and the renamed variables.

## 2026-08-28 — Prune package to code actually used by example/data/analysis

Removed code inherited from [smalldiffusion](https://github.com/yuanchenyang/smalldiffusion)
that this project never used, and refreshed the docs/dependency files to match.
`src/wavediffusion/` went from 2,948 lines across 10 files to 1,702 lines across 8
files. Verified by re-running `pytest` and re-doing `pip install -e .` after the
changes; both succeeded.

### `src/wavediffusion/` — deleted files (fully unused)
- `data.py` — smalldiffusion's toy datasets (`Swissroll`, `DatasaurusDozen`,
  `TreeDataset`, `MappedDataset`, image transform helpers).
- `model_dit.py` — unused DiT transformer model.
- Stale `.ipynb_checkpoints/` and `__pycache__/` inside the package directory.

### `src/wavediffusion/` — trimmed within kept files
- `model.py`: removed `TimeInputMLP`, `ConditionalMLP`, `IdealDenoiser`, `sq_norm`,
  `CondEmbedderLabel`, `PredV` (all unused). Kept `ModelMixin`, `get_sigma_embeds`,
  `SigmaEmbedderSinCos`, `alpha`, `Scaled`, `PredX0`, `CondSequential`, `Attention`.
- `model_unet.py`: removed the base `Unet` class — never used or subclassed;
  `myUnet` (the preconditioned variant actually used everywhere) is untouched.
- `diffusion.py`: removed `ScheduleSigmoid`, the unconditional `training_loop`,
  `masked_training_loop_lp`, `samples_onestep`, and a ~35-line commented-out dead
  draft of `masked_training_loop`.
- `waveutils.py`: removed `plot_wave` (unused) and ~150 lines of commented-out
  legacy code (`evaluate_lp`, `sample_and_save_lp`, an old 4-field `plot_sample`).
- `wavedata.py`: removed a stale `if __name__ == "__main__":` demo block that
  called `MultiFileNpyData` with a `maskname=` kwarg that no longer exists
  (`landmaskname` is the real parameter) — dead and already broken.
- `plain_unet.py`, `waveana.py`: untouched, fully used as-is.
- `__init__.py`: rewritten to only re-export symbols that still exist
  (`Schedule`, `ScheduleLogLinear`, `ScheduleDDPM`, `ScheduleLDM`, `ScheduleCosine`,
  `samples`, `ModelMixin`, `Scaled`, `PredX0`, `get_sigma_embeds`,
  `SigmaEmbedderSinCos`, `myUnet`).

### `README.md`
Rewritten around the pruned codebase: repo layout (`src`, `example`, `data`,
`analysis`, `script`), install instructions, and a "Package overview" section
documenting the Model/Schedule/Training/Sampling/Data APIs as they exist now
(dropped documentation of removed toy datasets and schedules). Kept an
Attribution section crediting smalldiffusion for the core diffusion abstractions.

### Dependency files
- Added `environment.yml` — conda env covering the package plus `example/`,
  `data/`, `analysis/` (torch stack via pip for correct CUDA wheel selection,
  scientific stack via conda-forge, `roguewavespectrum`/`wavespectra` for NDBC
  buoy processing).
- `pyproject.toml`: added `matplotlib` and `xarray` to `[project] dependencies`
  (the package imports both at module load — `waveutils.py`, `waveana.py` — but
  they were missing from the declared deps). Removed `diffusers`, `transformers`,
  `datasets` from the `test`/`examples` extras — leftover smalldiffusion example
  dependencies not used anywhere in this repo.

### Known follow-up (not done)
`pyproject.toml`'s `[project.urls]` still point at the `smalldiffusion` GitHub
repo. Left as-is since no wavediffusion-specific repo URL was given.

## 2026-08-28 — Verified environment.yml from scratch, independent of pytorch/2.6.0

Built a fresh conda env directly from `environment.yml` with no `pytorch` module
loaded, to confirm the dependency file is actually sufficient on its own (motivated
by the `huggingface-hub`/`transformers` conflict found earlier in
`script/slurm-57637075.out`, which came from the NERSC `pytorch/2.6.0` module
interacting badly with packages in `~/.local`).

- Used the NERSC-provided `module load conda/Miniforge3-25.11.0-1` (ships `mamba`)
  rather than a self-installed Miniconda, per
  [NERSC's Python docs](https://docs.nersc.gov/development/languages/python/nersc-python/).
- Built the env on `$PSCRATCH` (`/pscratch/sd/j/jiarongw/conda_envs/wavediffusion-test`),
  not `$HOME` — NERSC docs flag `$HOME` as unsuitable for large conda envs due to
  small quotas / many-small-files overhead.
- First attempt failed with `Disk quota exceeded`: `~/.condarc` (pre-existing,
  unrelated to this project) points `pkgs_dirs`/`envs_dirs` at a `/global/common/software`
  project allocation that was already full. Worked around it for this run via
  `CONDA_PKGS_DIRS=$PSCRATCH/conda_pkgs_cache` rather than editing the global
  `.condarc`. Documented this as a gotcha in the README.
- `mamba env create -f environment.yml -p $PSCRATCH/conda_envs/wavediffusion-test`
  then succeeded (~8.3GB total: 6.7GB env + 1.6GB package cache); resolved
  `torch==2.13.0+cu130`, `python==3.12.14`.
- With that env activated (no `module load pytorch`) and `PYTHONNOUSERSITE=1` set
  (to rule out `~/.local` leaking in, same root cause as the earlier
  `huggingface-hub` bug), `pip install -e .` and `pytest test/` both passed cleanly
  — confirms the package and its test suite have no hidden dependency on the NERSC
  `pytorch` module.

## 2026-08-28 — Cleaned up `~/.condarc` and moved wavediffusion env to project storage

The `Disk quota exceeded` above turned out to be worth fixing properly rather than
routing around: `~/.condarc` had `envs_dirs` and `pkgs_dirs` both pointing at the
same flat `/global/common/software/m4874/jiarongw/conda` directory, which was 26GB
and full of unrelated leftovers:
- `waveml` — an old, unrelated conda environment (15GB).
- A flat `pip install --target=.../conda torch ...` tree dumped directly into that
  directory (not a real conda env) — `torch`, `torchvision`, the full `nvidia-*`
  CUDA wheel closure, `numpy`, `sympy`, `huggingface_hub`, `accelerate`, etc.,
  plus a stale editable-install pointer to the pre-fork upstream
  `smalldiffusion==0.4.4` (~7GB).
- ~12GB of stale conda package cache (old `pytorch-2.3.0` builds, `cudatoolkit-11.8`,
  `cudnn`, `mkl`, `torchtriton`, ...).

Removed all of it (`mamba clean --all` for the cache, `conda env remove` for
`waveml`, `rm -rf` on the directory for the rest — confirmed nothing referenced
that path from shell startup files or this repo's job scripts first), then:
- Split `~/.condarc`: `envs_dirs` → `/global/common/software/m4874/jiarongw/conda/envs`
  (persistent, survives scratch purges), `pkgs_dirs` → `$PSCRATCH/conda_pkgs_cache`
  (disposable, keeps future cache growth off the small project quota).
- Recreated the env at `/global/common/software/m4874/jiarongw/conda/envs/wavediffusion`
  from `environment.yml` (now reachable as `mamba activate wavediffusion` from
  anywhere) and re-verified `pip install -e .` + `pytest test/` there.
- Removed the now-superseded `$PSCRATCH/conda_envs/wavediffusion-test` copy from
  the earlier verification run.

## 2026-08-28 — Registered a Jupyter kernel; added ipywidgets

Registered the persistent env as a Jupyter kernel:
`python -m ipykernel install --user --name wavediffusion --display-name wavediffusion`,
with `PYTHONNOUSERSITE=1` added to the resulting `kernel.json`'s `env` block (same
`~/.local` leakage concern as above). Also removed a leftover `waveml` kernelspec
that pointed at the now-deleted env and would have failed to launch.

Fixed a `TqdmWarning: IProgress not found` seen in a notebook using this kernel —
`ipywidgets` wasn't installed (only `jupyterlab`/`ipykernel` were). Installed it
into the live env (`mamba install ipywidgets`) and added `ipywidgets` to
`environment.yml` so a fresh env build doesn't need this fixed up again.

## 2026-08-28 — Found and fixed the actual root cause: `~/.bashrc` leaks `PYTHONPATH`

While updating `script/perlmutter-gpu.sh` to use the new env, `accelerate launch`
would have failed: `which accelerate` resolved to `~/.local/bin/accelerate`, whose
shebang pointed at the now-deleted `waveml` env's python.

Digging in, this account's `~/.bashrc` has:
```
export PYTHONPATH=/global/homes/j/jiarongw/.local/lib/python3.11/site-packages:$PYTHONPATH
```
This is sourced for every shell, unconditionally, regardless of which conda env or
module is active. **This is the real root cause of the original `huggingface-hub`
conflict from the very first entry in this changelog's history** (the `pytorch/2.6.0`
job's python3.12 interpreter was loading modules out of a `.../python3.11/...`
directory — that only makes sense via an explicit `PYTHONPATH` entry, not the
normal per-version user-site mechanism).

It also silently broke the "verified independent of pytorch/2.6.0" claims made in
the two entries above. `PYTHONNOUSERSITE=1` blocks the *implicit* user-site
addition to `sys.path`, but an explicit `PYTHONPATH` entry bypasses that entirely.
Because this leak was present in every verification shell (it's unconditional in
`~/.bashrc`), `mamba env create -f environment.yml`'s pip step saw `accelerate` and
`einops` as "already satisfied" via the leaked `~/.local/lib/python3.11/site-packages`
copies and skipped installing them into the env — so the persistent env at
`.../conda/envs/wavediffusion` never actually had its own `accelerate`/`einops`,
and every `pytest`/import check that "passed" in the two entries above was quietly
running against the `~/.local` copies instead, not the env's own packages.

Fix:
- `unset PYTHONPATH` (in addition to `PYTHONNOUSERSITE=1`) before reinstalling:
  `python3 -m pip install --no-cache-dir accelerate einops` — both are now
  actually present in the env's own `site-packages`, with `accelerate`'s console
  scripts correctly in the env's own `bin/`. (Checked the other pip-installed
  packages too — `torch`, `torchvision`, `torch_ema`, `roguewavespectrum`,
  `wavespectra` were unaffected, since none of them happened to already exist
  under `~/.local`.)
- Re-ran `pytest test/` with `PYTHONPATH` genuinely unset (not just
  `PYTHONNOUSERSITE=1`) — passes for real this time.
- `script/perlmutter-gpu.sh` now does both `export PYTHONNOUSERSITE=1` and
  `unset PYTHONPATH` before `accelerate launch`.
- Documented both env vars (and why both are needed) in `README.md`.
- **Not done**: `~/.bashrc` itself still has the unconditional `PYTHONPATH` export.
  It affects every conda/venv environment this account creates, not just this
  project — worth fixing at the source, but that's a global dotfile change, so
  left for the user to decide.
