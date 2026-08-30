# Changelog

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
