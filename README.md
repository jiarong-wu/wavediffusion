# wavediffusion

[![Tutorial blog post][blog-img]][blog-url]
[![Paper link][arxiv-img]][arxiv-url]
[![Open in Colab][colab-img]][colab-url]
[![Pypi project][pypi-img]][pypi-url]
[![Build Status][build-img]][build-url]

A lightweight diffusion model adopted from [blog-url] for sea state estimation. Below are the original README. Install locally in editable mode using `make install-local`.

`/data` provides scripts that process the raw data. In particular we use the [ifremer-hindcast] dataset.

`/script` provides a few script for downloading data, postprocessing, and cluster job scheduling.

A few additional packages needed for data analysis: [roguewavespectrum], [wavespectra].

## The original smalldiffusion repo
The core of smalldiffusion depends on the interaction between `data`, `model`
and `schedule` objects. Here we give a specification of these objects. For a
detailed introduction to diffusion models and the notation used in the code, see
the [accompanying tutorial][blog-url].

### Data
For training diffusion models, smalldiffusion supports pytorch [`Datasets` and
`DataLoaders`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
The training code expects the iterates from a `DataLoader` object to be batches
of data, without labels. To remove labels from existing datasets, extract the
data with the provided `MappedDataset` wrapper before constructing a
`DataLoader`.

Three 2D toy datasets, `Swissroll`,
[`DatasaurusDozen`](https://www.research.autodesk.com/publications/same-stats-different-graphs/),
and `TreeDataset`are provided.

### Model
All model objects should be a subclass of `torch.nn.Module`. Models should have:
  - A parameter `input_dims`, a tuple containing the dimensions of the input to
    the model (not including batch-size).
  - A method `rand_input(batchsize)` which takes in a batch-size and returns an
    i.i.d. standard normal random input with shape `[batchsize,
    *input_dims]`. This method can be inherited from the provided `ModelMixin`
    class when the `input_dims` parameter is set.

Models are called with arguments:
 - `x` is a batch of data of batch-size `B` and shape `[B, *model.input_dims]`.
 - `sigma` is either a singleton or a batch.
   1. If `sigma.shape == []`, the same value will be used for each `x`.
   2. Otherwise `sigma.shape == [B, 1, ..., 1]`, and `x[i]` will be paired with
      `sigma[i]`.
 - Optionally, `cond` of shape `[B, ...]`, if the model is conditional.

Models should return a predicted noise value with the same shape as `x`.

<!-- TODO: add note on xt and zt change of variables -->

### Schedule
A `Schedule` object determines the rate at which the noise level `sigma`
increases during the diffusion process. It is constructed by simply passing in a
tensor of increasing `sigma` values. `Schedule` objects have the methods
  - `sample_sigmas(steps)` which subsamples the schedule for sampling.
  - `sample_batch(batchsize)` which generates batch of `sigma` values selected
    uniformly at random, for use in training.

The following schedules are provided:
  1. `ScheduleLogLinear` is a simple schedule which works well on small
     datasets and toy models.
  2. `ScheduleDDPM` is commonly used in pixel-space image diffusion models.
  3. `ScheduleLDM` is commonly used in latent diffusion models,
     e.g. StableDiffusion.
  4. `ScheduleSigmoid` introduced in [GeoDiff][geodiff] for molecular conformal generation
  5. `ScheduleCosine` introduced in [iDDPM][iddpm]

The following plot shows these three schedules with default parameters.
<p align="center">
  <img src="https://raw.githubusercontent.com/yuanchenyang/smalldiffusion/main/imgs/schedule.png" width=40%>
</p>

### Training
The `training_loop` generator function provides a simple training loop for
training a diffusion model , given `loader`, `model` and `schedule` objects
described above. It yields a namespace with the local variables, for easy
evaluation during training. For example, to print out the loss every iteration:

```
for ns in training_loop(loader, model, schedule):
    print(ns.loss.item())
```

Multi-GPU training and sampling is also supported via
[`accelerate`](https://github.com/huggingface/accelerate).


### Sampling
To sample from a diffusion model, the `samples` generator function takes in a
`model` and a decreasing list of `sigmas` to use during sampling. This list is
usually created by calling the `sample_sigmas(steps)` method of a `Schedule`
object. The generator will yield a sequence of `xt`s produced during
sampling. The sampling loop generalizes most commonly-used samplers:
 - For DDPM [[Ho et. al. ]](https://arxiv.org/abs/2006.11239), use `gam=1, mu=0.5`.
 - For DDIM [[Song et. al. ]](https://arxiv.org/abs/2010.02502), use `gam=1, mu=0`.
 - For accelerated sampling [[Permenter and Yuan]][arxiv-url], use `gam=2`.

For more details on how these sampling algorithms can be simplified, generalized
and implemented in only 5 lines of code, see Appendix A of [[Permenter and
Yuan]][arxiv-url].



[ifremer-hindcast]:https://data-dataref.ifremer.fr/ww3/GLOBMULTI_ERA5_GLOBCUR_01/GLOB-30M/

[diffusion-py]:https://github.com/yuanchenyang/smalldiffusion/blob/main/src/smalldiffusion/diffusion.py
[unet-py]:https://github.com/yuanchenyang/smalldiffusion/blob/main/src/smalldiffusion/model_unet.py
[diffusers-wrapper]:https://github.com/yuanchenyang/smalldiffusion/blob/main/examples/diffusers_wrapper.py
[stablediffusion]:https://github.com/yuanchenyang/smalldiffusion/blob/main/examples/stablediffusion.py
[build-img]:https://github.com/yuanchenyang/smalldiffusion/workflows/CI/badge.svg
[build-url]:https://github.com/yuanchenyang/smalldiffusion/actions?query=workflow%3ACI
[pypi-img]:https://img.shields.io/badge/pypi-blue
[pypi-url]:https://pypi.org/project/smalldiffusion/
[dit-paper]:https://arxiv.org/abs/2212.09748
[model-code]:https://github.com/yuanchenyang/smalldiffusion/blob/main/src/smalldiffusion/model.py
[blog-img]:https://img.shields.io/badge/Tutorial-blogpost-blue
[blog-url]:https://www.chenyang.co/diffusion.html
[arxiv-img]:https://img.shields.io/badge/Paper-arxiv-blue
[arxiv-url]:https://arxiv.org/abs/2306.04848
[colab-url]:https://colab.research.google.com/drive/1So1lb9fG-AnDeSXNbosCnDbxbzf5xbor?usp=sharing
[colab-img]:https://colab.research.google.com/assets/colab-badge.svg
[geodiff]:https://arxiv.org/abs/2203.02923
[iddpm]:https://arxiv.org/abs/2102.09672
[cfg-paper]:https://arxiv.org/abs/2207.12598

[roguewavespectrum]: https://sofarocean.github.io/oceanwavespectrum/roguewavespectrum.html
[wavespectra]: https://wavespectra.readthedocs.io/en/latest/
