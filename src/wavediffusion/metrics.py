''' Latitude-weighted, masked evaluation metrics for ensemble wave forecasts.

    Shared by analysis/paper_figure/compare_checkpoints_stats.py and stats.ipynb.
    All `mask` arguments are boolean arrays (True = ocean / ice-free, i.e. valid);
    all `w` arguments are per-pixel weights broadcastable to the field (e.g. wlat).
    Masked-out pixels are dropped by boolean indexing rather than multiplied by
    zero, so NaNs on land don't propagate into the result.

    Variables flagged `circular=True` are in degrees (e.g. peak/mean direction);
    differences are wrapped to [-180, 180) and ensemble statistics use the
    circular mean, so members straddling the 0/360 wrap aren't penalized.
'''
import numpy as np


############### Helpers ###############
def circular_diff(a, b):
    """Signed difference of a degrees-valued circular quantity, wrapped to [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0


def _diff(a, b, circular=False):
    return circular_diff(a, b) if circular else (a - b)


############### Per-pixel statistics ###############
def ensemble_mean_var(x_ensem, circular=False):
    """Per-pixel ensemble mean and (population, ddof=0) variance across axis 0.
    For a circular variable, uses the circular mean (mean resultant vector) and
    the mean squared circular_diff from it."""
    if not circular:
        return x_ensem.mean(axis=0), x_ensem.var(axis=0)
    ang = np.deg2rad(x_ensem)
    mean_deg = np.rad2deg(np.arctan2(np.mean(np.sin(ang), axis=0), np.mean(np.cos(ang), axis=0))) % 360.0
    var = np.mean(circular_diff(x_ensem, mean_deg[None]) ** 2, axis=0)
    return mean_deg, var


def ensemble_crps(x_ensem, obs, circular=False, fair=True):
    """Per-pixel ensemble CRPS, in the kernel (energy) form
        CRPS = E|X - y| - 0.5 * E|X - X'|
    with X, X' independent ensemble members (axis 0 of x_ensem) and y = obs.

    fair=True uses the unbiased spread estimator (divides the pairwise sum by
    N(N-1) instead of N^2; Ferro 2014, as in WeatherBench 2), which removes the
    finite-ensemble penalty so CRPS is comparable across ensemble sizes.
    For a circular variable, |.| is the angular distance |circular_diff|, which
    keeps the score proper on the circle. Units are those of the variable.
    For a single member (N=1) there is no spread term and it reduces to the
    absolute error, regardless of `fair`."""
    n = x_ensem.shape[0]
    skill = np.mean(np.abs(_diff(x_ensem, obs[None], circular)), axis=0)
    if n < 2:
        return skill
    pair_sum = np.zeros_like(skill)
    for i in range(n - 1):  # loop over pairs i<j to avoid an (N, N, H, W) intermediate
        pair_sum += np.sum(np.abs(_diff(x_ensem[i + 1:], x_ensem[i][None], circular)), axis=0)
    # pair_sum counts each unordered pair once; E|X-X'| over ordered pairs is 2*pair_sum / norm
    norm = n * (n - 1) if fair else n * n
    return skill - pair_sum / norm


############### Masked, weighted spatial means ###############
def weighted_mean(a, mask, w):
    mask = np.asarray(mask, dtype=bool)
    w = np.broadcast_to(w, a.shape)
    return np.sum(a[mask] * w[mask]) / np.sum(w[mask])


def weighted_mse(a, b, mask, w, circular=False):
    return weighted_mean(_diff(a, b, circular) ** 2, mask, w)


def weighted_mae(a, b, mask, w, circular=False):
    """Masked, weighted mean absolute error. This is the CRPS of a deterministic
    (single-member) forecast, so it's what weighted_crps reduces to for N=1."""
    return weighted_mean(np.abs(_diff(a, b, circular)), mask, w)


def weighted_acc(fcst_anom, obs_anom, mask, w):
    """Uncentered anomaly correlation coefficient (anomalies w.r.t. climatology
    are passed in; their spatial mean is not removed)."""
    valid = np.asarray(mask, dtype=bool) & np.isfinite(fcst_anom) & np.isfinite(obs_anom)
    if not np.any(valid):
        return np.nan
    wv, f, o = np.broadcast_to(w, fcst_anom.shape)[valid], fcst_anom[valid], obs_anom[valid]
    cov = np.sum(f * o * wv) / np.sum(wv)
    var_f = np.sum(f ** 2 * wv) / np.sum(wv)
    var_o = np.sum(o ** 2 * wv) / np.sum(wv)
    denom = np.sqrt(var_f * var_o)
    return cov / denom if denom > 0 else np.nan


def weighted_crps(x_ensem, obs, mask, w, circular=False, fair=True):
    """Masked, weighted spatial mean of the per-pixel ensemble CRPS. x_ensem has
    the member axis first, (N, H, W). A deterministic forecast -- either a single
    member (1, H, W) or a plain field with the same shape as obs -- scores as
    weighted_mae, which is its CRPS."""
    if x_ensem.shape == obs.shape:
        return weighted_mae(x_ensem, obs, mask, w, circular=circular)
    if x_ensem.shape[0] == 1:
        return weighted_mae(x_ensem[0], obs, mask, w, circular=circular)
    return weighted_mean(ensemble_crps(x_ensem, obs, circular=circular, fair=fair), mask, w)
