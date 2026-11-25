"""Marginal spike likelihood with a gamma gain shared by a neural population."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GainModel:
    view: str
    windows: tuple[tuple[int, int], ...]
    alpha: torch.Tensor


def _repeat_products(counts):
    repeats = counts.shape[1]
    summed = counts.sum(dim=1)
    return (summed.square() - counts.square().sum(dim=1)) / (repeats * (repeats - 1))


def _gain_moments(units):
    total = units.sum(dim=-1)
    # Off-diagonal covariance excludes independent overdispersion of any unit.
    covariance = total.var(dim=1, unbiased=True) - units.var(dim=1, unbiased=True).sum(dim=-1)
    exposure = (_repeat_products(total) - _repeat_products(units).sum(dim=-1)).clamp_min(0)
    return torch.stack((covariance.sum(), exposure.sum(), covariance.square().sum(),
                        (covariance * exposure).sum(), exposure.square().sum()))


@torch.no_grad()
def fit_gain_model(training, view, *, temporal=False, prior_groups=64.0):
    """Estimate Gamma gain variance from same-stimulus training repeat covariance.

    Each trial's gain has mean one, variance alpha, and shape/rate 1/alpha.
    Consecutive trial gains are modeled independently. The default uses one
    gain over the evoked 50–250 ms window. The timed alternative separates a
    50–100 ms onset response from the 100–250 ms sustained response.
    """
    if not training or view not in ("rate", "timed") or (temporal and view != "timed") or prior_groups < 0:
        raise ValueError("Expected repeated training data and a valid population gain view")
    windows = ((2, 4), (4, 10)) if temporal else ((2, 10),)
    moments = []
    for counts in training.values():
        if counts.ndim != 3 or counts.shape[1] < 2 or counts.shape[-1] % 10:
            raise ValueError("Expected [glyphs, repeats >= 2, units × 10] training counts")
        bins = counts.float().unflatten(-1, (-1, 10))
        moments.append(torch.stack([_gain_moments(bins[..., start:end].sum(dim=-1)) for start, end in windows]))

    covariance, exposure, squared, cross, exposure_squared = torch.stack(moments).sum(dim=0).T
    denominator = exposure.clamp_min(1e-8)
    point = (covariance / denominator).clamp_min(0)
    residual = (squared - 2 * point * cross + point.square() * exposure_squared).clamp_min(0)
    error = residual.sqrt() / denominator
    effective_groups = exposure.square() / exposure_squared.clamp_min(1e-8)
    shrinkage = effective_groups / (effective_groups + prior_groups).clamp_min(1e-8)
    return GainModel(view, windows, (point - error).clamp(0, 1) * shrinkage)


@torch.no_grad()
def population_gain_logits(observed, expected, alpha):
    """Score one population window using an analytically integrated Gamma gain.

    For Lambda = sum(mu), N = sum(y), and k = 1/alpha, drop candidate-independent
    terms from log p(y|mu) to obtain y·log(mu) − (N+k)·log1p(Lambda/k).
    The sum is averaged across independently presented repeats.
    """
    expected = expected.clamp_min(1e-5)
    mean = observed.mean(dim=1)
    alpha = alpha.to(expected).clamp_min(0)
    total = expected.sum(dim=1)
    penalty = torch.log1p(alpha * total)
    safe = torch.where(alpha > 0, alpha, torch.ones_like(alpha))
    offset = torch.where(alpha > 0, penalty / safe, total)
    return mean @ expected.log().T - mean.sum(dim=1, keepdim=True) * penalty - offset


@torch.no_grad()
def gain_template_logits(observed, rates, model):
    """Apply training-calibrated population gains to a known glyph gallery."""
    if (observed.ndim != 3 or rates.ndim != 2 or observed.shape[-1] != rates.shape[-1]
            or observed.shape[-1] % 10):
        raise ValueError("Expected repeated counts and gallery means with aligned units × 10 bins")
    observed = observed.unflatten(-1, (-1, 10))
    rates = rates.unflatten(-1, (-1, 10))
    logits = 0
    for (start, end), alpha in zip(model.windows, model.alpha):
        counts = observed[..., start:end]
        expected = rates[..., start:end]
        if model.view == "rate":
            counts, expected = counts.sum(dim=-1), expected.sum(dim=-1)
        else:
            counts, expected = counts.flatten(-2), expected.flatten(-2)
        expected = 0.98 * expected + 0.02 * expected.mean(dim=0)
        logits = logits + population_gain_logits(counts, expected, alpha)
    return logits
