"""Predict single-trial counts while integrating uncertainty in glyph rates."""

from dataclasses import dataclass

import torch

from glyph_encoding import _evoked_counts


PRIOR_EXPOSURES = (0.5, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0)


@dataclass(frozen=True)
class GammaCountTemplates:
    shape: torch.Tensor
    rate: float

    @property
    def mean(self):
        return self.shape / self.rate


@torch.no_grad()
def build_gamma_prior(rates, training, session):
    """Combine forward rates with class-matched empirical responses on other days."""
    forward = _evoked_counts(rates[session].float(), "timed")
    center = forward.mean(dim=0)
    pooled = torch.zeros_like(forward)
    repeats = 0
    for name, counts in training.items():
        if name == session:
            continue
        source_mean = _evoked_counts(rates[name].float(), "timed").mean(dim=0)
        adjustment = ((center + 0.08) / (source_mean + 0.08)).clamp(0.5, 2.0)
        pooled += _evoked_counts(counts.sum(dim=1), "timed") * adjustment
        repeats += counts.shape[1]
    if not repeats:
        raise ValueError("A class-rate prior needs at least one aligned source session.")
    prior = 0.50 * forward + 0.50 * (pooled / repeats)
    return (0.98 * prior + 0.02 * center).clamp_min(1e-5)


@torch.no_grad()
def gamma_templates(prior_mean, local, exposure):
    """Update a class-specific Gamma prior with the local labeled spike counts."""
    if exposure <= 0 or local.ndim != 3 or not local.shape[1]:
        raise ValueError("Provide a positive Gamma prior exposure and labeled local repeats.")
    total = _evoked_counts(local.sum(dim=1).float(), "timed")
    if prior_mean.shape != total.shape:
        raise ValueError("The prior and labeled counts must share the gallery feature order.")
    return GammaCountTemplates(total + exposure * prior_mean, exposure + local.shape[1])


@torch.no_grad()
def paired_count_log_probability(counts, templates):
    """Score held trials against their corresponding true, independently fitted glyph."""
    y = _evoked_counts(counts.double(), "timed")
    shape = templates.shape.double()
    rate = shape.new_tensor(templates.rate)
    factorial = torch.lgamma(1 + y)
    log_gamma = (torch.lgamma(shape + y) - torch.lgamma(shape) - factorial
                 - shape * (1 / rate).log1p() - y * rate.log1p())
    mean = shape / rate
    log_poisson = y * mean.log() - mean - factorial
    return log_gamma.sum(dim=-1), log_poisson.sum(dim=-1)


@torch.no_grad()
def gamma_template_logits(counts, templates):
    """Score a complete gallery by exact matched-mean and Gamma predictive likelihood.

    For integer count y, log Γ(a+y) - log Γ(a) is y log(a) plus
    log(1+k/a) for each positive integer k strictly below y. Masked matrix
    products therefore compute candidate-specific shape without a Q×C×F tensor.
    Dropped terms are constant across candidates for a fixed query and rate.
    """
    observed = _evoked_counts(counts.float(), "timed")
    if observed.ndim != 2 or observed.shape[1] != templates.shape.shape[1]:
        raise ValueError("Expected one aligned spike count vector per independent trial.")
    mean, concentration = templates.mean, templates.shape
    dot = observed @ mean.log().T
    poisson = dot - mean.sum(dim=1)
    predictive = dot - concentration.sum(dim=1) * dot.new_tensor(1 / templates.rate).log1p()
    for count in range(1, int(observed.max())):
        indicator = (observed > count).to(dot.dtype)
        predictive = predictive + indicator @ (count / concentration).log1p().T
    return poisson, predictive
