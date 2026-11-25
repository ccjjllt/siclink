"""Gamma–Poisson decoding with dispersion estimated from labeled training repeats."""

import torch

from glyph_encoding import _evoked_counts


@torch.no_grad()
def fit_count_dispersion(training, view, *, prior_pairs=32.0):
    """Estimate per-feature alpha in Var(Y|glyph, session) = mu + alpha * mu².

    Training blocks have shape [glyphs, repeats, units × time bins]. Within a
    same-glyph recording group, sample variance minus sample mean estimates
    alpha * mu². Distinct-repeat count products estimate mu² without counting
    repeat noise as stimulus variance. Pool these moments across recording days.
    """
    if view not in ("rate", "timed") or not training or prior_pairs < 0:
        raise ValueError("Provide repeated training counts and a rate or timed count view")
    moments = []
    for counts in training.values():
        if counts.ndim != 3 or counts.shape[1] < 2 or counts.shape[-1] % 10:
            raise ValueError("Expected [glyphs, repeats >= 2, units × 10] training counts")
        projected = _evoked_counts(counts.float(), view)
        repeats = projected.shape[1]
        total = projected.sum(dim=1)
        squared = projected.square().sum(dim=1)
        excess = projected.var(dim=1, unbiased=True) - total / repeats
        cross_repeat = ((total.square() - squared) / (repeats * (repeats - 1))).clamp_min(0)
        moments.append(torch.stack((excess.sum(dim=0), cross_repeat.sum(dim=0), excess.square().sum(dim=0),
                                    (excess * cross_repeat).sum(dim=0), cross_repeat.square().sum(dim=0))))

    excess, exposure, squared, cross, exposure_squared = torch.stack(moments).sum(dim=0)
    denominator = exposure.clamp_min(1e-8)
    estimate = (excess / denominator).clamp_min(0)
    residual = (squared - 2 * estimate * cross + estimate.square() * exposure_squared).clamp_min(0)
    error = residual.sqrt() / denominator
    evidence = (estimate - error).clamp(0, 2.0)
    return evidence * exposure / (exposure + prior_pairs).clamp_min(1e-8)


@torch.no_grad()
def negative_binomial_logits(observed, expected, alpha):
    """Candidate-dependent log likelihood per repeat, including the Poisson limit.

    Removing terms independent of the candidate gives weights
    log(mu) - log1p(alpha * mu) and offsets -log1p(alpha * mu) / alpha.
    With fixed feature shape 1 / alpha, the score is one matrix product.
    """
    if observed.ndim != 3 or expected.ndim != 2 or alpha.shape != (expected.shape[-1],):
        raise ValueError("Expected repeated counts, candidate means and one dispersion per feature")
    expected = expected.clamp_min(1e-5)
    alpha = alpha.to(expected).clamp_min(0)
    positive = alpha > 0
    safe = torch.where(positive, alpha, torch.ones_like(alpha))
    penalty = torch.log1p(expected * alpha)
    offset = torch.where(positive, penalty / safe, expected)
    return observed.mean(dim=1) @ (expected.log() - penalty).T - offset.sum(dim=1)


def negative_binomial_template_logits(observed, rates, view, alpha):
    counts = _evoked_counts(observed, view)
    expected = _evoked_counts(rates, view)
    expected = 0.98 * expected + 0.02 * expected.mean(dim=0)
    return negative_binomial_logits(counts, expected, alpha)
