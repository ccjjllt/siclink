"""Adapt a count-template gallery to an unlabeled recording with repeated trials."""

from dataclasses import dataclass

import torch

from glyph_encoding import _evoked_counts, poisson_logits


@dataclass(frozen=True)
class RepeatMoments:
    mean: torch.Tensor
    within_variance: torch.Tensor
    signal_variance: torch.Tensor
    signal_error: torch.Tensor
    trials: int


@torch.no_grad()
def repeat_moments(counts):
    """Estimate stimulus variance after removing the variance of the repeat mean."""
    if counts.ndim != 3 or min(counts.shape[:2]) < 2 or not counts.shape[2]:
        raise ValueError("Adaptation needs at least two groups and two repeats per group")
    groups, repeats, _ = counts.shape
    group_mean = counts.mean(dim=1)
    mean = group_mean.mean(dim=0)
    within_group = (counts - group_mean[:, None]).square().sum(dim=1) / (repeats - 1)
    between_group = (group_mean - mean).square() * groups / (groups - 1)
    contributions = between_group - within_group / repeats
    return RepeatMoments(mean, within_group.mean(dim=0), contributions.mean(dim=0),
                         contributions.std(dim=0, unbiased=True) / groups**0.5, groups * repeats)


def _match_rates(rates, source_mean, target_mean, gain):
    fitted = (target_mean + gain * (rates - source_mean)).clamp_min(1e-6)
    # Clipping sparse units should not shift the requested recording mean.
    return fitted * (target_mean / fitted.mean(dim=0).clamp_min(1e-6))


@torch.no_grad()
def session_adaptation_logits(observed, rates, view):
    """Return the original scores and two moment-adapted Poisson alternatives.

    Observations have shape [unlabeled groups, repeats, units × time bins].
    The learned gallery has shape [glyphs, units × time bins]. Neither glyph
    ordering nor query labels enter the estimates of firing-rate drift.
    """
    counts = _evoked_counts(observed.float(), view)
    expected = _evoked_counts(rates.float(), view)
    source_mean = expected.mean(dim=0)
    source_signal = expected.var(dim=0, unbiased=True)
    moments = repeat_moments(counts)

    # Keep a small prior at the baseline template rate for sparse features.
    prior_trials = 128
    fraction = moments.trials / (moments.trials + prior_trials)
    target_mean = ((1 - fraction) * source_mean + fraction * moments.mean).clamp_min(1e-6)

    # Correct amplitude only when the estimated shift exceeds sampling noise.
    difference = (moments.signal_variance - source_signal).abs()
    strength = 0.75 * (difference - 2 * moments.signal_error).clamp_min(0) / difference.clamp_min(1e-8)
    floor = (0.04 * (target_mean + source_mean)).square() + 1e-8
    target_signal = moments.signal_variance.clamp_min(0)
    raw_gain = ((target_signal + floor) / (source_signal + floor)).sqrt().clamp(0.60, 1.65)
    gain = (strength * raw_gain.log()).exp()

    templates = {"baseline": expected,
                 "mean_shrink": _match_rates(expected, source_mean, target_mean, 1),
                 "mean_variance_shrink": _match_rates(expected, source_mean, target_mean, gain)}
    logits = {name: poisson_logits(counts, 0.98 * fitted + 0.02 * fitted.mean(dim=0))
              for name, fitted in templates.items()}
    summary = {"gain_below_one_fraction": float((gain < 0.999).float().mean()),
               "gain_above_one_fraction": float((gain > 1.001).float().mean()),
               "mean_absolute_gain_shift": float((gain - 1).abs().mean()),
               "mean_absolute_rate_shift": float((target_mean - source_mean).abs().mean()),
               "target_signal_to_within_noise": float(target_signal.sum() /
                                                       moments.within_variance.sum().clamp_min(1e-8))}
    return logits, summary
