"""Cross-fitted unlabeled population adaptation for complete Task 2 blocks."""

import torch

from glyph_encoding import count_template_logits
from glyph_mbr import glyph_posterior
from glyph_retrieval import (_moments, _scaled_logits, _standardize, _template_logits, _variance_stabilize,
                             _whiten, combine_logits, retrieval_score_components)


def _score_scale(scores):
    return (scores - scores.mean(dim=1, keepdim=True)).square().mean().sqrt().clamp_min(1e-6)


def _spike_coordinates(counts, model):
    values = _variance_stabilize(counts, n_bins=model.n_bins)
    pooled = _standardize(values, *_moments(values)).mean(dim=1)
    return _whiten(pooled, model.noise_factor) @ model.reliable_axes


def _cross_fit(apply_to, fit_from, targets, weights, *, shrinkage, maximum_change):
    weighted = fit_from * weights[:, None]
    gram = fit_from.T @ weighted
    prior = shrinkage * fit_from.square().sum(dim=0).mean().clamp_min(1e-6)
    gram.diagonal().add_(prior)
    factor = torch.linalg.cholesky(gram)
    residual_target = targets - fit_from
    adjustment = torch.cholesky_solve(weighted.T @ residual_target, factor)

    # Analytic leave-one-query-out ridge prediction. The fitted map does not
    # use the query's own mapped target/repeat as a regression row at inference.
    influence = torch.cholesky_solve(weighted.T.contiguous(), factor).T
    leverage = (influence * fit_from).sum(dim=1).clamp(max=0.95)
    unseen_cross = (influence * apply_to).sum(dim=1)
    residual = residual_target - fit_from @ adjustment
    change = apply_to @ adjustment - (unseen_cross / (1 - leverage))[:, None] * residual
    allowed = maximum_change * apply_to.norm(dim=1, keepdim=True)
    return change * (allowed / change.norm(dim=1, keepdim=True).clamp_min(1e-8)).clamp_max(1)


@torch.inference_mode()
def crossday_logits(model, counts, *, poisson_bank=None, rank=32, temperature=0.5,
                    shrinkage=2.0, maximum_change=0.20, correction_weight=0.35):
    """Return unchanged initializer and adapted logits using raw spikes only.

    Each query has four repeats. A posterior from repeats A supervises the
    stable coordinates of disjoint repeats B, and the reverse model uses
    B's posterior with A's coordinates. The two fitted changes are applied
    to their correspondence halves and averaged into the original score.
    """
    if (counts.ndim != 3 or counts.shape[1] != 4 or len(counts) != len(model.center)
            or not bool(model.available.all())):
        raise ValueError("Cross-day Sinkhorn needs a complete four-repeat glyph block")
    if rank < 1 or temperature <= 0 or shrinkage <= 0 or maximum_change < 0 or correction_weight < 0:
        raise ValueError("Provide positive adaptation rank, temperature and shrinkage")

    def original(values):
        if poisson_bank is not None:
            rates = poisson_bank.for_recording(None, values)["shrinkage"]
            return count_template_logits(values, rates, "rate"), None
        components = retrieval_score_components(model, values, None, task="task2")
        return combine_logits(components, "task2"), components

    baseline, components = original(counts)
    halves = tuple(counts.split(2, dim=1))
    full_centers = _whiten(model.center, model.noise_factor) @ model.reliable_axes
    centers = full_centers[:, -rank:]
    features, targets, confidences, peak_mass = [], [], [], []
    for half in halves:
        logits, _ = original(half)
        features.append(_spike_coordinates(half, model)[:, -rank:])
        posterior = glyph_posterior(logits / _score_scale(logits), temperature, balance=True)
        mass = posterior.amax(dim=1)
        confidence = (mass - 1 / len(centers)).clamp_min(0)
        peak_mass.append(float(mass.mean()))
        targets.append(posterior @ centers)
        confidences.append(confidence / (confidence + 0.02))

    a, b = features
    change_a = _cross_fit(a, b, targets[0], confidences[0], shrinkage=shrinkage, maximum_change=maximum_change)
    change_b = _cross_fit(b, a, targets[1], confidences[1], shrinkage=shrinkage, maximum_change=maximum_change)
    correction = (0.5 * (change_a + change_b)) @ centers.T
    reliable = _template_logits(_spike_coordinates(counts, model), full_centers)
    if components is not None:
        components["reliable"] = _scaled_logits(reliable + correction, model.available)
        adapted = combine_logits(components, "task2")
    else:
        adapted = baseline + correction_weight * (_score_scale(baseline) / _score_scale(reliable)) * correction
    relative_change = (change_a.square().sum() + change_b.square().sum()) / (
        a.square().sum() + b.square().sum()).clamp_min(1e-8)
    return baseline, adapted, {"rank": centers.shape[1], "mean_peak_probability": peak_mass,
                               "relative_coordinate_rms_change": float(relative_change.sqrt())}
