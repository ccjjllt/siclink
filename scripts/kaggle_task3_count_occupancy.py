"""Test single-trial class evidence from activation probability at a fixed count mean."""

import json
from pathlib import Path
import time

import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_mbr import contour_signatures, glyph_posterior
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import standard_error, standardized
from kaggle_posterior_utility_validation import fit_neural, v11_logits
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_mice2_full_balance_replication import cohorts
from kaggle_task3_mice2_utility_candidate import MOUSE, SHORTLIST, TABLE_SHA, validated_utility
from kaggle_task3_rankband_ocf_validation import NeuralRows
from kaggle_task3_relational_consensus import INDEPENDENT, UTILITY_WEIGHT, choose_images
from kaggle_task3_v20_candidate_headroom import DAYS, GLYPHS_PER_DAY, SEED, held_repeats
from kaggle_task3_v20_exact_utility import held_panels, original_panel, paired_summary


V20 = "v20"
POISSON = "same_mean_poisson"
NB = "same_mean_negative_binomial"
SHARED = "rate_calibrated_hurdle"
SHUFFLED = "rate_matched_shuffled_hurdle"
GLYPH = "glyph_conditioned_hurdle"
ATTRIBUTION = (POISSON, NB, SHARED, SHUFFLED, GLYPH)
WINDOWS = ((50, 100), (100, 250))
MEAN_PRIOR_REPEATS = 2.0
RATE_STRATA = 12
OCCUPANCY_WEIGHT = 0.20
POSTERIOR_WEIGHT = 0.20
OUTPUT = Path("/kaggle/working/task3_count_occupancy.json")


def counts_in_windows(raw):
    counts = raw.unflatten(-1, (-1, 10))
    return torch.stack([counts[..., start // 25:end // 25].sum(-1) for start, end in WINDOWS], dim=-1).flatten(-2)


def nb_dispersion(support):
    means = support.mean(2)
    within = support.var(2, unbiased=True)
    numerator = (within - means).mean((0, 1))
    denominator = means.square().mean((0, 1)).clamp_min(0.05)
    return (numerator / denominator).clamp(0.02, 2.0)


def nb_active_probability(means, dispersion):
    return -torch.expm1(-torch.log1p(dispersion * means) / dispersion)


def rate_strata(means, residual, rng=None):
    """Center each unit-window within fitted rate groups, optionally permuting within groups."""
    columns = torch.arange(means.shape[1])
    conditional = torch.empty_like(residual)
    centered = torch.empty_like(residual)
    shuffled = torch.empty_like(residual) if rng is not None else None
    for group in means.argsort(dim=0).tensor_split(RATE_STRATA):
        values = residual[group, columns]
        center = values.mean(0)
        conditional[group, columns] = center
        values = values - center
        centered[group, columns] = values
        if shuffled is not None:
            order = torch.rand(group.shape, generator=rng).argsort(dim=0)
            shuffled[group.gather(0, order), columns] = values.gather(0, order.roll(1, dims=0))
    return conditional, centered, shuffled


def fit_rates(recordings, expected):
    trials = torch.stack(tuple(recordings.values()))
    expected = torch.stack(tuple(expected.values()))
    alpha = nb_dispersion(trials)
    sums = trials.sum(2)
    pseudo = MEAN_PRIOR_REPEATS * expected
    mean = ((sums + pseudo) / (trials.shape[2] + MEAN_PRIOR_REPEATS)).clamp_min(1e-4)
    leave_one_out = ((sums[:, :, None] - trials + pseudo[:, :, None])
                     / (trials.shape[2] - 1 + MEAN_PRIOR_REPEATS)).clamp_min(1e-4)
    errors = ((trials > 0).float() - nb_active_probability(leave_one_out, alpha)).mean(2)
    by_day = {}
    centered = []
    for day, rates, residual in zip(recordings, mean, errors, strict=True):
        nuisance, remaining, _ = rate_strata(rates, residual)
        centered.append(remaining)
        by_day[day] = (rates, nuisance)
    peer_residuals = torch.stack(centered)
    return alpha, by_day, peer_residuals


def truncated_mean_minus_one(rate):
    """Stable E[Poisson(rate) | positive] - 1, including rates near zero."""
    squared = rate.square()
    series = 0.5 * rate + squared / 12 - squared.square() / 720
    return torch.where(rate < 0.04, series, rate / (-torch.expm1(-rate)) - 1)


def positive_rate(mean, active):
    # Subtract before dividing to retain the displacement from one in low-rate, near-Bernoulli regimes.
    target = (mean - active) / active
    lower, upper = torch.zeros_like(target), 1.0 + target
    for _ in range(29):
        center = 0.5 * (lower + upper)
        too_small = truncated_mean_minus_one(center) < target
        lower = torch.where(too_small, center, lower)
        upper = torch.where(too_small, upper, center)
    return 0.5 * (lower + upper)


def hurdle_parameters(mean, active):
    mean, active = mean.double(), active.double()
    theta = positive_rate(mean, active)
    restored = active * (1 + truncated_mean_minus_one(theta))
    relative_error = float(((restored - mean) / mean).abs().max())
    if relative_error > 1e-6 or not bool(torch.isfinite(theta).all()):
        raise RuntimeError("Hurdle occupancy must conserve the same mean counts.")
    return theta.float(), relative_error


def activation_models(day, day_rates, peer_residuals, alpha):
    target = tuple(day_rates).index(day)
    mean, nuisance = day_rates[day]
    prior = nb_active_probability(mean, alpha)
    maximum = mean.clamp_max(1.0) * 0.9999
    shared = (prior + nuisance).clamp(min=0.20 * prior, max=maximum)
    peers = peer_residuals[torch.arange(len(peer_residuals)) != target].mean(0)
    rng = torch.Generator().manual_seed(SEED + 1100 + target)
    _, centered, shuffled = rate_strata(mean, peers, rng)
    lower = 0.5 * shared
    upper = shared + 0.5 * (maximum - shared)
    arms = {SHARED: shared, SHUFFLED: (shared + OCCUPANCY_WEIGHT * shuffled).clamp(lower, upper),
            GLYPH: (shared + OCCUPANCY_WEIGHT * centered).clamp(lower, upper)}
    return mean, arms


def count_logit(observed, mean, dispersion=None):
    if dispersion is None:
        return observed @ mean.log().T - mean.sum(1), -torch.lgamma(observed + 1).sum(1)
    ratio = mean * dispersion
    norm = torch.log1p(ratio)
    weight = ratio.log() - norm
    shape = dispersion.reciprocal()
    score = observed @ weight.T - (shape * norm).sum(1)
    constant = (torch.lgamma(observed + shape) - torch.lgamma(shape) - torch.lgamma(observed + 1)).sum(1)
    return score, constant


def hurdle_logit(observed, mean, active):
    theta, error = hurdle_parameters(mean, active)
    no_spike = (-active).log1p()
    positive_weight = active.log() - no_spike - theta - (-torch.expm1(-theta)).log()
    logits = observed @ theta.log().T + (observed > 0).float() @ positive_weight.T + no_spike.sum(1)
    return logits, -torch.lgamma(observed + 1).sum(1), error


def paired_delta(values, reference):
    delta = (values - reference).numpy()
    return {"delta": float(delta.mean()), "paired_se": standard_error(delta)}


def posterior_metrics(mass, truth):
    leading = mass.topk(32, dim=1)
    hits = leading.indices == truth[:, None]
    loss = -mass.gather(1, truth[:, None])[:, 0].clamp_min(1e-25).log()
    return {"nll": float(loss.mean()), "mean_mass_top32": float(leading.values.sum(1).mean()),
            **{f"top{k}": float(hits[:, :k].any(1).float().mean()) for k in (1, 7, 16, 32)}}, loss


def measure_day(day, query, order, scoring_rows, baseline, utility, temperature, alpha, rates, peers):
    mean, probabilities = activation_models(day, rates, peers, alpha)
    observed = counts_in_windows(query[:, 0])
    rows = torch.arange(len(order))
    base_posterior = glyph_posterior(baseline, temperature)
    baseline_report, baseline_identity = posterior_metrics(base_posterior, order)
    reports, hypotheses = {V20: baseline_report}, {V20: base_posterior[scoring_rows]}
    held_losses, occupancy_losses, identity_losses = {}, {}, {}
    for name in ATTRIBUTION:
        result = {}
        if name in probabilities:
            active = probabilities[name]
            logits, constants, error = hurdle_logit(observed, mean, active)
            p_correct = active[order]
            activation_loss = -torch.where(observed > 0, p_correct.log(), (-p_correct).log1p()).sum(1)
            occupancy_losses[name] = activation_loss
            result.update(max_relative_mean_error=error, held_activation_nll=float(activation_loss.mean()),
                          occupancy_update_rms=float((active - probabilities[SHARED]).square().mean().sqrt()))
        else:
            logits, constants = count_logit(observed, mean, alpha if name == NB else None)
        held_losses[name] = -(logits[rows, order] + constants)
        result["held_count_nll"] = float(held_losses[name].mean())
        likelihood = standardized(logits)
        standalone, _ = posterior_metrics(glyph_posterior(likelihood, temperature), order)
        fused = glyph_posterior(baseline + POSTERIOR_WEIGHT * likelihood, temperature)
        fused_report, identity = posterior_metrics(fused, order)
        identity_losses[name] = identity
        hypotheses[name] = fused[scoring_rows]
        reports[name] = {**result, "fused_identity": fused_report, "standalone_identity": standalone,
                         "fused_identity_nll_vs_v20": paired_delta(identity, baseline_identity)}
    for name in ATTRIBUTION:
        reports[name]["count_nll_vs_shared"] = paired_delta(held_losses[name], held_losses[SHARED])
        if name in occupancy_losses:
            reports[name]["activation_nll_vs_shared"] = paired_delta(occupancy_losses[name], occupancy_losses[SHARED])
    reports[GLYPH].update(count_nll_vs_shuffle=paired_delta(held_losses[GLYPH], held_losses[SHUFFLED]),
                          activation_nll_vs_shuffle=paired_delta(occupancy_losses[GLYPH], occupancy_losses[SHUFFLED]),
                          identity_nll_vs_shuffle=paired_delta(identity_losses[GLYPH], identity_losses[SHUFFLED]),
                          identity_nll_vs_shared=paired_delta(identity_losses[GLYPH], identity_losses[SHARED]))

    fixed_logits = baseline[scoring_rows]
    actions, _, incumbent = original_panel(fixed_logits, temperature, utility)
    panel = NeuralRows(fixed_logits, fixed_logits, actions)
    alternatives = choose_images(panel, hypotheses, utility)
    choices = {name: alternatives[name] for name in hypotheses}
    if not torch.equal(incumbent, choices[V20]):
        raise RuntimeError("The original posterior must reproduce the inherited seven-action decision.")
    return reports, choices


def publish(report, started, stage):
    report.update(stage=stage, seconds=time.monotonic() - started)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{stage} after {report['seconds']:.1f}s", flush=True)


@torch.no_grad()
def main():
    if not Path("/kaggle/input").is_dir() or not OUTPUT.parent.is_dir():
        raise RuntimeError("Use private Kaggle CPU with the official training and verified utility sources.")
    device = torch.device("cpu")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, device)
    masks = images >= 128
    utility, temperature, _ = validated_utility(images, device)
    visual, previous, calibration, _ = cohorts(len(images), device)
    references, orders = held_panels(previous, torch.cat((visual, calibration)), GLYPHS_PER_DAY)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images), bin_ms=25)
    support, queries, overlap = held_repeats(recordings, previous)
    del recordings
    with torch.enable_grad():
        model = fit_neural(support, images, *contour_signatures(images), MOUSE, 211)
    encoder = model[0]
    observed = {day: counts_in_windows(values) for day, values in support.items()}
    priors = {day: counts_in_windows(encoder.expected_counts(images, day)) for day in observed}
    alpha, day_rates, peers = fit_rates(observed, priors)
    del support, observed, priors
    report = {"mouse": MOUSE, "device": str(device), "seed": SEED, "days": list(DAYS), "windows_ms": WINDOWS,
              "paired_glyphs_per_day": GLYPHS_PER_DAY, "gallery": len(images), "candidate_budget": SHORTLIST,
              "trial_overlap": overlap, "rate_strata": RATE_STRATA, "mean_prior_repeats": MEAN_PRIOR_REPEATS,
              "mean_dispersion": float(alpha.mean()), "occupancy_residual_weight": OCCUPANCY_WEIGHT,
              "posterior_likelihood_weight": POSTERIOR_WEIGHT, "utility_weight": UTILITY_WEIGHT,
              "utility_sha256": TABLE_SHA, "protocol": "Use identical train-only expected window counts in every likelihood. "
                          "Adjust shared activation probability with the cross-day same-glyph residual from independent "
                          "support repeats. Its control permutes glyph residuals within each unit-window's fitted rate strata. "
                          "The mean of the zero-truncated positive count model is fixed at expected count divided by "
                          "activation probability. Preserve v20's actual candidates, neural anchor and visual utility. "
                          "Each alternative supplies only a separate posterior. Held labeled repeat spikes are queried once "
                          "per glyph. The headroom study's preselected 96 independent glyph/day cases supply exact scores.",
              "recordings": {}}
    scorer = official_scorer(root)
    decisions = {}
    for day in DAYS:
        order, truth = orders[day], references[day]
        rows = order.argsort()[truth]
        query = queries[day][order]
        baseline = v11_logits(model, query, day, images)
        rank, choices = measure_day(day, query, order, rows, baseline, utility, temperature[day]["temperature"],
                                    alpha, day_rates, peers)
        report["recordings"][day] = {"posterior_and_counts": rank,
                                      "changed_scored_actions": {name: int((choice != choices[V20]).sum())
                                                                 for name, choice in choices.items()}}
        decisions[day] = choices
        publish(report, started, f"{day}_posteriors")
        print(json.dumps(rank), flush=True)

    values = {}
    for day, choices in decisions.items():
        selected = torch.stack(tuple(choices.values()))
        exact, unique = unique_pair_ocf(selected, references[day].expand_as(selected), masks, scorer, device, 96)
        values[day] = dict(zip(choices, exact, strict=True))
        exact_summary = paired_summary(values[day])
        exact_summary["glyph_vs_shared"] = paired_delta(values[day][GLYPH], values[day][SHARED])
        exact_summary["glyph_vs_shuffle"] = paired_delta(values[day][GLYPH], values[day][SHUFFLED])
        report["recordings"][day].update(exact_ocf=exact_summary, distinct_exact_pairs=unique)
        publish(report, started, f"{day}_ocf")
        print(json.dumps(exact_summary), flush=True)
    pooled = {name: torch.cat([scores[name] for scores in values.values()]) for name in next(iter(values.values()))}
    report["pooled_ocf"] = paired_summary(pooled)
    report["pooled_ocf"]["glyph_vs_shared"] = paired_delta(pooled[GLYPH], pooled[SHARED])
    report["pooled_ocf"]["glyph_vs_shuffle"] = paired_delta(pooled[GLYPH], pooled[SHUFFLED])
    publish(report, started, "complete")
    print(json.dumps(report["pooled_ocf"]), flush=True)


if __name__ == "__main__":
    main()
