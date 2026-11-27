"""Transfer stimulus-specific population rate residuals between labeled recording days."""

import json
from pathlib import Path
import time

import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import _evoked_counts
from glyph_kernel_encoding import descriptors, image_kernel
from glyph_mbr import contour_signatures, glyph_posterior
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import standardized
from kaggle_posterior_utility_validation import fit_neural, v11_logits
from kaggle_task1_microns_transfer import normalized_views, template_rates
from kaggle_task3_count_occupancy import paired_delta, posterior_metrics
from kaggle_task3_event_attention import fresh_cohort as previous_event_cohort
from kaggle_task3_glyph_covariance import matched_permutation, near_shape_quality, split_glyphs as previous_covariance_cohort
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_mice2_full_balance_replication import cohorts, early_cohorts
from kaggle_task3_mice2_utility_candidate import MOUSE, SHORTLIST, TABLE_SHA, validated_utility
from kaggle_task3_native_hanzi_nomination import fresh_glyphs
from kaggle_task3_rankband_ocf_validation import NeuralRows
from kaggle_task3_relational_consensus import choose_images
from kaggle_task3_v20_candidate_headroom import held_repeats
from kaggle_task3_v20_exact_utility import original_panel, paired_summary


SEED, HELD, EXACT = 373271, 320, 72
DAYS = ("session1", "session3")
RANK, RIDGE, FOLDS = 64, 0.35, 4
PRIOR_REPEATS, WEIGHT, RATE_FLOOR = 3.0, 0.30, 0.005
V20 = "v20"
BLIND, IMAGE, DIAGONAL = "class_blind", "visual_prior", "diagonal_peer"
LEARNED, SHUFFLED = "reduced_rank_peer", "reduced_rank_peer_shuffled"
OUTPUT = Path("/kaggle/working/task3_peer_rate_transfer.json")


def eligible_glyphs(size, cpu):
    image_fit, previous_v20, temperature, _ = cohorts(size, cpu)
    native, _, _ = fresh_glyphs(size, cpu)
    _, early = early_cohorts(size, cpu)
    _, covariance, _, _, _ = previous_covariance_cohort(size, cpu)
    _, event, _, _, _ = previous_event_cohort(size, 1728)
    excluded = torch.unique(torch.cat((image_fit, previous_v20, temperature, native, early, covariance, event)))
    gallery = torch.arange(size)
    eligible = gallery[~torch.isin(gallery, excluded)]
    return eligible, previous_v20


def new_fold(size, cpu):
    eligible, previous_v20 = eligible_glyphs(size, cpu)
    gallery = torch.arange(size)
    if len(eligible) < HELD:
        raise ValueError("The independent cohort requires 320 glyphs outside all named exclusions.")
    rng = torch.Generator().manual_seed(SEED)
    held = eligible[torch.randperm(len(eligible), generator=rng)[:HELD]]
    fit = gallery[~torch.isin(gallery, held)]
    fit = fit[torch.randperm(len(fit), generator=rng)]
    endpoints = dict(zip(DAYS, held[:len(DAYS) * EXACT].split(EXACT), strict=True))
    return fit, held, endpoints, previous_v20, len(eligible)


@torch.no_grad()
def crossfit_visual_rates(images, fitting, held, measured):
    """Assign every glyph an image-rate prediction fit without its measured spikes."""
    heads, size, features = measured.shape
    arrays = [descriptors(batch) for batch in images.split(96)]
    views = tuple(torch.cat(parts) for parts in zip(*arrays, strict=True))
    del arrays
    stabilized = (measured + 0.10).sqrt().transpose(0, 1).reshape(size, -1)
    rates = torch.empty_like(measured)
    destinations = []
    fold_details = []
    for inner_held, outer_held in zip(fitting.tensor_split(FOLDS), held.tensor_split(FOLDS), strict=True):
        source = fitting[~torch.isin(fitting, inner_held)]
        gallery = torch.cat((inner_held, outer_held))
        basis = normalized_views(views, source)
        anchor = basis[source]
        predicted = template_rates(image_kernel(anchor, anchor), image_kernel(basis[gallery], anchor), stabilized[source])
        rates[:, gallery] = predicted.reshape(len(gallery), heads, features).transpose(0, 1)
        destinations.append(gallery)
        fold_details.append({"fit_glyphs": len(source), "independent_predicted_glyphs": len(gallery)})
    if not torch.equal(torch.cat(destinations).sort().values, torch.arange(size)):
        raise ValueError("Each known glyph must have exactly one out-of-fold visual rate prediction.")
    return rates.clamp_min(RATE_FLOOR), fold_details


def fit_diagonal_transfer(peer, target, fit):
    x_center = peer[fit].mean(0)
    x_scale = peer[fit].std(0, unbiased=False).clamp_min(0.10)
    y_center = target[fit].mean(0)
    x = (peer - x_center) / x_scale
    x_train = x[fit]
    y = target[fit] - y_center
    slope = ((x_train * y).mean(0) / (x_train.square().mean(0) + 0.15)).clamp(-0.50, 2.0)
    return x, x * slope + y_center


def fit_joint_transfer(x, target, diagonal, fit):
    """Fit reduced-rank ridge on the residual left by same-feature transfer."""
    x_train = x[fit]
    remainder = target[fit] - diagonal[fit]
    gram = x_train.T @ x_train / len(fit)
    gram.diagonal().add_(RIDGE)
    factor = torch.linalg.cholesky(gram)
    cross = x_train.T @ remainder / len(fit)
    whitened = torch.linalg.solve_triangular(factor, cross, upper=False)
    eigenvalues, directions = torch.linalg.eigh(whitened.T @ whitened)
    axes = directions[:, -RANK:]
    mapping = torch.linalg.solve_triangular(factor.T, whitened @ axes, upper=True)
    correction = (x @ mapping) @ axes.T
    spectrum = eigenvalues.clamp_min(0)
    diagnostics = {"effective_rank": len(axes.T), "spectrum_fraction": float(spectrum[-RANK:].sum() / spectrum.sum()),
                   "training_diagonal_residual_mse": float(remainder.square().mean()),
                   "training_reduced_rank_residual_mse": float((remainder - correction[fit]).square().mean()),
                   "learned_shift_rms": float(correction.square().mean().sqrt())}
    return correction, diagnostics


def infer_rates(index, measured, image, fitting, scramble=None):
    residual_scale = (image + 0.12).sqrt()
    response = (measured - image) / residual_scale
    peer = response[torch.arange(len(response)) != index].mean(0)
    x, diagonal = fit_diagonal_transfer(peer, response[index], fitting)
    prior = image[index]
    target_scale = residual_scale[index]
    means = {BLIND: measured[index, fitting].mean(0).expand_as(prior), IMAGE: prior,
             DIAGONAL: prior + target_scale * diagonal}
    diagnostic = {}
    if scramble is not None:
        joint, diagnostic = fit_joint_transfer(x, response[index], diagonal, fitting)
        means[LEARNED] = prior + target_scale * (diagonal + joint)
        means[SHUFFLED] = prior + target_scale * (diagonal + joint[scramble])
        diagnostic["different_glyph_residual_rms"] = float((joint - joint[scramble]).square().mean().sqrt())
    repeats = 3.0
    expected = {name: (repeats * measured[index] + PRIOR_REPEATS * rate.clamp_min(RATE_FLOOR))
                      / (repeats + PRIOR_REPEATS) for name, rate in means.items()}
    diagnostic["mean_prior_rate"] = {name: float(rate.clamp_min(RATE_FLOOR).mean()) for name, rate in means.items()}
    return expected, diagnostic


def publish(report, started, stage):
    report.update(stage=stage, seconds=time.monotonic() - started)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{stage} after {report['seconds']:.1f}s", flush=True)


@torch.no_grad()
def evaluate(day, query, truth, endpoints, stock_logits, expected, neighbors, utility, temperature):
    rows = torch.where(torch.isin(truth, endpoints))[0]
    scores = stock_logits[rows]
    actions, original_mass, incumbent = original_panel(scores, temperature, utility)
    baseline, baseline_identity = posterior_metrics(glyph_posterior(stock_logits, temperature), truth)
    baseline.update(near_shape_quality(stock_logits, truth, neighbors))
    posteriors = {V20: original_mass}
    decisions = {V20: incumbent}
    ranks, density = {V20: baseline}, {}
    factorial = torch.lgamma(query + 1).sum(1)
    for name, rates in expected.items():
        log_probability = query @ rates.log().T - rates.sum(1)
        density[name] = log_probability.gather(1, truth[:, None])[:, 0] - factorial
        normalized = standardized(log_probability)
        fused = stock_logits + WEIGHT * normalized
        mass = glyph_posterior(fused, temperature)
        fusion_rank, identity = posterior_metrics(mass, truth)
        raw_rank, _ = posterior_metrics(glyph_posterior(normalized, temperature), truth)
        fusion_rank.update(near_shape_quality(fused, truth, neighbors))
        raw_rank.update(near_shape_quality(normalized, truth, neighbors))
        posteriors[name] = mass[rows]
        new_actions, _, winner = original_panel(fused[rows], temperature, utility)
        ranks[name] = {"anchored": fusion_rank, "standalone": raw_rank, "held_true_log_probability": float(density[name].mean()),
                       "identity_nll_vs_v20": paired_delta(identity, baseline_identity),
                       "renominated7_truth_in_pool": float((new_actions == truth[rows, None]).any(1).float().mean())}
        decisions[name + "_renominated7"] = winner
    same = choose_images(NeuralRows(scores, scores, actions), posteriors, utility)
    if not torch.equal(same[V20], incumbent):
        raise ValueError(f"{day} must retain the v20 fixed-seven decision rule.")
    decisions.update({name + "_fixed7": same[name] for name in expected})
    density_delta = {name: paired_delta(density[LEARNED], density[name]) for name in expected if name != LEARNED}
    return decisions, truth[rows], {"rank": ranks, "reduced_rank_count_log_p_vs_peers_nats": density_delta,
                                    "action_changes_vs_v20": {name: int((choice != incumbent).sum())
                                                              for name, choice in decisions.items()}}


def main():
    torch.set_num_threads(4)
    started = time.monotonic()
    cpu = torch.device("cpu")
    root = locate_data()
    images = load_images(root, cpu)
    masks = images >= 128
    fitting, held, endpoints, previous, eligible = new_fold(len(images), cpu)
    utility, temperatures, _ = validated_utility(images, cpu)
    neighbors = (utility + utility.T).fill_diagonal_(-torch.inf).topk(16).indices
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, cpu, len(images), bin_ms=25)
    support, query, overlap = held_repeats(recordings, previous)
    del recordings
    if any(values.shape[1] != 3 for values in support.values()) or any(values.shape[1] != 1 for values in query.values()):
        raise ValueError("Use three independent labeled supports and one untouched query in each recording.")
    report = {"mouse": MOUSE, "seed": SEED, "held_glyphs": len(held), "eligible_new_glyphs": eligible,
              "exact_glyphs_per_day": EXACT, "fitting_glyphs": len(fitting), "days": DAYS, "gallery": len(images),
              "candidate_budget": SHORTLIST, "input_window_ms": [50, 250], "bin_ms": 25, "v20_evidence_weight": WEIGHT,
              "rank": RANK, "ridge": RIDGE, "prior_exposure": PRIOR_REPEATS, "verified_utility": TABLE_SHA,
              "trial_overlap": overlap, "protocol": "Use 25 ms unit counts and the identical three measured target-day "
              "support trials in all models. Predict each glyph's visual rate out of fold from contours and use "
              "other-day observed residuals to predict its target firing rates. A rank-64 ridge transfers joint "
              "unit-time evidence beyond the same-feature diagonal predictor. The shuffled control keeps matched "
              "disjoint supports, diagonal transfer, and low-rank magnitude distribution, but swaps joint predictions "
              "within activity/ink strata. Held glyphs are absent from visual and transfer fits. Their independent "
              "class-labeled supports remain available as templates; all queries are genuine single measured trials. "
              "Compare through raw full-gallery Poisson likelihood and frozen v20 decision utility, with the original "
              "seven actions and an equal-seven re-nomination. No formal inputs are read.", "sessions": {}}
    publish(report, started, "split_ready")

    model = fit_neural(support, images, *contour_signatures(images), MOUSE, 211)
    baseline = {day: v11_logits(model, query[day][held], day, images) for day in DAYS}
    del model
    measured = torch.stack([_evoked_counts(values.float().mean(1), "timed") for values in support.values()])
    order = tuple(support)
    del support
    image_rates, report["visual_fold_exclusion"] = crossfit_visual_rates(images, fitting, held, measured)
    scramble = matched_permutation(measured.mean(0).sum(1), masks.float().mean((1, 2)), SEED + 13)
    publish(report, started, "cross_fitted_visual_rates")
    scorer = official_scorer(root)

    scores = {}
    for day in DAYS:
        rate, model_details = infer_rates(order.index(day), measured, image_rates, fitting, scramble)
        trials = _evoked_counts(query[day][held, 0], "timed")
        decisions, exact_truth, measures = evaluate(day, trials, held, endpoints[day], baseline[day], rate, neighbors,
                                                    utility, temperatures[day]["temperature"])
        report["sessions"][day] = {**measures, "rate_transfer": model_details, "exact_glyph_ids": exact_truth.tolist()}
        print(json.dumps({"day": day, "held_log_probability_comparisons": measures["reduced_rank_count_log_p_vs_peers_nats"]}),
              flush=True)
        publish(report, started, f"{day}_likelihoods")
        selections = torch.stack(tuple(decisions.values()))
        official, unique = unique_pair_ocf(selections, exact_truth.expand_as(selections), masks, scorer, cpu, 96)
        scores[day] = dict(zip(decisions, official, strict=True))
        report["sessions"][day].update(exact_ocf=paired_summary(scores[day]), unique_exact_pairs=unique)
        publish(report, started, f"{day}_official_ocf")
    combined = {arm: torch.cat([daily[arm] for daily in scores.values()]) for arm in next(iter(scores.values()))}
    report["pooled_ocf"] = paired_summary(combined)
    report["learned_vs_matched_controls_ocf"] = {
        policy: {name: paired_delta(combined[f"{LEARNED}_{policy}"], combined[f"{name}_{policy}"])
                 for name in (BLIND, IMAGE, DIAGONAL, SHUFFLED)} for policy in ("fixed7", "renominated7")
    }
    publish(report, started, "complete")
    print(json.dumps(report["pooled_ocf"]), flush=True)


if __name__ == "__main__":
    main()
