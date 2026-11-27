"""Test one shared spatial shift per blind Task 2 recording using independent repeats."""

from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs, combine_logits, fit_retrieval, retrieval_score_components
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_retinal_shift_validation import translate
from kaggle_task2_anchored_cdf import evaluation_sample as anchored_cohort
from kaggle_task2_peer_mouse import logit_scale, row_diagnostics
from kaggle_task2_rank_marginals import (BLEND, EPOCHS, MODEL_SEED, MOUSE, TEMPERATURE, V9,
                                         normal_scores, responses, source_cdf, summarize, whiten)
from kaggle_task2_rank_marginals_recheck import PROVENANCE, fit_marginal_templates
from kaggle_v8_metric_validation import SESSIONS


SEED = 286301
DAYS = ("session1", "session3")
OCF_GLYPHS, ORACLE_GLYPHS = 112, 96
RIDGE, LOG_RATE_LIMIT, BALANCE_STEPS = 0.5, 0.35, 16
CORRECTION_WEIGHT = 0.25
SHIFT_PIXELS = 3
STATES = ((0, 0), *(state for state in product((-SHIFT_PIXELS, 0, SHIFT_PIXELS), repeat=2) if state != (0, 0)))
REAL, SHUFFLED, ORACLE = "day_shift", "shuffled_shift", "known_pair_shift_oracle"
CDF = "target_cdf_fixed_zero_shift"
ARMS = (V9, CDF, REAL, SHUFFLED, ORACLE)
OUTPUT = Path("/kaggle/working/task2_day_shift.json")


def sample_glyphs(size):
    from kaggle_task2_constrained_ceiling import fresh_samples as ceiling_cohorts
    from kaggle_task2_native_hanzi import SEED as HANZI_SEED, scoring_glyphs as hanzi_cohorts
    from kaggle_task2_rank_marginals import CEILING_CROWDED, evaluation_glyphs
    from kaggle_task2_rank_marginals_recheck import evaluation_sample as rank_recheck

    prior, _ = evaluation_glyphs(size)
    recheck, _ = rank_recheck(size)
    anchoring, _ = anchored_cohort(size)
    followup, _ = anchored_cohort(size, followup=True)
    ceiling, _, _ = ceiling_cohorts(size)
    hanzi, _ = hanzi_cohorts(size, np.random.default_rng(HANZI_SEED))
    excluded = np.unique(np.concatenate((prior.numpy(), recheck.numpy(), anchoring.numpy(), followup.numpy(),
                                         *ceiling.values(), *hanzi.values(), np.asarray(CEILING_CROWDED))))
    eligible = np.setdiff1d(np.arange(size), excluded)
    drawn = np.random.default_rng(SEED).choice(eligible, OCF_GLYPHS + ORACLE_GLYPHS, replace=False)
    return tuple(torch.from_numpy(np.sort(indices)) for indices in np.split(drawn, [OCF_GLYPHS]))


def evoked(counts):
    return counts.unflatten(-1, (-1, 10))[..., 2:].sum(-1).mean(1)


def spatial_features(images):
    # Keep the forward model linear to make the spatial diagnostic cheap on CPU.
    pooled = F.avg_pool2d(images[:, None].float() / 255, 4)
    horizontal = F.pad(pooled.diff(dim=-1).abs(), (0, 1, 0, 0))
    vertical = F.pad(pooled.diff(dim=-2).abs(), (0, 0, 0, 1))
    return torch.cat((pooled, horizontal, vertical), dim=1).flatten(1)


def spatial_templates(sources, images, rng):
    """Anchor shift derivatives at empirical source rates and permute derivatives within strata."""
    by_day = torch.stack([evoked(counts) for counts in sources.values()])
    relative = (by_day + 0.05) / (by_day.mean(dim=1, keepdim=True) + 0.05)
    empirical = 0.86 * relative.mean(0) + 0.14
    features = spatial_features(images)
    feature_scale = features.std(dim=0).clamp_min(0.10)
    centered = (features - features.mean(0)) / feature_scale
    targets = empirical.log()
    regularized = centered.T @ centered / len(images)
    regularized.diagonal().add_(RIDGE)
    coefficients = torch.linalg.solve(regularized, centered.T @ (targets - targets.mean(0)) / len(images))
    derivatives = torch.stack([((spatial_features(translate(images, dx, dy, SHIFT_PIXELS)) - features)
                                / feature_scale) @ coefficients for dx, dy in STATES]).clamp(-LOG_RATE_LIMIT, LOG_RATE_LIMIT)
    if bool(derivatives[0].count_nonzero()):
        raise ValueError("The zero-shift image must reproduce the unshifted spatial features exactly.")

    ink = (images >= 128).float().mean(dim=(1, 2)).argsort(stable=True).numpy()
    rates = by_day.mean(dim=(0, 2)).numpy()
    permutation = np.empty(len(images), dtype=np.int64)
    for ink_group in np.array_split(ink, 4):
        sorted_by_rate = ink_group[np.argsort(rates[ink_group], kind="stable")]
        for group in np.array_split(sorted_by_rate, 4):
            visit = rng.permutation(group)
            permutation[visit] = np.roll(visit, 1)
    if np.any(permutation == np.arange(len(images))):
        raise ValueError("The fixed shuffle must derange all glyphs within rate and ink strata.")
    magnitude = derivatives.norm(dim=-1, keepdim=True)
    shuffled = derivatives[:, permutation]
    direction = F.normalize(shuffled, dim=-1)
    direction = torch.where(shuffled.norm(dim=-1, keepdim=True) > 0, direction,
                            torch.full_like(direction, direction.shape[-1] ** -0.5))
    shuffled = direction * magnitude
    diagnostics = {"max_shuffle_derivative_norm_error": float((shuffled.norm(dim=-1, keepdim=True) - magnitude).abs().max()),
                   "nonzero_shift_log_rate_rms": float(derivatives[1:].square().mean().sqrt())}
    return empirical, {REAL: derivatives, SHUFFLED: shuffled}, diagnostics


def shift_bank(empirical, derivative, calibration):
    shifted = empirical[None] * derivative.exp()
    # Match observed means per unit under every shift and shuffle.
    return (shifted / shifted.mean(dim=1, keepdim=True)) * calibration.mean(0).clamp_min(1e-4)


def hybrid_scores(retrieval, hierarchy, query):
    retrieval_scores = combine_logits(retrieval_score_components(retrieval, query, None, task="task2"), "task2")
    empirical = hierarchy.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
    return blend_hierarchical_logits(retrieval_scores, empirical)


def cdf_scores(hybrid, query, templates):
    _, gallery, factor, _ = templates
    raw = responses(query)
    coordinates = whiten(normal_scores(raw, source_cdf(raw)), factor)
    evidence = coordinates @ gallery.T - 0.5 * gallery.square().sum(1)
    return logit_scale(hybrid).lerp(logit_scale(evidence), BLEND)


def soft_matches(scores):
    scores = logit_scale(scores) / TEMPERATURE
    weights = (scores - scores.amax(dim=1, keepdim=True)).exp_().clamp_min_(1e-25)
    for _ in range(BALANCE_STEPS):
        weights /= weights.sum(dim=0, keepdim=True)
        weights /= weights.sum(dim=1, keepdim=True)
    return weights


def fitted_shift(bank, averaged_counts, glyph_mass):
    nats = (bank.log() * averaged_counts - bank * glyph_mass[:, None]).sum(dim=(1, 2)) / glyph_mass.sum()
    return int(nats.argmax()), nats - nats[0]


def shift_corrections(retrieval, hierarchy, templates, query, empirical, derivatives, order, oracle_rows, split):
    """Fit one shift on A, evaluate its correction on disjoint B, then swap."""
    halves = (query.gather(1, split[..., None].expand_as(query)).split(2, dim=1))
    corrections, diagnostics = {}, []
    for fitting, scoring in (halves, halves[::-1]):
        observed_fit, observed_score = evoked(fitting), evoked(scoring)
        assignment_logits = cdf_scores(hybrid_scores(retrieval, hierarchy, fitting), fitting, templates)
        posterior = soft_matches(assignment_logits)
        averaged_counts, glyph_mass = posterior.T @ observed_fit, posterior.sum(0)
        del assignment_logits, posterior
        banks = {name: shift_bank(empirical, values, observed_fit) for name, values in derivatives.items()}
        bank = banks[REAL]
        selectors = {name: fitted_shift(values, averaged_counts, glyph_mass) for name, values in banks.items()}
        selectors[ORACLE] = fitted_shift(bank[:, order[oracle_rows]], observed_fit[oracle_rows],
                                        torch.ones(len(oracle_rows)))
        fixed_zero = bank[0].clamp_min(1e-6)
        zero_score = observed_score @ fixed_zero.log().T - fixed_zero.sum(1)
        zero_scale = (zero_score - zero_score.mean(dim=1, keepdim=True)).square().mean().sqrt().clamp_min(1e-6)
        detail = {"max_soft_column_mass_error": float((glyph_mass - 1).abs().max()), "shifts": {}}
        for name, (state, objective) in selectors.items():
            variants = banks[SHUFFLED] if name == SHUFFLED else bank
            chosen = variants[state].clamp_min(1e-6)
            difference = observed_score @ (chosen.log() - fixed_zero.log()).T - (chosen - fixed_zero).sum(1)
            if name not in corrections:
                corrections[name] = difference / (2 * zero_scale)
            else:
                corrections[name].add_(difference / (2 * zero_scale))
            paired_rows = torch.arange(len(order))
            diagnostic_rows = paired_rows[~torch.isin(paired_rows, oracle_rows)]
            detail["shifts"][name] = {"delta_x_y_pixels": STATES[state], "calibration_nats_vs_zero": objective.tolist(),
                                       "independent_true_pair_nats_vs_zero": float(difference[diagnostic_rows, order[diagnostic_rows]].mean())}
        diagnostics.append(detail)
    return corrections, diagnostics


def publish(report, stage, started):
    report.update(stage=stage, seconds=time.monotonic() - started)
    pending = OUTPUT.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(OUTPUT)
    print(json.dumps({"stage": stage, "seconds": report["seconds"]}), flush=True)


@torch.no_grad()
def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Use a private Kaggle CPU notebook with the official labeled recordings.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if len(images) != 2748 or tuple(recordings) != SESSIONS or any(part.shape[:2] != (len(images), 4)
                                                                  for part in recordings.values()):
        raise ValueError("Require four complete labeled recordings with four measured repeats per glyph.")
    scored, oracle = sample_glyphs(len(images))
    metric = PairedOCF(official_scorer(root), images, (images[scored] >= 128).numpy(), device, 12)
    rng = np.random.default_rng(SEED + 1)
    report = {"mouse": MOUSE, "seed": SEED, "archive_provenance": PROVENANCE, "held_days": DAYS,
              "labeled_sources_per_fold": 3, "measured_repeats": 4, "gallery_size": len(images),
              "scored_glyph_indices": scored.tolist(), "known_pair_oracle_fit_indices": oracle.tolist(),
              "states_xy_pixels": STATES, "spatial_ridge": RIDGE, "derivative_limit": LOG_RATE_LIMIT,
              "v9_seed": MODEL_SEED, "v9_epochs": EPOCHS, "cdf_weight": BLEND, "shift_correction_weight": CORRECTION_WEIGHT,
              "primary_arm": REAL, "matched_control_arm": CDF, "formal_test_accessed": False, "recordings": {},
              "protocol": "Fit v9 and CDF from the other three labeled recordings. Independently permute each held "
                          "recording. Within each target query randomly divide real repeats 2/2. Fit one shift using "
                          "balanced soft CDF identities on either half and add its scaled shifted-minus-zero-rate "
                          "evidence from the other half. Source glyph rates anchor the forward model and all states "
                          "match the same target-half unit means. Compare fixed zero and ink/rate-stratified norm-matched "
                          "shuffled derivatives. The oracle uses target labels of disjoint diagnostic glyphs only. "
                          "All final arms use one full Hungarian assignment and four-repeat v9 and CDF scores."}
    publish(report, "inputs_ready", started)
    exact = {name: [] for name in ARMS}
    for day in DAYS:
        source = {key: value for key, value in recordings.items() if key != day}
        publish(report, f"{day}_fitting", started)
        torch.manual_seed(MODEL_SEED)
        with torch.enable_grad():
            retrieval = fit_retrieval(as_trial_blocks(source), len(images), images=images, device=device,
                                      epochs=EPOCHS, seed=MODEL_SEED)
        hierarchy = fit_hierarchical_gallery(source, view="total")
        templates = fit_marginal_templates(source)
        spatial, derivatives, spatial_checks = spatial_templates(source, images, rng)
        order = torch.from_numpy(rng.permutation(len(images)))
        query = recordings[day][order]
        split = torch.from_numpy(rng.random(query.shape[:2]).argsort(axis=1))
        rows = order.argsort()
        hybrid = hybrid_scores(retrieval, hierarchy, query)
        fixed = cdf_scores(hybrid, query, templates)
        changes, shifts = shift_corrections(retrieval, hierarchy, templates, query, spatial, derivatives,
                                            order, rows[oracle], split)
        scores = {V9: hybrid, CDF: fixed, **{name: fixed + CORRECTION_WEIGHT * correction
                                             for name, correction in changes.items()}}
        selected = {name: assign_glyphs(scores[name]) for name in ARMS}
        if any(value.unique().numel() != len(images) for value in selected.values()):
            raise ValueError("Every assignment must preserve the complete known gallery.")
        report["recordings"][day] = {"labeled_source_days": list(source), "split_half_shifts": shifts,
                                     "spatial_checks": spatial_checks, "full_gallery_identity": {
                                         name: {**row_diagnostics(scores[name], order),
                                                "assigned_top1": float((selected[name] == order).float().mean()),
                                                "changed_vs_cdf": int((selected[name] != selected[CDF]).sum()),
                                                "distinct_outputs": len(selected[name])} for name in ARMS}}
        publish(report, f"{day}_assignments_ready", started)
        measured = {name: metric.evaluate(selected[name][rows[scored], None])[:, 0] for name in ARMS}
        for name, values in measured.items():
            exact[name].append(values)
        report["recordings"][day]["ocf_vs_cdf"] = {name: summarize(values, measured[CDF]) for name, values in measured.items()}
        publish(report, f"{day}_scored", started)
    mean = {name: np.stack(values).mean(0) for name, values in exact.items()}
    report.update(ocf_vs_v9={name: summarize(values, mean[V9]) for name, values in mean.items()},
                  ocf_vs_cdf={name: summarize(values, mean[CDF]) for name, values in mean.items()},
                  shift_vs_shuffled=summarize(mean[REAL], mean[SHUFFLED]),
                  ocf_rows_per_arm=len(DAYS) * len(scored), unique_exact_pairs=len(metric.values))
    publish(report, "complete", started)


if __name__ == "__main__":
    main()
