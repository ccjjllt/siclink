"""Check whether an independent 2/2 spike forecast explains the v13 OCF change."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import fit_glyph_encoder
from glyph_mbr import contour_signatures, similarity_from_signatures
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_task1_crossfit_candidate import fit_crossfit_head
from kaggle_task1_crossfit_metric_scale import fresh_outer, fold_indices
from kaggle_task1_eight_fold_metric_scale import new_outer
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE, independent_holdout
from kaggle_task1_repeat_agreement import splits as agreement_splits
from kaggle_task1_repeat_weight_validation import independent_split
from kaggle_task1_v11_repeat_weight_transfer import (POISSON_AMPLIFICATION, decisions, normalized_sources,
                                                      weighted_fusion)
from kaggle_task1_v13_repeat_weight_recheck import independent_outer as prior_outer
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, nested_ranker, rendered_bank


OUTER_SEED = 26371
REPEAT_SEED = 25391
RATER_SEED = 4471
RATER_EPOCHS = 18
COUNT_MATCHES = 8
BATCH = 48
MIN_INFORMATIVE_GLYPHS = 150
WINDOWS = ((50, 100), (100, 250))
OUTPUT = Path("/kaggle/working/task1_poisson_bridge.json")


def fresh_split(count, device):
    cpu = torch.device("cpu")
    _, first, _ = independent_split(count, 6109)
    _, second, _ = independent_split(count, 6947, exclude_first_pilot=True)
    _, selector_fit, selector_held = agreement_splits(count, 18917, 144)
    already_used = [first, second, selector_fit, selector_held, independent_holdout(count, 17641, cpu)[1],
                    fresh_outer(count, 18313, cpu)[1], new_outer(count, 19661, cpu)[1], prior_outer(count, cpu)[1]]
    excluded = np.unique(np.concatenate([rows.numpy() for rows in already_used]))
    available = np.setdiff1d(np.arange(count), excluded)
    if len(available) < 320:
        raise ValueError("Need 320 glyphs outside previous whole-glyph weight cohorts.")
    chosen = np.sort(np.random.default_rng(OUTER_SEED).choice(available, size=320, replace=False))
    fitted = np.setdiff1d(np.arange(count), chosen)
    return torch.as_tensor(fitted, device=device), torch.as_tensor(chosen, device=device), len(excluded)


def save(result, started):
    result["seconds"] = time.monotonic() - started
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def halves(counts, rng):
    order = torch.as_tensor(rng.random(counts.shape[:2]).argsort(axis=1), device=counts.device)
    arranged = counts.gather(1, order[..., None].expand_as(counts))
    return arranged[:, :2], arranged[:, 2:]


@torch.no_grad()
def nominate(model, ranker, likelihood, trials, session, rates, rng):
    unshuffled = []
    for chunk in trials.split(BATCH):
        order = torch.as_tensor(rng.permutation(len(chunk)), device=trials.device)
        base = atlas_scores(model, likelihood, chunk[order], session, rates=rates)
        amended = {**base, **weighted_fusion(model, normalized_sources(base), POISSON_AMPLIFICATION)}
        picks, _ = decisions(base, amended, ranker)
        unshuffled.append(torch.stack((picks["v11"], picks["poisson_same_budget"]))[:, order.argsort()])
    return torch.cat(unshuffled, dim=1)


def band_counts(flat):
    counts = flat.unflatten(-1, (-1, 10))
    return torch.stack([counts[..., start // 25:stop // 25].sum(dim=-1) for start, stop in WINDOWS], dim=-1)


def log_unit_probability(rates):
    log = band_counts(rates.float()).clamp_min(1e-5).log()
    return log - log.logsumexp(dim=-2, keepdim=True)


def matched_exposures(trials):
    """Leave the paired group out of its comparison at similar total counts."""
    counts = band_counts(trials).mean(dim=1)
    total = counts.sum(dim=1).log1p()
    normalized = (total - total.mean(dim=0)) / total.std(dim=0, unbiased=False).clamp_min(0.03)
    distances = torch.cdist(normalized, normalized).fill_diagonal_(torch.inf)
    neighbors = distances.topk(COUNT_MATCHES, dim=1, largest=False).indices
    return counts, counts[neighbors].mean(dim=1)


def score_pair(log_ratio, observed, context):
    matched = (log_ratio * context).sum(dim=(-2, -1))
    raw = (log_ratio * observed).sum(dim=(-2, -1))
    return raw, matched


@torch.no_grad()
def neural_pairs(model, evaluator, noise, ranker, query, held_images, rendered, session, rng):
    half = halves(query, rng)
    with torch.autocast(device_type="cuda"):
        block_rates = model.encoder.expected_counts(model.atlas.images, session, batch_size=192).float()
    choices = torch.stack([nominate(model, ranker, noise, trials, session, block_rates, rng) for trials in half])
    changed = choices[:, 0] != choices[:, 1]

    # The evaluator sees candidate renders and the separately scored true-image
    # control only after both independent spike halves have made their choices.
    images_of_interest = choices.masked_select(changed[:, None].expand_as(choices)).unique()
    candidate_log = None
    if len(images_of_interest):
        candidate_log = log_unit_probability(evaluator.expected_counts(rendered[images_of_interest], session))
    actual_log = log_unit_probability(evaluator.expected_counts(held_images, session))
    signature, energy = contour_signatures(held_images)
    similarity = similarity_from_signatures(signature, energy).fill_diagonal_(-torch.inf)
    # The eighth neighbor is a similar, nonidentical image with a fixed rule.
    distractor = similarity.topk(COUNT_MATCHES, dim=1).indices[:, -1]
    control_log = actual_log - actual_log[distractor]
    controls, ll_real, ll_matched = [], [], []
    for direction, held_trials in enumerate(reversed(half)):
        actual, reference = matched_exposures(held_trials)
        raw_control, matched_control = score_pair(control_log, actual, reference)
        controls.append(torch.stack((raw_control, matched_control)))
        masked = changed[direction]
        margins = query.new_zeros((2, len(query)))
        if bool(masked.any()):
            low, high = torch.searchsorted(images_of_interest, choices[direction][:, masked].contiguous())
            margin = candidate_log[high] - candidate_log[low]
            real, matched = score_pair(margin, actual[masked], reference[masked])
            margins[:, masked] = torch.stack((real, matched))
        ll_real.append(margins[0])
        ll_matched.append(margins[1])
    return choices, torch.stack(ll_real), torch.stack(ll_matched), torch.stack(controls)


@torch.no_grad()
def exact_changed(choices, rendered, held_images, scorer):
    low, high = choices.unbind(dim=1)
    changed = low != high
    direction, row = changed.nonzero(as_tuple=True)
    width = len(rendered)
    keys = torch.stack((row * width + low[direction, row], row * width + high[direction, row]))
    unique, inverse = torch.unique(keys.flatten(), return_inverse=True)
    scores = []
    for chunk in unique.split(768):
        reserve = rendered[chunk.remainder(width)] >= 128
        reference = held_images[chunk.div(width, rounding_mode="floor")] >= 128
        values = scorer(reserve.cpu().numpy(), reference.cpu().numpy(), device=str(rendered.device), batch_size=12)
        scores.append(torch.as_tensor(values, device=rendered.device, dtype=torch.float32))
    result = rendered.new_zeros(low.shape, dtype=torch.float32)
    if len(unique):
        paired = torch.cat(scores)[inverse].reshape(keys.shape)
        result[direction, row] = paired[1] - paired[0]
    return changed, result, len(unique)


def grouped_average(values, selected=None):
    """Treat all held recordings and both directions of one glyph as a cluster."""
    selected = torch.ones_like(values, dtype=torch.bool) if selected is None else selected
    count = selected.sum(dim=0)
    values = (values * selected).sum(dim=0)[count > 0] / count[count > 0]
    if not len(values):
        return {"mean": None, "glyph_se": None, "scored_glyphs": 0, "scored_pairs": 0}
    mean = float(values.mean())
    error = float(values.std(unbiased=True) / len(values)**0.5) if len(values) > 1 else None
    return {"mean": mean, "glyph_se": error, "scored_glyphs": len(values), "scored_pairs": int(selected.sum())}


def report_pairs(changed, ocf, raw, matched, controls):
    likelihood = raw - matched
    informative = changed & (ocf != 0)
    raw_control, comparator = controls.unbind(dim=1)
    contrast = raw_control - comparator
    chosen = likelihood.sign() * ocf.sign()
    concordance = 0.5 + 0.5 * chosen
    return {"selection_changes": int(changed.sum()), "selection_change_rate": float(changed.float().mean()),
            "changed_glyphs": int(changed.any(dim=0).sum()), "ocf_ties_on_changed_rows": int((changed & ~informative).sum()),
            "paired_ocf_delta_all_queries": grouped_average(ocf),
            "paired_ocf_delta_changed_queries": grouped_average(ocf, changed),
            "held_neural_delta_changed_vs_count_matched": grouped_average(likelihood, changed),
            "positive_control_true_minus_similar_glyph_raw": grouped_average(raw_control),
            "positive_control_true_minus_similar_glyph_corrected": grouped_average(contrast),
            "positive_control_correct_sign_rate": grouped_average((contrast > 0).float()),
            "informative_bridge_glyphs": int(informative.any(dim=0).sum()),
            "bridge_concordance": grouped_average(concordance, informative),
            "raw_concordance": grouped_average(0.5 + 0.5 * raw.sign() * ocf.sign(), informative),
            "count_matched_concordance": grouped_average(0.5 + 0.5 * matched.sign() * ocf.sign(), informative),
            "signed_independent_margin": grouped_average(likelihood * ocf.sign(), informative),
            "corrected_when_ocf_rises": grouped_average(likelihood, informative & (ocf > 0)),
            "corrected_when_ocf_falls": grouped_average(likelihood, informative & (ocf < 0))}


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use a private Kaggle T4 with official training data and public atlas access.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fitted, held, earlier = fresh_split(len(images), device)
    groups = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    if tuple(groups) != SESSIONS or any(counts.shape[1] != 4 for counts in groups.values()):
        raise RuntimeError("Require the four complete training sessions and four measured repeats per glyph.")
    training = {name: trials[fitted] for name, trials in groups.items()}
    queries = {name: trials[held] for name, trials in groups.items()}
    del groups
    train_images = images[fitted]
    scorer = official_scorer(root)
    split = split_calibration(len(fitted), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    noise_ids, folds = fold_indices(len(fitted), device)
    if ANCHOR != 4 or not all(torch.equal(a, b) for a, b in zip(split, (folds[0][0], noise_ids, folds[0][1]), strict=True)):
        raise RuntimeError("Expected the literal production fold-zero selection and four-point anchor.")
    _, uncertainty, original_fit, columns = nested_ranker(training, train_images, split, scorer, MODEL_SEED)
    ranker, calibration = fit_crossfit_head(training, train_images, scorer, columns)
    model = fit_v8(training, train_images, MODEL_SEED)
    if not torch.equal(columns, model.codepoints.cpu()):
        raise RuntimeError("Expected the same fixed public atlas column order in all stages.")
    rendered = rendered_bank(model)
    result = {"mouse": "mice1", "outer_seed": OUTER_SEED, "repeat_seed": REPEAT_SEED, "model_seed": MODEL_SEED,
              "evaluator_seed": RATER_SEED, "evaluator_epochs": RATER_EPOCHS, "held_glyphs": len(held),
              "earlier_union_excluded": earlier, "fitted_glyphs": len(fitted), "poisson_weights": [1.0, POISSON_AMPLIFICATION],
              "anchor": ANCHOR, "nominations": NOMINATIONS, "evoked_windows_ms": WINDOWS, "matched_donors": COUNT_MATCHES,
              "minimum_effective_held_glyphs": MIN_INFORMATIVE_GLYPHS, "v13_fit_seconds": time.monotonic() - started,
              "calibration": calibration, "literal_stock_noise_fit": original_fit, "ranker": ranker.diagnostics,
              "method": "The v13 four-fold head changes only Poisson weight and nominations at a fixed candidate budget. "
                        "For each preassigned, disjoint 2/2 split, either half nominates and the other is scored with a "
                        "separately fitted true-image RF. Conditional scores hold fixed the total population count "
                        "within each evoked band and subtract a same-session comparison at similar totals. OCF uses the "
                        "official scorer on changed candidates and zero delta for unchanged candidates. Held images are "
                        "used only in scoring and the true-image positive control. Uncertainty clusters all directions "
                        "and days by glyph. No official test input was accessed.", "days": {}}
    save(result, started)
    print(json.dumps({"stage": "v13_fitted", "seconds": result["seconds"]}), flush=True)

    evaluator = fit_glyph_encoder(training, train_images, epochs=RATER_EPOCHS, seed=RATER_SEED)
    del training
    held_images = images[held]
    results = []
    for index, session in enumerate(SESSIONS):
        rng = np.random.default_rng(np.random.SeedSequence((REPEAT_SEED, index)))
        choices, raw, compare, controls = neural_pairs(model, evaluator, uncertainty[session], ranker, queries[session],
                                                      held_images, rendered, session, rng)
        changed, ocf, exact_pairs = exact_changed(choices, rendered, held_images, scorer)
        parts = changed, ocf, raw, compare, controls
        report = report_pairs(*parts)
        report.update({"exact_pairs": exact_pairs, "by_direction": {name: report_pairs(*(one[row:row + 1] for one in parts))
                       for row, name in enumerate(("a_to_b", "b_to_a"))}})
        results.append(parts)
        result["days"][session] = report
        save(result, started)
        print(json.dumps({"session": session, "new_choices": report["selection_changes"],
                          "bridge_concordance": report["bridge_concordance"], "seconds": result["seconds"]}), flush=True)
    result["paired_summary"] = report_pairs(*(torch.cat(entries, dim=0) for entries in zip(*results)))
    result["meets_effective_sample_target"] = result["paired_summary"]["informative_bridge_glyphs"] >= MIN_INFORMATIVE_GLYPHS
    save(result, started)


if __name__ == "__main__":
    main()
