"""Recheck the fixed 1.4× Poisson weight with the v13 four-fold Task 1 head."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import pack_atlas_candidates
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_task1_crossfit_candidate import fit_crossfit_head
from kaggle_task1_crossfit_metric_scale import fresh_outer, fold_indices
from kaggle_task1_eight_fold_metric_scale import new_outer
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE, independent_holdout
from kaggle_task1_repeat_agreement import splits as agreement_splits
from kaggle_task1_repeat_weight_validation import independent_split
from kaggle_task1_v11_repeat_weight_transfer import POISSON_AMPLIFICATION, decisions, normalized_sources, pooled_scores, weighted_fusion
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, nested_ranker, rendered_bank, summarize


OUTER_SEED = 22943
REPEAT_SEED = 22949
OUTER_GLYPHS = 320
BATCH = 48
POLICY = "v13"
ARMS = (POLICY, "poisson_same_pool", "poisson_proposals_only", "poisson_same_budget")
OUTPUT = Path("/kaggle/working/task1_v13_repeat_weight_recheck.json")


def stage(name, started, **fields):
    print(json.dumps({"stage": name, "seconds": round(time.monotonic() - started, 2), **fields}), flush=True)


def independent_outer(n, device):
    cpu = torch.device("cpu")
    _, first_weight, _ = independent_split(n, 6109)
    _, second_weight, _ = independent_split(n, 6947, exclude_first_pilot=True)
    _, agreement_cal, agreement_outer = agreement_splits(n, 18917, 144)
    groups = {"repeat_weight_6109": first_weight, "repeat_weight_6947": second_weight,
              "repeat_agreement_18917": agreement_outer, "repeat_agreement_calibration": agreement_cal,
              "metric_scale_17641": independent_holdout(n, 17641, cpu)[1],
              "four_fold_18313": fresh_outer(n, 18313, cpu)[1], "eight_fold_19661": new_outer(n, 19661, cpu)[1]}
    excluded = np.unique(np.concatenate([values.numpy() for values in groups.values()]))
    eligible = np.setdiff1d(np.arange(n), excluded)
    if len(eligible) < OUTER_GLYPHS:
        raise RuntimeError("The predeclared previous cohorts leave fewer than 320 independent glyphs.")
    outer = np.sort(np.random.default_rng(OUTER_SEED).permutation(eligible)[:OUTER_GLYPHS])
    overlap = {name: int(np.intersect1d(values.numpy(), outer).size) for name, values in groups.items()}
    if any(overlap.values()):
        raise RuntimeError("A new held glyph belongs to a prior weight or calibration-scale experiment.")
    fit = np.setdiff1d(np.arange(n), outer)
    report = {"fit_glyphs": len(fit), "held_glyphs": len(outer), "eligible_before_sampling": len(eligible),
              "overlap_with_previous_cohorts": overlap, "previous_union_excluded": len(excluded)}
    return torch.as_tensor(fit, device=device), torch.as_tensor(outer, device=device), report


def original_head_matches(model, original, sources, ranker):
    reconstructed = {**original, **weighted_fusion(model, sources, 1.0)}
    names = ("v4", "v8")
    drift = {name: float((original[name] - reconstructed[name]).abs().max()) for name in names}
    literal = pack_atlas_candidates(original, NOMINATIONS, baseline="v8")
    rebuilt = pack_atlas_candidates(reconstructed, NOMINATIONS, baseline="v8")
    identical = torch.equal(literal.indices, rebuilt.indices) and torch.equal(literal.valid, rebuilt.valid)
    selected = int((ranker.select(literal, anchor_ocf=ANCHOR) != ranker.select(rebuilt, anchor_ocf=ANCHOR)).sum())
    if not identical or selected or max(drift.values()) > 2e-5:
        raise RuntimeError("The reconstructed weight-one anchor does not match the literal production proposal graph.")
    return {"max_full_atlas_logit_error": drift, "same_v13_candidate_graph": identical, "changed_v13_choices": selected}


def choices_and_metadata(base, amplified, ranker):
    selected, details = decisions(base, amplified, ranker)
    selected = {POLICY if name == "v11" else name: choices for name, choices in selected.items()}
    return selected, {"candidates_matched": details["same_unique_count_every_row"],
                      "mean_candidates": details["mean_candidates_per_glyph"],
                      "mean_replaced": details["mean_replaced_candidates"],
                      "rows_with_replacements": details["rows_with_replacements"],
                      "changes_vs_v13": {name: int((choice != selected[POLICY]).sum()) for name, choice in selected.items()}}


def add_sources(total, incoming, reorder):
    for name, source in incoming.items():
        values = source[reorder]
        if name in total:
            total[name].add_(values)
        else:
            total[name] = values


def keep(accumulator, selected):
    for name, values in selected.items():
        accumulator.setdefault(name, []).append(values)


def summarize_meta(parts):
    counts = sum(part["rows"] for part in parts)
    return {"candidate_count_matched_every_query": all(part["candidates_matched"] for part in parts),
            "mean_candidates": sum(part["mean_candidates"] * part["rows"] for part in parts) / counts,
            "mean_replaced": sum(part["mean_replaced"] * part["rows"] for part in parts) / counts,
            "rows_with_replacements": sum(part["rows_with_replacements"] for part in parts),
            "changes_vs_v13": {name: sum(part["changes_vs_v13"][name] for part in parts) for name in ARMS}}


@torch.no_grad()
def score_queries(model, ranker, noise, queries, started):
    device = model.atlas.images.device
    rng = np.random.default_rng(REPEAT_SEED)
    length = len(next(iter(queries.values())))
    repeat_order = {name: torch.as_tensor(rng.random(counts.shape[:2]).argsort(axis=1), device=device)
                    for name, counts in queries.items()}
    with torch.autocast(device_type="cuda"):
        rates = {name: model.encoder.expected_counts(model.atlas.images, name, batch_size=192).float() for name in SESSIONS}

    groups = (*SESSIONS, "pooled16", *(f"matched_four_{index}" for index in range(4)))
    all_choices = {name: {} for name in groups}
    metadata = {name: [] for name in groups}
    anchor_checks = {}

    for start in range(0, length, BATCH):
        end = min(start + BATCH, length)
        size = end - start
        sixteen, matched = {}, ({}, {}, {}, {})
        for name, counts in queries.items():
            order = torch.as_tensor(rng.permutation(size), device=device)
            reorder = order.argsort()
            block = counts[start:end]
            shuffled = block.gather(1, repeat_order[name][start:end, :, None].expand_as(block))[order]
            original = atlas_scores(model, noise[name], shuffled, name, rates=rates[name])
            signals = normalized_sources(original)
            if start == 0:
                anchor_checks[name] = original_head_matches(model, original, signals, ranker)
            amplified = {**original, **weighted_fusion(model, signals, POISSON_AMPLIFICATION)}
            selected, details = choices_and_metadata(original, amplified, ranker)
            keep(all_choices[name], {arm: picks[reorder] for arm, picks in selected.items()})
            metadata[name].append({"rows": size, **details})
            add_sources(sixteen, signals, reorder)
            del original, signals, amplified

            for fold, evidence in enumerate(matched):
                one = atlas_scores(model, noise[name], shuffled[:, fold:fold + 1], name, rates=rates[name])
                add_sources(evidence, normalized_sources(one), reorder)

        for name, sources in (("pooled16", sixteen), *((f"matched_four_{index}", sums) for index, sums in enumerate(matched))):
            maps = {key: value / len(SESSIONS) for key, value in sources.items()}
            base, amended = pooled_scores(model, maps)
            selected, details = choices_and_metadata(base, amended, ranker)
            keep(all_choices[name], selected)
            metadata[name].append({"rows": size, **details})
        stage("prediction_chunk", started, completed_glyphs=end, total_glyphs=length)

    return ({name: {arm: torch.cat(picks) for arm, picks in rows.items()} for name, rows in all_choices.items()},
            {name: summarize_meta(parts) for name, parts in metadata.items()}, anchor_checks)


def compare(scores):
    return {name: summarize(values, scores[POLICY]) for name, values in scores.items()}


def combine(scores, blocks):
    return {arm: [scores[name][arm] for name in blocks] for arm in ARMS}


def extra_gain(scores, blocks):
    full = scores["pooled16"]
    return {arm: summarize([full[arm] - full[POLICY]],
                           [np.stack([scores[g][arm] - scores[g][POLICY] for g in blocks]).mean(axis=0)])
            for arm in ARMS if arm != POLICY}


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use a private Kaggle T4 with training data and the public font available.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, held, partition = independent_outer(len(images), device)
    groups = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    if tuple(groups) != SESSIONS or any(counts.shape[1] != 4 for counts in groups.values()):
        raise RuntimeError("Expected four recordings with four measured repeats per glyph.")
    queries = {name: counts[held] for name, counts in groups.items()}
    training = {name: counts[fit] for name, counts in groups.items()}
    del groups

    scorer = official_scorer(root)
    original_split = split_calibration(len(fit), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    noise_rows, four_folds = fold_indices(len(fit), device)
    fold_zero = (four_folds[0][0], noise_rows, four_folds[0][1])
    if not all(torch.equal(a, b) for a, b in zip(original_split, fold_zero, strict=True)) or ANCHOR != 4.0:
        raise RuntimeError("The literal production fold-zero split or OCF anchor changed.")
    _, noise, production_fit, codepoints = nested_ranker(training, images[fit], original_split, scorer, MODEL_SEED)
    ranker, folds = fit_crossfit_head(training, images[fit], scorer, codepoints)
    model = fit_v8(training, images[fit], MODEL_SEED)
    del training
    if not torch.equal(codepoints, model.codepoints.cpu()):
        raise RuntimeError("Expected the original public atlas column ordering in all models.")
    rendered = rendered_bank(model)
    stage("v13_fitted", started, folds=len(folds), atlas=len(rendered), **partition)

    report = {"protocol": {"mouse": "mice1", "glyph_seed": OUTER_SEED, "partition_seed": REPEAT_SEED,
                           "model_seed": MODEL_SEED, "poisson_weight_base": 1.0, "poisson_weight_variant": POISSON_AMPLIFICATION,
                           "anchor_ocf": ANCHOR, "nominations": NOMINATIONS, "formal_test_accessed": False,
                           "known_same_glyph_four_day_pooling": True,
                           "repeat_control": "The within-day 4-repeat result averages four separate day decisions. The same-day-coverage "
                                             "4-repeat control randomly partitions the 16 measured trials into four disjoint sets, "
                                             "each with one trial from each day. The 16-repeat query is the union, so differences "
                                             "are paired by glyph rather than independent 16- and 4-repeat groups.",
                           "policy": "The same v13 four-fold head, exact OCF labels, original fold-zero noise, same public atlas "
                                     "and rendered candidates are fixed. Only the standardized Poisson component of the v4 mix "
                                     "is multiplied by 1.4, affecting both v4 and v8. The primary test changes candidate scores "
                                     "and nominates at the same count; unchanged pools and proposal-only results diagnose the mechanism."},
              "partition": partition, "fold_zero_split_matches_production": True,
              "literal_v11_upstream_fit_for_production_noise": production_fit,
              "calibration_folds": folds, "crossfit_ranker": ranker.diagnostics, "outer_model": model.report}
    report["seconds"] = time.monotonic() - started
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    selections, metadata, anchor_checks = score_queries(model, ranker, noise, queries, started)
    report["baseline_weight_one_checks"] = anchor_checks
    report["conditions"] = {}
    exact = PairedOCF(scorer, rendered, (images[held] >= 128).cpu().numpy(), device, 12)
    measured = {}
    for name, block in selections.items():
        measured[name] = {arm: exact.evaluate(ids[:, None])[:, 0] for arm, ids in block.items()}
        outcome = compare({arm: [value] for arm, value in measured[name].items()})
        report["conditions"][name] = {"ocf": outcome, **metadata[name]}
        report["seconds"] = time.monotonic() - started
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        stage("exact_ocf", started, condition=name, **outcome)

    random_four = tuple(f"matched_four_{index}" for index in range(4))
    report.update({"single_day_four": compare(combine(measured, SESSIONS)),
                   "matched_day_coverage_four": compare(combine(measured, random_four)),
                   "poisson_gain_16_minus_mean_single_day_four": extra_gain(measured, SESSIONS),
                   "poisson_gain_16_minus_mean_matched_four": extra_gain(measured, random_four),
                   "exact_selected_image_pairs": len(exact.values), "seconds": time.monotonic() - started})
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    stage("complete", started, pooled16=report["conditions"]["pooled16"]["ocf"],
          single_day_four=report["single_day_four"], matched_four=report["matched_day_coverage_four"])


if __name__ == "__main__":
    main()
