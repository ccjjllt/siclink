"""Transfer the fixed 1.4× Task 1 Poisson mix to the frozen exact v11 policy."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import _deduplicate, describe_atlas_candidates, pack_atlas_candidates
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_gallery_validation import standardized
from kaggle_task1_repeat_agreement import splits
from kaggle_v8_metric_validation import (ATLAS_WEIGHT, CONTOUR_WEIGHT, NOMINATIONS, SESSIONS, atlas_scores, fit_v8,
                                         nested_ranker, rendered_bank, summarize)


MODEL_SEED = 211
OUTER_SEED = 18917
OUTER_PAIRS = 144
EXCLUDED_AFFINITY_CALIBRATION = 144
POISSON_AMPLIFICATION = 1.40
ANCHOR_OCF = 4.0
OUTPUT = Path("/kaggle/working/task1_v11_repeat_weight_transfer.json")
RAW_SOURCES = ("poisson_rate", "poisson_timed", "decoded_contour", "ridge_contour_raw", "predictive_error", "noise_raw")


def progress(stage, started, **fields):
    print(json.dumps({"stage": stage, "seconds": round(time.monotonic() - started, 2), **fields}), flush=True)


@torch.no_grad()
def normalized_sources(scores):
    result = {key: standardized(scores[key]) for key in RAW_SOURCES}
    result["poisson_mix"] = standardized(0.5 * (result["poisson_rate"] + result["poisson_timed"]))
    return result


@torch.no_grad()
def weighted_fusion(model, sources, poisson_weight):
    v4 = standardized(0.5 * (sources["decoded_contour"] + poisson_weight * sources["poisson_mix"]))
    v8 = ATLAS_WEIGHT * v4 + CONTOUR_WEIGHT * sources["ridge_contour_raw"]
    return {key: model.prior.variants(score)["usage_unseen"] for key, score in (("v4", v4), ("v8", v8))}


@torch.no_grad()
def pooled_scores(model, sources):
    shape = sources["decoded_contour"]
    noise_blend = 0.5 * (shape + sources["noise_raw"])
    common = {name: sources[name] for name in RAW_SOURCES}
    common.update({"ridge_contour": model.prior.variants(sources["ridge_contour_raw"])["usage_unseen"],
                   "noise": model.prior.variants(noise_blend)["usage_unseen"],
                   "lexical": model.prior.lexical_log.expand_as(shape), "unseen": model.prior.unseen_log.expand_as(shape)})
    return {**common, **weighted_fusion(model, sources, 1.0)}, {**common, **weighted_fusion(model, sources, POISSON_AMPLIFICATION)}


@torch.no_grad()
def matched_pool(original, proposed):
    # Keep every shared proposal, including all proposals from unchanged sources.
    common = ((proposed.indices[:, :, None] == original.indices[:, None]) & original.valid[:, None]).any(dim=2)
    sentinel = original.atlas_size
    shared = proposed.indices.masked_fill(~(common & proposed.valid), sentinel)
    fresh = proposed.indices.masked_fill(~(~common & proposed.valid), sentinel)
    fill = original.indices.masked_fill(~original.valid, sentinel)
    indices, valid = _deduplicate(torch.cat((shared, fresh, fill), dim=1), sentinel)
    width = original.indices.shape[1]
    indices, valid = indices[:, :width], valid[:, :width] & original.valid
    indices = indices.masked_fill(~valid, 0)
    if not torch.equal(valid.sum(dim=1), original.valid.sum(dim=1)):
        raise RuntimeError("Renomination changed the number of distinct candidates in a row.")
    novel = ((indices[:, :, None] != original.indices[:, None]) | ~original.valid[:, None]).all(dim=2) & valid
    return indices, valid, novel.sum(dim=1)


@torch.no_grad()
def decisions(original_scores, amplified_scores, ranker):
    original = pack_atlas_candidates(original_scores, NOMINATIONS, baseline="v8")
    amended = pack_atlas_candidates(amplified_scores, NOMINATIONS, baseline="v8")
    ids, valid, novel = matched_pool(original, amended)
    unchanged_features = describe_atlas_candidates(amplified_scores, NOMINATIONS, original.baseline,
                                                  original.rank_depth, original.indices, original.valid)
    original_features = describe_atlas_candidates(original_scores, NOMINATIONS, original.baseline,
                                                 original.rank_depth, ids, valid)
    amended_features = describe_atlas_candidates(amplified_scores, NOMINATIONS, original.baseline,
                                                original.rank_depth, ids, valid)
    views = {"v11": original, "poisson_same_pool": unchanged_features,
             "poisson_proposals_only": original_features, "poisson_same_budget": amended_features}
    chosen = {name: ranker.select(view, anchor_ocf=ANCHOR_OCF) for name, view in views.items()}
    diagnostics = {"same_unique_count_every_row": True, "mean_candidates_per_glyph": float(valid.sum(dim=1).float().mean()),
                   "mean_replaced_candidates": float(novel.float().mean()), "rows_with_replacements": int((novel > 0).sum()),
                   "changes_vs_v11": {key: int((value != chosen["v11"]).sum()) for key, value in chosen.items()}}
    return chosen, diagnostics


def score_selected(exact, chosen):
    return {name: exact.evaluate(indices[:, None])[:, 0] for name, indices in chosen.items()}


def compare(ocf):
    result = {name: summarize(values, ocf["v11"]) for name, values in ocf.items()}
    result["poisson_same_budget_vs_same_pool"] = summarize(ocf["poisson_same_budget"], ocf["poisson_same_pool"])
    return result


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the v11 transfer on Kaggle T4 with competition data and public font downloads enabled.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, earlier_weight_calibration, holdout = splits(len(images), OUTER_SEED, EXCLUDED_AFFINITY_CALIBRATION)
    fit, scored = fit.to(device), holdout[:OUTER_PAIRS].to(device)
    groups = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    if tuple(groups) != SESSIONS or any(group.shape[1] != 4 for group in groups.values()):
        raise ValueError("Require exactly the original four recording sessions and four measured repeats per glyph per session.")
    training = {day: group[fit] for day, group in groups.items()}
    query = {day: group[scored] for day, group in groups.items()}
    del groups
    scorer = official_scorer(root)
    split = split_calibration(len(fit), 96, 384, device, MODEL_SEED)
    ranker, likelihood, nested, nested_codes = nested_ranker(training, images[fit], split, scorer, MODEL_SEED)
    outer = fit_v8(training, images[fit], MODEL_SEED)
    del training
    if not torch.equal(nested_codes, outer.codepoints.cpu()):
        raise RuntimeError("The nested and outer public atlas columns must coincide.")
    rendered = rendered_bank(outer)
    progress("fixed_v11_fit", started, fit_glyphs=len(fit), metric_glyphs=len(split[0]), atlas_columns=len(rendered))

    report = {"protocol": {"mouse": "mice1", "glyph_seed": OUTER_SEED, "model_seed": MODEL_SEED,
                           "same_144_glyphs_as_repeat_agreement_1": True, "paired_per_budget": len(scored),
                           "outer_fit_glyphs": len(fit), "metric_calibration_glyphs": len(split[0]),
                           "noise_calibration_glyphs": len(split[1]), "prior_weight_calibration_glyphs_excluded": len(earlier_weight_calibration),
                           "session_repeats": 4, "known_correspondence_pooled_repeats": 16,
                           "poisson_multiplier": POISSON_AMPLIFICATION, "v11_anchor_ocf": ANCHOR_OCF,
                           "v11_nominations": NOMINATIONS, "actual_test_accessed": False,
                           "method": "The fixed ranker, noise calibration, full public atlas and styled output are shared. "
                                     "Only the Poisson coefficient inside v4 changes, including its v8 contribution. "
                                     "The matched re-nomination preserves every shared proposal and fills any shortfall "
                                     "from the original list at exactly the same per-row distinct candidate count. "
                                     "The pooled proxy aligns known held training glyphs in four recordings, averages within-day "
                                     "standardized components before fusion, and adds the fixed atlas prior once."},
              "ranker": ranker.diagnostics, "nested_fit": nested, "outer_fit": outer.report, "recordings": {}}
    exact = PairedOCF(scorer, rendered, (images[scored] >= 128).cpu().numpy(), device, 12)
    rng = np.random.default_rng(OUTER_SEED + 101)
    pooled, per_day = {}, {}

    for session, counts in query.items():
        permutation = torch.as_tensor(rng.permutation(len(counts)), device=device)
        inverse = permutation.argsort()
        base = atlas_scores(outer, likelihood[session], counts[permutation], session)
        sources = normalized_sources(base)
        amplified = {**base, **weighted_fusion(outer, sources, POISSON_AMPLIFICATION)}
        choices, diagnostics = decisions(base, amplified, ranker)
        values = score_selected(exact, {name: selected[inverse] for name, selected in choices.items()})
        for name, scores in values.items():
            per_day.setdefault(name, []).append(scores)
        for name, values_in_day in sources.items():
            vote = values_in_day[inverse]
            if name in pooled:
                pooled[name].add_(vote)
            else:
                pooled[name] = vote
        report["recordings"][session] = {"ocf": compare({name: [value] for name, value in values.items()}), **diagnostics}
        report["seconds"] = time.monotonic() - started
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        progress("four_repeats", started, recording=session, **report["recordings"][session])
        del base, sources, amplified

    pooled = {name: value / len(SESSIONS) for name, value in pooled.items()}
    base, amplified = pooled_scores(outer, pooled)
    chosen, diagnostics = decisions(base, amplified, ranker)
    pooled_result = score_selected(exact, chosen)
    report["single_recording_four_repeats"] = compare(per_day)
    report["sixteen_repeats_known_cross_recording_correspondence"] = {
        "ocf": compare({name: [value] for name, value in pooled_result.items()}), **diagnostics}
    report["poisson_increment_sixteen_vs_mean_four"] = {
        name: summarize([value - pooled_result["v11"]],
                        [(np.stack(per_day[name]) - np.stack(per_day["v11"])).mean(axis=0)])
        for name, value in pooled_result.items() if name != "v11"}
    report["exact_selected_image_pairs"] = len(exact.values)
    report["seconds"] = time.monotonic() - started
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    progress("complete", started, four=report["single_recording_four_repeats"],
             sixteen=report["sixteen_repeats_known_cross_recording_correspondence"])


if __name__ == "__main__":
    main()
