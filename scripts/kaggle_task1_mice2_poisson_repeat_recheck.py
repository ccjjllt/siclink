"""Recheck 1.4× Poisson evidence on independent mice2 Task 1 glyphs."""

from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_task1_bipartite_validation import fraction, image_proxy
from kaggle_task1_crossfit_candidate import fit_crossfit_head
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE
from kaggle_task1_mice2_incumbent_crossfit import independent_split
from kaggle_task1_mice2_microns_spatial_atlas import fresh_glyphs as spatial_atlas_split
from kaggle_task1_microns_open_atlas import publish
from kaggle_task1_microns_repeat_recheck import accumulate
from kaggle_task1_microns_spatial_readout import new_whole_glyphs as closed_spatial_split
from kaggle_task1_v11_repeat_weight_transfer import (ANCHOR_OCF, POISSON_AMPLIFICATION, decisions, normalized_sources,
                                                      pooled_scores, score_selected, weighted_fusion)
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, nested_ranker, rendered_bank, summarize


SEED = 97139
MOUSE = "mice2"
HOLDOUT = 320
BASE = "v14"
FULL = "poisson_same_budget"
FIXED = "poisson_same_pool"
OUTPUT = Path("/kaggle/working/task1_mice2_poisson_repeat_recheck.json")


def fresh_holdout(size, device):
    cpu = torch.device("cpu")
    previous = {"mice2_microns_spatial_atlas": spatial_atlas_split(size, cpu)[1].numpy(),
                "closed_spatial": closed_spatial_split(size, cpu)[1].numpy(),
                "mice2_incumbent_calibration": independent_split(size, 19391, cpu)[1].numpy()}
    excluded = np.unique(np.concatenate(tuple(previous.values())))
    eligible = np.setdiff1d(np.arange(size), excluded)
    held = np.sort(np.random.default_rng(SEED).choice(eligible, HOLDOUT, replace=False))
    overlap = {name: int(np.intersect1d(held, earlier).size) for name, earlier in previous.items()}
    if any(overlap.values()):
        raise ValueError("The evaluation cohort must be independent of all three earlier reference cohorts.")
    fit = np.setdiff1d(np.arange(size), held)
    return torch.as_tensor(fit, device=device), torch.as_tensor(held, device=device), overlap


def compare(values):
    contrasts = {name: summarize(score, values[BASE]) for name, score in values.items()}
    contrasts["same_budget_vs_same_pool"] = summarize(values[FULL], values[FIXED])
    return contrasts


@torch.no_grad()
def evaluate(original, amplified, ranker, image_score, visual_proxy, trusted, inverse):
    chosen, matched = decisions(original, amplified, ranker)
    chosen[BASE] = chosen.pop("v11")
    result = score_selected(image_score, {name: index[inverse] for name, index in chosen.items()})
    recall = {}
    for name, features in ((BASE, original), (FULL, amplified)):
        top = features["v8"].topk(32, dim=1).indices
        correct = top == visual_proxy[:, None]
        recall[name] = {f"top_{depth}": fraction(correct[:, :depth].any(1), trusted) for depth in (1, 5, 16, 32)}
        recall[name]["chosen"] = fraction(chosen[name] == visual_proxy, trusted)
    return result, {"recall": recall, "same_unique_count_every_row": matched["same_unique_count_every_row"],
                    "mean_candidates_per_glyph": matched["mean_candidates_per_glyph"],
                    "mean_replaced_candidates": matched["mean_replaced_candidates"],
                    "rows_with_replacements": matched["rows_with_replacements"],
                    "changed_predictions": {name: int((index != chosen[BASE]).sum()) for name, index in chosen.items()}}


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run on one private Kaggle T4 with labeled competition input and public fonts.")
    if ANCHOR != ANCHOR_OCF:
        raise RuntimeError("The fixed equal-budget policy must use the exact v14 OCF anchor.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, held, overlap = fresh_holdout(len(images), device)
    groups = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if tuple(groups) != SESSIONS or any(count.shape[1] != 4 for count in groups.values()):
        raise ValueError("Require four labeled mice2 recordings with four measured repeats per glyph.")
    training = {day: counts[fit] for day, counts in groups.items()}
    queries = {day: counts[held] for day, counts in groups.items()}
    del groups

    report = {"mouse": MOUSE, "outer_seed": SEED, "model_seed": MODEL_SEED, "held_glyphs": len(held),
              "fit_glyphs": len(fit), "prior_overlap": overlap, "anchor_ocf": ANCHOR, "nominations": NOMINATIONS,
              "poisson_multiplier": POISSON_AMPLIFICATION, "sessions": {},
              "protocol": "Train one exact mice2 v14 four-fold ranker and standard atlas strictly outside the 320 held glyphs. "
                          "Compare the original neural readout with its Poisson contribution multiplied by 1.4, sharing the model, "
                          "rendering, and per-row distinct candidate budget. Fixed-pool and proposals-only contrasts use the same "
                          "raw neural scores and separate rescoring from new nominations. The pooled diagnostic averages scores "
                          "from four held recordings of four trials using known correspondence, which is unavailable at official "
                          "inference. References are used only for exact OCF and a post hoc atlas-recall proxy. No formal test "
                          "recording and no MICrONS features are used."}
    publish(report, OUTPUT, started, "source_verified")
    scorer = official_scorer(root)
    split = split_calibration(len(fit), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    _, noise, nested, codes = nested_ranker(training, images[fit], split, scorer, MODEL_SEED)
    ranker, folds = fit_crossfit_head(training, images[fit], scorer, codes)
    model = fit_v8(training, images[fit], MODEL_SEED)
    del training
    if not torch.equal(codes, model.codepoints.cpu()):
        raise ValueError("The v14 calibration folds and the outer model must use identical atlas columns.")
    report.update(atlas_glyphs=len(codes), calibration_folds=folds, ranker=ranker.diagnostics, v14_nested_fit=nested)
    publish(report, OUTPUT, started, "v14_ready")

    proxy, trusted = image_proxy(model, images[held])
    report["trusted_proxy_glyphs"] = int(trusted.sum())
    exact = PairedOCF(scorer, rendered_bank(model), (images[held] >= 128).cpu().numpy(), device, 12)
    rng = np.random.default_rng(SEED + 2)
    total, measured = {}, {}
    for day, counts in queries.items():
        order = torch.as_tensor(rng.permutation(len(held)), device=device)
        reverse = order.argsort()
        original = atlas_scores(model, noise[day], counts[order], day)
        sources = normalized_sources(original)
        amplified = {**original, **weighted_fusion(model, sources, POISSON_AMPLIFICATION)}
        outcome, detail = evaluate(original, amplified, ranker, exact, proxy[order], trusted[order], reverse)
        accumulate(total, sources, reverse)
        for name, values in outcome.items():
            measured.setdefault(name, []).append(values)
        report["sessions"][day] = {"ocf": compare({name: [value] for name, value in outcome.items()}), **detail}
        report["four_repeat_ocf"] = compare(measured)
        del original, sources, amplified
        publish(report, OUTPUT, started, day)

    original, amplified = pooled_scores(model, {key: value / len(SESSIONS) for key, value in total.items()})
    outcome, detail = evaluate(original, amplified, ranker, exact, proxy, trusted, torch.arange(len(held), device=device))
    report["known_correspondence_16"] = {"ocf": compare({name: [value] for name, value in outcome.items()}), **detail}
    report["delta_repeat_interaction"] = summarize([outcome[FULL] - outcome[BASE]],
                                                    [(np.stack(measured[FULL]) - np.stack(measured[BASE])).mean(0)])
    report["unique_exact_image_pairs"] = len(exact.values)
    publish(report, OUTPUT, started, "complete")


if __name__ == "__main__":
    main()
