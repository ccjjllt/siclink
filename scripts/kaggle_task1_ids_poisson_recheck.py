"""Recheck Task 1 IDS nominations and their interaction with fixed 1.4x Poisson."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import describe_atlas_candidates, pack_atlas_candidates
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_gallery_validation import standardized
from kaggle_task1_crossfit_candidate import fit_crossfit_head
from kaggle_task1_crossfit_metric_scale import fresh_outer
from kaggle_task1_eight_fold_metric_scale import new_outer
from kaggle_task1_ids_retrieval import (ADD, HELD_COUNT, IDS_PROPOSAL_WEIGHT, MARGIN, MOUSE, QUALITY,
                                       component_dictionary, ids_scores, learn_component_scores, matched_training,
                                       projected_counts, replaced_pool, split_glyphs, visual_atlas)
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE, independent_holdout
from kaggle_task1_poisson_bridge import fresh_split as bridge_split
from kaggle_task1_repeat_agreement import splits as agreement_splits
from kaggle_task1_repeat_weight_validation import independent_split
from kaggle_task1_repertoire_coverage import shape_matches
from kaggle_task1_v11_repeat_weight_transfer import (POISSON_AMPLIFICATION, matched_pool, normalized_sources,
                                                      pooled_scores, progress, weighted_fusion)
from kaggle_task1_v13_repeat_weight_recheck import independent_outer as prior_outer
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, nested_ranker, rendered_bank, summarize


SEED = 36371
OUTPUT = Path("/kaggle/working/task1_ids_poisson_recheck.json")


def new_holdout(count, device):
    cpu = torch.device("cpu")
    _, first, _ = independent_split(count, 6109)
    _, second, _ = independent_split(count, 6947, exclude_first_pilot=True)
    _, selector_fit, selector_held = agreement_splits(count, 18917, 144)
    previous = {
        "poisson_initial_6109": first,
        "poisson_replication_6947": second,
        "poisson_agreement_18917_fit": selector_fit,
        "poisson_agreement_18917_held": selector_held,
        "metric_scale_17641": independent_holdout(count, 17641, cpu)[1],
        "four_fold_18313": fresh_outer(count, 18313, cpu)[1],
        "eight_fold_19661": new_outer(count, 19661, cpu)[1],
        "poisson_v13_recheck": prior_outer(count, cpu)[1],
        "poisson_bridge": bridge_split(count, cpu)[1],
        "ids_primary_33643": split_glyphs(count, cpu)[1],
    }
    excluded = np.unique(np.concatenate([rows.numpy() for rows in previous.values()]))
    eligible = np.setdiff1d(np.arange(count), excluded)
    if len(eligible) < HELD_COUNT:
        raise ValueError(f"Only {len(eligible)} glyphs are outside all specified earlier cohorts; need {HELD_COUNT}.")
    held = np.sort(np.random.default_rng(SEED).choice(eligible, size=HELD_COUNT, replace=False))
    counts = {name: {"excluded": len(rows), "overlap": len(np.intersect1d(held, rows.numpy()))}
              for name, rows in previous.items()}
    fit = np.setdiff1d(np.arange(count), held)
    return torch.as_tensor(fit, device=device), torch.as_tensor(held, device=device), {
        "eligible": len(eligible), "unique_exclusions": len(excluded), "cohorts": counts}


@torch.no_grad()
def decisions(original, amplified, component_scores, ranker):
    base = pack_atlas_candidates(original, NOMINATIONS, baseline="v8")
    higher = pack_atlas_candidates(amplified, NOMINATIONS, baseline="v8")
    indices, valid, novel_counts = matched_pool(base, higher)
    weighted = describe_atlas_candidates(amplified, NOMINATIONS, base.baseline, base.rank_depth, indices, valid)
    base_rank = ranker.rank(base, anchor_ocf=ANCHOR)
    weighted_rank = ranker.rank(weighted, anchor_ocf=ANCHOR)
    pools = {
        "v14": base,
        "v15_poisson": weighted,
        "ids_nomination": replaced_pool(base, base_rank, original,
                                        standardized(original["v8"]) + IDS_PROPOSAL_WEIGHT * component_scores),
        "joint": replaced_pool(weighted, weighted_rank, amplified,
                               standardized(amplified["v8"]) + IDS_PROPOSAL_WEIGHT * component_scores),
        "v14_depth_control": replaced_pool(base, base_rank, original, original["v8"]),
        "v15_depth_control": replaced_pool(weighted, weighted_rank, amplified, amplified["v8"]),
    }
    picks = {name: ranker.select(pool, anchor_ocf=ANCHOR) for name, pool in pools.items()}
    for name, pool in pools.items():
        budget = pool.valid.sum(dim=1)
        if not torch.equal(budget, base.valid.sum(dim=1)) or bool((picks[name] < 0).any()):
            raise RuntimeError(f"Unequal candidate budget or no selected glyph in {name}.")
    for name, incumbent in (("ids_nomination", "v14"), ("joint", "v15_poisson"),
                            ("v14_depth_control", "v14"), ("v15_depth_control", "v15_poisson")):
        pool, old = pools[name], pools[incumbent]
        shared = ((pool.indices[:, :, None] == old.indices[:, None, :]) & old.valid[:, None, :]).any(dim=2)
        retains_winner = ((pool.indices == picks[incumbent][:, None]) & pool.valid).any(dim=1)
        if bool((((~shared & pool.valid).sum(dim=1) != ADD) | ~retains_winner).any()):
            raise RuntimeError(f"{name} must add exactly {ADD} new edges while retaining the incumbent selection.")
    return picks, pools, {"mean_unique_budget": float(valid.sum(dim=1).float().mean()),
                          "poisson_mean_new_edges": float(novel_counts.float().mean()),
                          "poisson_rows_with_replacement": int((novel_counts > 0).sum())}


def score(exact, original, amplified, components, ranker, proxy, trusted):
    picks, pools, audit = decisions(original, amplified, components, ranker)
    values = {name: exact.evaluate(selected[:, None])[:, 0] for name, selected in picks.items()}
    for name, pool in pools.items():
        recalled = ((pool.indices == proxy[:, None]) & pool.valid).any(dim=1)
        audit[name] = {"changed_vs_v14": int((picks[name] != picks["v14"]).sum()),
                       "changed_vs_v15": int((picks[name] != picks["v15_poisson"]).sum()),
                       "nearest_rendering_proxy_recall": float(recalled[trusted].float().mean()) if bool(trusted.any()) else None}
    return values, audit


def compare(arms):
    result = {name: summarize(values, arms["v14"]) for name, values in arms.items()}
    for name, baseline in (("joint", "v15_poisson"), ("joint", "ids_nomination"),
                           ("ids_nomination", "v14_depth_control"), ("joint", "v15_depth_control")):
        result[f"{name}_vs_{baseline}"] = summarize(arms[name], arms[baseline])
    values = np.stack(arms["joint"]) - np.stack(arms["v15_poisson"])
    unweighted = np.stack(arms["ids_nomination"]) - np.stack(arms["v14"])
    result["interaction_joint_minus_additive_gains"] = summarize(values, unweighted)
    return result


def save(report, started):
    report["seconds"] = time.monotonic() - started
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle T4 and official training data, with pinned public font and IDS access.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, held, exclusions = new_holdout(len(images), device)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if tuple(recordings) != SESSIONS or any(block.shape[1] != 4 for block in recordings.values()):
        raise ValueError("Require four measured repeats and four labeled training recordings.")
    training = {day: trials[fit] for day, trials in recordings.items()}
    query = {day: trials[held] for day, trials in recordings.items()}
    del recordings
    report = {"protocol": {"mouse": MOUSE, "held_glyphs": len(held), "glyph_seed": SEED,
                           "model_seed": MODEL_SEED, "fit_glyphs": len(fit), "exclusions": exclusions,
                           "proposal_slots": ADD, "ids_proposal_weight": IDS_PROPOSAL_WEIGHT,
                           "poisson_multiplier": POISSON_AMPLIFICATION, "anchor_ocf": ANCHOR,
                           "affinity_style_visibility": "fit glyphs only", "formal_input_accessed": False,
                           "primary": "Four real repeats within each recording; paired by glyph across all four recordings.",
                           "secondary": "Sixteen measured repeats pooled across days using known labeled co-index only. "
                                        "Normalize neural scores within day, average, and apply the glyph prior once."},
              "recordings": {}}
    save(report, started)
    progress("split", started, **exclusions)
    scorer = official_scorer(root)
    split = split_calibration(len(fit), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    _, noise, _, codes = nested_ranker(training, images[fit], split, scorer, MODEL_SEED)
    progress("noise_fit", started)
    ranker, folds = fit_crossfit_head(training, images[fit], scorer, codes)
    progress("metric_fit", started, folds=folds)
    model = fit_v8(training, images[fit], MODEL_SEED)
    if not torch.equal(model.codepoints.cpu(), codes):
        raise ValueError("The rate, ranker, style and IDS models need one common public atlas order.")
    bank = rendered_bank(model)
    atlas = visual_atlas(bank)
    entries, provenance = component_dictionary(codes)
    fit_matches = shape_matches(atlas, images[fit].cpu())
    rows, targets, _, components, confident = matched_training(fit_matches, entries, images[fit], device)
    training_response = {day: projected_counts(model.contour, trials[rows], day) for day, trials in training.items()}
    learned = learn_component_scores(training_response, targets)
    del training, training_response
    held_matches = shape_matches(atlas, images[held].cpu())
    del atlas
    proxy = held_matches.closest.to(device)
    trusted = held_matches.confident(QUALITY, MARGIN).to(device)
    report.update({"public_atlas_size": len(bank), "ids_source": provenance, "training_folds": folds,
                   "fit_confident_public_code_proxy": confident, "fit_ids_rows": len(rows),
                   "features_with_fit_support": components.shape[1], "held_proxy_confident_rows": int(trusted.sum())})
    save(report, started)
    progress("ids_fit", started, fit_ids_rows=len(rows), feature_count=components.shape[1])

    exact = PairedOCF(scorer, bank, (images[held] >= 128).cpu().numpy(), device, 12)
    measured, accumulated, accumulated_ids = {}, {}, None
    for day in SESSIONS:
        counts = query[day]
        original = atlas_scores(model, noise[day], counts, day)
        sources = normalized_sources(original)
        amplified = {**original, **weighted_fusion(model, sources, POISSON_AMPLIFICATION)}
        component_scores = ids_scores(learned, projected_counts(model.contour, counts, day), components)
        values, audit = score(exact, original, amplified, component_scores, ranker, proxy, trusted)
        for name, measure in values.items():
            measured.setdefault(name, []).append(measure)
        for name, source in sources.items():
            if name in accumulated:
                accumulated[name].add_(source)
            else:
                accumulated[name] = source
        if accumulated_ids is None:
            accumulated_ids = component_scores
        else:
            accumulated_ids.add_(component_scores)
        report["recordings"][day] = {"ocf": compare({name: [row] for name, row in values.items()}), **audit}
        report["four_repeats"] = compare(measured)
        save(report, started)
        progress("four_repeats", started, session=day, ocf=report["recordings"][day]["ocf"])
        del original, amplified, sources, component_scores

    ordinary, higher = pooled_scores(model, {name: source / len(SESSIONS) for name, source in accumulated.items()})
    supported = components.sum(dim=1) > 0
    accumulated_ids /= len(SESSIONS)
    accumulated_ids[:, supported] = standardized(accumulated_ids[:, supported]).clamp(-4, 4)
    values, audit = score(exact, ordinary, higher, accumulated_ids, ranker, proxy, trusted)
    report["sixteen_repeats_known_coindex"] = {"ocf": compare({name: [row] for name, row in values.items()}), **audit}
    report["unique_exact_selected_image_pairs"] = len(exact.values)
    report["completed"] = True
    save(report, started)
    progress("complete", started, four=report["four_repeats"], sixteen=report["sixteen_repeats_known_coindex"]["ocf"])


if __name__ == "__main__":
    main()
