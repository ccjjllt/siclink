"""Test frozen OCR nominations within the seven-action mice2 Task 3 policy."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_bipartite_encoding import timing_view
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_mbr import contour_signatures, glyph_posterior
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import standard_error
from kaggle_posterior_utility_validation import fit_neural, v11_logits
from kaggle_task1_native_hanzi import OCR_CHECKSUM, OCR_REVISION, OCR_SOURCE, PC_COUNT, RIDGE
from kaggle_task1_native_hanzi import descriptors, feature_extractors, fitted_pca
from kaggle_task3_half_balance_recheck import cohorts as half_cohorts
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_mice2_full_balance_replication import (capacity_cohorts, cohorts as previous_cohort, early_cohorts,
                                                         prior_ocf_cohorts, scoring_glyphs, split_trials as previous_trials)
from kaggle_task3_mice2_utility_candidate import MOUSE, SHORTLIST, TABLE_SHA, validated_utility
from kaggle_task3_part_discovery import standardize_rows
from kaggle_task3_rankband_ocf_recheck import gather_repeats
from kaggle_task3_rankband_ocf_validation import NeuralRows
from kaggle_task3_relational_consensus import INDEPENDENT, UTILITY_WEIGHT, choose_images


SEED = 312599
V19 = "v19"
EVALUATE_DAYS = ("session1", "session3")
SHAPE_WEIGHT = 0.25
NEW_SLOTS = 2
ORACLE_COUNT = 16
OUTPUT = Path("/kaggle/working/task3_native_hanzi_nomination.json")


def fresh_glyphs(size, device):
    visual, previous_held, calibration, diagnostics = previous_cohort(size, device)
    _, half, _ = half_cohorts(size, device)
    _, _, capacity = capacity_cohorts(size, device)
    _, early = early_cohorts(size, device)
    _, _, earlier_exact, _ = prior_ocf_cohorts(size, device)
    older = (visual, half[MOUSE][0], half[MOUSE][1], scoring_glyphs(size, 128)[0], earlier_exact,
             capacity, early, torch.from_numpy(np.random.default_rng(1531).permutation(size)[2432:2560]),
             torch.from_numpy(np.random.default_rng(352).permutation(size)[192:288]))
    available = torch.arange(size)[~torch.isin(torch.arange(size), torch.cat(older))]
    if len(available) != diagnostics["eligible_glyphs"]:
        raise RuntimeError("The pinned baseline cohort and its earlier exclusions must agree.")
    fresh = available[~torch.isin(available, torch.cat((previous_held, calibration)))]
    return fresh, previous_held, size - len(fresh)


def new_trials(recordings, legacy_held):
    indices = {day: torch.arange(4).view(1, 4, 1).expand(len(counts), -1, -1)
               for day, counts in recordings.items()}
    _, _, _, old_query, _ = previous_trials(indices, legacy_held)
    support, queries = {}, {}
    for offset, (day, counts) in enumerate(recordings.items()):
        ids = indices[day][..., 0]
        old = old_query[day][..., 0]
        rng = torch.Generator().manual_seed(SEED + offset)
        draw = torch.rand(ids.shape, generator=rng).masked_fill(ids == old, -1)
        query_id = draw.argmax(dim=1, keepdim=True)
        support_ids = ids[ids != query_id].reshape(len(counts), 3)
        if (old == query_id).any() or (support_ids == query_id).any():
            raise RuntimeError("Query repeats must be disjoint from training and the previous probe.")
        support[day], queries[day] = gather_repeats(counts, support_ids), gather_repeats(counts, query_id)
    return support, queries


def image_codes(images, scored):
    fit = torch.arange(len(images))[~torch.isin(torch.arange(len(images)), scored)]
    order = torch.cat((fit, scored))
    codes = {name: fitted_pca(features, len(fit))[order.argsort()] for name, features in
             descriptors(images[order], feature_extractors()).items()}
    return fit, codes


def shape_scores(support, query, fit, codes):
    samples = timing_view(support[fit].unflatten(-1, (-1, 10))).flatten(0, 1)
    observations = timing_view(query.unflatten(-1, (-1, 10)))[:, 0]
    center, scale = samples.mean(0), samples.std(0).clamp_min(0.04)
    design = (samples - center) / scale
    test = (observations - center) / scale
    gram = design.T @ design
    gram.diagonal().add_(RIDGE * len(design))
    mapping = torch.cholesky_solve(design.T, torch.linalg.cholesky(gram))
    return {name: F.normalize(test @ (mapping @ target[fit].repeat_interleave(support.shape[1], dim=0)))
            @ F.normalize(target).T for name, target in codes.items()}


def nominate(logits, morphology, baseline, incumbent):
    positions = torch.arange(SHORTLIST).expand_as(baseline)
    others = positions.masked_fill(baseline == incumbent[:, None], SHORTLIST).topk(
        SHORTLIST - NEW_SLOTS - 1, largest=False).values
    kept = torch.cat((incumbent[:, None], baseline.gather(1, others)), dim=1)
    row_z = standardize_rows(logits)
    scores = {"deepening": row_z, **{name: row_z + SHAPE_WEIGHT * standardize_rows(values)
                                      for name, values in morphology.items()}}
    pools = {V19: baseline}
    for name, values in scores.items():
        new = values.clone().scatter_(1, baseline, -torch.inf).topk(NEW_SLOTS, dim=1).indices
        pools[name] = torch.cat((kept, new), dim=1)
    if any((values.sort(dim=1).values.diff(dim=1) == 0).any() for values in pools.values()):
        raise RuntimeError("Every policy must retain seven distinct gallery candidates.")
    return pools, scores


def select_candidates(logits, posterior, action, utility):
    return choose_images(NeuralRows(logits, logits, action), {INDEPENDENT: posterior}, utility)[INDEPENDENT]


def rank_stats(pools, scores, order, sample):
    output = {}
    for name, pool in pools.items():
        found = (pool == order[:, None]).any(1)
        rank = scores[name].topk(32, dim=1).indices == order[:, None]
        output[name] = {"gallery_top32": float(rank.any(1).float().mean()), "seven_action_recall": float(found.float().mean()),
                        "paired_glyph_seven_action_recall": float(found[sample].float().mean()),
                        "new_truth_vs_v19": int((found & ~(pools[V19] == order[:, None]).any(1)).sum()),
                        "lost_truth_vs_v19": int((~found & (pools[V19] == order[:, None]).any(1)).sum())}
    return output


def paired(values, anchor):
    delta = (values - anchor).numpy()
    return {"ocf": float(values.mean()), "paired_delta": float(delta.mean()), "paired_se": standard_error(delta),
            "scored_glyphs": len(values)}


def summarize(values):
    measured = {name: paired(scores, values[V19]) for name, scores in values.items()}
    for control in ("deepening", "contour", "random_mobilenet"):
        measured[f"ocr_vs_{control}"] = paired(values["frozen_hanzi"], values[control])
    return measured


def publish(report, started, stage):
    report.update(stage=stage, seconds=time.monotonic() - started)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{stage} after {report['seconds']:.1f}s", flush=True)


@torch.no_grad()
def main():
    if not Path("/kaggle/input").is_dir() or not OUTPUT.parent.is_dir():
        raise RuntimeError("Use private Kaggle CPU with official training and the verified utility source mounted.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    root = locate_data()
    images = load_images(root, device)
    masks = images >= 128
    scored, prior, excluded = fresh_glyphs(len(images), device)
    utility, temperatures, _ = validated_utility(images, device)
    print(f"Verified v19-inherited utility with {len(scored)} fresh scoring glyphs", flush=True)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images), bin_ms=25)
    support, queries = new_trials(recordings, prior)
    del recordings
    shape_fit, codes = image_codes(images, scored)
    with torch.enable_grad():
        model = fit_neural(support, images, *contour_signatures(images), MOUSE, 211)
    report = {"mouse": MOUSE, "seed": SEED, "held_days": list(EVALUATE_DAYS), "gallery": len(images),
              "fresh_ocf_glyphs_per_day": len(scored), "excluded_earlier_glyphs": excluded,
              "shape_fit_glyphs": len(shape_fit), "shape_fit_repeats": 3, "query_repeats": 1,
              "candidate_budget": SHORTLIST, "new_slots": NEW_SLOTS, "shape_nomination_weight": SHAPE_WEIGHT,
              "fixed_utility_weight": UTILITY_WEIGHT, "utility_table_sha256": TABLE_SHA,
              "fixed_temperature_by_day": {day: temperatures[day]["temperature"] for day in EVALUATE_DAYS},
              "ocr": {"checkpoint": OCR_SOURCE, "revision": OCR_REVISION, "sha256": OCR_CHECKSUM,
                      "license": "Apache-2.0", "pca_components": PC_COUNT, "ridge_per_trial": RIDGE},
              "protocol": "The v19-inherited mice2 Task 3 policy uses an identical calibrated utility table and "
                          "temperature, refitting its neural model on three labeled repeats per day. The single "
                          "query repeat is disjoint from those and from the previous confirmation. Each new pool "
                          "protects the v19 winner and the four other highest incumbent candidates, then proposes "
                          "two images outside the original seven. The unchanged v19 posterior, utility, and scope "
                          "select from each seven-image pool. Adapted shape heads and PCA exclude the 47 exact-scored "
                          "reference glyphs, while the stock v19 model may use their other labeled repeats as usual. "
                          "All query spike recordings are independent shuffled draws from the official training set.",
              "recordings": {}}
    evaluations = {}
    for day in EVALUATE_DAYS:
        order = torch.from_numpy(rng.permutation(len(images)))
        spikes = queries[day][order]
        source = v11_logits(model, spikes, day, images)
        posterior = glyph_posterior(source, temperatures[day]["temperature"], balance=False)
        leader = source.argmax(dim=1, keepdim=True)
        following = source.clone().scatter_(1, leader, -torch.inf).topk(SHORTLIST - 1, dim=1).indices
        original = torch.cat((leader, following), dim=1)
        incumbent = select_candidates(source, posterior, original, utility)
        morphology = shape_scores(support[day], spikes, shape_fit, codes)
        pools, scores = nominate(source, morphology, original, incumbent)
        rows = order.argsort()[scored]
        rank = rank_stats(pools, {V19: source, **scores}, order, rows)
        chosen = {name: select_candidates(source[rows], posterior[rows], pool[rows], utility) for name, pool in pools.items()}
        report["recordings"][day] = {"rank": rank, "scored_changed_from_v19": {name: int((selected != chosen[V19]).sum())
                                                                             for name, selected in chosen.items()}}
        evaluations[day] = (chosen, {name: labels[rows] for name, labels in pools.items()})
        publish(report, started, f"{day}_ranks")
        print(json.dumps(rank), flush=True)

    scorer = official_scorer(root)
    actual, oracle = {}, {}
    for day, (chosen, pools) in evaluations.items():
        choices = torch.stack(tuple(chosen.values()))
        exact, pairs = unique_pair_ocf(choices, scored.expand_as(choices), masks, scorer, device, 96)
        actual[day] = dict(zip(chosen, exact, strict=True))
        report["recordings"][day].update(actual_ocf=summarize(actual[day]), actual_ocf_pairs=pairs)
        publish(report, started, f"{day}_ocf")
        print(json.dumps(report["recordings"][day]["actual_ocf"]), flush=True)
    oracle_rows = torch.from_numpy(rng.choice(len(scored), ORACLE_COUNT, replace=False))
    for day, (_, pools) in evaluations.items():
        actions = torch.stack([values[oracle_rows] for values in pools.values()])
        references = scored[oracle_rows, None].expand_as(actions)
        exact, pairs = unique_pair_ocf(actions, references, masks, scorer, device, 96)
        oracle[day] = dict(zip(pools, exact.amax(-1), strict=True))
        report["recordings"][day].update(oracle_ocf=summarize(oracle[day]), oracle_ocf_pairs=pairs)
        publish(report, started, f"{day}_oracle")
        print(json.dumps(report["recordings"][day]["oracle_ocf"]), flush=True)
    for name, measurements in (("actual_ocf", actual), ("oracle_ocf", oracle)):
        arms = next(iter(measurements.values()))
        report[name] = summarize({arm: torch.stack([row[arm] for row in measurements.values()]).mean(0) for arm in arms})
    publish(report, started, "complete")


if __name__ == "__main__":
    main()
