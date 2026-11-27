"""Separate candidate coverage and selection headroom in the v20 Task 3 policy."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_mbr import contour_signatures, glyph_posterior
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import standard_error
from kaggle_posterior_utility_validation import fit_neural, v11_logits
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_mice2_full_balance_replication import cohorts, split_trials
from kaggle_task3_mice2_utility_candidate import MOUSE, SHORTLIST, TABLE_SHA, validated_utility
from kaggle_task3_native_hanzi_nomination import new_trials as prior_native_trials
from kaggle_task3_rankband_ocf_recheck import gather_repeats
from kaggle_task3_rankband_ocf_validation import NeuralRows
from kaggle_task3_relational_consensus import INDEPENDENT, UTILITY_WEIGHT, choose_images
from kaggle_task3_repeat_posterior import expected_utility


SEED = 340741
DAYS = ("session1", "session3")
GLYPHS_PER_DAY = 48
DEPTH = 16
RAW_KEEP = 12
NOMINATION_DEPTH = 64
OUTPUT = Path("/kaggle/working/task3_v20_candidate_headroom.json")


def held_repeats(recordings, prior_glyphs):
    """Use neither the previous replication repeat nor the OCR nomination repeat."""
    ids = {day: torch.arange(4).view(1, 4, 1).expand(len(counts), -1, -1)
           for day, counts in recordings.items()}
    prior_replication = split_trials(ids, prior_glyphs)[3]
    prior_ocr = prior_native_trials(ids, prior_glyphs)[1]
    support, queries, checks = {}, {}, {}
    for offset, (day, counts) in enumerate(recordings.items()):
        index = ids[day][..., 0]
        blocked = (index == prior_replication[day][..., 0]) | (index == prior_ocr[day][..., 0])
        rng = torch.Generator().manual_seed(SEED + offset)
        held = torch.rand(index.shape, generator=rng).masked_fill(blocked, -1).argmax(dim=1, keepdim=True)
        rest = index[index != held].reshape(len(counts), 3)
        previous_overlap = int(blocked.gather(1, held).sum())
        support_overlap = int((rest == held).sum())
        if previous_overlap or support_overlap:
            raise RuntimeError("A query trial entered its calibration or matched an earlier query split.")
        support[day], queries[day] = gather_repeats(counts, rest), gather_repeats(counts, held)
        checks[day] = {"previous_query_overlap": previous_overlap, "support_overlap": support_overlap}
    return support, queries, checks


def standardized_from_original(values, floor):
    original = values[:, :SHORTLIST]
    center = original.mean(dim=1, keepdim=True)
    scale = (original - center).square().mean(dim=1, keepdim=True).sqrt().clamp_min(floor)
    return (values - center) / scale


def make_actions(logits, utility, temperature):
    posterior = glyph_posterior(logits, temperature, balance=False)
    leader = logits.argmax(dim=1, keepdim=True)
    tail = logits.clone().scatter_(1, leader, -torch.inf).topk(NOMINATION_DEPTH - 1, dim=1).indices
    ordered = torch.cat((leader, tail), dim=1)
    original = ordered[:, :SHORTLIST]
    incumbent = choose_images(NeuralRows(logits, logits, original), {INDEPENDENT: posterior}, utility)[INDEPENDENT]
    anchor = logits.gather(1, ordered)
    excess = expected_utility(posterior, ordered, utility) - utility.mean(dim=1)[ordered]
    score = (1 - UTILITY_WEIGHT) * standardized_from_original(anchor, 0.30)
    score += UTILITY_WEIGHT * standardized_from_original(excess, 0.50)
    if not torch.equal(incumbent, original.gather(1, score[:, :SHORTLIST].argmax(dim=1, keepdim=True))[:, 0]):
        raise RuntimeError("Frozen scaling must reproduce the original v20 decision.")
    novel_count = DEPTH - RAW_KEEP
    extra = score[:, DEPTH:].topk(novel_count, dim=1).indices + DEPTH
    combined = torch.cat((ordered[:, :DEPTH], ordered.gather(1, extra)), dim=1)
    mixed = torch.cat((ordered[:, :RAW_KEEP], ordered.gather(1, extra)), dim=1)
    # Keep every original action score and only add opportunities to select another image.
    expanded_score = torch.cat((score[:, :RAW_KEEP], score.gather(1, extra)), dim=1)
    selected = mixed.gather(1, expanded_score.argmax(dim=1, keepdim=True))[:, 0]
    return combined, incumbent, selected


def selected_value(scores, candidates, chosen):
    corresponding = candidates == chosen[:, None]
    if not bool((corresponding.sum(1) == 1).all()):
        raise RuntimeError("Every chosen candidate must appear exactly once in the shared exact-scoring batch.")
    return scores[corresponding]


def exact_result(candidates, truth, incumbent, selected, masks, scorer, device):
    exact, unique = unique_pair_ocf(candidates, truth[:, None].expand_as(candidates), masks, scorer, device, 96)
    original = exact[:, :SHORTLIST]
    raw = exact[:, :DEPTH]
    mixed = torch.cat((raw[:, :RAW_KEEP], exact[:, DEPTH:]), dim=1)
    scores = {"actual_v20": selected_value(exact, candidates, incumbent),
              "original_7_oracle": original.amax(1), "raw_16_oracle": raw.amax(1),
              "mixed_16_oracle": mixed.amax(1), "frozen_scale_mixed_16": selected_value(exact, candidates, selected)}
    return scores, unique


def summarize(values):
    comparisons = {"selection_headroom_within_7": ("original_7_oracle", "actual_v20"),
                   "extra_coverage_raw_16": ("raw_16_oracle", "original_7_oracle"),
                   "extra_coverage_mixed_16": ("mixed_16_oracle", "original_7_oracle"),
                   "mixed_versus_raw_16_coverage": ("mixed_16_oracle", "raw_16_oracle"),
                   "expanded_actual_gain": ("frozen_scale_mixed_16", "actual_v20"),
                   "remaining_selection_headroom_mixed_16": ("mixed_16_oracle", "frozen_scale_mixed_16")}
    paired = {}
    for name, (left, right) in comparisons.items():
        delta = (values[left] - values[right]).numpy()
        paired[name] = {"delta": float(delta.mean()), "paired_se": standard_error(delta),
                        "fraction_with_gain_ge_5": float((delta >= 5).mean())}
    return {"paired_glyphs": len(values["actual_v20"]),
            "mean_ocf": {name: float(scores.mean()) for name, scores in values.items()}, "comparisons": paired}


def publish(report, started, stage):
    report.update(stage=stage, seconds=time.monotonic() - started)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{stage} after {report['seconds']:.1f}s", flush=True)


@torch.no_grad()
def main():
    if not Path("/kaggle/input").is_dir() or not OUTPUT.parent.is_dir():
        raise RuntimeError("Use private Kaggle CPU with official training and the verified utility source mounted.")
    device = torch.device("cpu")
    torch.set_num_threads(4)
    started = time.monotonic()
    rng = np.random.default_rng(SEED)
    root = locate_data()
    images = load_images(root, device)
    masks = images >= 128
    utility, temperatures, _ = validated_utility(images, device)
    visual_fit, previous, temperature_glyphs, _ = cohorts(len(images), device)
    sampled = torch.from_numpy(rng.choice(previous.numpy(), len(DAYS) * GLYPHS_PER_DAY, replace=False))
    if torch.isin(sampled, torch.cat((visual_fit, temperature_glyphs))).any():
        raise RuntimeError("No scoring glyph may enter the pinned visual utility or temperature fitting cohort.")
    references = dict(zip(DAYS, sampled.split(GLYPHS_PER_DAY), strict=True))
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images), bin_ms=25)
    train, query, overlap = held_repeats(recordings, previous)
    del recordings
    with torch.enable_grad():
        model = fit_neural(train, images, *contour_signatures(images), MOUSE, 211)
    del train
    result = {"mouse": MOUSE, "seed": SEED, "utility_table_sha256": TABLE_SHA, "gallery_glyphs": len(images),
              "held_repeats_per_query": 1, "source_repeats_per_glyph": 3, "held_sessions": list(DAYS),
              "held_glyphs_per_session": GLYPHS_PER_DAY, "repeat_overlap": overlap, "original_width": SHORTLIST,
              "expanded_width": DEPTH, "near_candidate_depth": NOMINATION_DEPTH, "mixed_preserved_raw_ranks": RAW_KEEP,
              "mixed_new_ranks": [DEPTH + 1, NOMINATION_DEPTH],
              "protocol": "Reproduce v20-inherited mice2 Task 3 utility, temperatures, and neural recipe on "
                          "three labeled support repeats. Rank held single trials. Each scored glyph belongs to the "
                          "prior 256-glyph evaluation cohort, outside pinned utility and temperature fits. The "
                          "scoring days use disjoint glyphs and every queried repeat is new relative to the earlier "
                          "mice2 balance and native OCR experiments. Compare the original seven to the first 16 raw "
                          "ranks and to a matched-width pool retaining ranks 1–12 plus four high fixed-utility ranks "
                          "from raw ranks 17–64. Normalize every decision against original seven action values, and "
                          "allow only the expanded action set to vary in the practical policy. Official true images "
                          "are used for retrospective oracles and paired OCF. All neural spikes come from training.",
              "recordings": {}}
    decisions = {}
    for day in DAYS:
        ordering = torch.from_numpy(rng.permutation(len(images)))
        truth = references[day]
        rows = ordering.argsort()[truth]
        matrix = v11_logits(model, query[day][ordering], day, images)
        raw_hit = matrix.topk(32, dim=1).indices == ordering[:, None]
        candidates, incumbent, updated = make_actions(matrix[rows], utility, temperatures[day]["temperature"])
        decisions[day] = candidates, truth, incumbent, updated
        match = candidates == truth[:, None]
        result["recordings"][day] = {"paired_indices": truth.tolist(),
                                      "full_gallery_recall": {str(k): float(raw_hit[:, :k].any(1).float().mean())
                                                              for k in (1, 7, 16, 32)},
                                      "paired_recall": {"original_7": float(match[:, :SHORTLIST].any(1).float().mean()),
                                                        "raw_16": float(match[:, :DEPTH].any(1).float().mean()),
                                                        "mixed_16": float(torch.cat((match[:, :RAW_KEEP], match[:, DEPTH:]),
                                                                                    dim=1).any(1).float().mean())},
                                      "policy_changes": int((incumbent != updated).sum())}
        publish(result, started, f"{day}_ranks")
        print(json.dumps(result["recordings"][day]), flush=True)

    scorer = official_scorer(root)
    measurements = []
    for day, decision in decisions.items():
        exact, pairs = exact_result(*decision, masks, scorer, device)
        measurements.append(exact)
        result["recordings"][day].update(exact_ocf=summarize(exact), official_ocf_pairs=pairs)
        publish(result, started, f"{day}_ocf")
        print(json.dumps(result["recordings"][day]["exact_ocf"]), flush=True)
    result["pooled"] = summarize({arm: torch.cat([m[arm] for m in measurements]) for arm in measurements[0]})
    publish(result, started, "complete")
    print(json.dumps(result["pooled"]), flush=True)


if __name__ == "__main__":
    main()
