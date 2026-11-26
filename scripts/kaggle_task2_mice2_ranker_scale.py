"""Compare matched single-day and three-day exact-OCF fits for mice2 Task 2."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import concat_atlas_candidates, fit_atlas_metric_ranker
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task2_metric_validation import (CALIBRATION_SESSIONS, NOMINATIONS, SESSIONS, metric_labels, score_block,
                                            shortlist, shortlisted_v9_score, sparse_assignment)
from kaggle_task3_metric_validation import unique_pair_ocf


MOUSE = "mice2"
ANCHOR = 8.0
SINGLE_GLYPHS = 96
MULTI_GLYPHS_PER_DAY = 128
ROWS_PER_EPOCH = 96
BATCH_ROWS = 48
EPOCHS = 80
V9 = "v9_actual_full_assignment"
SINGLE = "single_source_96"
MULTI = "three_sources_384"


@dataclass
class Gallery:
    images: torch.Tensor
    masks: torch.Tensor
    signatures: torch.Tensor
    energy: torch.Tensor
    ink: torch.Tensor
    scorer: object

    @property
    def device(self):
        return self.images.device


def previous_mice2_cohorts(count):
    gallery = np.arange(count)
    literal = np.random.default_rng(211)
    exact_scored = literal.choice(count, size=96, replace=False)
    exact_fit = literal.choice(np.setdiff1d(gallery, exact_scored), size=128, replace=False)
    temporal_nomination = np.random.default_rng(7193).permutation(count)[:256]
    peer_first = np.random.default_rng(17691).permutation(count)[:160]
    no_first = np.setdiff1d(gallery, peer_first)
    peer_second = np.random.default_rng(63729).choice(no_first, size=160, replace=False)
    relational = np.random.default_rng(22177).choice(no_first, size=192, replace=False)
    no_peer = np.setdiff1d(no_first, peer_second)
    temporal_first = np.random.default_rng(77317).choice(no_peer, size=160, replace=False)
    newer = np.setdiff1d(no_peer, np.union1d(temporal_first, relational))
    temporal_recheck = np.random.default_rng(23971).choice(newer, size=256, replace=False)
    excluded = np.unique(np.concatenate((exact_scored, exact_fit, temporal_nomination, peer_first, peer_second,
                                         relational, temporal_first, temporal_recheck)))
    return excluded


def samples(count, opts, device):
    earlier = previous_mice2_cohorts(count)
    eligible = np.setdiff1d(np.arange(count), earlier)
    if len(eligible) < opts.ocf_glyphs:
        raise ValueError("Previous Task 2 cohorts leave too few new scoring glyphs.")
    rng = np.random.default_rng(opts.split_seed)
    held = rng.choice(eligible, size=opts.ocf_glyphs, replace=False)
    outside = np.setdiff1d(np.arange(count), held)
    source = rng.choice(outside, size=len(SESSIONS) * MULTI_GLYPHS_PER_DAY, replace=False)
    panels = {name: torch.as_tensor(rows, device=device) for name, rows in zip(SESSIONS, np.split(source, len(SESSIONS)))}
    return torch.as_tensor(held, device=device), panels, len(earlier)


def permute(recording, rng, device):
    order = torch.as_tensor(rng.permutation(len(recording)), device=device)
    return recording[order], order


def score(source, observed, gallery, args):
    return score_block(MOUSE, source, observed, gallery.images, gallery.signatures, gallery.energy,
                       gallery.device, args)


def fit_heads(held, source, panels, gallery, args, rng):
    candidate_parts, target_parts, calibration = {}, {}, {}
    for day, counts in source.items():
        inner = {name: trials for name, trials in source.items() if name != day}
        permuted, order = permute(counts, rng, gallery.device)
        matrices, assignment = score(inner, permuted, gallery, args)
        labels = panels[day]
        rows = order.argsort()[labels]
        graph = shortlist({key: values[rows] for key, values in matrices.items()}, assignment[rows],
                          gallery.energy, gallery.ink)
        targets, exact_pairs = metric_labels(graph, labels, gallery.masks, gallery.scorer, gallery.device, 384)
        candidate_parts[day], target_parts[day] = graph, targets
        calibration[day] = {"source_training_days": list(inner), "glyphs": len(labels), "unique_exact_pairs": exact_pairs,
                            "rowwise_ocf_oracle": float(targets.masked_fill(~graph.valid, -torch.inf).amax(dim=1).mean())}
        print(json.dumps({"held": held, "metric_fit_day": day, **calibration[day]}), flush=True)
        del matrices, assignment

    stock_day = CALIBRATION_SESSIONS[held]
    stock_rows = torch.arange(SINGLE_GLYPHS, device=gallery.device)
    baseline_graph = candidate_parts[stock_day].take_rows(stock_rows)
    baseline_targets = target_parts[stock_day][stock_rows]
    combined = concat_atlas_candidates(list(candidate_parts.values()))
    all_targets = torch.cat(tuple(target_parts.values()))
    fit_options = {"seed": args.seed, "epochs": EPOCHS, "batch_rows": BATCH_ROWS, "rows_per_epoch": ROWS_PER_EPOCH}
    heads = {SINGLE: fit_atlas_metric_ranker(baseline_graph, baseline_targets, **fit_options),
             MULTI: fit_atlas_metric_ranker(combined, all_targets, **fit_options)}
    return heads, {"reference_calibration_day": stock_day, "per_day": calibration,
                   "heads": {name: head.diagnostics for name, head in heads.items()}}


def paired(values, anchor):
    delta = (values - anchor).mean(dim=0)
    return {"ocf": float(values.mean()), "paired_delta": float(delta.mean()),
            "paired_glyph_se": float(delta.std(unbiased=True) / len(delta)**0.5),
            "scored_pairs": values.numel(), "distinct_scored_glyphs": len(delta)}


def comparisons(scores):
    return {"against_v9": {name: paired(values, scores[V9]) for name, values in scores.items()},
            "three_vs_single": paired(scores[MULTI], scores[SINGLE])}


@torch.no_grad()
def evaluate(outer, query_order, matrices, baseline, heads, gallery, opts):
    graph = shortlist(matrices, baseline, gallery.energy, gallery.ink)
    anchor = ANCHOR * shortlisted_v9_score(matrices["v9"], graph)
    choices = {V9: baseline}
    fallback = {}
    for name, head in heads.items():
        objective = head.predict_ocf(graph) + anchor
        choices[name], fallback[name] = sparse_assignment(graph, objective, baseline)
    if any(len(predicted.unique()) != len(query_order) for predicted in choices.values()):
        raise ValueError("Every experiment arm must produce exactly one of each complete-gallery glyph.")

    # Only post-decision evaluation uses the outer recording's held correspondence.
    query_rows = query_order.argsort()[outer]
    guesses = torch.stack([value[query_rows] for value in choices.values()])
    scores, unique = unique_pair_ocf(guesses, outer.expand_as(guesses), gallery.masks, gallery.scorer,
                                   gallery.device, 384)
    measured = dict(zip(choices, scores))
    oracle = {}
    if opts.oracle_glyphs:
        n = opts.oracle_glyphs
        subset = graph.take_rows(query_rows[:n])
        labels, count = metric_labels(subset, outer[:n], gallery.masks, gallery.scorer, gallery.device, 384)
        best = labels.masked_fill(~subset.valid, -torch.inf).amax(dim=1)
        top8 = matrices["v9"][query_rows[:n]].topk(NOMINATIONS["v9"], dim=1).indices
        from_base = (subset.indices[:, :, None] == top8[:, None, :]).any(dim=2) & subset.valid
        best_v9_top8 = labels.masked_fill(~from_base, -torch.inf).amax(dim=1)
        oracle = {"scored_images": n, "unique_pairs": count, "graph_rowwise_oracle": float(best.mean()),
                  "v9_top8_rowwise_oracle": float(best_v9_top8.mean()),
                  "gap_from_each_assignment": {name: paired(best[None], values[None, :n])
                                               for name, values in measured.items()}}
    report = {**comparisons({name: values[None] for name, values in measured.items()}),
              "exact_unique_pairs": unique, "per_query_mean_candidates": float(graph.valid.sum(dim=1).float().mean()),
              "truth_in_shared_candidates": float(((graph.indices == query_order[:, None]) & graph.valid).any(dim=1).float().mean()),
              "retrospective_oracle": oracle,
              "assignments": {name: {"distinct_glyphs": len(values.unique()),
                                     "identity_top1": float((values == query_order).float().mean()),
                                     "changes_from_v9": int((values != baseline).sum()),
                                     "sparse_fallback": fallback.get(name, False)} for name, values in choices.items()}}
    return {name: values.cpu() for name, values in measured.items()}, report


def save(path, result, started):
    result["seconds"] = time.monotonic() - started
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--split-seed", type=int, default=41721)
    parser.add_argument("--ocf-glyphs", type=int, default=256)
    parser.add_argument("--oracle-glyphs", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task2_mice2_ranker_scale.json"))
    parser.set_defaults(encoder_epochs=18, retrieval_epochs=12)
    args = parser.parse_args()
    if args.ocf_glyphs < 192 or not (args.oracle_glyphs == 0 or 2 <= args.oracle_glyphs <= args.ocf_glyphs):
        parser.error("Score at least 192 paired glyphs per held day and a valid retrospective oracle subset.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use one private Kaggle GPU with competition training data mounted.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    device = torch.device("cuda:0")
    images = load_images(root, device)
    masks = images >= 128
    signatures, energy = contour_signatures(images)
    gallery = Gallery(images, masks, signatures, energy, masks.float().mean(dim=(1, 2)), official_scorer(root))
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if tuple(recordings) != SESSIONS or any(group.shape[1] != 4 for group in recordings.values()):
        raise RuntimeError("Expected four complete mice2 training days with four measured repeats per class.")
    outer, panels, excluded = samples(len(images), args, device)
    report = {"mouse": MOUSE, "seed": args.seed, "split_seed": args.split_seed,
              "scored_glyphs_per_day": len(outer), "earlier_calibration_or_scoring_glyphs_excluded": excluded,
              "library_size": len(images), "nominations": NOMINATIONS, "common_anchor": ANCHOR,
              "single_source_glyphs": SINGLE_GLYPHS, "multi_source_glyphs_per_day": MULTI_GLYPHS_PER_DAY,
              "optimizer_epochs": EPOCHS, "rows_per_epoch": ROWS_PER_EPOCH,
              "batch_rows": BATCH_ROWS, "optimizer_updates_per_head": EPOCHS * ROWS_PER_EPOCH // BATCH_ROWS,
              "method": "The v14 mice2 Task 2 incumbent is the full v9 assignment, reproduced on each held "
                        "training day. In each outer fold, all three source days take turns as the labeled OCF "
                        "calibration day, with the other two source dates training the neural score. The single "
                        "head uses 96 glyphs from its predeclared calibration day. The pooled head uses 128 distinct "
                        "glyphs on each of three calibration days. All held OCF glyphs are disjoint from every metric "
                        "calibration panel. Both heads use the same architecture, update count, candidate graph, "
                        "anchor and complete one-to-one objective after the neural model is refit on the three "
                        "source days. The graph contains full v9's solution. The rowwise oracle is evaluated "
                        "only after assignment and does not enforce one-to-one. Formal test blocks are unused.",
              "days": {}}
    results = {name: [] for name in (V9, SINGLE, MULTI)}
    for day in SESSIONS:
        fold_time = time.monotonic()
        source = {name: trials for name, trials in recordings.items() if name != day}
        rng = np.random.default_rng(np.random.SeedSequence((args.split_seed, SESSIONS.index(day))))
        heads, calibration = fit_heads(day, source, panels, gallery, args, rng)
        query, order = permute(recordings[day], rng, device)
        matrices, baseline = score(source, query, gallery, args)
        measured, scores = evaluate(outer, order, matrices, baseline, heads, gallery, args)
        report["days"][day] = {"outer_training_days": list(source), "calibration": calibration, **scores,
                               "seconds": time.monotonic() - fold_time}
        for name, values in measured.items():
            results[name].append(values)
        report["paired_summary"] = comparisons({name: torch.stack(values) for name, values in results.items()})
        save(args.output, report, started)
        print(json.dumps({"held": day, "three_vs_single": scores["three_vs_single"],
                          "vs_v9": scores["against_v9"][MULTI], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
