"""Test temporal candidates with a frozen Task 2 exact-OCF ranker and matched graph size."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import _row_features, concat_atlas_candidates, fit_atlas_metric_ranker
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_mbr import contour_signatures
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import official_scorer
from kaggle_task2_conditional_precision_validation import (WINDOWS, fit_band, fit_window_weights,
                                                           normalize_assignment_scores, paired_results, score_band)
from kaggle_task2_metric_validation import (NOMINATIONS, SESSIONS, metric_labels, score_block, shortlist,
                                            shortlisted_v9_score, sparse_assignment)
from kaggle_task3_metric_validation import unique_pair_ocf


ANCHOR = 8.0
TEMPORAL_FUSION = 0.25
SWAPS = 2


@torch.no_grad()
def global_windows(training, query):
    return torch.stack([score_band(fit_band(training, window, None, 32), query, window, global_only=True)
                        for window in WINDOWS])


@torch.no_grad()
def temporal_calibration(training, query, order, sample):
    windows = global_windows(training, query)[:, order.argsort()[sample]]
    reference = windows.mean(dim=0)
    reference.scatter_(1, sample[:, None], -torch.inf)
    shortlist_indices = torch.cat((sample[:, None], reference.topk(32, dim=1).indices), dim=1)
    return windows.gather(2, shortlist_indices[None].expand(len(WINDOWS), -1, -1)).permute(1, 2, 0)


def fit_source_episodes(mouse, sources, glyph_bank, ocf_glyphs, temporal_glyphs, rng, args):
    images, masks, signature, energy, ink, scorer = glyph_bank
    device = images.device
    ranker_inputs, ranker_ocf, calibration_features, reports = [], [], [], {}
    for simulated, observed in sources.items():
        fitting = {name: counts for name, counts in sources.items() if name != simulated}
        permutation = torch.as_tensor(rng.permutation(len(images)), device=device)
        query = observed[permutation]
        matrices, v9 = score_block(mouse, fitting, query, images, signature, energy, device, args)
        rows = permutation.argsort()[ocf_glyphs]
        candidates = shortlist({name: matrix[rows] for name, matrix in matrices.items()}, v9[rows], energy, ink)
        del matrices, v9
        labels, pairs = metric_labels(candidates, ocf_glyphs, masks, scorer, device, 384)
        ranker_inputs.append(candidates)
        ranker_ocf.append(labels)
        calibration_features.append(temporal_calibration(fitting, query, permutation, temporal_glyphs))
        reports[simulated] = {"template_sessions": list(fitting), "exact_ocf_pairs": pairs,
                              "temporal_calibration_queries": len(temporal_glyphs)}
        print(json.dumps({"mouse": mouse, "source_calibration": simulated, **reports[simulated]}), flush=True)
    ranker = fit_atlas_metric_ranker(concat_atlas_candidates(ranker_inputs), torch.cat(ranker_ocf), seed=args.seed)
    return ranker, fit_window_weights(calibration_features), reports


@torch.no_grad()
def swap_slots(base, matrices, v9_assignment, metric_assignment):
    preserve = torch.cat([matrices[name].topk(NOMINATIONS[name], dim=1).indices for name in ("v9", "validated", "eb")]
                         + [v9_assignment[:, None], metric_assignment[:, None]], dim=1)
    protected = (base.indices[:, :, None] == preserve[:, None]).any(dim=2)
    columns = torch.arange(base.indices.shape[1], device=base.indices.device).expand_as(base.indices)
    evict = columns.masked_fill(protected | ~base.valid, -1).topk(SWAPS, dim=1).values
    return evict


def new_candidates(base, proposal):
    top = proposal.topk(base.indices.shape[1] + SWAPS, dim=1).indices
    old = (top[:, :, None] == base.indices[:, None]).logical_and(base.valid[:, None]).any(dim=2)
    priority = torch.arange(top.shape[1], device=top.device).expand_as(top)
    eligible = priority.masked_fill(old, top.shape[1]).topk(SWAPS, dim=1, largest=False).values
    return top.gather(1, eligible)


@torch.no_grad()
def v11_features(matrices, v9_assignment, energy, ink, proposed, active, base):
    """Score new edges with v11's existing feature definitions and agreements."""
    assigned = torch.full_like(matrices["v9"], -torch.inf)
    assigned.scatter_(1, v9_assignment[:, None], 0.0)
    raw = {**matrices, "v9_assigned": assigned, "contour_energy": energy.log().expand_as(assigned),
           "ink_fraction": ink.expand_as(assigned)}
    depth = base.rank_depth
    features, ranks = [], []
    for name, matrix in sorted(raw.items()):
        values, indexes = matrix.masked_fill(~torch.isfinite(matrix), -torch.inf).topk(depth, dim=1)
        indexes = indexes.masked_fill(~torch.isfinite(values), base.atlas_size)
        fields, rank = _row_features(matrix, proposed, active, values, indexes)
        features.extend(fields)
        if name in NOMINATIONS:
            ranks.append(rank)
        if name == "v9":
            baseline_z = fields[0]
    nominations = torch.stack(ranks)
    features.extend([(nominations <= limit).float().mean(dim=0) for limit in (1, min(5, depth), depth)])
    return torch.stack(features, dim=-1), baseline_z


@torch.no_grad()
def swap_pool(base, evict, proposed, matrices, v9_assignment, energy, ink):
    active = evict >= 0
    fields, z = v11_features(matrices, v9_assignment, energy, ink, proposed, active, base)
    rows, slots = active.nonzero(as_tuple=True)
    columns = evict[rows, slots]
    indices, features, baseline_z = base.indices.clone(), base.features.clone(), base.baseline_z.clone()
    indices[rows, columns] = proposed[rows, slots]
    features[rows, columns] = fields[rows, slots]
    baseline_z[rows, columns] = z[rows, slots]
    return replace(base, indices=indices, features=features, baseline_z=baseline_z)


def exact_assignment(candidates, ranker, anchor, prior):
    expected = ranker.predict_ocf(candidates)
    return sparse_assignment(candidates, expected + ANCHOR * shortlisted_v9_score(anchor, candidates), prior)


def pool_hit(candidates, assignment):
    return (candidates.valid & (candidates.indices == assignment[:, None])).any(dim=1)


def report_pools(pools, old, temporal_proposal, active, choices, truth):
    previous = pool_hit(old, truth)
    intro = (temporal_proposal == choices["temporal_nomination"][:, None]).logical_and(active)
    return {"original_mean_candidates": float(old.valid.sum(dim=1).float().mean()),
            "mean_replacements_per_row": float(active.sum(dim=1).float().mean()),
            "rows_with_replacements": int(active.any(dim=1).sum()),
            "winning_temporal_added_edges": int(intro.any(dim=1).sum()),
            "known_identity_coverage": {name: {"shortlist_hit": float((hit := pool_hit(pool, truth)).float().mean()),
                                               "true_labels_added": int((hit & ~previous).sum()),
                                               "true_labels_removed": int((~hit & previous).sum())}
                                        for name, pool in pools.items()}}


def shortlist_oracle(pools, rows, targets, masks, scorer):
    indices = torch.stack([pool.indices[rows] for pool in pools.values()])
    valid = torch.stack([pool.valid[rows] for pool in pools.values()])
    references = targets[None, :, None].expand_as(indices)
    exact, unique = unique_pair_ocf(indices[valid], references[valid], masks, scorer, masks.device, 384)
    full = exact.new_full(indices.shape, -torch.inf)
    full[valid] = exact
    by_method = full.amax(dim=2).cpu().numpy()
    return {"unique_ocf_pairs": unique, "methods": {name: paired_results(scores, by_method[0])
                                                   for name, scores in zip(pools, by_method)}}


def evaluate(mouse, source, query, order, selected, glyph_bank, ranker, weights, args):
    images, masks, signature, energy, ink, scorer = glyph_bank
    matrices, v9_assignment = score_block(mouse, source, query, images, signature, energy, images.device, args)
    base = shortlist(matrices, v9_assignment, energy, ink)
    v9 = matrices["v9"]
    v11_style, fallback = exact_assignment(base, ranker, v9, v9_assignment)
    global_temporal = (global_windows(source, query) * weights[:, None, None]).sum(dim=0)
    temporal_blend = v9.lerp(normalize_assignment_scores(global_temporal), TEMPORAL_FUSION)
    prior, prior_fallback = exact_assignment(base, ranker, temporal_blend, v11_style)
    choices = {"v9": v9_assignment, "matched_original": v11_style, "original_pool_temporal_prior": prior}
    fallbacks = {"matched_original": fallback, "original_pool_temporal_prior": prior_fallback}
    if not args.prior_only:
        evict = swap_slots(base, matrices, v9_assignment, v11_style)
        temporal_proposal = new_candidates(base, temporal_blend)
        depth_proposal = new_candidates(base, v9)
        pools = {"original": base,
                 "temporal": swap_pool(base, evict, temporal_proposal, matrices, v9_assignment, energy, ink),
                 "v9_depth": swap_pool(base, evict, depth_proposal, matrices, v9_assignment, energy, ink)}
        for pool in pools.values():
            if not bool((pool_hit(pool, v9_assignment) & pool_hit(pool, v11_style)).all()):
                raise ValueError("Every candidate pool must retain both complete baseline assignments.")
            if not torch.equal(pool.valid, base.valid):
                raise ValueError("The candidate count must match in every row.")
        for name in ("temporal", "v9_depth"):
            method = f"{name}_nomination"
            choices[method], fallbacks[method] = exact_assignment(pools[name], ranker, v9, v11_style)
        choices["full_gallery_direct_temporal"] = assign_glyphs(temporal_blend)
    if any(len(choice.unique()) != len(images) for choice in choices.values()):
        raise ValueError("Every experiment arm must make a complete one-to-one assignment.")

    # Only evaluation below consumes the held correspondence.
    rows = order.argsort()[selected]
    predicted = torch.stack([choice[rows] for choice in choices.values()])
    exact, num_pairs = unique_pair_ocf(predicted, selected.expand_as(predicted), masks, scorer, images.device, 384)
    observed = dict(zip(choices, exact.cpu().numpy()))
    v11_arm = "matched_original" if mouse == "mice1" else "v9"
    observed["actual_v11_policy"] = observed[v11_arm]
    metrics = {name: {**paired_results(observed[name], observed["matched_original"]),
                      "paired_delta_vs_v11_policy": float((observed[name] - observed[v11_arm]).mean()),
                      "assignment_top1": float((choice == order).float().mean()),
                      "assignment_changes_vs_matched_original": int((choice != v11_style).sum())}
               for name, choice in choices.items()}
    result = {"metrics": metrics, "unique_ocf_pairs": num_pairs, "actual_v11_policy_arm": v11_arm,
              "sparse_fallback": fallbacks, "complete_assignments": len(choices),
              "original_mean_candidates": float(base.valid.sum(dim=1).float().mean())}
    if not args.prior_only:
        result["pool"] = report_pools(pools, base, temporal_proposal, evict >= 0, choices, order)
        result["shortlist_oracle"] = shortlist_oracle(pools, rows[:args.oracle_rows], selected[:args.oracle_rows], masks, scorer)
    return observed, result


def summarize(measurements):
    values = {name: np.stack(rows) for name, rows in measurements.items()}
    summary = {"vs_matched_original": {name: paired_results(rows, values["matched_original"]) for name, rows in values.items()},
               "prior_vs_actual_v11_policy": paired_results(values["original_pool_temporal_prior"], values["actual_v11_policy"])}
    if "temporal_nomination" in values:
        summary["temporal_vs_matched_v9_depth"] = paired_results(values["temporal_nomination"], values["v9_depth_nomination"])
        summary["temporal_vs_actual_v11_policy"] = paired_results(values["temporal_nomination"], values["actual_v11_policy"])
    return summary


def select_glyphs(count, rng, args):
    order = rng.permutation(count)
    if args.reference_calibration_seed is None:
        calibration = order[args.ocf_images:][:640]
        return order[:args.ocf_images], calibration[:128], calibration
    previous = np.random.default_rng(args.reference_calibration_seed).permutation(count)
    reserved = previous[:128 + 640]
    selected = order[~np.isin(order, reserved)][:args.ocf_images]
    calibration = previous[128:128 + 640]
    if len(selected) != args.ocf_images:
        raise ValueError("The previous scored and calibration cohorts leave too few disjoint evaluation glyphs.")
    return selected, calibration[:128], calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--held-sessions", nargs="+", choices=SESSIONS, default=("session2", "session4"))
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--split-seed", type=int, default=7193)
    parser.add_argument("--reference-calibration-seed", type=int,
                        help="reuse a previous 128/640 calibration split and exclude its first 128 scored glyphs")
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--oracle-rows", type=int, default=12)
    parser.add_argument("--prior-only", action="store_true", help="ablate only the fixed-graph temporal prior")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task2_temporal_nomination_validation.json"))
    parser.set_defaults(encoder_epochs=18, retrieval_epochs=12)
    args = parser.parse_args()
    if (not args.prior_only and not 2 <= args.oracle_rows <= args.ocf_images) or args.ocf_images < 128:
        parser.error("Use at least 128 paired OCF rows and a valid paired oracle subset.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the exact OCF experiment on Kaggle GPU.")

    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    signature, energy = contour_signatures(images)
    masks = images >= 128
    glyph_bank = images, masks, signature, energy, masks.float().mean(dim=(1, 2)), official_scorer(root)
    rng = np.random.default_rng(args.split_seed)
    selected, ocf_fit, temporal_fit = (torch.as_tensor(rows, device=device)
                                       for rows in select_glyphs(len(images), rng, args))
    output = {"seed": args.seed, "split_seed": args.split_seed, "anchor_ocf": ANCHOR,
              "reference_calibration_seed": args.reference_calibration_seed, "prior_only": args.prior_only,
              "nomination_slots_per_row_at_most": 0 if args.prior_only else SWAPS,
              "complete_gallery": len(images), "ocf_rows_per_block": len(selected),
              "baseline_policy": {"mice1": "v11 sparse anchor8 policy refit in outer folds",
                                  "mice2": "v9 full assignment shipped; sparse anchor8 fitted for matched nomination control"},
              "temporal_fusion": TEMPORAL_FUSION, "original_nominations": NOMINATIONS, "mice": {}}
    pooled = {}
    for mouse in args.mice:
        recordings = grouped_blocks(root, catalog, mouse, device, len(images))
        output["mice"][mouse] = record = {"folds": {}}
        measured = {}
        for held in args.held_sessions:
            sources = {name: counts for name, counts in recordings.items() if name != held}
            print(f"{mouse}/{held}: fit OCF ranker and global temporal scores on source episodes", flush=True)
            ranker, weights, calibration = fit_source_episodes(mouse, sources, glyph_bank, ocf_fit, temporal_fit, rng, args)
            order = torch.as_tensor(rng.permutation(len(images)), device=device)
            scores, detail = evaluate(mouse, sources, recordings[held][order], order, selected, glyph_bank, ranker, weights, args)
            record["folds"][held] = {"source_calibration": calibration, "ranker": ranker.diagnostics,
                                     "temporal_weights": weights.tolist(), **detail}
            for name, values in scores.items():
                pooled.setdefault(name, []).append(values)
                measured.setdefault(name, []).append(values)
            record["summary"] = summarize(measured)
            output["summary"] = summarize(pooled)
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"mouse": mouse, "held_session": held, **detail}), flush=True)
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
