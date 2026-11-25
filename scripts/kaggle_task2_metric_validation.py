"""Validate an exact-OCF reranker for Task 2's complete cross-day gallery.

For each outer held recording, a separate calibration recording supplies query
responses and exact candidate OCF labels. The calibration scorer sees the other
two source recordings. The outer scorer sees all three non-outer recordings.
Query rows are shuffled before scoring and assignment. Calibration and outer
OCF are measured on disjoint glyph subsets from the complete known gallery.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from glyph_atlas_metric_ranker import fit_atlas_metric_ranker, pack_atlas_candidates
from glyph_contour_retrieval import contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_mbr import contour_signatures
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import official_scorer
from kaggle_hierarchical_gallery_validation import paired_delta, validated_logits
from kaggle_task3_metric_validation import unique_pair_ocf


SESSIONS = ("session1", "session2", "session3", "session4")
CALIBRATION_SESSIONS = {"session1": "session3", "session2": "session4",
                        "session3": "session1", "session4": "session2"}
NOMINATIONS = {"v9": 8, "validated": 4, "eb": 4, "contour": 6, "v9_assigned": 1}
ANCHORS = (4, 8)


def score_block(mouse, training, query, images, signature, energy, device, args):
    torch.manual_seed(args.seed)
    validated = validated_logits(mouse, training, query, images, device, args)
    hierarchy = fit_hierarchical_gallery(training, view="total")
    eb = hierarchy.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
    v9 = blend_hierarchical_logits(validated, eb)
    baseline = assign_glyphs(v9)
    contour = fit_contour_retrieval(training, signature, energy, task="task2", seed=args.seed)
    directional = contour_retrieval_scores(contour, query, None)["contour_ridge"]
    return {"v9": v9, "validated": validated, "eb": eb, "contour": directional}, baseline


@torch.no_grad()
def shortlist(matrices, assignment, energy, ink):
    assigned = torch.full_like(matrices["v9"], -torch.inf)
    assigned.scatter_(1, assignment[:, None], 0.0)
    scores = {**matrices, "v9_assigned": assigned,
              "contour_energy": energy.log().expand_as(assigned), "ink_fraction": ink.expand_as(assigned)}
    return pack_atlas_candidates(scores, NOMINATIONS, baseline="v9")


@torch.no_grad()
def sparse_assignment(candidates, objective, fallback):
    """Solve on shortlisted edges, retaining the complete v9 perfect matching."""
    assigned_edges = candidates.valid & (candidates.indices == fallback[:, None])
    if not bool(assigned_edges.any(dim=1).all()):
        raise ValueError("The full v9 assignment must be present in the shortlist.")
    rows, columns = candidates.valid.nonzero(as_tuple=True)
    weights = objective[rows, columns].double().cpu().numpy()
    if not np.isfinite(weights).all():
        raise ValueError("Assignment needs finite objectives on all shortlisted edges.")
    glyphs = candidates.indices[rows, columns].cpu().numpy()
    # Sparse matching ignores explicit zero weights. All costs remain positive.
    cost = weights.max() + 1.0 - weights
    graph = csr_matrix((cost, (rows.cpu().numpy(), glyphs)), shape=(len(fallback), candidates.atlas_size))
    try:
        matched_rows, selected = min_weight_full_bipartite_matching(graph)
    except ValueError:
        return fallback, True
    prediction = torch.empty_like(fallback)
    matched_rows = torch.as_tensor(matched_rows, device=fallback.device, dtype=torch.long)
    prediction[matched_rows] = torch.as_tensor(selected, device=fallback.device, dtype=torch.long)
    return prediction, False


def metric_labels(candidates, references, masks, scorer, device, chunk_size):
    rows, columns = candidates.valid.nonzero(as_tuple=True)
    exact, unique_pairs = unique_pair_ocf(candidates.indices[rows, columns], references[rows], masks,
                                         scorer, device, chunk_size)
    target = candidates.features.new_full(candidates.valid.shape, torch.nan)
    target[rows, columns] = exact
    return target, unique_pairs


def raw_pair_metrics(values, baseline):
    delta, error = paired_delta(values, baseline)
    return {"ocf": float(values.mean()), "paired_delta_vs_v9": delta, "paired_glyph_se": error}


def aggregate(measurements):
    baseline = np.stack(measurements["v9_assignment"])
    result = {}
    for name, blocks in measurements.items():
        values = np.stack(blocks)
        result[name] = {**raw_pair_metrics(values.mean(axis=0), baseline.mean(axis=0)),
                        "ocf_n": values.size, "unique_scored_glyphs": values.shape[1]}
    return result


def evaluate(query_matrices, baseline, ranker, permutation, selected, masks, energy, ink, scorer, device, args):
    candidates = shortlist(query_matrices, baseline, energy, ink)
    expected = ranker.predict_ocf(candidates)
    v9 = query_matrices["v9"]
    centered = v9 - v9.mean(dim=1, keepdim=True)
    global_z = centered / centered.square().mean().sqrt().clamp_min(1e-6)
    global_z = global_z.gather(1, candidates.indices)
    choices = {"v9_assignment": baseline, "v9_independent": v9.argmax(dim=1),
               "metric_independent_anchor4": ranker.select(candidates, anchor_ocf=4)}
    fallback = {}
    for strength in ANCHORS:
        name = f"metric_assignment_anchor{strength}"
        choices[name], fallback[name] = sparse_assignment(candidates, expected + strength * global_z, baseline)

    # The held correspondence enters only after every decision is fixed.
    sample_rows = permutation.argsort()[selected]
    selected_guesses = torch.stack([choice[sample_rows] for choice in choices.values()])
    exact, _ = unique_pair_ocf(selected_guesses, selected.expand_as(selected_guesses), masks, scorer,
                              device, args.ocf_chunk_size)
    measurements = dict(zip(choices, exact.cpu().numpy()))
    metrics = {name: {**raw_pair_metrics(values, measurements["v9_assignment"]), "ocf_n": len(values),
                      "full_gallery_top1": float((choices[name] == permutation).float().mean()),
                      "changed_fraction": float((choices[name] != baseline).float().mean()),
                      "distinct_glyphs": int(choices[name].unique().numel()),
                      "sparse_fallback": fallback.get(name, False)} for name, values in measurements.items()}

    oracle = {}
    if args.depth_images:
        depth_rows = sample_rows[:args.depth_images]
        target = selected[:args.depth_images]
        pool = candidates.take_rows(depth_rows)
        reference, num_pairs = metric_labels(pool, target, masks, scorer, device, args.ocf_chunk_size)
        v9_top8 = v9[depth_rows].topk(NOMINATIONS["v9"], dim=1).indices
        v9_support = (pool.indices[:, :, None] == v9_top8[:, None]).any(dim=2) & pool.valid
        v9_best = reference.masked_fill(~v9_support, -torch.inf).amax(dim=1)
        union_best = reference.masked_fill(~pool.valid, -torch.inf).amax(dim=1)
        oracle = {"reference_images": len(target), "unique_pairs": num_pairs,
                  "v9_top8_oracle": float(v9_best.mean()), "shortlist_union_oracle": float(union_best.mean()),
                  "union_vs_v9_top8": float((union_best - v9_best).mean())}
    return measurements, metrics, oracle


def run_fold(mouse, held, recordings, images, masks, signature, energy, ink, calibration_ids, outer_ids,
             scorer, device, args):
    started = time.monotonic()
    calibration = CALIBRATION_SESSIONS[held]
    inner_sources = {name: counts for name, counts in recordings.items() if name not in (held, calibration)}
    outer_sources = {name: counts for name, counts in recordings.items() if name != held}
    rng = np.random.default_rng(args.seed + 1000 + 10 * SESSIONS.index(held) + 100 * (mouse == "mice2"))

    inner_permutation = torch.as_tensor(rng.permutation(len(images)), device=device)
    print(f"{mouse}/{held}: calibrate on {calibration} after fitting {list(inner_sources)}", flush=True)
    matrices, baseline = score_block(mouse, inner_sources, recordings[calibration][inner_permutation],
                                    images, signature, energy, device, args)
    calibration_rows = inner_permutation.argsort()[calibration_ids]
    candidates = shortlist({name: matrix[calibration_rows] for name, matrix in matrices.items()},
                           baseline[calibration_rows], energy, ink)
    del matrices, baseline
    target, num_pairs = metric_labels(candidates, calibration_ids, masks, scorer, device, args.ocf_chunk_size)
    ranker = fit_atlas_metric_ranker(candidates, target, seed=args.seed, epochs=args.ranker_epochs)
    calibration_summary = {**ranker.diagnostics, "unique_pairs": num_pairs,
                           "shortlist_oracle_ocf": float(target.masked_fill(~candidates.valid, -torch.inf).amax(1).mean())}
    print(json.dumps({"mouse": mouse, "outer_session": held, "calibration": calibration_summary}), flush=True)
    del candidates, target

    outer_permutation = torch.as_tensor(rng.permutation(len(images)), device=device)
    print(f"{mouse}/{held}: outer scoring after fitting {list(outer_sources)}", flush=True)
    matrices, baseline = score_block(mouse, outer_sources, recordings[held][outer_permutation],
                                    images, signature, energy, device, args)
    measured, metrics, oracle = evaluate(matrices, baseline, ranker, outer_permutation, outer_ids,
                                         masks, energy, ink, scorer, device, args)
    return measured, {"mouse": mouse, "outer_session": held, "calibration_session": calibration,
                      "calibration_score_sources": list(inner_sources), "outer_score_sources": list(outer_sources),
                      "calibration": calibration_summary, "metrics": metrics, "reference_best_candidates": oracle,
                      "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--held-sessions", nargs="+", choices=SESSIONS, default=("session2", "session4"))
    parser.add_argument("--calibration-images", type=int, default=128)
    parser.add_argument("--ocf-images", type=int, default=96)
    parser.add_argument("--depth-images", type=int, default=16)
    parser.add_argument("--ocf-chunk-size", type=int, default=384)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--retrieval-epochs", type=int, default=12)
    parser.add_argument("--ranker-epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task2_metric_validation.json"))
    args = parser.parse_args()
    if min(args.calibration_images, args.ocf_images) < 2 or not 0 <= args.depth_images <= args.ocf_images:
        parser.error("Use valid calibration, outer and reference-candidate sample counts.")
    if min(args.encoder_epochs, args.retrieval_epochs, args.ranker_epochs, args.ocf_chunk_size) <= 0:
        parser.error("Use positive fit settings and chunk size.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run this exact OCF validation on a Kaggle GPU with competition data mounted.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    masks = images >= 128
    signature, energy = contour_signatures(images)
    ink = masks.float().mean(dim=(1, 2))
    scorer = official_scorer(root)
    rng = np.random.default_rng(args.seed)
    if args.calibration_images + args.ocf_images > len(images):
        parser.error("Calibration and outer reference glyphs must be disjoint.")
    outer = rng.choice(len(images), args.ocf_images, replace=False)
    remaining = np.setdiff1d(np.arange(len(images)), outer)
    calibration = rng.choice(remaining, args.calibration_images, replace=False)
    calibration_ids, outer_ids = (torch.as_tensor(rows, device=device) for rows in (calibration, outer))
    output = {"seed": args.seed, "encoder_epochs": args.encoder_epochs, "retrieval_epochs": args.retrieval_epochs,
              "ranker_epochs": args.ranker_epochs, "calibration_images_per_fold": len(calibration_ids),
              "outer_ocf_images_per_fold": len(outer_ids), "nominations": NOMINATIONS,
              "baseline": "Exact v9 Task 2 policy, 65% validated per-mouse score and 35% repeat-noise EB, full assignment.",
              "assignment": "Sparse matching over shortlisted edges, objective predicted OCF plus anchor times "
                            "global-RMS v9 margin. All v9 assignment edges are included; fallback is the full v9 assignment.",
              "balanced_anchor_weights": ANCHORS, "disjoint_calibration_and_scored_glyphs": True,
              "complete_gallery_size": len(images), "folds": []}
    overall = {}
    for mouse in dict.fromkeys(args.mice):
        recordings = grouped_blocks(root, catalog, mouse, device, len(images))
        if set(recordings) != set(SESSIONS) or any(block.shape[1] != 4 for block in recordings.values()):
            raise ValueError("Expected four training recordings with four repeats per glyph.")
        per_mouse = {}
        for held in dict.fromkeys(args.held_sessions):
            measured, details = run_fold(mouse, held, recordings, images, masks, signature, energy, ink,
                                         calibration_ids, outer_ids, scorer, device, args)
            for name, values in measured.items():
                overall.setdefault(name, []).append(values)
                per_mouse.setdefault(name, []).append(values)
            output["folds"].append(details)
            output.setdefault("per_mouse", {})[mouse] = aggregate(per_mouse)
            output["summary"] = aggregate(overall)
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(details), flush=True)
        del recordings
    print(json.dumps({"summary": output["summary"], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
