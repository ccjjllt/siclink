"""Probe whether held-day neural geometry helps Task 2 glyph assignment."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import fit_hierarchical_gallery, response_features
from glyph_retrieval import _whiten, assign_glyphs
from kaggle_diagnostics import official_scorer, standardize


def projected_views(training, query, gallery, dimension):
    """Find stable glyph axes from aligned labeled source recordings only."""
    source = []
    for counts in training.values():
        values, _ = standardize(response_features(counts, gallery.view))
        source.append(_whiten(values.mean(1), gallery.noise_whitener))
    source = torch.stack(source)
    source = source - source.mean(1, keepdim=True)
    combined = source.sum(0)
    flat = source.flatten(0, 1)
    cross_session = combined.T @ combined - flat.T @ flat
    _, vectors = torch.linalg.eigh(cross_session)
    axes = vectors[:, -dimension:]
    values, _ = standardize(response_features(query, gallery.view), gallery.normalization)
    projected = (_whiten(values.flatten(0, 1), gallery.noise_whitener) @ axes).unflatten(0, values.shape[:2])
    return source @ axes, projected[:, :2].mean(1), projected[:, 2:].mean(1)


def crossnobis_profiles(a, b, a_anchor, b_anchor):
    """Calculate all signed split-half distances to the selected anchors."""
    return ((a * b).sum(1, keepdim=True) + (a_anchor * b_anchor).sum(1)
            - a @ b_anchor.T - b @ a_anchor.T)


def source_geometry(source, anchors):
    n_sessions = len(source)
    profiles = [crossnobis_profiles(source[i], source[j], source[i, anchors], source[j, anchors])
                for i in range(n_sessions) for j in range(i + 1, n_sessions)]
    return torch.stack(profiles).mean(0)


def pair_distances(a, b, left, right):
    return ((a[left] - a[right]) * (b[left] - b[right])).sum(1)


def geometry_agreement(source, a, b, truth, rng):
    """Score the cross-day geometry using known labels only after projection."""
    left = torch.as_tensor(rng.integers(len(truth), size=12_000))
    right = (left + torch.as_tensor(rng.integers(1, len(truth), size=len(left)))) % len(truth)
    source_distances = [pair_distances(source[i], source[j], truth[left], truth[right])
                        for i in range(len(source)) for j in range(i + 1, len(source))]
    target_distances = pair_distances(a, b, left, right)
    shuffled = truth[torch.as_tensor(rng.permutation(len(truth)))]
    control = [pair_distances(source[i], source[j], shuffled[left], shuffled[right])
               for i in range(len(source)) for j in range(i + 1, len(source))]
    return {"pairs": len(left),
            "spearman": float(spearmanr(target_distances, torch.stack(source_distances).mean(0)).statistic),
            "shuffled_spearman": float(spearmanr(target_distances, torch.stack(control).mean(0)).statistic)}


def select_anchors(gallery, query, count):
    """Choose unique glyphs that independently win on both repeat halves."""
    half_leaders = []
    for repeats in (query[:, :2], query[:, 2:]):
        scores = gallery.logits(repeats, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
        values, indices = scores.topk(2, dim=1)
        half_leaders.append((values[:, 0] - values[:, 1], indices[:, 0]))
    (margin_a, glyph_a), (margin_b, glyph_b) = half_leaders
    agreeing = torch.where(glyph_a == glyph_b)[0]
    margin = torch.minimum(margin_a[agreeing], margin_b[agreeing])
    chosen_rows, chosen_glyphs, seen = [], [], set()
    for row in agreeing[margin.argsort(descending=True)].tolist():
        glyph = int(glyph_a[row])
        if glyph in seen:
            continue
        seen.add(glyph)
        chosen_rows.append(row)
        chosen_glyphs.append(glyph)
        if len(chosen_rows) == count:
            break
    return torch.as_tensor(chosen_rows, dtype=torch.long), torch.as_tensor(chosen_glyphs, dtype=torch.long)


def profile_cosines(query_profiles, source_profiles):
    a = F.normalize(query_profiles - query_profiles.mean(1, keepdim=True), dim=1)
    b = F.normalize(source_profiles - source_profiles.mean(1, keepdim=True), dim=1)
    return a @ b.T


def blend_shortlist(unary, geometry, rows, shortlist, weight):
    """Add an anchor score to the shortlist without reranking anchor rows."""
    leading, candidates = unary.topk(shortlist, dim=1)
    adjustment = geometry.gather(1, candidates)
    adjustment -= adjustment.mean(1, keepdim=True)
    scale = (leading - leading.mean(1, keepdim=True)).square().mean().sqrt()
    adjustment *= weight * scale / adjustment.square().mean().sqrt().clamp_min(1e-8)
    adjustment[rows] = 0
    return unary.clone().scatter_add_(1, candidates, adjustment)


def assignment_metrics(logits, truth):
    chosen = assign_glyphs(logits)
    leaders = logits.topk(5, dim=1).indices
    return chosen, {"one_to_one_top1": float((chosen == truth).float().mean()),
                    "map_top1": float((leaders[:, 0] == truth).float().mean()),
                    "map_top5": float((leaders == truth[:, None]).any(1).float().mean())}


def analyze_fold(training, query, truth, args, rng):
    gallery = fit_hierarchical_gallery(training)
    unary = gallery.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
    source, a, b = projected_views(training, query, gallery, args.dimensions)
    comparison = geometry_agreement(source, a, b, truth, rng)
    rows, glyphs = select_anchors(gallery, query, args.anchors)
    comparison["anchors"] = {"count": len(rows),
                             "precision": float((truth[rows] == glyphs).float().mean()) if len(rows) else None}
    predictions, comparison["metrics"] = {}, {}
    predictions["hierarchical"], comparison["metrics"]["hierarchical"] = assignment_metrics(unary, truth)
    if len(rows) < 4:
        return comparison, predictions

    query_profiles = crossnobis_profiles(a, b, a[rows], b[rows])
    gallery_profiles = source_geometry(source, glyphs)
    unanchored = torch.ones(len(query), dtype=torch.bool)
    unanchored[rows] = False
    shuffled = torch.as_tensor(rng.permutation(len(rows)))
    for name, profiles in (("geometry", gallery_profiles), ("shuffled_anchors", gallery_profiles[:, shuffled])):
        geometry = profile_cosines(query_profiles, profiles)
        ranks = geometry.topk(5, dim=1).indices
        comparison[name + "_only_top5"] = float((ranks[unanchored] == truth[unanchored, None]).any(1).float().mean())
        blended = blend_shortlist(unary, geometry, rows, args.shortlist, args.weight)
        predictions[name], comparison["metrics"][name] = assignment_metrics(blended, truth)
    return comparison, predictions


def score_ocf(masks, pairs, selected, scorer):
    summaries = {}
    for results, truth, predictions in pairs:
        if len(predictions) < 3:
            continue
        selected_rows = truth.argsort()[selected]
        predicted = torch.stack([p[selected_rows] for p in predictions.values()])
        glyph_pairs = torch.stack((predicted, selected.expand_as(predicted)), dim=-1)
        unique, inverse = torch.unique(glyph_pairs.flatten(0, 1), dim=0, return_inverse=True)
        values = scorer(masks[unique[:, 0]].float().numpy(), masks[unique[:, 1]].numpy(), batch_size=4)
        results["ocf"] = {}
        for name, scores in zip(predictions, np.asarray(values)[inverse.numpy()].reshape(tuple(predicted.shape))):
            summaries.setdefault(name, []).append(scores)
            results["ocf"][name] = float(scores.mean())
    if not summaries:
        return {}
    baseline = np.stack(summaries["hierarchical"])
    summary = {}
    for name, parts in summaries.items():
        scores = np.stack(parts)
        delta = (scores - baseline).mean(0)
        summary[name] = {"ocf": float(scores.mean()), "n": int(scores.size),
                         "paired_delta": float(delta.mean()),
                         "paired_glyph_se": float(delta.std(ddof=1) / np.sqrt(len(delta)))}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-sessions", nargs="+", default=("session1", "session2", "session3", "session4"))
    parser.add_argument("--mice", nargs="+", default=("mice1", "mice2"))
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--anchors", type=int, default=128)
    parser.add_argument("--shortlist", type=int, default=32)
    parser.add_argument("--weight", type=float, default=0.15)
    parser.add_argument("--ocf-images", type=int, default=32)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/crossnobis_validation.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    rng = np.random.default_rng(args.seed)
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, "cpu")
    selected = torch.as_tensor(rng.choice(len(images), args.ocf_images, replace=False))
    output = {"baseline": "hierarchical empirical Bayes with 0.5 query-repeat noise, followed by Hungarian",
              "view_ms": [50, 250], "dimensions": args.dimensions, "anchors_max": args.anchors,
              "shortlist": args.shortlist, "weight": args.weight, "seed": args.seed,
              "device": "cpu", "ocf_images_per_fold": len(selected), "folds": {}}
    pairs = []
    for mouse in args.mice:
        groups = grouped_blocks(root, catalog, mouse, "cpu", len(images))
        for session in args.held_sessions:
            fold_started = time.monotonic()
            training = {name: counts for name, counts in groups.items() if name != session}
            truth = torch.as_tensor(rng.permutation(len(images)))
            block = groups[session][truth]
            order = torch.as_tensor(rng.random((len(block), block.shape[1])).argsort(1))
            query = block.gather(1, order[..., None].expand_as(block))
            comparison, predictions = analyze_fold(training, query, truth, args, rng)
            comparison["seconds"] = time.monotonic() - fold_started
            output["folds"][f"{mouse}/{session}"] = comparison
            pairs.append((comparison, truth, predictions))
            print(json.dumps({"mouse": mouse, "session": session, **comparison}), flush=True)
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    output["ocf_summary"] = score_ocf(images >= 128, pairs, selected, official_scorer(root))
    output["seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ocf_summary": output["ocf_summary"], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
