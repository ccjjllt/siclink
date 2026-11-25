"""Validate cross-recording links using GCCA spikes and public-atlas posterior agreement."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs
from glyph_dual_alignment import edge_features, fit_edge_reliability, pair_cues, posterior_view
from glyph_multiview_alignment import calibrate_margin, fit_multiview, reciprocal_matches
from kaggle_atlas_prior_validation import neural_blocks
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import align_training_glyphs, build_font_atlas, standard_error, standardized
from kaggle_multiview_alignment import match_counts, nearest, shuffled_embeddings


METHODS = ("gcca", "atlas", "dual")
SESSIONS = ("session1", "session2", "session3", "session4")


def mean_reciprocal_rank(scores, left_labels, right_labels):
    correct_columns = right_labels.argsort()[left_labels]
    matching_scores = scores.gather(1, correct_columns[:, None])
    rank = (scores > matching_scores).sum(dim=1) + 0.5 * (1 + (scores == matching_scores).sum(dim=1))
    return float(rank.reciprocal().mean())


def select_spike_weight(cues, labels):
    def quality(weight):
        scores = cues.method("dual", weight)
        left = mean_reciprocal_rank(scores, *labels)
        right = mean_reciprocal_rank(scores.T, *reversed(labels))
        return (left + right) * 0.5

    options = [(weight, quality(weight)) for weight in (0.50, 0.25, 0.75)]
    return max(options, key=lambda item: item[1])


def fitted_pair(name, cues, labels, spike_weight, args):
    blend, calibration, held = cues
    blend_labels, calibration_labels, held_labels = labels
    scores = blend.method(name, spike_weight)
    rows, columns, margins = reciprocal_matches(scores)
    correct = blend_labels[0][rows] == blend_labels[1][columns]
    confidence = fit_edge_reliability(edge_features(name, blend, scores, rows, columns, margins), correct)

    scores = calibration.method(name, spike_weight)
    rows, columns, margins = reciprocal_matches(scores)
    correct = calibration_labels[0][rows] == calibration_labels[1][columns]
    rank = confidence(edge_features(name, calibration, scores, rows, columns, margins))
    minimum = {"validated": args.minimum_precision, "coverage_probe": args.coverage_precision}
    thresholds = {rule: calibrate_margin(rank, correct, minimum_precision=precision,
                                         minimum_matches=args.minimum_matches)
                  for rule, precision in minimum.items()}

    scores = held.method(name, spike_weight)
    rows, columns, margins = reciprocal_matches(scores)
    rank = confidence(edge_features(name, held, scores, rows, columns, margins))
    accepted = {rule: (rank >= value["threshold"] if value["threshold"] is not None
                       else torch.zeros_like(rank, dtype=torch.bool)) for rule, value in thresholds.items()}
    result = {"left_to_right": nearest(scores, *held_labels),
              "right_to_left": nearest(scores.T, *reversed(held_labels)),
              "reciprocal": match_counts(rows, columns, *held_labels), "calibration": thresholds,
              "confidence_fit": "ridge" if confidence.weights is not None else "bilateral_margin"}
    for rule, keep in accepted.items():
        result[rule] = match_counts(rows[keep], columns[keep], *held_labels)
    keep = accepted["validated"]
    return result, (rows[keep], columns[keep])


def pair_validation(spikes, atlas, shuffles, args):
    results, links = [], {block: [] for block in spikes["held"]}
    for left, right in combinations(spikes["held"], 2):
        cues = [pair_cues(spikes[split][left], spikes[split][right], atlas[split][left], atlas[split][right])
                for split in ("blend", "calibration", "held")]
        labels = [(shuffles[split][left], shuffles[split][right]) for split in ("blend", "calibration", "held")]
        spike_weight, blend_mrr = select_spike_weight(cues[0], labels[0])
        record = {"left": left, "right": right, "same_mouse": left[0] == right[0], "spike_weight": spike_weight,
                  "inner_blend_mrr": blend_mrr, "methods": {}}
        for name in METHODS:
            record["methods"][name], (rows, columns) = fitted_pair(name, cues, labels, spike_weight, args)
            if name == "dual" and len(rows):
                links[left].append((right, rows, columns))
                links[right].append((left, columns, rows))
        results.append(record)
        print(json.dumps(record), flush=True)
    return results, links


def summarize_pairs(results):
    summary = {}
    groups = {"all": results, "same_mouse": [row for row in results if row["same_mouse"]],
              "cross_mouse": [row for row in results if not row["same_mouse"]]}
    for group, selected in groups.items():
        if not selected:
            continue
        methods = {}
        for name in METHODS:
            rows = [record["methods"][name] for record in selected]
            directions = [row[direction] for row in rows for direction in ("left_to_right", "right_to_left")]
            measured = {"top1": float(np.mean([row["top1"] for row in directions])),
                        "top5": float(np.mean([row["top5"] for row in directions]))}
            for rule in ("reciprocal", "validated", "coverage_probe"):
                n = sum(row[rule]["matches"] for row in rows)
                correct = sum(row[rule]["correct"] for row in rows)
                measured[rule] = {"matches": n, "correct": correct, "precision": correct / n if n else None,
                                  "coverage": float(np.mean([row[rule]["coverage"] for row in rows]))}
            methods[name] = measured
        summary[group] = {"pairs": len(selected), "methods": methods}
    return summary


def linked_queries(block, links, n_queries, device):
    linked = torch.zeros(n_queries, dtype=torch.bool, device=device)
    for _, rows, _ in links[block]:
        linked[rows] = True
    return torch.where(linked)[0]


def pooled_logits(base, queries, links, block, device):
    own = standardized(base[block][queries.cpu()].to(device))
    row_lookup = torch.full((len(base[block]),), -1, dtype=torch.long, device=device)
    row_lookup[queries] = torch.arange(len(queries), device=device)
    peers = torch.zeros_like(own)
    votes = torch.zeros(len(own), device=device)
    for peer, rows, columns in links[block]:
        positions = row_lookup[rows]
        valid = positions >= 0
        if bool(valid.any()):
            row = positions[valid]
            logits = base[peer][columns[valid].cpu()].to(device)
            peers.index_add_(0, row, standardized(logits))
            votes.index_add_(0, row, torch.ones(len(row), dtype=votes.dtype, device=device))
    correction = torch.where(votes[:, None] > 0, peers / votes[:, None].clamp_min(1) - own, 0)
    return own, correction


def ocf_validation(root, atlas, base, held, images, shuffles, links, args, device):
    scorer = official_scorer(root)
    fixed = np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_per_session, replace=False)
    fixed = torch.as_tensor(fixed, device=device)
    rng = np.random.default_rng(args.seed + 15)
    output = {"blocks": [], "summary": {}}
    fixed_by_weight = {weight: [] for weight in args.pool_weights}
    linked_by_weight = {weight: [] for weight in args.pool_weights}
    for block in base:
        order = shuffles[block]
        linked = linked_queries(block, links, len(order), device)
        linked_rows = rng.choice(linked.cpu().numpy(), min(len(linked), args.ocf_per_session), replace=False)
        linked_rows = torch.as_tensor(linked_rows, device=device)
        fixed_rows = order.argsort()[fixed]
        queries = torch.cat((fixed_rows, linked_rows)).unique(sorted=True)
        fixed_position = torch.searchsorted(queries, fixed_rows).cpu().numpy()
        linked_position = torch.searchsorted(queries, linked_rows).cpu().numpy()
        own, correction = pooled_logits(base, queries, links, block, device)
        original = own.argmax(dim=1)

        # Held reference images are available only in this scoring function.
        reference = (images[held[order[queries]]] >= 128).cpu().numpy()
        baseline_images = (atlas.images[original] >= 128).cpu().numpy()
        baseline = np.asarray(scorer(baseline_images, reference, device=str(device), batch_size=8))
        report = {"block": block, "eligible_queries": len(linked), "queries": len(order), "weights": {}}
        for weight in args.pool_weights:
            predicted = (own + weight * correction).argmax(dim=1)
            changed = predicted != original
            values = baseline.copy()
            if bool(changed.any()):
                mask = changed.cpu().numpy()
                selected = (atlas.images[predicted[changed]] >= 128).cpu().numpy()
                values[mask] = scorer(selected, reference[mask], device=str(device), batch_size=8)
            fixed_score, fixed_base = values[fixed_position], baseline[fixed_position]
            fixed_by_weight[weight].append((fixed_score, fixed_base))
            row = {"fixed_ocf": float(fixed_score.mean()), "fixed_baseline": float(fixed_base.mean()),
                   "fixed_delta": float((fixed_score - fixed_base).mean()), "fixed_n": len(fixed_position),
                   "linked_n": len(linked_position), "changed_n": int(changed.sum())}
            if len(linked_position):
                matched_score, matched_base = values[linked_position], baseline[linked_position]
                row.update(linked_ocf=float(matched_score.mean()), linked_baseline=float(matched_base.mean()),
                           linked_delta=float((matched_score - matched_base).mean()))
                linked_by_weight[weight].append((order[linked_rows].cpu().numpy(), matched_score, matched_base))
            report["weights"][str(weight)] = row
        output["blocks"].append(report)
        print(json.dumps({"pooling": report}), flush=True)

    for weight in args.pool_weights:
        fixed_scores, fixed_base = (np.stack(values) for values in zip(*fixed_by_weight[weight]))
        delta = (fixed_scores - fixed_base).mean(axis=0)
        result = {"fixed_ocf": float(fixed_scores.mean()), "fixed_baseline": float(fixed_base.mean()),
                  "fixed_delta": float(delta.mean()), "fixed_delta_se": standard_error(delta), "fixed_n": fixed_scores.size}
        if linked_by_weight[weight]:
            glyphs, scores, baseline = (np.concatenate(values) for values in zip(*linked_by_weight[weight]))
            unique, inverse = np.unique(glyphs, return_inverse=True)
            paired = np.bincount(inverse, weights=scores - baseline) / np.bincount(inverse)
            result.update(linked_ocf=float(scores.mean()), linked_baseline=float(baseline.mean()),
                          linked_delta=float((scores - baseline).mean()), linked_glyph_delta_se=standard_error(paired),
                          linked_n=len(scores), linked_unique_glyphs=len(unique))
        output["summary"][str(weight)] = result
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", choices=SESSIONS, default=("session2", "session4"))
    parser.add_argument("--view", choices=("evoked-total", "evoked-multiscale", "early-late"), default="evoked-total")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=0.35)
    parser.add_argument("--blend-glyphs", type=int, default=320)
    parser.add_argument("--calibration-glyphs", type=int, default=320)
    parser.add_argument("--minimum-precision", type=float, default=0.65)
    parser.add_argument("--coverage-precision", type=float, default=0.45)
    parser.add_argument("--minimum-matches", type=int, default=8)
    parser.add_argument("--shortlist", type=int, default=48)
    parser.add_argument("--posterior-temperature", type=float, default=0.50)
    parser.add_argument("--pool-weights", nargs="+", type=float, default=(0.25, 0.45))
    parser.add_argument("--ocf-per-session", type=int, default=64, help="zero runs correspondence only")
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--refine-epochs", type=int, default=12)
    parser.add_argument("--contour-epochs", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/dual_alignment_validation.json"))
    args = parser.parse_args()
    args.mice = ("mice1", "mice2")
    args.sessions = tuple(dict.fromkeys(args.sessions))
    args.pool_weights = tuple(dict.fromkeys(args.pool_weights))
    if (len(args.sessions) < 2 or not 0 < args.coverage_precision <= args.minimum_precision < 1
            or not 2 <= args.minimum_matches <= args.calibration_glyphs or args.blend_glyphs < 4
            or min(args.shortlist, args.dimension, args.encoder_epochs, args.refine_epochs) < 1
            or not 0 <= args.contour_epochs <= args.refine_epochs or not 0 <= args.ocf_per_session <= 320
            or args.ridge <= 0 or args.posterior_temperature <= 0
            or not 0 < min(args.pool_weights) <= max(args.pool_weights) <= 0.5):
        parser.error("Choose two sessions and valid nested splits, match thresholds and fits.")

    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Use Kaggle CUDA or CPU.")
    images = load_images(root, device)
    outer_fit, held = split_glyphs(len(images), 320, args.seed, device)
    boundary = args.blend_glyphs + args.calibration_glyphs
    if len(outer_fit) - boundary < 128:
        parser.error("Keep at least 128 fit glyphs outside nested calibration and blend folds.")
    order = np.random.default_rng(args.seed + 3).permutation(len(outer_fit))
    arranged = outer_fit[torch.as_tensor(order, device=device)]
    fit = arranged[boundary:]
    splits = {"blend": arranged[:args.blend_glyphs],
              "calibration": arranged[args.blend_glyphs:boundary], "held": held}
    recordings = {}
    for mouse in args.mice:
        groups = grouped_blocks(root, catalog, mouse, device, len(images))
        recordings.update({(mouse, session): counts for session, counts in groups.items() if session in args.sessions})
    alignment = fit_multiview({block: counts[fit] for block, counts in recordings.items()}, view=args.view,
                              ridge=args.ridge, dimension=args.dimension)
    spikes, shuffles = {}, {}
    for offset, (name, glyphs) in enumerate(splits.items()):
        spikes[name], shuffles[name] = shuffled_embeddings(alignment, recordings, glyphs, args.seed + 10 + offset)
    del recordings, groups

    fit_images = images[fit]
    style = np.random.default_rng(args.seed + 2).choice(len(fit), 128, replace=False)
    atlas, provenance = build_font_atlas(fit_images[torch.as_tensor(style, device=device)], device)
    if args.shortlist > len(atlas.images):
        parser.error("The atlas shortlist must fit inside the fixed public catalogue.")
    aligned, agreement = align_training_glyphs(atlas, fit_images)
    atlas_views, base = {name: {} for name in splits}, {}
    queries = torch.cat(tuple(splits.values()))
    for block, scores in neural_blocks(atlas, root, catalog, len(images), fit_images, aligned, fit, queries, args, device):
        for name, subset in zip(splits, scores.split([len(glyphs) for glyphs in splits.values()])):
            shuffled = subset[shuffles[name][block]]
            atlas_views[name][block] = posterior_view(shuffled, atlas.contours, top_k=args.shortlist,
                                                       temperature=args.posterior_temperature)
            if name == "held" and args.ocf_per_session:
                base[block] = shuffled.detach().float().cpu()
    del scores, shuffled, subset

    results, links = pair_validation(spikes, atlas_views, shuffles, args)
    summary = summarize_pairs(results)
    output = {"seed": args.seed, "device": str(device), "mice": args.mice, "sessions": args.sessions,
              "fit_glyphs": len(fit), "split_sizes": {name: len(values) for name, values in splits.items()},
              "view": args.view, "dimension": len(alignment.spectrum), "ridge": args.ridge,
              "minimum_precision": args.minimum_precision, "coverage_probe_precision": args.coverage_precision,
              "minimum_matches": args.minimum_matches, "shortlist": args.shortlist,
              "posterior_temperature": args.posterior_temperature, "atlas": provenance,
              "fit_atlas_agreement": agreement, "pair_details": results, "pair_summary": summary,
              "protocol": "GCCA and public atlas models fit on disjoint inner fit glyphs. Inner blend labels choose each pair's mix and confidence scorer, and separate inner calibration labels select reciprocal-match thresholds. Coverage probe links are measured only. The held replay uses the same labeled glyph set in each recording for scoring. Its rows are independently shuffled, and the matcher sees neither this membership nor the held labels. Held images are used only for OCF. No official test data is accessed.",
              "seconds": time.monotonic() - started}
    if args.ocf_per_session:
        output["pooling"] = {"status": "pending" if any(links.values()) else "no_validated_links"}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pair_summary": summary}), flush=True)
    if args.ocf_per_session and any(links.values()):
        output["pooling"] = ocf_validation(root, atlas, base, held, images, shuffles["held"], links, args, device)
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
