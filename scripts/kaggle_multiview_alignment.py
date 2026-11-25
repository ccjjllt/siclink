"""Measure spike-only correspondence of held glyphs across Task 1 recordings."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs
from glyph_multiview_alignment import calibrate_margin, fit_multiview, pair_similarity, reciprocal_matches
from glyph_retrieval import assign_glyphs


SESSIONS = ("session1", "session2", "session3", "session4")
GCCA = "https://arxiv.org/abs/1605.09459"


def shuffled_embeddings(model, recordings, glyphs, seed):
    rng = np.random.default_rng(seed)
    positions, embeddings = {}, {}
    for block, observed in recordings.items():
        permutation = torch.as_tensor(rng.permutation(len(glyphs)), device=observed.device)
        positions[block] = permutation
        embeddings[block] = model.encode(block, observed[glyphs[permutation]])
    return embeddings, positions


def nearest(scores, row_labels, column_labels):
    selected = scores.topk(min(5, scores.shape[1]), dim=1).indices
    correct = column_labels[selected] == row_labels[:, None]
    return {"top1": float(correct[:, 0].float().mean()), "top5": float(correct.any(1).float().mean()),
            "queries": len(row_labels), "bank_rows": len(column_labels)}


def match_counts(rows, columns, source, target):
    hits = int((source[rows] == target[columns]).sum())
    return {"matches": len(rows), "correct": hits, "precision": hits / len(rows) if len(rows) else None,
            "coverage": len(rows) / len(source)}


def compare_pairs(calibration, calibration_labels, held, held_labels, args):
    pairs, neighbors = [], {block: [] for block in held}
    for left, right in combinations(held, 2):
        cal_scores = pair_similarity(calibration[left], calibration[right])
        rows, columns, margins = reciprocal_matches(cal_scores)
        correct = calibration_labels[left][rows] == calibration_labels[right][columns]
        fitted = calibrate_margin(margins, correct, minimum_precision=args.minimum_precision,
                                  minimum_matches=args.minimum_calibration_matches)

        scores = pair_similarity(held[left], held[right])
        left_labels, right_labels = held_labels[left], held_labels[right]
        rows, columns, margins = reciprocal_matches(scores)
        accepted = (margins >= fitted["threshold"] if fitted["threshold"] is not None
                    else torch.zeros_like(margins, dtype=torch.bool))
        chosen_left, chosen_right = rows[accepted], columns[accepted]
        neighbors[left].append((right, chosen_left, chosen_right))
        neighbors[right].append((left, chosen_right, chosen_left))
        balanced = assign_glyphs(scores)
        result = {"left": left, "right": right, "same_mouse": left[0] == right[0],
                  "left_to_right": nearest(scores, left_labels, right_labels),
                  "right_to_left": nearest(scores.T, right_labels, left_labels),
                  "hungarian_top1": float((right_labels[balanced] == left_labels).float().mean()),
                  "calibration": fitted, "reciprocal": match_counts(rows, columns, left_labels, right_labels),
                  "calibrated": match_counts(chosen_left, chosen_right, left_labels, right_labels)}
        pairs.append(result)
        print(json.dumps(result), flush=True)
    return pairs, neighbors


def bank_reports(embeddings, labels):
    output = []
    for block, values in embeddings.items():
        peers = [key for key in embeddings if key != block]
        within = [peer for peer in peers if peer[0] == block[0]]
        across = [peer for peer in peers if peer[0] != block[0]]
        result = {"block": block}
        for name, collection in (("all_peers", peers), ("same_mouse", within), ("cross_mouse", across)):
            if collection:
                scores = torch.cat([values @ embeddings[peer].T for peer in collection], dim=1)
                candidate_labels = torch.cat([labels[peer] for peer in collection])
                result[name] = nearest(scores, labels[block], candidate_labels)
        output.append(result)
    return output


def summarize_pairs(pairs):
    result = {}
    for name, group in (("all", pairs), ("same_mouse", [pair for pair in pairs if pair["same_mouse"]]),
                        ("cross_mouse", [pair for pair in pairs if not pair["same_mouse"]])):
        if not group:
            continue
        directions = [pair[direction] for pair in group for direction in ("left_to_right", "right_to_left")]
        row = {"block_pairs": len(group), "top1": float(np.mean([item["top1"] for item in directions])),
               "top5": float(np.mean([item["top5"] for item in directions])),
               "hungarian_top1": float(np.mean([pair["hungarian_top1"] for pair in group]))}
        for key in ("reciprocal", "calibrated"):
            matches = sum(pair[key]["matches"] for pair in group)
            correct = sum(pair[key]["correct"] for pair in group)
            row[key] = {"matches": matches, "correct": correct, "precision": correct / matches if matches else None,
                        "mean_coverage": float(np.mean([pair[key]["coverage"] for pair in group]))}
        result[name] = row
    return result


def pool_atlas(root, catalog, images, fit, held, shuffles, neighbors, args, device):
    from kaggle_atlas_prior_validation import neural_blocks
    from kaggle_diagnostics import official_scorer
    from kaggle_gallery_validation import (align_training_glyphs, build_font_atlas, standard_error, standardized)

    started = time.monotonic()
    style = np.random.default_rng(args.seed + 2).choice(len(fit), 128, replace=False)
    selected = np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_per_session, replace=False)
    selected = torch.as_tensor(selected, device=device)
    fit_images = images[fit]
    atlas, provenance = build_font_atlas(fit_images[torch.as_tensor(style, device=device)], device)
    aligned, agreement = align_training_glyphs(atlas, fit_images)
    # Apply the same independent input-row permutation used for each embedding block.
    base = {block: scores[shuffles[block]] for block, scores in neural_blocks(
        atlas, root, catalog, len(images), fit_images, aligned, fit, held, args, device)}
    score = official_scorer(root)
    reference = (images[held[selected]] >= 128).cpu().numpy()
    far = max(args.pool_weights)
    output = {"alignment_agreement": agreement, "atlas": provenance, "rows": []}
    all_values = {str(strength): [] for strength in (0.0, *args.pool_weights)}
    for block, logits in base.items():
        rows = shuffles[block].argsort()[selected]
        own = standardized(logits[rows])
        row_positions = torch.full((len(logits),), -1, dtype=torch.long, device=device)
        row_positions[rows] = torch.arange(len(rows), device=device)
        supports = torch.zeros_like(own)
        votes = torch.zeros(len(rows), device=device)
        for peer, target, source in neighbors[block]:
            positions = row_positions[target]
            used = positions >= 0
            supports.index_add_(0, positions[used], standardized(base[peer][source[used]]))
            votes.index_add_(0, positions[used], torch.ones_like(positions[used], dtype=votes.dtype))
        correction = torch.where(votes[:, None] > 0, supports / votes[:, None].clamp_min(1) - own, 0)
        baseline_id = own.argmax(1)
        baseline_img = (atlas.images[baseline_id] >= 128).cpu().numpy()
        baseline = np.asarray(score(baseline_img, reference, device=str(device)))
        all_values["0.0"].append(baseline)
        row = {"block": block, "baseline_ocf": float(baseline.mean()), "ocf_n": len(rows),
               "glyphs_with_accepted_peer": int((votes > 0).sum()), "max_pool_weight": far, "variants": {}}
        for strength in args.pool_weights:
            corrected_id = (own + strength * correction).argmax(1)
            changed = corrected_id != baseline_id
            values = baseline.copy()
            if bool(changed.any()):
                changed_np = changed.cpu().numpy()
                prediction = (atlas.images[corrected_id[changed]] >= 128).cpu().numpy()
                values[changed_np] = score(prediction, reference[changed_np], device=str(device))
            all_values[str(strength)].append(values)
            row["variants"][str(strength)] = {"ocf": float(values.mean()), "changed": int(changed.sum()),
                                              "paired_delta": float((values - baseline).mean())}
        output["rows"].append(row)
        print(json.dumps({"atlas_pool": row}), flush=True)

    output["summary"] = {}
    baseline = np.stack(all_values["0.0"])
    for name, records in all_values.items():
        values = np.stack(records)
        delta = (values - baseline).mean(axis=0)
        output["summary"][name] = {"ocf": float(values.mean()), "ocf_n": values.size,
                                    "paired_delta": float(delta.mean()), "paired_delta_se": standard_error(delta),
                                    "independent_held_glyphs": len(delta)}
    output["seconds"] = time.monotonic() - started
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--sessions", nargs="+", choices=SESSIONS, default=SESSIONS)
    parser.add_argument("--view", choices=("evoked-total", "evoked-multiscale", "early-late"), default="evoked-total")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=0.35)
    parser.add_argument("--calibration-glyphs", type=int, default=320)
    parser.add_argument("--minimum-precision", type=float, default=0.65,
                        help="minimum overlap in a conservative inner-fit calibration")
    parser.add_argument("--minimum-calibration-matches", type=int, default=12)
    parser.add_argument("--atlas-pool", action="store_true", help="validate public atlas fusion on accepted peer links only")
    parser.add_argument("--pool-weights", nargs="+", type=float, default=(0.25, 0.45))
    parser.add_argument("--ocf-per-session", type=int, default=64)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--refine-epochs", type=int, default=12)
    parser.add_argument("--contour-epochs", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/multiview_alignment.json"))
    args = parser.parse_args()
    if not (len(set(args.mice)) * len(set(args.sessions)) >= 2 and 2 <= args.minimum_calibration_matches
            <= args.calibration_glyphs and 0 < args.minimum_precision < 1 and 0 < args.ridge
            and args.dimension > 0 and 1 <= args.ocf_per_session <= 320 and min(args.pool_weights) > 0
            and max(args.pool_weights) <= 0.5 and 0 <= args.contour_epochs <= args.refine_epochs
            and args.encoder_epochs > 0 and args.refine_epochs > 0):
        parser.error("Choose at least two recording blocks and valid calibration, ridge and atlas settings.")
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Use Kaggle CPU or CUDA for alignment.")
    started = time.monotonic()
    torch.set_num_threads(4)
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    fit, held = split_glyphs(len(images), 320, args.seed, device)
    if args.calibration_glyphs > len(fit) - 3:
        parser.error("Leave at least three fitted glyphs outside the calibration subset.")
    permutations = np.random.default_rng(args.seed + 3).permutation(len(fit))
    permutations = torch.as_tensor(permutations, device=device)
    inner = fit[permutations[args.calibration_glyphs:]]
    calibration_glyphs = fit[permutations[:args.calibration_glyphs]]
    recordings = {}
    for mouse in args.mice:
        blocks = grouped_blocks(root, catalog, mouse, device, len(images))
        recordings.update({(mouse, session): values for session, values in blocks.items() if session in args.sessions})
    alignment = fit_multiview({block: values[inner] for block, values in recordings.items()},
                              view=args.view, dimension=args.dimension, ridge=args.ridge)
    calibration, calibration_labels = shuffled_embeddings(alignment, recordings, calibration_glyphs, args.seed + 7)
    embeddings, labels = shuffled_embeddings(alignment, recordings, held, args.seed + 11)
    del recordings
    pairs, neighbors = compare_pairs(calibration, calibration_labels, embeddings, labels, args)
    summary = summarize_pairs(pairs)
    banks = bank_reports(embeddings, labels)
    bank_summary = {}
    for name in ("all_peers", "same_mouse", "cross_mouse"):
        measured = [row[name] for row in banks if name in row]
        if measured:
            bank_summary[name] = {"blocks": len(measured), "top1": float(np.mean([row["top1"] for row in measured])),
                                  "top5": float(np.mean([row["top5"] for row in measured]))}
    output = {"paper": GCCA, "device": str(device), "seed": args.seed, "blocks": len(embeddings),
              "fit_glyphs": len(inner), "calibration_glyphs": len(calibration_glyphs), "held_glyphs": len(held),
              "dimensions": len(alignment.spectrum), "ridge": args.ridge, "view": args.view,
              "minimum_calibration_precision": args.minimum_precision,
              "minimum_calibration_matches": args.minimum_calibration_matches,
              "chance_pair_top1": 1 / len(held), "maxvar_eigenvalues": alignment.spectrum.tolist(),
              "fit_cross_mouse_axis_agreement": alignment.cross_mouse_agreement.tolist()
              if alignment.cross_mouse_agreement is not None else None,
              "protocol": "Regularized GCCA uses aligned labeled fit responses. Every held block is independently shuffled and uses spike-only embeddings. Held labels enter match accuracy and OCF only. Hungarian is a diagnostic on the known complete held set, and peer fusion accepts calibrated reciprocal links without requiring shared official repertoires.",
              "pair_details": pairs, "pair_summary": summary, "union_bank_by_block": banks,
              "union_bank_summary": bank_summary,
              "seconds": time.monotonic() - started}
    if args.atlas_pool:
        output["atlas_pool"] = {"status": "pending" if summary["all"]["calibrated"]["matches"] else "no_calibrated_links"}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pair_summary": summary}), flush=True)
    if args.atlas_pool and summary["all"]["calibrated"]["matches"]:
        output["atlas_pool"] = pool_atlas(root, catalog, images, fit, held, labels, neighbors, args, device)
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
