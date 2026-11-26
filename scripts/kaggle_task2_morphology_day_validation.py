"""Validate cross-fitted glyph-specific target-day residuals on the v11 assignment graph."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import concat_atlas_candidates, fit_atlas_metric_ranker
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import fit_hierarchical_gallery
from glyph_mbr import contour_signatures
from glyph_morphology_day_adapter import adapt, image_coordinates, shuffled_coordinates
from kaggle_diagnostics import official_scorer
from kaggle_task2_metric_validation import (SESSIONS, metric_labels, score_block, shortlist, shortlisted_v9_score,
                                            sparse_assignment)
from kaggle_task3_metric_validation import unique_pair_ocf


ANCHOR = 8.0
ADAPT_WEIGHT = 4.0
ADAPT_CAP = 3.0


def fit_ranker(sources, images, masks, signature, energy, ink, calibration, scorer, args):
    device = images.device
    rng = np.random.default_rng(args.seed)
    proposals, targets, info = [], [], {}
    for session, counts in sources.items():
        fit = {name: data for name, data in sources.items() if name != session}
        order = torch.as_tensor(rng.permutation(len(images)), device=device)
        evidence, v9 = score_block("mice1", fit, counts[order], images, signature, energy, device, args)
        rows = order.argsort()[calibration]
        candidates = shortlist({name: value[rows] for name, value in evidence.items()}, v9[rows], energy, ink)
        exact, pairs = metric_labels(candidates, calibration, masks, scorer, device, 384)
        proposals.append(candidates)
        targets.append(exact)
        info[session] = {"fit_sessions": list(fit), "exact_pairs": pairs}
        print(f"Calibration {session}: {pairs} exact image pairs", flush=True)
        del evidence
    ranker = fit_atlas_metric_ranker(concat_atlas_candidates(proposals), torch.cat(targets), seed=args.seed)
    return ranker, info


def summarize(values):
    baseline = np.stack(values["v11"])
    result = {}
    for name, rows in values.items():
        scores = np.stack(rows)
        paired = (scores - baseline).mean(axis=0)
        entry = {"ocf": float(scores.mean()), "paired_delta_vs_v11": float(paired.mean()),
                 "paired_glyph_se": float(paired.std(ddof=1) / np.sqrt(len(paired))), "ocf_n": scores.size}
        if name == "morphology":
            difference = (scores - np.stack(values["shuffled"])).mean(axis=0)
            entry.update(paired_delta_vs_shuffled=float(difference.mean()),
                         paired_se_vs_shuffled=float(difference.std(ddof=1) / np.sqrt(len(difference))))
        result[name] = entry
    return result


@torch.no_grad()
def score_adaptations(query, sources, matrices, v9, ranker, energy, ink, morphology, shuffled, seed):
    candidates = shortlist(matrices, v9, energy, ink)
    fixed = ranker.predict_ocf(candidates) + ANCHOR * shortlisted_v9_score(matrices["v9"], candidates)
    baseline, baseline_fallback = sparse_assignment(candidates, fixed, v9)
    hierarchy = fit_hierarchical_gallery(sources, view="total")
    zero, corrections, scale, adapter = adapt(query, hierarchy, morphology, shuffled, seed)
    choices, fallback, diagnostics = {"v11": baseline}, {"v11": baseline_fallback}, {}
    for name, delta in corrections.items():
        offset = ADAPT_WEIGHT * delta.gather(1, candidates.indices) / scale[:, None]
        adjusted = fixed + offset.clamp(-ADAPT_CAP, ADAPT_CAP)
        choices[name], fallback[name] = sparse_assignment(candidates, adjusted, v9)
        diagnostics[name] = {"clipped_edge_fraction": float((offset[candidates.valid].abs() > ADAPT_CAP).float().mean()),
                             "mean_absolute_edge_offset": float(offset[candidates.valid].abs().mean())}
    return choices, fallback, zero, corrections, adapter, diagnostics


def run_fold(held, recording, data, calibration, held_glyphs, args):
    started = time.monotonic()
    images, masks, signature, energy, ink, morphology, shuffled, scorer = data
    device = images.device
    sources = {name: counts for name, counts in recording.items() if name != held}
    print(f"Hold out all of {held}; fit v11 on {list(sources)}", flush=True)
    ranker, fit_report = fit_ranker(sources, images, masks, signature, energy, ink, calibration, scorer, args)
    rng = np.random.default_rng(args.split_seed + SESSIONS.index(held))
    row_order = torch.as_tensor(rng.permutation(len(images)), device=device)
    query = recording[held][row_order]
    trial_order = torch.as_tensor(rng.random(query.shape[:2]).argsort(axis=1), device=device)
    split_query = query.gather(1, trial_order[:, :, None].expand_as(query))
    matrices, v9_assignment = score_block("mice1", sources, query, images, signature, energy, device, args)
    choices, fallback, zero, corrections, report, offsets = score_adaptations(
        split_query, sources, matrices, v9_assignment, ranker, energy, ink, morphology, shuffled, args.split_seed
    )

    # Labels are consulted only after scores and assignments are fully determined.
    sample = row_order.argsort()[held_glyphs]
    images_chosen = torch.stack([indices[sample] for indices in choices.values()])
    scored, pairs = unique_pair_ocf(images_chosen, held_glyphs.expand_as(images_chosen), masks, scorer, device, 384)
    measurements = dict(zip(choices, scored.cpu().numpy()))
    matches, predictive = {}, {}
    for name, indices in choices.items():
        matches[name] = {"full_gallery_assignment_top1": float((indices == row_order).float().mean()),
                         "changed_fraction": float((indices != choices["v11"]).float().mean()),
                         "unique_glyphs": int(indices.unique().numel()), "fallback": fallback[name]}
    for name, scores in {"source_only": zero, **{key: zero + value for key, value in corrections.items()}}.items():
        ranked = scores.topk(5, dim=1).indices
        probability = scores.gather(1, row_order[:, None])[:, 0]
        gain = probability - zero.gather(1, row_order[:, None])[:, 0]
        predictive[name] = {"full_gallery_top1": float((ranked[:, 0] == row_order).float().mean()),
                            "full_gallery_top5": float((ranked == row_order[:, None]).any(dim=1).float().mean()),
                            "mean_true_class_log_likelihood_delta": float(gain.mean()),
                            "true_class_gain_standard_error": float(gain.std(unbiased=True) / len(gain) ** 0.5)}
    result = {"held_session": held, "source_sessions": list(sources), "ranker": ranker.diagnostics,
              "ocf_calibration": fit_report, "ocf": summarize({name: [values] for name, values in measurements.items()}),
              "assignment_identity": matches, "predictive_gallery": predictive, "adapter": report,
              "edge_offsets": offsets, "unique_outer_image_pairs": pairs, "seconds": time.monotonic() - started}
    return result, measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-sessions", nargs="+", choices=SESSIONS, default=("session1", "session3"))
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--split-seed", type=int, default=3593)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--calibration-images", type=int, default=128)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.set_defaults(retrieval_epochs=12)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task2_morphology_day_validation.json"))
    args = parser.parse_args()
    if min(args.calibration_images, args.ocf_images) < 2 or args.encoder_epochs < 1:
        parser.error("Choose at least two glyphs in both fit and evaluation sets.")
    if not Path("/kaggle/input").is_dir() or (args.device == "cuda:0" and not torch.cuda.is_available()):
        raise RuntimeError("Run the Task 2 leave-day-out validation on Kaggle with the requested accelerator.")

    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, args.device)
    masks = images >= 128
    signature, energy = contour_signatures(images)
    ink = masks.float().mean(dim=(1, 2))
    morphology = image_coordinates(masks)
    shuffled = shuffled_coordinates(morphology, ink, energy, args.split_seed)
    scorer = official_scorer(root)
    recording = grouped_blocks(root, load_catalog(root), "mice1", images.device, len(images))
    if any(counts.shape[1] != 4 for counts in recording.values()):
        raise ValueError("The leave-day-out proxy requires four measured repeats for each labeled glyph.")
    chosen = np.random.default_rng(args.seed).choice(len(images), args.calibration_images, replace=False)
    remaining = np.setdiff1d(np.arange(len(images)), chosen)
    held = np.random.default_rng(args.split_seed).choice(remaining, args.ocf_images, replace=False)
    calibration, scored = (torch.as_tensor(index, device=images.device) for index in (chosen, held))
    output = {"mouse": "mice1", "device": args.device, "model_seed": args.seed, "split_seed": args.split_seed,
              "held_sessions": args.held_sessions, "gallery_glyphs": len(images), "calibration_images": len(calibration),
              "paired_ocf_per_session": len(scored), "v11_anchor": ANCHOR, "adaptation_ocf_weight": ADAPT_WEIGHT,
              "adaptation_ocf_cap": ADAPT_CAP, "sessions": {},
              "protocol": "Keep the v11 graph, ranker and fallback fixed. Freeze shared response and noise estimates "
                          "from source days. Within each anonymous target day, fit glyph-conditioned residuals with "
                          "separate halves providing weak correspondence and response, excluding both halves of "
                          "every row in the applied fold from its correction and scale fit. All matching uses one "
                          "row per glyph from the documented grouped design. Exact OCF and identity use the "
                          "held recording's training labels after inference. No official test block or sample ID "
                          "is used."}
    combined = {}
    data = images, masks, signature, energy, ink, morphology, shuffled, scorer
    for held_session in dict.fromkeys(args.held_sessions):
        result, paired = run_fold(held_session, recording, data, calibration, scored, args)
        output["sessions"][held_session] = result
        for name, values in paired.items():
            combined.setdefault(name, []).append(values)
        output["summary"] = summarize(combined)
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
