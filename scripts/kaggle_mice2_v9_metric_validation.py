"""Compare v9-style contour exact reranking against the existing mice2 Task 1 policy."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import (SHORTLIST as LEGACY_NOMINATIONS, fit_calibrated_ranker,
                                           shortlist as legacy_shortlist, split_calibration)
from kaggle_gallery_validation import align_training_glyphs, score_block
from kaggle_v8_metric_validation import (ANCHORS, NOMINATIONS, SESSIONS, fit_v8, nested_ranker, rendered_bank,
                                          shortlist, summarize)


REFERENCE = "v9_existing_mice2"


def outer_split(image_count, seed, avoid_seeds, device):
    if not avoid_seeds:
        return split_glyphs(image_count, 320, seed, device)
    used = np.zeros(image_count, dtype=bool)
    for previous_seed in avoid_seeds:
        previous = np.random.default_rng(previous_seed).permutation(image_count)
        used[previous[~used[previous]][:320]] = True
    order = np.random.default_rng(seed).permutation(image_count)
    held = np.sort(order[~used[order]][:320])
    fit = np.setdiff1d(np.arange(image_count), held)
    return torch.as_tensor(fit, device=device), torch.as_tensor(held, device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2209)
    parser.add_argument("--avoid-held-seed", type=int, nargs="+", default=(961,),
                        help="previous 320-glyph holdout seeds in evaluation order")
    parser.add_argument("--metric-glyphs", type=int, default=96)
    parser.add_argument("--noise-glyphs", type=int, default=384)
    parser.add_argument("--ocf-images", type=int, default=160)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/mice2_v9_metric_validation.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle GPU with competition data mounted.")
    if not 2 <= args.ocf_images <= 320 or min(args.metric_glyphs, args.noise_glyphs) < 2:
        parser.error("Choose positive calibration counts and a paired held subset of at most 320 glyphs.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    if (1 + len(args.avoid_held_seed)) * 320 > len(images) or args.seed in args.avoid_held_seed:
        parser.error("Choose a new seed and leave enough glyphs for the disjoint holdout.")
    fit, held = outer_split(len(images), args.seed, args.avoid_held_seed, device)
    if args.metric_glyphs + args.noise_glyphs > len(fit) - 512:
        parser.error("Reserve at least 512 glyphs for the nested encoders.")
    split = split_calibration(len(fit), args.metric_glyphs, args.noise_glyphs, device, args.seed)
    selection = np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_images, replace=False)
    queries = held[torch.as_tensor(selection, device=device)]
    fitted_images = images[fit]
    data = grouped_blocks(root, catalog, "mice2", device, len(images))
    training = {session: counts[fit] for session, counts in data.items()}
    scorer = official_scorer(root)

    print("Fitting the new contour-union ranker with inner-only atlas and calibration", flush=True)
    transfer, transfer_likelihoods, nested, codepoints = nested_ranker(training, fitted_images, split, scorer, args.seed)
    print("Refitting shared final atlas and exact existing mice2 Task 1 control", flush=True)
    final = fit_v8(training, fitted_images, args.seed)
    if not torch.equal(codepoints, final.codepoints.cpu()):
        raise ValueError("Nested and outer atlas must retain the same public codepoint order.")
    rendered = rendered_bank(final)
    aligned, _ = align_training_glyphs(final.atlas, fitted_images)
    legacy, old_likelihoods, baseline_fit = fit_calibrated_ranker(
        final.atlas, final.codepoints, fitted_images, aligned, training, split, rendered, scorer, SESSIONS, args.seed
    )
    del aligned, training

    output = {"mouse": "mice2", "seed": args.seed, "avoid_held_seed": args.avoid_held_seed,
              "held_glyphs": len(held), "ocf_images_per_session": len(queries),
              "metric_calibration_glyphs": len(split[0]), "noise_calibration_glyphs": len(split[1]),
              "inner_prediction_glyphs": len(split[2]), "existing_nominations": LEGACY_NOMINATIONS,
              "existing_anchor_ocf": 2.0, "transfer_nominations": NOMINATIONS, "transfer_anchors": ANCHORS,
              "outer_fit": final.report, "transfer_nested_fit": nested,
              "existing_ranker": baseline_fit, "transfer_ranker": transfer.diagnostics,
              "protocol": "Paired whole-glyph holdout. Both policies use identical fit, noise, metric and held glyph indices and the same final atlas, encoder, decoder and rendering. The mice2 control reproduces the original v6 nested atlas/OCF path and its v4 anchor of 2, unchanged in v9. The new model calibrates on an inner-only atlas with contour-derived nominations and uses its 60/40 atlas-contour anchor. Neither uses the outer held images for fitting or ranking.",
              "blocks": []}
    measurements = {}
    reference = (images[queries] >= 128).cpu().numpy()
    for session in SESSIONS:
        observed = data[session][queries]
        existing = legacy_shortlist(final.atlas, final.prior, final.encoder, final.decoder, final.normalization,
                                    old_likelihoods[session], observed, session)
        new, standalone = shortlist(final, transfer_likelihoods[session], observed, session)
        selected = {REFERENCE: legacy.select(existing, anchor_ocf=2), "unranked_contour_anchor": standalone,
                    **{f"contour_ranker_anchor_{anchor:g}": transfer.select(new, anchor_ocf=anchor)
                       for anchor in ANCHORS}}
        rendered_choices = {method: rendered[index] for method, index in selected.items()}
        observed_scores = score_block(scorer, rendered_choices, reference, device)
        report = {"session": session, "mean_transfer_candidates": float(new.valid.sum(dim=1).float().mean()),
                  "mean_existing_candidates": float(existing.valid.sum(dim=1).float().mean()), "metrics": {}}
        for method, scores in observed_scores.items():
            measurements.setdefault(method, []).append(scores)
            report["metrics"][method] = {**summarize([scores], [observed_scores[REFERENCE]]),
                                         "changed_vs_v9": int((selected[method] != selected[REFERENCE]).sum())}
        output["blocks"].append(report)
        output["summary"] = {method: summarize(scores, measurements[REFERENCE])
                             for method, scores in measurements.items()}
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report), flush=True)
    print(json.dumps({"summary": output["summary"], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
