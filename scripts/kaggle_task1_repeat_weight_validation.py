"""Test Task 1 neural and lexical weights at fixed 4-, 8-, and 16-repeat budgets.

The same four labeled days and unseen glyphs provide all budgets. Public atlas
alignment, readouts, encoder, render style and priors are fit once without those
glyphs. This CPU experiment uses directional contour and multi-window Poisson
scores as its anchor. Held images are used only to evaluate fixed predictions.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_prior import fit_atlas_prior
from glyph_contour_retrieval import contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_font_style import fit_aligned_font_style
from glyph_mbr import contour_signatures
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import align_training_glyphs, build_font_atlas, standard_error, standardized


MOUSE = "mice1"
REPEATS_PER_DAY = (1, 2, 4)
REPEAT_BUDGETS = (4, 8, 16)
HOLDOUT = 320
FIRST_PILOT_SEED = 6109
DIRECTION_WEIGHT = 0.65
NEURAL_GAIN = 1.40
LEXICAL_WEIGHT = 0.55
WEAK_LEXICAL_WEIGHT = 0.3575
CONTROL = "matched_cpu"
NEURAL = "neural_weight"
LEXICAL = "lexical_weak"


def independent_split(size, seed, exclude_first_pilot=False):
    labels = np.arange(size)
    previous = np.unique(np.concatenate([np.random.default_rng(earlier).permutation(size)[:HOLDOUT]
                                         for earlier in (961, 2209)]))
    available = np.setdiff1d(labels, previous)
    # Reconstruct the prior SDF experiment's outer cohort before selecting a fresh one.
    sdf = np.random.default_rng(5431).permutation(available)[:HOLDOUT]
    available = np.setdiff1d(available, sdf)
    if exclude_first_pilot:
        first_pilot = np.random.default_rng(FIRST_PILOT_SEED).permutation(available)[:HOLDOUT]
        available = np.setdiff1d(available, first_pilot)
    if len(available) < HOLDOUT:
        raise ValueError("The previous pilots leave too few fresh held-out glyphs.")
    held = np.random.default_rng(seed).permutation(available)[:HOLDOUT]
    return torch.as_tensor(np.setdiff1d(labels, held)), torch.as_tensor(held), size - len(available)


def shuffled_queries(values, generator):
    day_order = torch.randperm(len(values), generator=generator)
    repeat_order = torch.rand(values.shape[:2], generator=generator).argsort(1)
    shuffled = values.gather(1, repeat_order[:, :, None].expand_as(values))
    return shuffled[day_order], day_order.argsort()


@torch.no_grad()
def pooled_modalities(directional, encoder, atlas, groups, sample, seed):
    """Average day scores in known-label order after independent position shuffling."""
    sums, diagnostics = {}, {}
    generator = torch.Generator().manual_seed(seed + 173)
    for session, counts in groups.items():
        started = time.monotonic()
        values, inverse = shuffled_queries(counts[sample], generator)
        rates = encoder.expected_counts(atlas.images, session, batch_size=192)
        scaling_check = {}
        for repeats in REPEATS_PER_DAY:
            query = values[:, :repeats]
            contour = contour_retrieval_scores(directional, query, session)["contour_ridge"]
            shape = standardized(contour)
            summed = count_template_logits(query, rates, "rate")
            timed = count_template_logits(query, rates, "timed")
            rate_z, timed_z = standardized(summed), standardized(timed)
            forward = standardized(0.5 * (rate_z + timed_z))
            scaling_check[str(repeats * len(groups))] = {
                "rate_max_abs_z_difference": float((rate_z - standardized(repeats * summed)).abs().max()),
                "timed_max_abs_z_difference": float((timed_z - standardized(repeats * timed)).abs().max()),
                "rate_median_raw_summed_row_sd": float((repeats * summed).std(dim=1, unbiased=False).median()),
                "timed_median_raw_summed_row_sd": float((repeats * timed).std(dim=1, unbiased=False).median()),
            }
            if repeats not in sums:
                sums[repeats] = shape[inverse], forward[inverse]
            else:
                sums[repeats][0].add_(shape[inverse])
                sums[repeats][1].add_(forward[inverse])
        diagnostics[session] = scaling_check
        print(json.dumps({"session": session, "z_scale_control": scaling_check,
                          "seconds": time.monotonic() - started}), flush=True)
    return {repeats * len(groups): (shape / len(groups), forward / len(groups))
            for repeats, (shape, forward) in sums.items()}, diagnostics


def top_choices(modalities, prior):
    predictions = {}
    for repeats, (shape, forward) in modalities.items():
        shape = DIRECTION_WEIGHT * shape
        neural = (1 - DIRECTION_WEIGHT) * forward
        base = shape + neural
        predictions[repeats] = {
            CONTROL: prior.variants(base)["usage_unseen"].argmax(dim=1),
            NEURAL: prior.variants(shape + NEURAL_GAIN * neural)["usage_unseen"].argmax(dim=1),
            LEXICAL: (standardized(base) + WEAK_LEXICAL_WEIGHT * prior.lexical_log + prior.unseen_log).argmax(dim=1),
        }
    return predictions


def paired_stats(delta):
    return {"paired_delta": float(delta.mean()), "paired_glyph_se": standard_error(delta), "glyphs": len(delta)}


def evaluate_predictions(predictions, scorer):
    metrics, deltas = {}, {}
    base_at_4 = None
    for repeats in REPEAT_BUDGETS:
        rows = predictions[repeats]
        exact = {name: scorer.evaluate(choice[:, None])[:, 0] for name, choice in rows.items()}
        base = exact[CONTROL]
        if repeats == 4:
            base_at_4 = base
        deltas[repeats] = {name: scores - base for name, scores in exact.items()}
        metrics[str(repeats)] = {
            "variants": {name: {"ocf": float(scores.mean()), **paired_stats(deltas[repeats][name]),
                                "changed_top1_fraction": float((rows[name] != rows[CONTROL]).float().mean())}
                         for name, scores in exact.items()},
            "anchor_vs_4": paired_stats(base - base_at_4),
        }
        print(json.dumps({"repeat_budget": repeats, **metrics[str(repeats)]}), flush=True)
    interactions = {str(repeats): {name: paired_stats(deltas[repeats][name] - deltas[4][name])
                                  for name in (NEURAL, LEXICAL)}
                    for repeats in REPEAT_BUDGETS[1:]}
    return metrics, interactions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=FIRST_PILOT_SEED)
    parser.add_argument("--fit-seed", type=int, default=FIRST_PILOT_SEED)
    parser.add_argument("--exclude-first-pilot", action="store_true",
                        help="exclude all 320 outer glyphs used in the first repeat-weight pilot")
    parser.add_argument("--ocf-images", type=int, default=96)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_repeat_weight_validation.json"))
    args = parser.parse_args()
    if not 2 <= args.ocf_images <= HOLDOUT:
        parser.error("Choose a valid number of paired OCF glyphs.")
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run on Kaggle CPU with competition input and public font downloads enabled.")
    torch.set_num_threads(4)
    torch.manual_seed(args.fit_seed)
    device = torch.device("cpu")
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, device)
    fit, held, prior_excluded = independent_split(len(images), args.seed, args.exclude_first_pilot)
    sample = held[:args.ocf_images]
    train_images = images[fit]
    style_rows = np.random.default_rng(args.fit_seed + 2).choice(len(fit), 128, replace=False)
    print(f"Fit public atlas and render style on {len(fit)} labeled glyphs", flush=True)
    atlas, provenance, codepoints = build_font_atlas(train_images[style_rows], device, return_codepoints=True)
    aligned, agreement = align_training_glyphs(atlas, train_images)
    prior = fit_atlas_prior(atlas, codepoints, train_images)
    style, style_fit = fit_aligned_font_style(aligned, train_images, seed=args.fit_seed)
    rendered = torch.cat([(style.apply(rows[:, None].float() / 255)[:, 0] >= 0.5).to(torch.uint8) * 255
                          for rows in atlas.images.split(1024)])
    catalog = load_catalog(root)
    groups = grouped_blocks(root, catalog, MOUSE, device, len(images))
    if len(groups) != 4 or any(values.shape[1] != 4 for values in groups.values()):
        raise ValueError("This proxy requires all four labeled days with four observed repeats each.")
    training = {session: values[fit] for session, values in groups.items()}
    sessions = tuple(training)
    print("Fit directional contour readout and Poisson encoder", flush=True)
    directional = fit_contour_retrieval(training, *contour_signatures(aligned), task="task2", seed=args.fit_seed)
    signature, energy = contour_signatures(atlas.images)
    directional = replace(directional, signature=signature, energy=energy)
    encoder = fit_glyph_encoder(training, aligned, epochs=18, seed=args.fit_seed)
    del training
    modalities, diagnostics = pooled_modalities(directional, encoder, atlas, groups, sample, args.seed)
    choices = top_choices(modalities, prior)
    del modalities, groups
    score = PairedOCF(official_scorer(root), rendered, (images[sample] >= 128).numpy(), device, 12)
    metrics, interactions = evaluate_predictions(choices, score)
    result = {
        "seed": args.seed, "fit_seed": args.fit_seed, "mouse": MOUSE, "device": str(device), "fit_glyphs": len(fit),
        "exclude_first_pilot": args.exclude_first_pilot,
        "held_glyphs": len(held), "paired_glyphs": len(sample), "prior_pilot_glyphs_excluded_from_held": prior_excluded,
        "atlas": provenance, "style_fit": style_fit, "fit_alignment_agreement": agreement,
        "prior": prior.diagnostics, "sessions": sessions,
        "weights": {"direction": DIRECTION_WEIGHT, "neural": 1 - DIRECTION_WEIGHT,
                    "neural_multiplier": NEURAL_GAIN, "lexical": LEXICAL_WEIGHT,
                    "weak_lexical": WEAK_LEXICAL_WEIGHT, "unseen": 1.0},
        "protocol": "One frozen CPU directional and forward Poisson atlas comparator, with neither direct CNN "
                    "nor exact-OCF ranker. Each group of held repeats is permuted independently, and nested 1/2/4 "
                    "trial prefixes from each of four days form budgets 4/8/16. Known training labels co-index "
                    "the independent day scores. The same 96 held glyphs by default are scored at all budgets; their "
                    "images appear only after ranking. Official unlabeled recordings are not accessed. The cross-day "
                    "assay does not reproduce a same-recording 20-repeat group.",
        "primary_interaction": "Neural-weight gain at 16 repeats minus neural-weight gain at 4 repeats, using "
                               "the same per-glyph exact top-1 OCF changes.",
        "primary_fixed_blend_recheck": "Neural-weight gain at 4 repeats." if args.exclude_first_pilot else None,
        "z_scale_negative_control": diagnostics, "ocf_by_repeat_budget": metrics,
        "delta_of_deltas_vs_4": interactions, "unique_exact_ocf_pairs": len(score.values),
        "seconds": time.monotonic() - started,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"delta_of_deltas_vs_4": interactions, "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
