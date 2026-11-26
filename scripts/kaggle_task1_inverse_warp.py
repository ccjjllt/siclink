"""Evaluate a spike-driven smooth deformation of frozen Task 1 atlas winners."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import fit_glyph_encoder
from glyph_local_warp import (EPOCHS, KNOTS, LEARNING_RATE, MAX_CENTROID_SHIFT, MAX_INK_FRACTION, MAX_SHIFT_PIXELS,
                              invert_local, observed_evoked, poisson_scores, warp_statistics)
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_mice2_v9_metric_validation import outer_split
from kaggle_task1_half_proposal_validation import ANCHOR_OCF, MODEL_SEED, twos
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, fit_v8, nested_ranker, rendered_bank, shortlist, summarize


PRIOR_OUTER_SEEDS = (961, 2209, 3931, 4931, 5147, 8081)
INVERSE_ENCODER_EPOCHS = 18
CONTROL_GROUP_SIZE = 16


@torch.no_grad()
def initial_glyphs(model, ranker, likelihood, counts, session, rendered):
    candidates, _ = shortlist(model, likelihood, counts, session)
    winner = ranker.select(candidates, anchor_ocf=ANCHOR_OCF)
    return rendered[winner] >= 128


def shuffled_responses(initial, counts, seed):
    """Permute fitting rows among comparable spike totals and template ink."""
    total = observed_evoked(counts).sum(1).log1p()
    ink = initial.float().flatten(1).sum(1).log1p()
    sorting = ((total - total.mean()) / total.std(unbiased=False).clamp_min(0.01)
               + 0.25 * ((ink - ink.mean()) / ink.std(unbiased=False).clamp_min(0.01)))
    order = sorting.argsort()
    shuffled = torch.empty_like(order)
    rng = np.random.default_rng(seed)
    groups = max(1, len(order) // CONTROL_GROUP_SIZE)
    for group in order.tensor_split(groups):
        order_inside = torch.as_tensor(rng.permutation(len(group)), device=order.device)
        rows = group[order_inside]
        shuffled[rows] = rows.roll(1)
    return counts[shuffled]


def distort(encoder, initial, fit, session, seed):
    shuffled = shuffled_responses(initial, fit, seed)
    variants = {"real": invert_local(encoder, initial, fit, session),
                "shuffled": invert_local(encoder, initial, shuffled, session)}
    masks = {"initial": initial, **{name: variant.images for name, variant in variants.items()}}
    return masks, {name: warp_statistics(initial, variant) for name, variant in variants.items()}


def evaluate_ocf(masks, refs, scorer):
    initial = masks["initial"]
    device = str(initial.device)
    reference = refs.cpu().numpy()
    base = scorer(initial.cpu().numpy(), reference, device=device, batch_size=12)
    result = {"initial": base}
    for name, images in masks.items():
        if name == "initial":
            continue
        changed = (images != initial).flatten(1).any(1)
        values = base.copy()
        if bool(changed.any()):
            keep = changed.cpu().numpy()
            values[keep] = scorer(images[changed].cpu().numpy(), reference[keep], device=device, batch_size=12)
        result[name] = values
    return result


def paired_ocf(values):
    metrics = {name: summarize(rows, values["initial"]) for name, rows in values.items()}
    metrics["real_vs_shuffled"] = summarize(values["real"], values["shuffled"])
    return metrics


def response_values(encoder, images, counts, session):
    return {name: poisson_scores(encoder, pixels, counts, session).cpu().numpy() for name, pixels in images.items()}


def response_metrics(values):
    pairs = {"real_vs_initial": ("real", "initial"), "shuffled_vs_initial": ("shuffled", "initial"),
             "real_vs_shuffled": ("real", "shuffled")}
    results = {}
    for name, (first, second) in pairs.items():
        delta = np.stack(values[first]) - np.stack(values[second])
        per_glyph = delta.mean(0)
        results[name] = {"nats_per_trial": float(per_glyph.mean()), "paired_glyph_se": float(per_glyph.std(ddof=1) / len(per_glyph)**0.5),
                         "likelihood_comparisons": delta.size, "scored_glyphs": len(per_glyph),
                         "fraction_positive": float((delta > 0).mean())}
    return results


def combine(collector, observed):
    for name, values in observed.items():
        collector.setdefault(name, []).append(values)


def run_recording(model, ranker, likelihood, encoder, rendered, counts, refs, session, seed, scorer):
    started = time.monotonic()
    initial = initial_glyphs(model, ranker, likelihood, counts, session, rendered)
    masks, geometry = distort(encoder, initial, counts, session, seed)
    primary = evaluate_ocf(masks, refs, scorer)
    ocf = paired_ocf({name: [values] for name, values in primary.items()})
    full_fit = response_values(encoder, masks, counts, session)
    full_metrics = response_metrics({name: [values] for name, values in full_fit.items()})
    print(json.dumps({"session": session, "full_repeats_ocf": ocf, "geometry": geometry}), flush=True)

    held_observed, in_fit_observed = {}, {}
    half_reports = {}
    halves = twos(counts, seed + 1)
    for index, (fit, held) in enumerate(zip(halves, halves[::-1])):
        # The other half contributes neither to the atlas winner nor to optimizing its deformation.
        initial_half = initial_glyphs(model, ranker, likelihood, fit, session, rendered)
        warped, changes = distort(encoder, initial_half, fit, session, seed + 2 + index)
        check = response_values(encoder, warped, held, session)
        in_fit = response_values(encoder, warped, fit, session)
        combine(held_observed, check)
        combine(in_fit_observed, in_fit)
        half_reports[f"half{index + 1}_fits"] = {"held_likelihood": response_metrics({n: [v] for n, v in check.items()}),
                                                 "in_fit_likelihood": response_metrics({n: [v] for n, v in in_fit.items()}),
                                                 "geometry": changes}
    report = {"four_repeat_ocf": ocf, "four_repeat_in_fit_likelihood": full_metrics, "four_repeat_geometry": geometry,
              "strict_held_half_likelihood": response_metrics(held_observed), "fit_half_likelihood": response_metrics(in_fit_observed),
              "half_directions": half_reports, "seconds": time.monotonic() - started}
    return report, primary, held_observed, in_fit_observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12457)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_inverse_warp.json"))
    args = parser.parse_args()
    if not 2 <= args.ocf_images <= 320:
        parser.error("Select between two and 320 paired held glyphs in each recording.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use a Kaggle GPU with the official competition input and the public font enabled.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, held = outer_split(len(images), args.seed, PRIOR_OUTER_SEEDS, device)
    rng = np.random.default_rng(args.seed + 1)
    chosen = held[torch.as_tensor(rng.choice(len(held), args.ocf_images, replace=False), device=device)]
    recording = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    if tuple(recording) != SESSIONS or any(values.shape[1] != 4 for values in recording.values()):
        raise ValueError("Expected four mice1 recordings with four trials per labeled glyph.")
    scorer = official_scorer(root)
    training = {name: values[fit] for name, values in recording.items()}
    split = split_calibration(len(fit), 96, 384, device, MODEL_SEED)
    print("Fitting the v11 atlas and inner leave-glyph-out ranker", flush=True)
    ranker, likelihoods, nested, codepoints = nested_ranker(training, images[fit], split, scorer, MODEL_SEED)
    model = fit_v8(training, images[fit], MODEL_SEED)
    if not torch.equal(model.codepoints.cpu(), codepoints):
        raise ValueError("Nested and outer public candidate columns must match exactly.")
    print("Fitting the independent native-image spike encoder", flush=True)
    encoder = fit_glyph_encoder(training, images[fit], epochs=INVERSE_ENCODER_EPOCHS, seed=MODEL_SEED + 499)
    encoder.eval().requires_grad_(False)
    del training
    rendered = rendered_bank(model)

    result = {"mouse": "mice1", "outer_seed": args.seed, "earlier_outer_seeds": PRIOR_OUTER_SEEDS,
              "fit_glyphs": len(fit), "held_glyphs": len(held), "paired_ocf_per_recording": len(chosen),
              "repeats": 4, "v11_anchor": ANCHOR_OCF, "v11_nomination": NOMINATIONS, "ranker": ranker.diagnostics,
              "nested_fit": nested, "encoder_epochs": INVERSE_ENCODER_EPOCHS, "inverse_seed": MODEL_SEED + 499,
              "deformation": {"boundary_fixed_lattice": KNOTS, "maximum_displacement_px": MAX_SHIFT_PIXELS,
                              "maximum_relative_ink_with_one_pixel_floor": MAX_INK_FRACTION,
                              "maximum_centroid_shift_px": MAX_CENTROID_SHIFT, "adam_steps": EPOCHS,
                              "adam_learning_rate": LEARNING_RATE, "poisson_scoring_window_ms": [50, 250]},
              "protocol": "Freeze both the v11 atlas ranking and the native-image spike encoder using only "
                          "whole training glyphs. In the primary arm choose the original v11 glyph using all "
                          "four repeats, fit bounded local deformation with those same counts, and score the "
                          "admissible binary output by official OCF. Shuffled fitting counts have comparable "
                          "population totals and template ink and use identical budgets. In a separate strict "
                          "diagnostic each fitting half both selects the initial glyph and fits the deformation. "
                          "The independent other half evaluates the binary output with Poisson nats per trial. "
                          "Swap halves and cluster by glyph across both directions and recordings. No reference "
                          "images for held glyphs are used to predict, and no official test block is loaded.",
              "sessions": {}}
    pooled_ocf, pooled_held, pooled_in_fit = {}, {}, {}
    for index, session in enumerate(SESSIONS):
        counts = recording[session][chosen]
        reference = images[chosen] >= 128
        report, exact, half_held, half_fit = run_recording(model, ranker, likelihoods[session], encoder, rendered, counts,
                                                         reference, session, args.seed + 200 + 50 * index, scorer)
        result["sessions"][session] = report
        combine(pooled_ocf, exact)
        for collector, values in ((pooled_held, half_held), (pooled_in_fit, half_fit)):
            for arm, by_half in values.items():
                collector.setdefault(arm, []).extend(by_half)
        result["summary"] = {"four_repeat_ocf": paired_ocf(pooled_ocf),
                             "strict_held_half_likelihood": response_metrics(pooled_held),
                             "fit_half_likelihood": response_metrics(pooled_in_fit)}
        result["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": session, "independent_likelihood": report["strict_held_half_likelihood"],
                          "seconds": report["seconds"]}), flush=True)
    print(f"Saved {args.output} in {result['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
