"""Choose between fixed Task 1 neural weights using independent repeat evidence."""

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
from kaggle_gallery_validation import align_training_glyphs, build_font_atlas, standardized
from kaggle_task1_repeat_weight_validation import independent_split
from kaggle_v8_metric_validation import summarize


MODEL_SEED = 211
COUNTS = (1, 2, 4)
BASE = (0.65, 0.35)
AMPLIFIED = (0.65, 0.35 * 1.40)
RIDGE = 0.25


def splits(n, seed, calibration_size):
    _, first, _ = independent_split(n, 6109)
    _, second, _ = independent_split(n, 6947, exclude_first_pilot=True)
    available = np.setdiff1d(np.arange(n), torch.cat((first, second)).numpy())
    order = np.random.default_rng(seed).permutation(available)
    held, calibration = order[:320], order[320:320 + calibration_size]
    model = np.setdiff1d(np.arange(n), np.concatenate((held, calibration)))
    return tuple(torch.as_tensor(rows) for rows in (model, calibration, held))


def pack_signal(contour, rates, counts, session):
    shape = standardized(contour_retrieval_scores(contour, counts, session)["contour_ridge"])
    pooled = standardized(count_template_logits(counts, rates, "rate"))
    timed = standardized(count_template_logits(counts, rates, "timed"))
    return shape, standardized(0.5 * (pooled + timed))


@torch.no_grad()
def decisions(groups, rows, contour, encoder, atlas, prior, seed):
    generator = torch.Generator().manual_seed(seed)
    halves = {}
    for day, (session, counts) in enumerate(groups.items()):
        selected = counts[rows]
        order = torch.rand(selected.shape[:2], generator=generator).argsort(dim=1)
        selected = selected.gather(1, order[..., None].expand_as(selected))
        rates = encoder.expected_counts(atlas.images, session, batch_size=192)
        for n in COUNTS:
            shape, neural = pack_signal(contour, rates, selected[:, :n], session)
            key = (n, day % 2)
            if key in halves:
                halves[key][0].add_(shape)
                halves[key][1].add_(neural)
            else:
                halves[key] = (shape, neural)

    result = {}
    for n in COUNTS:
        left, right = halves.pop((n, 0)), halves.pop((n, 1))
        pooled_shape, pooled_neural = ((first + second) / 4 for first, second in zip(left, right))
        base = prior.variants(BASE[0] * pooled_shape + BASE[1] * pooled_neural)["usage_unseen"].argmax(dim=1)
        amplified = prior.variants(AMPLIFIED[0] * pooled_shape + AMPLIFIED[1] * pooled_neural)["usage_unseen"].argmax(dim=1)
        # The two halves contain disjoint recording days. Compare raw
        # spike-derived evidence without the shared lexical prior.
        shape_agreement, neural_agreement = (
            (standardized(first) * standardized(second)).mean(1) for first, second in zip(left, right)
        )
        repeat_log = torch.full_like(shape_agreement, float(np.log2(n)))
        features = torch.stack((repeat_log, shape_agreement, neural_agreement), dim=1)
        result[4 * n] = (features, base, amplified)
    return result


def fit_difference(features, target):
    center = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-3)
    x = (features - center) / scale
    y = target - target.mean()
    gram = x.T @ x / len(x)
    gram.diagonal().add_(RIDGE)
    coefficient = torch.linalg.solve(gram, x.T @ y / len(x))
    return center, scale, coefficient, target.mean()


def predict_difference(features, model):
    center, scale, coefficient, intercept = model
    return ((features - center) / scale) @ coefficient + intercept


def paired_outcomes(entries, exact):
    result = {}
    for n, (features, base, amplified) in entries.items():
        first = exact.evaluate(base[:, None])[:, 0]
        second = exact.evaluate(amplified[:, None])[:, 0]
        result[n] = (features, first, second, base == amplified)
    return result


def train_weights(calibration):
    examples = torch.cat([values[0] for values in calibration.values()])
    truth = torch.as_tensor(np.concatenate([values[2] - values[1] for values in calibration.values()]), dtype=examples.dtype)
    return fit_difference(examples, truth), fit_difference(examples[:, :1], truth)


def evaluate(outer, full_model, count_model, seed):
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for n, (features, first, amplified, equal) in outer.items():
        pieces = {"base": first, "amplified": amplified}
        shuffle = torch.randperm(len(features), generator=generator)
        actions = {"repeat_count_only": predict_difference(features[:, :1], count_model) > 0,
                   "repeat_and_agreement": predict_difference(features, full_model) > 0,
                   "shuffled_agreement": predict_difference(features[shuffle], full_model) > 0}
        for name, chosen in actions.items():
            pieces[name] = np.where(chosen.numpy(), amplified, first)
        pieces["reference_best_of_two"] = np.maximum(first, amplified)
        result[str(n)] = {
            "ocf": {name: summarize([scores], [first]) for name, scores in pieces.items()},
            "agreement_vs_fixed_amplified": summarize([pieces["repeat_and_agreement"]], [amplified]),
            "agreement_vs_counts_only": summarize([pieces["repeat_and_agreement"]], [pieces["repeat_count_only"]]),
            "agreement_vs_shuffled": summarize([pieces["repeat_and_agreement"]], [pieces["shuffled_agreement"]]),
            "switched_to_amplified_fraction": {name: float(active.float().mean()) for name, active in actions.items()},
            "weight_options_same_glyph_fraction": float(equal.float().mean()),
            "mean_shape_agreement": float(features[:, 1].mean()),
            "mean_neural_agreement": float(features[:, 2].mean()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=18917)
    parser.add_argument("--calibration-glyphs", type=int, default=144)
    parser.add_argument("--ocf-glyphs", type=int, default=144)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_repeat_agreement.json"))
    args = parser.parse_args()
    if not 2 <= args.ocf_glyphs <= 320 or not 2 <= args.calibration_glyphs <= 512:
        parser.error("Choose valid disjoint OCF partitions.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use the mounted Kaggle competition input.")
    torch.set_num_threads(4)
    torch.manual_seed(MODEL_SEED)
    started = time.monotonic()
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    fit, calibration, held = splits(len(images), args.seed, args.calibration_glyphs)
    held = held[:args.ocf_glyphs]
    rng = np.random.default_rng(args.seed + 44)
    model_images = images[fit]
    atlas, provenance, codepoints = build_font_atlas(model_images[rng.choice(len(fit), 128, replace=False)],
                                                    device, return_codepoints=True)
    aligned, _ = align_training_glyphs(atlas, model_images)
    prior = fit_atlas_prior(atlas, codepoints, model_images)
    style, _ = fit_aligned_font_style(aligned, model_images, seed=MODEL_SEED)
    rendered = torch.cat([(style.apply(batch[:, None].float() / 255)[:, 0] >= 0.5).to(torch.uint8) * 255
                          for batch in atlas.images.split(1024)])
    groups = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    training = {day: counts[fit] for day, counts in groups.items()}
    contour = fit_contour_retrieval(training, *contour_signatures(aligned), task="task2", seed=MODEL_SEED)
    signature, energy = contour_signatures(atlas.images)
    contour = replace(contour, signature=signature, energy=energy)
    encoder = fit_glyph_encoder(training, aligned, epochs=18, seed=MODEL_SEED)
    all_rows = torch.cat((calibration, held))
    choices = decisions(groups, all_rows, contour, encoder, atlas, prior, args.seed)
    scorer = PairedOCF(official_scorer(root), rendered, (images[all_rows] >= 128).numpy(), device, 12)
    outcomes = paired_outcomes(choices, scorer)
    cal, outer = ({n: tuple(value[selector] for value in values) for n, values in outcomes.items()}
                  for selector in (slice(None, len(calibration)), slice(len(calibration), None)))
    full_model, count_model = train_weights(cal)
    report = {"mouse": "mice1", "seed": args.seed, "training_glyphs": len(fit),
              "fit_weight_glyphs": len(calibration), "scored_glyphs": len(held), "atlas": provenance,
              "fixed_weights": {"base": BASE, "amplified": AMPLIFIED}, "ridge": RIDGE,
              "training_coefficients": {"repeat_and_agreement": full_model[2].tolist(),
                                        "repeat_count_only": count_model[2].tolist()},
              "protocol": "Fit the OCF difference between two fixed atlas decisions on glyphs excluded from "
                          "both the decoder fit and outer evaluation. Within each glyph measured trial "
                          "prefixes from four known recording days form 4/8/16 repeat proxies. Measure "
                          "agreement between disjoint pairs of days before adding the public lexical prior. "
                          "No sample identifiers, official test queries or official-test labels are read.",
              "exact_pairs": len(scorer.values), "held_ocf_by_repeat_count": evaluate(outer, full_model, count_model, args.seed + 7),
              "seconds": time.monotonic() - started}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ocf_by_repeat_budget": report["held_ocf_by_repeat_count"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
