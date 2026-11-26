"""Test local spike-conditioned stroke edits of an otherwise fixed Task 1 glyph.

The same atlas shortlist, registration, and edit budget are used for all edits.
A fit-only readout predicts local signed ink distances. Separate calibration
glyphs teach pixel risk given the spike-only shortlist. Exact OCF on new outer
glyphs compares learned neural evidence, atlas evidence, and shuffled spikes.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_atlas_prior import fit_atlas_prior
from glyph_contour_retrieval import contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import align_training_glyphs, standard_error, standardized
from kaggle_task1_distance_geometry import distance_fields, fitted_estimates, transform_spikes
from kaggle_task1_repertoire_coverage import independent_cohorts
from kaggle_task1_repertoire_ocf import fit_atlas


MOUSE = "mice1"
DEPTH = 8
EDIT_FRACTION = 0.08
DIRECTION_WEIGHT = 0.65
TEMPERATURE = 0.55
EDIT_NAMES = ("anchor", "atlas_edit", "neural_edit", "shuffled_neural_edit")


@torch.no_grad()
def shortlist(encoder, decoder, atlas, counts, session, prior):
    shape = standardized(contour_retrieval_scores(decoder, counts, session)["contour_ridge"])
    rates = encoder.expected_counts(atlas.images, session, batch_size=192)
    forward = standardized(0.5 * (standardized(count_template_logits(counts, rates, "rate"))
                                  + standardized(count_template_logits(counts, rates, "timed"))))
    fused = prior.variants(DIRECTION_WEIGHT * shape + (1 - DIRECTION_WEIGHT) * forward)["usage_unseen"]
    return fused.topk(DEPTH, dim=1)


def register(candidates):
    """Translate the same fixed eight bitmaps by at most two pixels to anchor them."""
    source = candidates[:, 1:].float()
    anchor = candidates[:, :1].float()
    padded = F.pad(source, (2, 2, 2, 2))
    source_mass = source.sum(dim=(-2, -1))
    anchor_mass = anchor.sum(dim=(-2, -1))
    best_score = torch.full_like(source_mass, -torch.inf)
    best = torch.zeros_like(source)
    for vertical in (0, 2, 4):
        for horizontal in (0, 2, 4):
            shifted = padded[..., vertical:vertical + 64, horizontal:horizontal + 64]
            intersection = (shifted * anchor).sum(dim=(-2, -1))
            penalty = 0.002 * (abs(vertical - 2) + abs(horizontal - 2))
            score = 2 * intersection / (source_mass + anchor_mass).clamp_min(1) - penalty
            better = score > best_score
            best = torch.where(better[..., None, None], shifted, best)
            best_score = torch.maximum(score, best_score)
    return torch.cat((anchor, best), dim=1)


def blur(value, width=3):
    return F.avg_pool2d(value, width, stride=1, padding=width // 2)


def atlas_columns(aligned, logits):
    anchor = aligned[:, :1]
    direction = 1 - 2 * anchor
    votes = (aligned * (logits / TEMPERATURE).softmax(dim=1)[..., None, None]).sum(dim=1, keepdim=True)
    count = aligned[:, 1:].mean(dim=1, keepdim=True)
    support = (votes - anchor) * direction
    alternative = (count - anchor) * direction
    available = (alternative > 0) & (blur((alternative > 0).float()) >= 1 / 3)
    gap = (logits[:, :1] - logits[:, 1:2])[:, :, None, None].expand_as(anchor)
    rows = torch.cat((anchor, support, alternative, blur(support), blur(alternative), blur(anchor, 5), gap), dim=1)
    return rows, available[:, 0], anchor[:, 0].bool()


def fit_geometry(training, images, decoder):
    geometry = distance_fields(images)[:, :320]
    center = geometry.mean(dim=0)
    projected = {name: transform_spikes(counts, decoder.moments[name], decoder.projection)[:, None]
                 for name, counts in training.items()}
    return fitted_estimates(projected, geometry - center), center


@torch.no_grad()
def estimate_geometry(model, center, decoder, counts, session):
    estimate = model.predict(transform_spikes(counts, decoder.moments[session], decoder.projection)) + center
    patches = (estimate[:, :64].reshape(-1, 1, 8, 8), estimate[:, 64:].reshape(-1, 1, 16, 16))
    return torch.cat([F.interpolate(field, size=(64, 64), mode="bilinear", align_corners=False)
                      for field in patches], dim=1)


def neural_columns(fields, anchor):
    direction = (1 - 2 * anchor.float())[:, None]
    return direction * fields


def calibration_pixels(columns, mask, target, generator):
    """Take at most 128 valid pixels per glyph, independently of their labels."""
    keys = torch.rand(mask.shape, generator=generator).flatten(1)
    valid = mask.flatten(1)
    sampled = keys.masked_fill(~valid, -1).topk(128, dim=1).indices
    keep = valid.gather(1, sampled)
    flat = columns.flatten(2).transpose(1, 2)
    picked = flat.gather(1, sampled[..., None].expand(-1, -1, flat.shape[-1]))
    return picked[keep], target.flatten(1).gather(1, sampled)[keep]


def fit_pixel_risk(columns, correct_flip):
    """Fit a regularized small conditional probability model for a beneficial flip."""
    center = columns.mean(dim=0)
    spread = columns.std(dim=0, unbiased=False).clamp_min(0.025)
    inputs = ((columns - center) / spread).clamp(-6, 6)
    weight = torch.nn.Parameter(torch.zeros(inputs.shape[1]))
    bias = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS((weight, bias), max_iter=24, lr=1.0, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        predicted = inputs @ weight + bias
        loss = F.binary_cross_entropy_with_logits(predicted, correct_flip.float()) + 0.015 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        predicted = inputs @ weight + bias
        report = {"pixels": len(inputs), "beneficial_flip_fraction": float(correct_flip.float().mean()),
                  "fit_pixel_log_loss": float(F.binary_cross_entropy_with_logits(predicted, correct_flip.float())),
                  "standardized_weights": weight.tolist()}
        model = (center[None, :, None, None], spread[None, :, None, None], weight.detach()[None, :, None, None],
                 bias.detach())
    return model, report


@torch.no_grad()
def edit_at_fixed_budget(model, columns, available, anchor):
    center, spread, weight, bias = model
    risk = ((((columns - center) / spread).clamp(-6, 6) * weight).sum(dim=1) + bias).sigmoid()
    local_valid = blur(available[:, None].float())[:, 0]
    neighborhood = blur((risk * available)[:, None])[:, 0] / local_valid.clamp_min(1e-6)
    priority = (0.70 * risk + 0.30 * neighborhood).masked_fill(~available, -torch.inf)
    budget = torch.minimum((anchor.sum(dim=(-2, -1)) * EDIT_FRACTION).round().long(), available.sum(dim=(-2, -1)))
    ranking = priority.flatten(1).argsort(dim=1, descending=True)
    flip = torch.zeros_like(anchor.flatten(1)).scatter_(1, ranking, torch.arange(64 * 64)[None] < budget[:, None])
    return anchor ^ flip.reshape_as(anchor), budget


def shuffled_rows(size, generator):
    order = torch.randperm(size, generator=generator)
    shuffled = torch.empty_like(order)
    shuffled[order] = order.roll(1)
    return shuffled


def exact_score(prediction, references, scorer, cache):
    # Reuse exact scores on unchanged bitmaps within a held glyph across arms and days.
    packed = np.packbits(prediction.numpy(), axis=None).reshape(len(prediction), -1)
    keys = [(row, mask.tobytes()) for row, mask in enumerate(packed)]
    missed = np.asarray([row for row, key in enumerate(keys) if key not in cache], dtype=np.int64)
    if len(missed):
        result = scorer(prediction[missed].numpy(), references[missed], device="cpu", batch_size=12)
        cache.update((keys[row], float(score)) for row, score in zip(missed, result))
    return np.asarray([cache[key] for key in keys])


def paired(values, anchor):
    values, anchor = np.stack(values), np.stack(anchor)
    diff = (values - anchor).mean(axis=0)
    return {"ocf": float(values.mean()), "paired_delta": float(diff.mean()),
            "paired_glyph_se": standard_error(diff), "pairs": int(values.size), "held_glyphs": len(diff)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=9103)
    parser.add_argument("--ocf-glyphs", type=int, default=128)
    parser.add_argument("--calibration-glyphs", type=int, default=160)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_stroke_edit_validation.json"))
    args = parser.parse_args()
    if not (2 <= args.ocf_glyphs <= 320 and 2 <= args.calibration_glyphs <= 256):
        parser.error("Use valid whole-glyph calibration and exact OCF sample sizes.")
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run on Kaggle CPU with competition data and Internet enabled.")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed + 2)
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, "cpu")
    core, inner, outer, previous = independent_cohorts(len(images), args.seed, exclude_coverage_seed=7933)
    _, _, prior_ocf, _ = independent_cohorts(len(images), 8291, exclude_coverage_seed=7933)
    eligible = outer[~torch.isin(outer, prior_ocf)]
    if len(eligible) < args.ocf_glyphs:
        raise ValueError("Too few fresh outer glyphs remain after prior validation exclusions.")
    cal = inner[:args.calibration_glyphs]
    fit = torch.cat((core, inner[args.calibration_glyphs:])).sort().values
    sample = eligible[:args.ocf_glyphs]
    training = images[fit]
    print(f"Fit atlas, directional/Poisson scores and local geometry on {len(fit)} glyphs", flush=True)
    atlas, rendered, codepoints, font = fit_atlas(training, args.seed)
    prior = fit_atlas_prior(atlas, codepoints, training)
    aligned, _ = align_training_glyphs(atlas, training)
    groups = grouped_blocks(root, load_catalog(root), MOUSE, "cpu", len(images))
    if len(groups) != 4 or any(values.shape[1] != 4 for values in groups.values()):
        raise ValueError("Four measured trials from each of four training days are required.")
    train_counts = {name: counts[fit] for name, counts in groups.items()}
    decoder = fit_contour_retrieval(train_counts, *contour_signatures(aligned), task="task2", seed=args.seed)
    signature, energy = contour_signatures(atlas.images)
    decoder = replace(decoder, signature=signature, energy=energy)
    encoder = fit_glyph_encoder(train_counts, aligned, epochs=18, seed=args.seed)
    distance, center = fit_geometry(train_counts, training, decoder)
    del train_counts

    cal_pixels, cal_labels, held = [], [], {}
    for session, counts in groups.items():
        query = counts[torch.cat((cal, sample))]
        logits, candidates = shortlist(encoder, decoder, atlas, query, session, prior)
        columns, available, anchor = atlas_columns(register(rendered[candidates] >= 128), logits)
        fields = estimate_geometry(distance, center, decoder, query, session)
        fit_row = slice(len(cal))
        outer_row = slice(len(cal), None)
        local = neural_columns(fields[fit_row], anchor[fit_row])
        correct_flip = anchor[fit_row] != (images[cal] >= 128)
        data, labels = calibration_pixels(torch.cat((columns[fit_row], local), dim=1), available[fit_row],
                                          correct_flip, generator)
        cal_pixels.append(data)
        cal_labels.append(labels)
        held[session] = columns[outer_row].clone(), available[outer_row].clone(), anchor[outer_row].clone(), fields[outer_row].clone()
        print(json.dumps({"session": session, "calibration_pixels": len(data)}), flush=True)
    evidence, label = torch.cat(cal_pixels), torch.cat(cal_labels)
    if len(label) == 0 or not label.any() or label.all():
        raise ValueError("The edit calibration requires both beneficial and harmful available pixel flips.")
    atlas_risk, atlas_fit = fit_pixel_risk(evidence[:, :-2], label)
    neural_risk, neural_fit = fit_pixel_risk(evidence, label)
    del groups, encoder, decoder, atlas, rendered, evidence
    print(json.dumps({"atlas_calibration": atlas_fit, "neural_calibration": neural_fit}), flush=True)

    refs = (images[sample] >= 128).numpy()
    score = official_scorer(root)
    cache, by_name, blocks = {}, {name: [] for name in EDIT_NAMES}, []
    result = {"seed": args.seed, "mouse": MOUSE, "fit_glyphs": len(fit), "calibration_glyphs": len(cal),
              "held_glyphs": len(outer), "exact_paired_glyphs_per_day": len(sample),
              "prior_holdout_exclusions": previous, "additional_exclusion_from_prior_repertoire_ocf_in_held": len(outer) - len(eligible),
              "depth": DEPTH, "direction_weight": DIRECTION_WEIGHT, "atlas_temperature": TEMPERATURE,
              "max_edited_fraction_of_anchor_ink": EDIT_FRACTION, "font": font, "public_prior": prior.diagnostics,
              "pixel_feature_order": ["anchor_ink", "posterior_disagreement", "unweighted_disagreement",
                                      "nearby_posterior_disagreement", "nearby_unweighted_disagreement",
                                      "nearby_anchor_ink", "anchor_score_margin", "signed_neural_sdf8", "signed_neural_sdf16"],
              "atlas_calibration": atlas_fit, "neural_calibration": neural_fit, "sessions": blocks,
              "primary_comparison": "Neural edits versus atlas-only edits at identical edit budgets and registration, "
                                    "with a separate exact contrast against the untouched anchor.",
              "protocol": "No official queries. Fit the CPU anchor and local signed distance readout on core glyphs, "
                          "then fit two regularized pixel models on separate glyphs with real spike-based anchor "
                          "errors. For each outer glyph, edit candidates must lie within a locally supported shortlist "
                          "disagreement. The same anchor and 8% pixel budget apply to all edits, including the "
                          "within-day deranged neural fields. Outer images enter only official OCF; paired errors "
                          "are clustered by glyph across days."}
    for name, (columns, mask, anchor, fields) in held.items():
        local = neural_columns(fields, anchor)
        permuted = neural_columns(fields[shuffled_rows(len(fields), generator)], anchor)
        atlas_edit, budget = edit_at_fixed_budget(atlas_risk, columns, mask, anchor)
        neural_edit, _ = edit_at_fixed_budget(neural_risk, torch.cat((columns, local), dim=1), mask, anchor)
        shuffled, _ = edit_at_fixed_budget(neural_risk, torch.cat((columns, permuted), dim=1), mask, anchor)
        choices = dict(zip(EDIT_NAMES, (anchor, atlas_edit, neural_edit, shuffled)))
        actual = {method: exact_score(bitmaps, refs, score, cache) for method, bitmaps in choices.items()}
        for method, value in actual.items():
            by_name[method].append(value)
        block = {"session": name, "vs_anchor": {method: paired([value], [actual["anchor"]]) for method, value in actual.items()},
                 "neural_vs_atlas": paired([actual["neural_edit"]], [actual["atlas_edit"]]),
                 "neural_vs_shuffled": paired([actual["neural_edit"]], [actual["shuffled_neural_edit"]]),
                 "glyphs_edited": int((budget > 0).sum()), "mean_edited_pixels": float(budget.float().mean()),
                 "neural_and_atlas_edited_bitmaps_differ": int((neural_edit != atlas_edit).flatten(1).any(1).sum())}
        blocks.append(block)
        result["summary"] = {"vs_anchor": {method: paired(values, by_name["anchor"]) for method, values in by_name.items()},
                             "neural_vs_atlas": paired(by_name["neural_edit"], by_name["atlas_edit"]),
                             "neural_vs_shuffled": paired(by_name["neural_edit"], by_name["shuffled_neural_edit"]),
                             "unique_mask_ocf_pairs": len(cache)}
        result["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(block), flush=True)
    print(json.dumps({"summary": result["summary"], "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
