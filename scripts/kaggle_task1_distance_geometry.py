"""Decode glyph distance geometry from spikes and retrieve a fixed public atlas.

Learn a coarse signed layout, a finer contour distance and the distance to
stroke ridges. Calibrate the prediction-error covariance on inner held glyphs.
Every public ranking sees only four measured repeats for a new glyph. A
matched CPU control uses directional contours, evoked Poisson likelihood,
the same public atlas, and the same priors and rendering.
"""

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter
import torch

from glyph_atlas_prior import fit_atlas_prior
from glyph_contour_retrieval import (_fit_readout, _project_training, _response_features,
                                     contour_retrieval_scores, fit_contour_retrieval)
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_font_style import fit_aligned_font_style
from glyph_mbr import contour_signatures, contour_utility
from glyph_retrieval import _standardize
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import align_training_glyphs, build_font_atlas, standard_error, standardized


GEOMETRY_WEIGHT = 0.25
DIRECTION_WEIGHT = 0.65
SCALES = (64, 256, 256)
SESSIONS = ("session2", "session4")
PRIMARY = "distance_covariance_blend"


def pool_field(values, width):
    tile = 64 // width
    return values.reshape(width, tile, width, tile).mean(axis=(1, 3)).ravel()


def distance_fields(images):
    """Encode normalized distances while retaining where strokes lie within the canvas."""
    masks = (images >= 128).cpu().numpy()
    features = np.empty((len(masks), sum(SCALES)), dtype=np.float32)
    for row, mask in enumerate(masks):
        interior = distance_transform_edt(mask)
        signed = interior - distance_transform_edt(~mask)
        ridge = mask & (interior >= maximum_filter(interior, size=3) - 1e-6)
        envelope = np.exp(-distance_transform_edt(~ridge) / 2.0)
        features[row] = np.concatenate((pool_field(np.tanh(signed / 8.0), 8),
                                        pool_field(np.tanh(signed / 2.5), 16),
                                        pool_field(envelope, 16)))
    return torch.from_numpy(features)


@dataclass
class GeometrySpace:
    center: torch.Tensor
    scale: torch.Tensor

    def transform(self, values):
        return (values - self.center) / self.scale


def fit_geometry_space(features):
    center = features.mean(dim=0)
    scale = torch.cat([band.square().sum(1).mean().sqrt().clamp_min(0.05).expand(band.shape[1])
                       for band in (features - center).split(SCALES, dim=1)])
    return GeometrySpace(center, scale)


def transform_spikes(counts, calibration, projection):
    value = _response_features(counts, "evoked-multiscale", 10)
    return _standardize(value, *calibration).mean(dim=1) @ projection


def fitted_estimates(views, targets):
    means = tuple(values.mean(dim=1, keepdim=True) for values in views.values())
    return _fit_readout(means, targets, ridge=0.09)


@torch.no_grad()
def fit_residual_geometry(training, targets, fit_rows, calibration_rows):
    nested = {name: counts[fit_rows] for name, counts in training.items()}
    projected, moments, projection = _project_training(nested, "evoked-multiscale", 10)
    model = fitted_estimates(projected, targets[fit_rows])
    errors, offsets = [], {}
    variances = {}
    for name, counts in training.items():
        estimate = model.predict(transform_spikes(counts[calibration_rows], moments[name], projection))
        residual = targets[calibration_rows] - estimate
        offsets[name] = residual.mean(dim=0)
        centered = residual - offsets[name]
        errors.append(centered.T @ centered / (len(centered) - 1))
        variances[name] = {"mean_prediction_error": float(residual.square().mean()),
                           "constant_prediction_error": float(targets[calibration_rows].square().mean())}
    covariance = torch.stack(errors).mean(dim=0)
    shrink = 0.50
    diagonal = covariance.diagonal().clamp_min(0.15 * covariance.diagonal().mean().clamp_min(1e-8))
    covariance = (1 - shrink) * covariance + shrink * torch.diag(diagonal)
    covariance.diagonal().add_(0.10 * diagonal.mean())
    factor = torch.linalg.cholesky(covariance)
    return factor, offsets, {"glyphs": len(calibration_rows), "diagonal_shrinkage": shrink, "blocks": variances}


def whiten(rows, lower):
    return torch.linalg.solve_triangular(lower, rows.T.contiguous(), upper=False).T.contiguous()


def distance_logits(prediction, bank):
    return prediction @ bank.T - 0.5 * bank.square().sum(dim=1)


@torch.no_grad()
def atlas_scores(encoder, contour, geometry, atlas, matrix, white_matrix, factor, offset, counts, session):
    directional = contour_retrieval_scores(contour, counts, session)["contour_ridge"]
    rate = encoder.expected_counts(atlas.images, session, batch_size=192)
    summed = standardized(count_template_logits(counts, rate, "rate"))
    temporal = standardized(count_template_logits(counts, rate, "timed"))
    likelihood = standardized(0.5 * (summed + temporal))
    directional_z = standardized(directional)
    baseline = DIRECTION_WEIGHT * directional_z + (1 - DIRECTION_WEIGHT) * likelihood
    estimate = geometry.predict(transform_spikes(counts, contour.moments[session], contour.projection))
    isotropic = standardized(distance_logits(estimate, matrix))
    adaptive = standardized(distance_logits(whiten(estimate + offset, factor), white_matrix))
    return {"matched_cpu_anchor": baseline, "directional_only": directional_z,
            "distance_only": adaptive,
            "distance_unweighted_blend": baseline.lerp(isotropic, GEOMETRY_WEIGHT),
            PRIMARY: baseline.lerp(adaptive, GEOMETRY_WEIGHT)}, estimate


@torch.no_grad()
def known_bank_diagnostic(contour, estimate, counts, session, features, image_signature, image_energy,
                          factor, offset, permutation):
    """Measure representation recovery only after public-atlas choices are fixed."""
    response = contour.readout.predict(transform_spikes(counts, contour.moments[session], contour.projection))
    energy = response[:, -1].clamp(image_energy.log().min(), image_energy.log().max()).exp()
    direction = contour_utility(response[:, :-1], energy, image_signature[permutation], image_energy[permutation])
    scores = {"directional": direction, "unweighted_distance": distance_logits(estimate, features[permutation]),
              "covariance_distance": distance_logits(whiten(estimate + offset, factor),
                                                     whiten(features[permutation], factor))}
    correct_column = permutation.argsort()
    return {name: {"top1": float((values.argmax(1) == correct_column).float().mean()),
                   "top5": float((values.topk(5, dim=1).indices == correct_column[:, None]).any(1).float().mean())}
            for name, values in scores.items()}


def summarize(observed, reference):
    values, control = np.stack(observed), np.stack(reference)
    delta = (values - control).mean(axis=0)
    return {"ocf": float(values.mean()), "delta": float(delta.mean()), "paired_glyph_se": standard_error(delta),
            "n": int(values.size)}


def independent_split(size, seed):
    avoided = set()
    for earlier in (961, 2209):
        avoided.update(np.random.default_rng(earlier).permutation(size)[:320].tolist())
    random = np.random.default_rng(seed)
    eligible = np.setdiff1d(np.arange(size), np.array(sorted(avoided)))
    outer = random.permutation(eligible)[:320]
    fit = np.setdiff1d(np.arange(size), outer)
    return torch.as_tensor(fit), torch.as_tensor(outer), len(avoided)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocf-images", type=int, default=96)
    parser.add_argument("--depth-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5431)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_distance_geometry.json"))
    args = parser.parse_args()
    if not 2 <= args.depth_images <= args.ocf_images <= 320 or args.ocf_images < 5:
        parser.error("Choose valid independent glyph and OCF sample sizes.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use an Internet-enabled Kaggle CPU Notebook with competition data mounted.")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    device = torch.device("cpu")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    fit, outer, excluded = independent_split(len(images), args.seed)
    sample = outer[:args.ocf_images]
    random = np.random.default_rng(args.seed + 2)
    training = images[fit]
    style_rows = torch.as_tensor(random.choice(len(training), 128, replace=False))
    atlas, provenance, codepoints = build_font_atlas(training[style_rows], device, return_codepoints=True)
    aligned, agreement = align_training_glyphs(atlas, training)
    prior = fit_atlas_prior(atlas, codepoints, training)
    style, style_fit = fit_aligned_font_style(aligned, training, seed=args.seed)
    rendered = torch.cat([(style.apply(chunk[:, None].float() / 255)[:, 0] >= 0.5).to(torch.uint8) * 255
                          for chunk in atlas.images.split(1024)])
    print("Extract three distance fields from the public atlas and fit glyphs", flush=True)
    arc = torch.as_tensor(random.permutation(len(fit)))
    covariance_rows, inner_rows = arc.split((320, len(fit) - 320))
    shape_features = distance_fields(aligned)
    shape_space = fit_geometry_space(shape_features[inner_rows])
    targets = shape_space.transform(shape_features)
    geometry_bank = shape_space.transform(distance_fields(atlas.images))

    grouped = grouped_blocks(root, catalog, "mice1", device, len(images))
    train_counts = {name: values[fit] for name, values in grouped.items()}
    if any(counts.shape[1] != 4 for counts in train_counts.values()):
        raise ValueError("The validation proxy must use the four real repeats per training glyph.")
    factor, offsets, diagnostics = fit_residual_geometry(train_counts, targets, inner_rows, covariance_rows)
    white_bank = whiten(geometry_bank, factor)
    print(json.dumps({"inner_distance_error": diagnostics}), flush=True)
    directional = fit_contour_retrieval(train_counts, *contour_signatures(aligned), task="task2", seed=args.seed)
    projected = {}
    for name, values in train_counts.items():
        projected[name] = transform_spikes(values, directional.moments[name], directional.projection)[:, None]
    geometry = fitted_estimates(projected, targets)
    contour_bank, contour_energy = contour_signatures(atlas.images)
    directional = replace(directional, signature=contour_bank, energy=contour_energy)
    encoder = fit_glyph_encoder(train_counts, aligned, epochs=18, seed=args.seed)
    del train_counts, projected
    output = {"seed": args.seed, "mouse": "mice1", "sdf_coordinates": sum(SCALES),
              "prior_validation_holdout_union_excluded": excluded, "fit_glyphs": len(fit), "held_glyphs": len(outer),
              "paired_glyphs_per_block": len(sample), "oracle_glyphs_per_block": args.depth_images,
              "atlas": provenance, "alignment_agreement": agreement, "style_fit": style_fit,
              "geometry_weight": GEOMETRY_WEIGHT, "direction_weight_in_matched_base": DIRECTION_WEIGHT,
              "primary": PRIMARY, "predictive_covariance": diagnostics,
              "matched_base": "65% standardized aligned directional contour ridge and 35% multi-window forward Poisson. "
                              "Shares lexical prior and style rendering with all variants. A CPU-matched proxy for the v8 family, "
                              "without the CNN and v9 exact-OCF reranker.",
              "held_protocol": "Use four observed repeats without synthetic pooling. Held glyph images enter only exact OCF "
                               "assessment and the separate known-bank representation diagnostic.", "folds": []}
    scorer = PairedOCF(official_scorer(root), rendered, (images[sample] >= 128).numpy(), device, 12)
    paired, oracle = {}, {}
    for session in SESSIONS:
        block = time.monotonic()
        query = grouped[session][sample]
        scores, estimate = atlas_scores(encoder, directional, geometry, atlas, geometry_bank, white_bank,
                                        factor, offsets[session], query, session)
        ranks = {name: prior.variants(values)["usage_unseen"].topk(16, dim=1).indices for name, values in scores.items()}
        top1 = {name: scorer.evaluate(candidates[:, :1])[:, 0] for name, candidates in ranks.items()}
        base = ranks["matched_cpu_anchor"][:args.depth_images]
        new_scores = prior.variants(scores[PRIMARY][:args.depth_images])["usage_unseen"].clone()
        novel = new_scores.scatter_(1, base, -torch.inf).topk(4, dim=1).indices
        union = torch.cat((base[:, :12], novel), dim=1)
        depth = {"base_top16": scorer.evaluate(base).max(axis=1), "same_budget_distance_union": scorer.evaluate(union).max(axis=1)}
        held_fields = shape_space.transform(distance_fields(images[sample]))
        held_signature, held_energy = contour_signatures(images[sample])
        known = known_bank_diagnostic(directional, estimate, query, session, held_fields, held_signature,
                                      held_energy, factor, offsets[session], torch.as_tensor(random.permutation(len(sample))))
        fold = {"session": session, "ocf": {name: summarize([values], [top1["matched_cpu_anchor"]]) for name, values in top1.items()},
                "known_image_probe": known, "oracle": {name: summarize([values], [depth["base_top16"]]) for name, values in depth.items()},
                "seconds": time.monotonic() - block}
        output["folds"].append(fold)
        for group, readings in ((paired, top1), (oracle, depth)):
            for name, values in readings.items():
                group.setdefault(name, []).append(values)
        output["summary"] = {"ocf": {name: summarize(values, paired["matched_cpu_anchor"]) for name, values in paired.items()},
                             "oracle": {name: summarize(values, oracle["base_top16"]) for name, values in oracle.items()}}
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(fold), flush=True)
    print(json.dumps({"summary": output["summary"], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
