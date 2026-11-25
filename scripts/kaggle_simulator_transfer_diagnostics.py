"""Measure whether calibrating simulated responses improves held-glyph retrieval.

Each glyph belongs to one of three disjoint groups inside a predefined Task 1
training fold. The first group fits the font transform, visual encoder and
neural projection. The second estimates the simulation-to-recording map and
noise. The third evaluates that frozen map on a closed set of labeled glyphs
and compares it with retrieval between disjoint halves of recorded trials.
This diagnostic uses competition training records only and saves aggregates.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_contour_retrieval import _project_training, _response_features
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs
from glyph_encoding import fit_glyph_encoder
from glyph_predictive_likelihood import covariance, regularize
from glyph_retrieval import _standardize
from glyph_synthetic_contour import N_BINS, VIEW, calibrate_response, simulated_views
from kaggle_gallery_validation import align_training_glyphs, build_font_atlas, standard_error


SESSIONS = ("session1", "session2", "session3", "session4")
HALF_SPLITS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
OUTCOMES = ("top1", "top8", "reciprocal_rank")


def project_trials(counts, moments, projection):
    features = _response_features(counts, VIEW, N_BINS)
    return _standardize(features, *moments) @ projection


def repeat_covariance(trials):
    glyphs, repeats, _ = trials.shape
    residual = (trials - trials.mean(dim=1, keepdim=True)).flatten(0, 1)
    return covariance(residual, glyphs * (repeats - 1))


def discrepancy_covariance(real, simulated, real_noise, sim_noise):
    residual = real.mean(dim=1) - simulated.mean(dim=1)
    centered = residual - residual.mean(dim=0)
    scatter = covariance(centered, len(centered) - 1)
    unexplained = scatter - real_noise / real.shape[1] - sim_noise / simulated.shape[1]
    eigenvalues, vectors = torch.linalg.eigh(regularize(unexplained, 0.35))
    return (vectors * eigenvalues.clamp_min(0)) @ vectors.T


def crossfit_calibration(simulated, real_mean):
    """Measure discrepancy on calibration glyphs excluded from their own maps."""
    simulated_mean = simulated.mean(dim=1)
    fold = torch.arange(len(simulated), device=simulated.device).remainder(4)
    adjusted = torch.empty_like(simulated)
    for index in range(4):
        held = fold == index
        mapping = calibrate_response(simulated_mean[~held], real_mean[~held])
        adjusted[held] = mapping.predict(simulated[held])
    return adjusted


def retrieval_outcomes(query, template, noise=None):
    if noise is not None:
        noise = noise.clone()
        noise.diagonal().add_(0.05 * noise.diagonal().mean().clamp_min(1e-5))
        factor = torch.linalg.cholesky(noise)
        query = torch.linalg.solve_triangular(factor, query.T.contiguous(), upper=False).T
        template = torch.linalg.solve_triangular(factor, template.T.contiguous(), upper=False).T
    scores = query @ template.T - 0.5 * template.square().sum(dim=1)
    correct = scores.diagonal()[:, None]
    better = (scores > correct).sum(dim=1)
    tied = (scores == correct).sum(dim=1)
    # Average over tied ranks, which can occur when two glyphs map to the same font image.
    harmonic = torch.arange(1, len(template) + 1, device=scores.device, dtype=scores.dtype).reciprocal().cumsum(0)
    harmonic = torch.cat((scores.new_zeros(1), harmonic))
    reciprocal = (harmonic[better + tied] - harmonic[better]) / tied
    top1 = ((1 - better) / tied).clamp(0, 1)
    top8 = ((8 - better) / tied).clamp(0, 1)
    return torch.stack((top1, top8, reciprocal), dim=1)


def mse(reference, prediction):
    return float((reference - prediction).square().mean())


def score_outcomes(outcomes, reference, reference_name):
    values = outcomes.mean(dim=0)
    delta = (outcomes - reference).mean(dim=0).cpu().numpy()
    return {"reference": reference_name,
            **{name: {"value": 100 * float(values[:, column].mean()),
                      "delta_vs_reference": 100 * float(delta[:, column].mean()),
                      "paired_glyph_se": 100 * standard_error(delta[:, column])}
               for column, name in enumerate(OUTCOMES)}}


@torch.no_grad()
def evaluate_session(encoder, session, real_b, real_c, masks_b, masks_c, moments, projection,
                     *, repeats, seed):
    generator = torch.Generator(device=projection.device).manual_seed(seed)
    rate = encoder.expected_counts(torch.cat((masks_b, masks_c)), session, batch_size=192)
    simulated = simulated_views(rate, moments, projection, generator, repeats=1, groups=repeats)
    sim_b, sim_c = simulated.split((len(real_b), len(real_c)))
    real_b_mean, real_c_mean = real_b.mean(dim=1), real_c.mean(dim=1)
    raw_b_mean, raw_c_mean = sim_b.mean(dim=1), sim_c.mean(dim=1)
    offset = (real_b_mean - raw_b_mean).mean(dim=0)
    affine = calibrate_response(raw_b_mean, real_b_mean)
    adjusted_b_mean = affine.predict(raw_b_mean)
    adjusted_c_mean = affine.predict(raw_c_mean)
    crossfit_b = crossfit_calibration(sim_b, real_b_mean)

    real_noise = repeat_covariance(real_b)
    sim_noise = repeat_covariance(crossfit_b)
    discrepancy = discrepancy_covariance(real_b, crossfit_b, real_noise, sim_noise)
    repeat_noise = regularize(real_noise, 0.30) / real_c.shape[1] + regularize(sim_noise, 0.30) / repeats
    expected_noise = repeat_noise + discrepancy
    metrics = {
        "calibration_in_sample_mse": {"raw": mse(real_b_mean, raw_b_mean),
                                      "offset": mse(real_b_mean, raw_b_mean + offset),
                                      "affine": mse(real_b_mean, adjusted_b_mean)},
        "calibration_crossfit_affine_mse": mse(real_b_mean, crossfit_b.mean(dim=1)),
        "independent_evaluation_mse": {"raw": mse(real_c_mean, raw_c_mean),
                                       "offset": mse(real_c_mean, raw_c_mean + offset),
                                       "affine": mse(real_c_mean, adjusted_c_mean)},
        "variance_per_axis": {"fit_recorded_trial": float(real_noise.diagonal().mean()),
                              "fit_affine_simulated_trial": float(sim_noise.diagonal().mean()),
                              "fit_model_discrepancy": float(discrepancy.diagonal().mean()),
                              "predicted_affine_residual": float(expected_noise.diagonal().mean()),
                              "observed_affine_residual": mse(real_c_mean, adjusted_c_mean)},
    }
    outcomes = {
        "raw_simulation": retrieval_outcomes(real_c_mean, raw_c_mean),
        "offset_simulation": retrieval_outcomes(real_c_mean, raw_c_mean + offset),
        "affine_simulation": retrieval_outcomes(real_c_mean, adjusted_c_mean),
        "affine_repeat_whitened": retrieval_outcomes(real_c_mean, adjusted_c_mean, repeat_noise),
        "affine_predictive": retrieval_outcomes(real_c_mean, adjusted_c_mean, expected_noise),
    }
    real_reference, raw_halves, affine_halves, repeat_agreement = [], [], [], []
    for left, right in HALF_SPLITS:
        a, b = real_c[:, left].mean(dim=1), real_c[:, right].mean(dim=1)
        centered_a, centered_b = a - a.mean(dim=0), b - b.mean(dim=0)
        agreement = (centered_a * centered_b).sum() / (centered_a.norm() * centered_b.norm()).clamp_min(1e-8)
        repeat_agreement.append(float(agreement))
        for query, other_half in ((a, b), (b, a)):
            real_reference.append(retrieval_outcomes(query, other_half))
            raw_halves.append(retrieval_outcomes(query, raw_c_mean))
            affine_halves.append(retrieval_outcomes(query, adjusted_c_mean))
    outcomes["real_split_half_reference"] = torch.stack(real_reference).mean(dim=0)
    outcomes["raw_simulation_two_repeat_queries"] = torch.stack(raw_halves).mean(dim=0)
    outcomes["affine_simulation_two_repeat_queries"] = torch.stack(affine_halves).mean(dim=0)
    metrics["split_half_recorded_response_correlation"] = float(np.mean(repeat_agreement))
    return metrics, outcomes


def comparison_reference(name):
    if name == "affine_predictive":
        return "affine_repeat_whitened"
    if name in {"real_split_half_reference", "raw_simulation_two_repeat_queries",
                "affine_simulation_two_repeat_queries"}:
        return "raw_simulation_two_repeat_queries"
    return "raw_simulation"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=961)
    parser.add_argument("--mouse", choices=("mice1", "mice2"), default="mice1")
    parser.add_argument("--sessions", choices=SESSIONS, nargs="+", default=SESSIONS)
    parser.add_argument("--calibration-glyphs", type=int, default=192)
    parser.add_argument("--evaluation-glyphs", type=int, default=188)
    parser.add_argument("--latent-dims", type=int, default=64)
    parser.add_argument("--simulation-repeats", type=int, default=32)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--output", type=Path,
                        default=Path("/kaggle/working/simulator_transfer_diagnostics.json"))
    args = parser.parse_args()
    if min(args.calibration_glyphs, args.evaluation_glyphs) < 8 or args.simulation_repeats < 2:
        parser.error("Use at least eight glyphs per calibration or evaluation fold and two simulation repeats.")
    if not 1 <= args.latent_dims <= 192 or args.encoder_epochs < 1:
        parser.error("Choose 1–192 stable dimensions and at least one encoder epoch.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run this diagnostic on a Kaggle GPU with competition input and Internet enabled.")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    outer_fit, _ = split_glyphs(len(images), 320, args.seed, device)
    order = np.random.default_rng(args.seed + 17).permutation(len(outer_fit))
    required = args.calibration_glyphs + args.evaluation_glyphs
    if required > len(outer_fit) - 128:
        parser.error("Leave at least 128 glyphs in the encoder fitting split.")
    calibration, evaluation, fit = outer_fit[torch.as_tensor(order, device=device)].split(
        (args.calibration_glyphs, args.evaluation_glyphs, len(outer_fit) - required))
    style_indices = np.random.default_rng(args.seed + 2).choice(len(fit), 128, replace=False)
    atlas, provenance = build_font_atlas(images[fit[torch.as_tensor(style_indices, device=device)]], device)
    aligned_fit, agreement_fit = align_training_glyphs(atlas, images[fit])
    aligned_b, agreement_b = align_training_glyphs(atlas, images[calibration])
    aligned_c, agreement_c = align_training_glyphs(atlas, images[evaluation])
    del atlas
    grouped = grouped_blocks(root, catalog, args.mouse, device, len(images))
    if any(values.shape[1] != 4 for values in grouped.values()):
        raise ValueError("The repeat reference requires the four original training trials per glyph.")
    training = {name: counts[fit] for name, counts in grouped.items()}
    print(f"Encoder fit on {len(fit)} glyphs, calibrate on {len(calibration)}, evaluate on {len(evaluation)}",
          flush=True)
    encoder = fit_glyph_encoder(training, aligned_fit, epochs=args.encoder_epochs, seed=args.seed)
    projected, moments, projection = _project_training(training, VIEW, N_BINS)
    projection = projection[:, -args.latent_dims:].contiguous()
    del projected, training
    output = {"seed": args.seed, "mouse": args.mouse, "atlas": provenance,
              "fit_glyphs": len(fit), "calibration_glyphs": len(calibration), "evaluation_glyphs": len(evaluation),
              "latent_dims": projection.shape[1], "simulation_repeats": args.simulation_repeats,
              "encoder_epochs": args.encoder_epochs, "recorded_repeats": 4,
              "alignment_agreement": {"fit": agreement_fit, "calibration": agreement_b, "evaluation": agreement_c},
              "metric_units": "percent of evaluation glyphs, or 100 times mean reciprocal rank",
              "protocol": "Use a disjoint inner fit, calibration and closed gallery evaluation, without using the "
                          "outer Task 1 glyphs. Record both directions of all three recorded two-trial half splits.",
              "blocks": {}}
    collected = {}
    for index, session in enumerate(dict.fromkeys(args.sessions)):
        real_b = project_trials(grouped[session][calibration], moments[session], projection)
        real_c = project_trials(grouped[session][evaluation], moments[session], projection)
        diagnostics, outcomes = evaluate_session(encoder, session, real_b, real_c, aligned_b, aligned_c,
                                                  moments[session], projection,
                                                  repeats=args.simulation_repeats, seed=args.seed + 40 + index)
        diagnostics["retrieval"] = {}
        for name, values in outcomes.items():
            reference = comparison_reference(name)
            diagnostics["retrieval"][name] = score_outcomes(values[None], outcomes[reference][None], reference)
            collected.setdefault(name, []).append(values)
        output["blocks"][session] = diagnostics
        output["summary"] = {name: score_outcomes(torch.stack(values),
                                                   torch.stack(collected[comparison_reference(name)]),
                                                   comparison_reference(name))
                             for name, values in collected.items()}
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": session, **diagnostics}), flush=True)


if __name__ == "__main__":
    main()
