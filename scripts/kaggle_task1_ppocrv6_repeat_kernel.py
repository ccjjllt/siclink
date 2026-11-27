"""Compare unbiased cross-repeat quadratic readouts of held PP-OCRv6 features."""

from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_atlas_prior import fit_atlas_prior
from glyph_bipartite_encoding import timing_view
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import build_font_atlas, standardized
from kaggle_task1_native_hanzi import RIDGE, feature_extractors, fit_response_ridge
from kaggle_task1_native_hanzi_open_atlas import image_coordinates, visual_proxy
from kaggle_task1_ppocrv6_readout import SEED as PPOCR_SEED, distinct_cohort, fit_coordinates, measure, read_glyphs, recognizer
from kaggle_task1_textverifier import publish
from kaggle_v8_metric_validation import SESSIONS


SEED, SCORED, DIMENSIONS = 225017, 112, 96
QUADRATIC_SHARE = 0.35
OUTPUT = Path("/kaggle/working/task1_ppocrv6_repeat_kernel.json")
STOCK, WHITE, SELF, CROSS = "ppocr_ridge", "repeat_whitened_linear", "pooled_square", "disjoint_half_product"


def precise_cohort(size, rng):
    _, mice2, _ = distinct_cohort(size, np.random.default_rng(PPOCR_SEED))
    _, mice1, _ = distinct_cohort(size, np.random.default_rng(PPOCR_SEED + 1), mice2)
    earlier = torch.cat((mice1, mice2))
    fit, held, overlaps = distinct_cohort(size, rng, earlier)
    overlaps["first_two_ppocr_pilots"] = overlaps.pop("ppocrv6_mice2")
    return fit, held, overlaps


def disjoint_halves(trials, rng):
    order = torch.from_numpy(rng.random(trials.shape[:2]).argsort(1))
    distinct = trials.gather(1, order[:, :, None].expand_as(trials))
    return distinct[:, :2].mean(1), distinct[:, 2:].mean(1)


def fit_repeat_whitening(training, rng):
    n, repeats, width = training.shape
    means = training.mean(1)
    center = means.mean(0)
    scale = means.std(0).clamp_min(0.04)
    train = (training - center) / scale
    residuals = (train - train.mean(1, keepdim=True)).flatten(0, 1)
    within = residuals.T @ residuals / (n * (repeats - 1))
    diagonal = within.diag().clamp_min(0.05)
    within.mul_(0.75).diagonal().add_(0.25 * diagonal + 0.05)
    lower = torch.linalg.cholesky(within)

    def white(values):
        return torch.linalg.solve_triangular(lower, values.T, upper=False).T

    left, right = map(white, disjoint_halves(train, rng))
    # Symmetric independent-half covariance estimates identity variation without repeat noise on its diagonal.
    signal = (left.T @ right + right.T @ left) / (2 * n)
    eigenvalues, vectors = torch.linalg.eigh(signal)
    axes = vectors[:, -min(width, DIMENSIONS):]
    radius = (0.5 * (left + right) @ axes).square().sum(1).mean().sqrt().clamp_min(1e-4)
    coordinates = ((left @ axes) / radius, (right @ axes) / radius)
    parameters = (center, scale, lower, axes, radius)
    return coordinates, parameters, {
        "rank": axes.shape[1], "last_retained_signal_eigenvalue": float(eigenvalues[-axes.shape[1]]),
        "first_signal_eigenvalue": float(eigenvalues[-1]), "mean_population_radius": float(radius),
    }


def whitened_projection(values, parameters):
    center, scale, lower, axes, radius = parameters
    standardized = (values - center) / scale
    white = torch.linalg.solve_triangular(lower, standardized.T, upper=False).T
    return (white @ axes) / radius


def whitened_halves(training, query, rng):
    (left, right), parameters, diagnostics = fit_repeat_whitening(training, rng)
    # Split after normalization to retain the original arithmetic for this experiment.
    center, scale, lower, axes, radius = parameters
    query_halves = disjoint_halves((query - center) / scale, rng)
    query_coordinates = tuple((torch.linalg.solve_triangular(lower, values.T, upper=False).T @ axes) / radius
                              for values in query_halves)
    return (left, right, *query_coordinates), diagnostics


def centered(gram, query):
    center = gram.mean(0, keepdim=True)
    centered_fit = gram - center - center.T + center.mean()
    centered_query = query - center - query.mean(1, keepdim=True) + center.mean()
    contrast = centered_fit.diag().mean().clamp_min(1e-5)
    return centered_fit / contrast, centered_query / contrast


def crossed_product(left, right, source_left, source_right):
    same = (left @ source_left.T) * (right @ source_right.T)
    alternate = (left @ source_right.T) * (right @ source_left.T)
    return 0.5 * (same + alternate)


def fit_kernel(fit, query, labels, penalty):
    fit = (fit + fit.T) * 0.5
    fit.diagonal().add_(penalty)
    coefficients = torch.cholesky_solve(labels, torch.linalg.cholesky(fit))
    return query @ coefficients


def nonlinear_readouts(training, query, labels, rng):
    (left, right, left_query, right_query), diagnostics = whitened_halves(training, query, rng)
    means = 0.5 * (left + right)
    queries = 0.5 * (left_query + right_query)
    linear_fit = means @ means.T
    linear_query = queries @ means.T
    linear = centered(linear_fit, linear_query)
    squared = centered(linear_fit.square(), linear_query.square())
    cross_fit = crossed_product(left, right, left, right)
    cross_query = crossed_product(left_query, right_query, left, right)
    cross = centered(cross_fit, cross_query)
    penalty = RIDGE * len(training) / means.shape[1]
    predictions = {WHITE: fit_kernel(*linear, labels, penalty)}
    for name, quadratic in ((SELF, squared), (CROSS, cross)):
        train, held = (base.lerp(extra, QUADRATIC_SHARE) for base, extra in zip(linear, quadratic, strict=True))
        predictions[name] = fit_kernel(train, held, labels, penalty)
    diagnostics["kernel_penalty"] = penalty
    return predictions, diagnostics


def compare_day(training, queries, base_coordinates, pp_coordinates, prior, rng):
    ridge, center, scale = fit_response_ridge(training.mean(1))
    observation = (queries.mean(1) - center) / scale
    old = [standardized(F.normalize(observation @ (ridge @ labels)) @ bank)
           for labels, bank in base_coordinates.values()]
    legacy = torch.stack(old).mean(0)
    pp_labels, pp_bank = pp_coordinates
    alternatives, diagnostics = nonlinear_readouts(training, queries, pp_labels, rng)
    predictions = {STOCK: observation @ (ridge @ pp_labels), **alternatives}
    result = {name: prior.variants(0.5 * (legacy + standardized(F.normalize(output) @ pp_bank)))["usage_unseen"]
              for name, output in predictions.items()}
    return result, diagnostics


@torch.inference_mode()
def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    root = locate_data()
    images, catalog = load_images(root, device), load_catalog(root)
    fit, held, overlaps = precise_cohort(len(images), rng)
    scored = torch.from_numpy(np.sort(rng.choice(len(held), SCORED, replace=False)))
    groups = grouped_blocks(root, catalog, "mice2", device, len(images))
    report = {"seed": SEED, "mouse": "mice2", "formal_queries_loaded": False, "fit_glyph_count": len(fit),
              "overlaps": overlaps, "held_gallery_indices": held.tolist(), "scored_gallery_indices": held[scored].tolist(),
              "quadratic_share": QUADRATIC_SHARE, "recordings": {},
              "protocol": "Freeze PP-OCRv6 visual features, contour/Hanzi scores, public atlas, render and prior. Use four "
              "independent measured repeats. In the same repeat-whitened coordinates compare linear, ordinary squared "
              "average and an unbiased cross-half degree-two feature. The quadratic kernels have equal trace scaling, "
              "ridge and linear mixing weight. Same-day four and known-correspondence sixteen use the same held glyphs."}
    session, metadata = recognizer()
    report["published_ocr"] = metadata
    publish(report, OUTPUT, started, "recognizer_ready")
    selected = rng.choice(fit.numpy(), size=128, replace=False)
    atlas, provenance, codepoints = build_font_atlas(images[selected], device, return_codepoints=True)
    encoded = read_glyphs(session, torch.cat((images[fit], atlas.images)), metadata)
    pp_coordinates = fit_coordinates(encoded["ppocrv6_global"], len(fit))
    del encoded, session
    extractors = feature_extractors()
    extractors.pop("random_mobilenet")
    base = image_coordinates(images[fit], atlas.images, extractors)
    prior = fit_atlas_prior(atlas, codepoints, images[fit])
    report.update(atlas=provenance, candidates=len(codepoints), unihan=prior.diagnostics)
    publish(report, OUTPUT, started, "frozen_representations_ready")

    scores = []
    for day in SESSIONS:
        counts = timing_view(groups[day].unflatten(-1, (-1, 10)))
        values, diagnostics = compare_day(counts[fit], counts[held], base, pp_coordinates, prior, rng)
        scores.append(values)
        report["recordings"][day] = diagnostics
        publish(report, OUTPUT, started, f"{day}_scored")
    stacked = {name: torch.stack([daily[name] for daily in scores]) for name in scores[0]}
    proxy, trusted, _ = visual_proxy(atlas, images[held])
    report["trusted_glyph_proxy_count"] = int(trusted.sum())
    day_order = torch.from_numpy(rng.permutation(len(held)) % len(SESSIONS))
    glyph_rows = torch.arange(len(held))
    exact = PairedOCF(official_scorer(root), atlas.images, (images[held[scored]] >= 128).numpy(), device, 12)
    report["same_day_four"] = measure({name: values[day_order, glyph_rows] for name, values in stacked.items()},
                                      scored, exact, proxy, trusted, baseline=STOCK)
    publish(report, OUTPUT, started, "same_day_exact")
    report["known_correspondence_sixteen"] = measure({name: values.mean(0) for name, values in stacked.items()},
                                                     scored, exact, proxy, trusted, baseline=STOCK)
    report["unique_exact_pairs"] = len(exact.values)
    publish(report, OUTPUT, started, "complete")


if __name__ == "__main__":
    main()
