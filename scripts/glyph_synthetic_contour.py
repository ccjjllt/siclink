"""Augment contour decoding with responses simulated on a public glyph atlas."""

import torch

from glyph_contour_retrieval import _fit_readout, _fourier_features, _project_training, _response_features
from glyph_reconstruction import RidgeReadout
from glyph_retrieval import _standardize


VIEW = "evoked-multiscale"
N_BINS = 10


@torch.no_grad()
def simulated_views(rates, moments, projection, generator, *, repeats, groups):
    """Project Poisson trials in small batches into the measured latent space."""
    results = []
    for batch in rates.split(128):
        intensity = batch.float()[:, None, None].expand(-1, groups, repeats, -1)
        samples = torch.poisson(intensity, generator=generator)
        counts = _response_features(samples, VIEW, N_BINS)
        features = _standardize(counts, *moments).mean(dim=2)
        results.append(features @ projection)
    return torch.cat(results)


def calibrate_response(simulated, measured):
    """Fit an affine correction toward measured means, regularized toward identity."""
    source_mean, target_mean = simulated.mean(dim=0), measured.mean(dim=0)
    centered = simulated - source_mean
    drift = measured - target_mean - centered
    gram = centered.T @ centered / len(centered)
    ridge = 0.35 * gram.diagonal().mean().clamp_min(0.02)
    gram.diagonal().add_(ridge)
    correction = torch.cholesky_solve(centered.T @ drift / len(centered), torch.linalg.cholesky(gram))
    matrix = torch.eye(len(source_mean), device=simulated.device) + correction
    return RidgeReadout(matrix, target_mean - source_mean @ matrix)


def moments(views, targets):
    count = sum(x.shape[0] * x.shape[1] for x in views)
    per_glyph = sum(x.sum(dim=1) for x in views)
    return (per_glyph.sum(dim=0) / count, targets.mean(dim=0),
            sum(x.flatten(0, 1).T @ x.flatten(0, 1) for x in views) / count,
            per_glyph.T @ targets / count)


def augment_readout(measured, simulated, real_targets, simulated_targets, fraction):
    real = moments(measured, real_targets)
    synthetic = moments(simulated, simulated_targets)
    mean_x, mean_y, gram, cross = [(1 - fraction) * a + fraction * b for a, b in zip(real, synthetic)]
    gram -= mean_x[:, None] * mean_x
    gram.diagonal().add_(0.09)
    cross -= mean_x[:, None] * mean_y
    coefficient = torch.cholesky_solve(cross, torch.linalg.cholesky(gram))
    return RidgeReadout(coefficient, mean_y - mean_x @ coefficient)


@torch.no_grad()
def fit_synthetic_contours(training, aligned, targets, atlas_masks, atlas_targets, encoder, *, seed, fraction):
    projected, normalization, projection = _project_training(training, VIEW, N_BINS)
    real = {name: values.mean(dim=1, keepdim=True) for name, values in projected.items()}
    generator = torch.Generator(device=projection.device).manual_seed(seed)
    synthetic, diagnostics = {}, {}
    for name, view in real.items():
        fit_rates = encoder.expected_counts(aligned, name, batch_size=192)
        fit_sim = simulated_views(fit_rates, normalization[name], projection, generator, repeats=16, groups=1)[:, 0]
        domain_map = calibrate_response(fit_sim, view[:, 0])
        sampled_rates = encoder.expected_counts(atlas_masks, name, batch_size=192)
        sampled = simulated_views(sampled_rates, normalization[name], projection, generator, repeats=4, groups=2)
        synthetic[name] = domain_map.predict(sampled)
        unmatched = (fit_sim - view[:, 0]).square().mean()
        corrected = (domain_map.predict(fit_sim) - view[:, 0]).square().mean()
        diagnostics[name] = {"fit_pair_raw_mse": float(unmatched), "fit_pair_calibrated_mse": float(corrected)}

    real_views, simulated = tuple(real.values()), tuple(synthetic.values())
    radius = torch.stack([view.square().sum(-1).mean() for view in real_views]).mean().sqrt().clamp_min(0.10)
    frequency = torch.randn((projection.shape[1], 256), generator=generator, device=projection.device) / radius
    phase = torch.rand(256, generator=generator, device=projection.device) * (2 * torch.pi)
    real_fourier = tuple(_fourier_features(x, frequency, phase) for x in real_views)
    synthetic_fourier = tuple(_fourier_features(x, frequency, phase) for x in simulated)
    readouts = {
        "v8": _fit_readout(real_views, targets, ridge=0.09),
        "real_fourier": _fit_readout(real_fourier, targets, ridge=0.09),
        "synthetic_linear": augment_readout(real_views, simulated, targets, atlas_targets, fraction),
        "synthetic_fourier": augment_readout(real_fourier, synthetic_fourier, targets, atlas_targets, fraction),
    }
    return (readouts, normalization, projection, frequency, phase), diagnostics


@torch.no_grad()
def estimate_contours(model, counts, session):
    readouts, normalization, projection, frequency, phase = model
    features = _response_features(counts, VIEW, N_BINS)
    values = _standardize(features, *normalization[session]).mean(dim=1) @ projection
    expanded = _fourier_features(values, frequency, phase)
    return {name: readout.predict(expanded if "fourier" in name else values) for name, readout in readouts.items()}
