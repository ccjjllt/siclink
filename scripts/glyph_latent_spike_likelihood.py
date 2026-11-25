"""Marginalize unit-specific shared variability in a Poisson glyph encoder."""

from dataclasses import dataclass

import torch


WINDOWS = ((2, 4), (4, 10))


@dataclass(frozen=True)
class LatentSpikeModel:
    loadings: tuple[torch.Tensor, ...]
    diagnostics: tuple[dict, ...]


def covariance_moments(counts):
    repeats = counts.shape[1]
    totals = counts.sum(dim=1)
    raw = counts.flatten(0, 1)
    within = raw.T @ raw
    between = totals.T @ totals
    exposure = (between - within) / (repeats * (repeats - 1))
    covariance = (within - between / repeats) / (repeats - 1)
    covariance.diagonal().sub_(totals.sum(dim=0) / repeats)
    return covariance, exposure


def normalized_covariance(covariance, exposure):
    scale = exposure.diagonal().clamp_min(0).sqrt()
    ratio = covariance / (exposure.clamp_min(0) + 0.10 * (scale[:, None] * scale) + 16)
    logged = ratio.clamp(-0.35, 1.5).log1p()
    return (logged + logged.T) * 0.5


@torch.no_grad()
def fit_latent_spikes(training, *, rank=4, seed=211):
    """Fit onset and sustained-response covariances from labeled repeat residuals."""
    first = next(iter(training.values()))
    if rank < 1 or rank > first.shape[-1] // 10 or len(first) < 4:
        raise ValueError("Use valid latent rank and at least four labeled glyphs.")
    generator = torch.Generator(device=first.device).manual_seed(seed)
    split = torch.randperm(len(first), device=first.device, generator=generator).chunk(2)
    loadings, diagnostics = [], []
    for start, end in WINDOWS:
        halves = []
        for glyphs in split:
            stats = [covariance_moments(counts[glyphs].unflatten(-1, (-1, 10))[..., start:end].sum(-1))
                     for counts in training.values()]
            halves.append(torch.stack([torch.stack(part) for part in stats]).sum(0))
        left, right = (normalized_covariance(*half) for half in halves)
        covariance = normalized_covariance(*(halves[0] + halves[1]))
        eigenvalues, vectors = torch.linalg.eigh(covariance)
        noise = torch.linalg.eigvalsh(0.5 * (left - right)).abs().amax()
        amplitudes = (eigenvalues[-rank:] - noise).clamp_min(0).sqrt()
        factors = vectors[:, -rank:] * amplitudes
        factors *= (0.55 / factors.norm(dim=1, keepdim=True).clamp_min(0.55))
        off_diagonal = torch.triu_indices(len(factors), len(factors), offset=1, device=factors.device)
        a, b = left[off_diagonal[0], off_diagonal[1]], right[off_diagonal[0], off_diagonal[1]]
        loadings.append(factors)
        diagnostics.append({"window_ms": [start * 25, end * 25], "split_offdiagonal_cosine": float(
            (a @ b) / (a.norm() * b.norm()).clamp_min(1e-6)), "raw_eigenvalues": eigenvalues[-rank:].tolist(),
            "covariance_noise_floor": float(noise), "active_factors": int((amplitudes > 0).sum()),
            "median_log_gain_sd": float(factors.norm(dim=1).median()),
            "max_log_gain_sd": float(factors.norm(dim=1).max())})
    return LatentSpikeModel(tuple(loadings), tuple(diagnostics))


@torch.no_grad()
def laplace_window(observed, expected, loading, *, batch_size=128):
    """Use the second-order integrated evidence at zero log-gain.

    Expected counts are the prior marginal means. The log-normal correction
    converts them to rates at zero latent value. Factorial terms are common
    across a query's candidate glyphs and cancel in scores and model deltas.
    """
    corrected = expected * (-0.5 * loading.square().sum(dim=1)).exp()
    corrected = corrected.clamp_min(1e-5)
    rank = loading.shape[1]
    identity = torch.eye(rank, device=loading.device, dtype=loading.dtype)
    hessian = torch.einsum("cf,fr,fs->crs", corrected, loading, loading) + identity
    # The small matrices use the Kaggle notebook's CPU linear algebra. The
    # shipped batched CUDA Cholesky inverse has no kernel for T4.
    factor = torch.linalg.cholesky(hessian.cpu())
    precision = torch.cholesky_inverse(factor).to(loading.device)
    curvature_cost = factor.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1).to(loading.device)
    center = corrected @ loading
    log_rate = corrected.log().T.contiguous()
    mean_cost = corrected.sum(dim=1) + curvature_cost
    result = []
    for trial in observed.split(batch_size):
        difference = (trial @ loading)[:, None] - center[None, :]
        gain = 0.5 * torch.einsum("bcr,crs,bcs->bc", difference, precision, difference)
        result.append(trial @ log_rate - mean_cost + gain)
    return torch.cat(result)


@torch.no_grad()
def latent_spike_logits(observed, rates, loadings):
    """Score independent repeats in both windows with gallery-conditioned rates."""
    shape = observed.shape[:2]
    counts = observed.float().unflatten(-1, (-1, 10)).flatten(0, 1)
    rates = rates.float().unflatten(-1, (-1, 10))
    scores = 0
    for (start, end), loading in zip(WINDOWS, loadings):
        expected = rates[..., start:end].flatten(-2)
        expected = 0.98 * expected + 0.02 * expected.mean(0)
        features = loading.repeat_interleave(end - start, dim=0)
        scores = scores + laplace_window(counts[..., start:end].flatten(-2), expected, features)
    return scores.reshape(*shape, len(rates)).mean(dim=1)
