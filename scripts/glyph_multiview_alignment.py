"""Align independently recorded spike responses with train-only regularized GCCA."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from kaggle_diagnostics import feature_view, standardize


@dataclass
class MultiviewAlignment:
    view: str
    normalization: dict
    centers: dict
    projections: dict
    spectrum: torch.Tensor
    cross_mouse_agreement: torch.Tensor | None

    def encode(self, block, counts):
        features, _ = standardize(feature_view(counts, self.view), self.normalization[block])
        values = (features.mean(dim=1) - self.centers[block]) @ self.projections[block]
        return F.normalize(values, dim=1)


@torch.no_grad()
def fit_multiview(training, *, view="evoked-total", dimension=64, ridge=0.35):
    """Fit shared coordinates using the known correspondence of training glyphs."""
    if len(training) < 2 or ridge <= 0 or dimension < 1:
        raise ValueError("Use at least two aligned blocks and positive ridge settings")
    n = len(next(iter(training.values())))
    if n < 3:
        raise ValueError("Use at least three fitted glyphs")
    normalization, centers, response, factors, whitened = {}, {}, {}, {}, []
    for block, counts in training.items():
        if counts.ndim != 3 or len(counts) != n or counts.shape[1] < 2:
            raise ValueError("Each block needs the same fitted glyphs with grouped spike repeats")
        features, normalization[block] = standardize(feature_view(counts, view))
        means = features.mean(dim=1)
        centers[block] = means.mean(dim=0)
        x = means - centers[block]
        response[block] = x
        within = (features - means[:, None]).flatten(0, 1)
        repeat_cov = within.T @ within / (n * (counts.shape[1] - 1))
        covariance = x.T @ x / n
        trace_scale = covariance.diagonal().mean().clamp_min(1e-4)
        covariance.add_(repeat_cov, alpha=0.5 * ridge / counts.shape[1])
        covariance.diagonal().add_(0.5 * ridge * trace_scale)
        factor = torch.linalg.cholesky(covariance)
        factors[block] = factor
        whitened.append(torch.linalg.solve_triangular(factor, x.T, upper=False).T / n**0.5)

    # Sum of ridge projection matrices in sample space is the MAXVAR objective.
    joint = torch.cat(whitened, dim=1)
    dimension = min(dimension, n - 1, joint.shape[1])
    if joint.shape[1] < n:
        eigenvalues, basis = torch.linalg.eigh(joint.T @ joint)
        eigenvalues, basis = eigenvalues[-dimension:], basis[:, -dimension:]
        latent = (joint @ basis) / eigenvalues.clamp_min(1e-6).sqrt()
    else:
        eigenvalues, latent = torch.linalg.eigh(joint @ joint.T)
        eigenvalues, latent = eigenvalues[-dimension:], latent[:, -dimension:]
    latent = latent * n**0.5
    projections = {block: torch.cholesky_solve(x.T @ latent / n, factors[block])
                   for block, x in response.items()}

    by_mouse = {}
    for block, x in response.items():
        by_mouse.setdefault(block[0], []).append(x @ projections[block])
    agreement = None
    weights = (eigenvalues / eigenvalues.max().clamp_min(1e-6)).clamp_min(0).sqrt()
    if len(by_mouse) == 2:
        left, right = (torch.stack(views).mean(dim=0) for views in by_mouse.values())
        scale = (left.square().sum(dim=0) * right.square().sum(dim=0)).sqrt().clamp_min(1e-8)
        agreement = (left * right).sum(dim=0) / scale
        # Do not let the strongest directions from one mouse dominate all views.
        weights *= (0.25 + 0.75 * agreement.clamp(0, 1)).sqrt()
    projections = {block: matrix * weights for block, matrix in projections.items()}
    return MultiviewAlignment(view, normalization, centers, projections, eigenvalues, agreement)


def pair_similarity(left, right, *, neighbors=10):
    """Cosine similarity corrected for each response's local neighborhood density."""
    score = left @ right.T
    row_density = score.topk(min(neighbors, score.shape[1]), dim=1).values.mean(dim=1)
    column_density = score.topk(min(neighbors, score.shape[0]), dim=0).values.mean(dim=0)
    return score - 0.5 * (row_density[:, None] + column_density)


def reciprocal_matches(scores):
    """Return mutual nearest-neighbor rows, columns and bilateral score margins."""
    left = scores.topk(2, dim=1)
    right = scores.topk(2, dim=0)
    columns = left.indices[:, 0]
    rows = torch.arange(len(columns), device=scores.device)
    mutual = right.indices[0, columns] == rows
    margin = torch.minimum(left.values[:, 0] - left.values[:, 1],
                           (right.values[0] - right.values[1])[columns])
    return rows[mutual], columns[mutual], margin[mutual]


def calibrate_margin(margin, correct, *, minimum_precision=0.65, minimum_matches=12):
    """Set a reciprocal-match threshold on independent labeled fit glyphs."""
    ordered, order = margin.sort(descending=True)
    if len(order) < minimum_matches:
        return {"threshold": None, "matches": 0, "precision": None, "lower_precision": None}
    n = torch.arange(1, len(order) + 1, device=ordered.device, dtype=ordered.dtype)
    precision = correct[order].cumsum(dim=0) / n
    z = 1.282
    lower = (precision + z**2 / (2 * n) - z * (precision * (1 - precision) / n + z**2 / (4 * n**2)).sqrt())
    lower /= 1 + z**2 / n
    # Count a whole group when several matches share the same margin.
    tie_end = torch.cat((ordered[:-1] > ordered[1:], ordered.new_ones((1,), dtype=torch.bool)))
    allowed = torch.where((n >= minimum_matches) & (lower >= minimum_precision) & tie_end)[0]
    if not len(allowed):
        return {"threshold": None, "matches": 0, "precision": None, "lower_precision": None}
    best = allowed[-1]
    return {"threshold": float(ordered[best]), "matches": int(n[best]), "precision": float(precision[best]),
            "lower_precision": float(lower[best])}
