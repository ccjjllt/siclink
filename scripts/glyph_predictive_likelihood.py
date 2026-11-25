"""Score an unseen glyph while accounting for uncertainty in its encoded response."""

from dataclasses import dataclass

import torch


def evoked_windows(raw_counts):
    """Pool each unit over the onset and sustained portions of the evoked response."""
    binned = raw_counts.float().unflatten(-1, (-1, 10))
    return torch.stack((binned[..., 2:4].sum(dim=-1), binned[..., 4:].sum(dim=-1)), dim=-1).flatten(-2)


def stabilized_spikes(raw_counts):
    return (evoked_windows(raw_counts) + 0.375).sqrt()


def stabilized_rate(expected_counts):
    rates = evoked_windows(expected_counts).clamp_min(0)
    root = (rates + 0.375).sqrt()
    # A delta-method correction for the mean of a transformed Poisson count.
    return root - rates / (8 * root.pow(3))


def covariance(values, degrees):
    return values.T @ values / degrees


def regularize(matrix, diagonal_weight):
    return (1 - diagonal_weight) * matrix + diagonal_weight * torch.diag(matrix.diagonal())


@dataclass
class PredictiveLikelihood:
    offset: torch.Tensor
    scale: torch.Tensor
    trial_covariance: torch.Tensor
    model_covariance: torch.Tensor
    diagnostics: dict

    @torch.no_grad()
    def logits(self, observed, expected, *, include_model_error):
        query = stabilized_spikes(observed).mean(1)
        templates = stabilized_rate(expected) + self.offset
        center = templates.mean(dim=0)
        query, templates = (query - center) / self.scale, (templates - center) / self.scale
        covariance = self.trial_covariance / observed.shape[1]
        if include_model_error:
            covariance = covariance + self.model_covariance
        covariance.diagonal().add_(0.05)
        lower = torch.linalg.cholesky(covariance)
        whitened_queries = torch.linalg.solve_triangular(lower, query.T.contiguous(), upper=False).T
        whitened_templates = torch.linalg.solve_triangular(lower, templates.T.contiguous(), upper=False).T
        return whitened_queries @ whitened_templates.T - 0.5 * whitened_templates.square().sum(dim=1)


@torch.no_grad()
def fit_predictive_likelihood(calibration_counts, calibration_rates, *, diagonal_weight=0.30):
    """Fit residual statistics using training glyphs held out of the forward model."""
    if (calibration_counts.ndim != 3 or calibration_rates.shape != calibration_counts[:, 0].shape
            or min(calibration_counts.shape[:2]) < 2 or not 0 < diagonal_weight <= 1):
        raise ValueError("Use aligned out-of-fold rates and at least two observations and repeats")
    observations = stabilized_spikes(calibration_counts)
    templates = stabilized_rate(calibration_rates)
    means = observations.mean(dim=1)
    residual = means - templates
    offset = residual.mean(dim=0)
    rows, repeats, _ = observations.shape
    trial_residuals = (observations - means[:, None]).flatten(0, 1)
    trial = covariance(trial_residuals, rows * (repeats - 1))
    scale = trial.diagonal().clamp_min(0.025).sqrt()
    trial = trial / (scale[:, None] * scale)
    estimation_error = covariance((residual - offset) / scale, rows - 1)
    model = regularize(estimation_error - trial / repeats, diagonal_weight)
    raw_eigenvalues, vectors = torch.linalg.eigh(model)
    # Repeat noise is removed before enforcing positive predictive variance.
    model = (vectors * raw_eigenvalues.clamp_min(0)) @ vectors.T
    trial = regularize(trial, diagonal_weight)
    diagnostics = {"glyphs": rows, "repeats": repeats,
                   "fraction_model_eigenvalues_clipped": float((raw_eigenvalues < 0).float().mean()),
                   "model_trace_over_trial_trace": float(model.trace() / trial.trace().clamp_min(1e-8)),
                   "diagonal_weight": diagonal_weight}
    return PredictiveLikelihood(offset, scale, trial, model, diagnostics)
