"""Learn conditional glyph contour utility directly from single-trial spikes."""

from dataclasses import dataclass

import torch

from glyph_contour_retrieval import _fit_readout, _project_training, _response_features
from glyph_mbr import similarity_from_signatures
from glyph_reconstruction import RidgeReadout
from glyph_retrieval import _standardize


@dataclass
class UtilityBasis:
    targets: torch.Tensor
    actions: torch.Tensor
    unconditional: torch.Tensor
    diagnostics: dict


@torch.no_grad()
def fit_utility_basis(signatures, energy, *, rank=128):
    """Factor the image-only contour utility without imposing positive definiteness."""
    if rank < 1 or len(signatures) < 2:
        raise ValueError("Provide multiple fitted glyphs and a positive rank")
    utility = similarity_from_signatures(signatures, energy)
    utility = 0.5 * (utility + utility.T)
    unconditional = utility.mean(dim=0)
    # The removed target-only term is constant across output glyphs, so the
    # conditional expected ranking can recover the action mean separately.
    utility.sub_(unconditional).sub_(unconditional[:, None]).add_(unconditional.mean())
    values, vectors = torch.linalg.eigh(utility)
    selected = values.abs().topk(min(rank, len(values) - 1)).indices
    retained = values[selected]
    targets = vectors[:, selected] * retained.abs().sqrt()
    actions = targets * retained.sign()
    explained = retained.square().sum() / values.square().sum().clamp_min(1e-8)
    diagnostics = {"rank": len(retained), "retained_negative": int((retained < 0).sum()),
                   "centered_utility_energy_retained": float(explained)}
    return UtilityBasis(targets, actions, unconditional, diagnostics)


@dataclass
class UtilityReadout:
    moments: dict
    projection: torch.Tensor
    basis: UtilityBasis
    pooled: RidgeReadout
    local: dict[str, RidgeReadout]


@torch.no_grad()
def fit_utility_readout(training, basis):
    """Train the same single-trial ridge readout on factored utility targets."""
    if not training or any(len(counts) != len(basis.targets) for counts in training.values()):
        raise ValueError("The training recordings and utility basis must share glyph order")
    projected, moments, projection = _project_training(training, "evoked-multiscale", 10)
    pooled = _fit_readout(tuple(projected.values()), basis.targets, ridge=0.09)
    local = {name: _fit_readout((values,), basis.targets, ridge=0.14) for name, values in projected.items()}
    return UtilityReadout(moments, projection, basis, pooled, local)


@torch.inference_mode()
def expected_utility_scores(model, query, session):
    features = _response_features(query, "evoked-multiscale", 10)
    features = _standardize(features, *model.moments[session]).mean(dim=1) @ model.projection
    coefficients = 0.80 * model.pooled.predict(features) + 0.20 * model.local[session].predict(features)
    return coefficients @ model.basis.actions.T + model.basis.unconditional
