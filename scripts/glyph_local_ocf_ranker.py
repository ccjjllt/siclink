"""Compare the local directional evidence for already nominated glyphs."""

from dataclasses import dataclass

import torch


SIGNED_SIZE = 4 * 16 * 16
SCALE_SIZE = SIGNED_SIZE + 10 * 8 * 8


def directional_tiles(signature):
    """Return signed direction vectors in 4×4 regions at each of three scales."""
    fields = signature.unflatten(-1, (3, SCALE_SIZE))[..., :SIGNED_SIZE]
    fields = fields.reshape(-1, 3, 4, 4, 4, 4, 4)
    return fields.permute(0, 1, 3, 5, 2, 4, 6).reshape(-1, 3, 16, 64)


@torch.no_grad()
def local_evidence(estimate, gallery_tiles, indices):
    """Track signed agreement and relative edge energy in each region."""
    query = directional_tiles(estimate)[:, None]
    candidate = gallery_tiles[indices]
    query_norm = query.norm(dim=-1)
    candidate_norm = candidate.norm(dim=-1)
    query_mass = query_norm / query_norm.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    candidate_mass = candidate_norm / candidate_norm.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    direction = (query * candidate).sum(dim=-1) / (query_norm * candidate_norm).clamp_min(1e-8)
    shared = torch.minimum(query_mass, candidate_mass)
    signed_agreement = shared * direction.clamp(-1, 1)
    energy_difference = (query_mass - candidate_mass).abs()
    group_totals = torch.cat((signed_agreement.sum(-1), energy_difference.sum(-1)), dim=-1)
    return torch.cat((signed_agreement.flatten(-2), energy_difference.flatten(-2), group_totals), dim=-1)


def _relative(features, anchor_columns):
    rows = torch.arange(len(features), device=features.device)
    return features - features[rows, anchor_columns][:, None, :]


@dataclass
class RelativeOCFRanker:
    scale: torch.Tensor
    weights: torch.Tensor
    diagnostics: dict

    @torch.no_grad()
    def select(self, candidates, local, anchor_columns, anchor_ocf):
        features = candidates.features if local is None else torch.cat((candidates.features, local), dim=-1)
        score = (_relative(features, anchor_columns) / self.scale).clamp(-8, 8) @ self.weights
        score += anchor_ocf * candidates.baseline_z
        selected = score.masked_fill(~candidates.valid, -torch.inf).argmax(dim=1, keepdim=True)
        return candidates.indices.gather(1, selected)[:, 0]


@torch.no_grad()
def fit_relative_ranker(candidates, local, anchor_columns, exact, *, ridge=0.18):
    """Fit OCF differences relative to the policy leader with bounded influence."""
    scalar = candidates.features
    features = scalar if local is None else torch.cat((scalar, local), dim=-1)
    valid = candidates.valid & torch.isfinite(exact)
    weight = valid.float() / valid.sum(dim=1, keepdim=True).clamp_min(1) / len(valid)
    features = _relative(features, anchor_columns)
    scale = (weight[:, :, None] * features.square()).sum((0, 1)).sqrt()
    scale[:scalar.shape[-1]].clamp_min_(0.12)
    scale[scalar.shape[-1]:].clamp_min_(0.015)
    features = (features[valid] / scale).clamp(-8, 8)
    target = (exact - exact.gather(1, anchor_columns[:, None]))[valid]
    weight = weight[valid]
    penalty = features.new_full((features.shape[-1],), ridge)
    penalty[scalar.shape[-1]:] *= 2
    weights = features.new_zeros(features.shape[-1])
    for _ in range(4):
        residual = target - features @ weights
        robust = (12 / residual.abs().clamp_min(12)) * weight
        gram = features.T @ (features * robust[:, None])
        gram.diagonal().add_(penalty)
        cross = features.T @ (target * robust)
        weights = torch.cholesky_solve(cross[:, None], torch.linalg.cholesky(gram))[:, 0]
    diagnostics = {"features": features.shape[-1], "labeled_candidates": len(target), "ridge": ridge,
                   "weighted_absolute_error": float((weight * (target - features @ weights).abs()).sum()),
                   "local_weight_norm": float(weights[scalar.shape[-1]:].norm())}
    return RelativeOCFRanker(scale, weights, diagnostics)
