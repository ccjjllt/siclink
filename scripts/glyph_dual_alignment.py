"""Combine spike embeddings and public-atlas posteriors for cross-block matching."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch.nn import functional as F

from glyph_multiview_alignment import pair_similarity


@dataclass
class AtlasPosterior:
    contours: torch.Tensor
    square_root_mass: csr_matrix


@torch.no_grad()
def posterior_view(logits, atlas_contours, *, top_k, temperature):
    """Construct expected contours and sparse Bhattacharyya features."""
    selected = logits.float().topk(top_k, dim=1)
    centered_scale = logits.float().std(dim=1, unbiased=False).clamp_min(1e-5)
    probability = (selected.values / (temperature * centered_scale[:, None])).softmax(dim=1)
    mixture = []
    for index, mass in zip(selected.indices.split(64), probability.split(64)):
        mixture.append((atlas_contours[index] * mass[:, :, None]).sum(dim=1))
    expected = F.normalize(torch.cat(mixture), dim=1)
    columns = selected.indices.flatten().cpu().numpy()
    mass = probability.sqrt().flatten().cpu().numpy()
    offsets = np.arange(len(logits) + 1, dtype=np.int32) * top_k
    sparse = csr_matrix((mass, columns, offsets), shape=logits.shape)
    return AtlasPosterior(expected, sparse)


def standardized_pair(values):
    return ((values - values.mean()) / values.std(unbiased=False).clamp_min(1e-5)).clamp(-5, 5)


@dataclass
class MatchCues:
    spike: torch.Tensor
    atlas: torch.Tensor

    def method(self, name, spike_weight):
        if name == "gcca":
            return self.spike
        if name == "atlas":
            return self.atlas
        if name == "dual":
            return self.atlas.lerp(self.spike, spike_weight)
        raise KeyError(name)


def pair_cues(left_spikes, right_spikes, left_atlas, right_atlas):
    spike = standardized_pair(pair_similarity(left_spikes, right_spikes))
    geometry = standardized_pair(pair_similarity(left_atlas.contours, right_atlas.contours))
    common = (left_atlas.square_root_mass @ right_atlas.square_root_mass.T).toarray()
    common = torch.as_tensor(common, device=spike.device)
    row_density = common.topk(min(10, common.shape[1]), dim=1).values.mean(dim=1)
    column_density = common.topk(min(10, common.shape[0]), dim=0).values.mean(dim=0)
    overlap = standardized_pair(common - 0.5 * (row_density[:, None] + column_density))
    atlas = standardized_pair(0.70 * geometry + 0.30 * overlap)
    return MatchCues(spike, atlas)


def edge_features(name, cues, scores, rows, columns, margin):
    values = [margin, scores[rows, columns]]
    if name == "dual":
        values.append(torch.minimum(cues.spike[rows, columns], cues.atlas[rows, columns]))
    return torch.stack(values, dim=1)


@dataclass
class EdgeReliability:
    mean: torch.Tensor | None
    scale: torch.Tensor | None
    weights: torch.Tensor | None

    def __call__(self, features):
        if self.weights is None:
            return features[:, 0]
        return ((features - self.mean) / self.scale) @ self.weights


def fit_edge_reliability(features, correctness, *, ridge=0.75):
    """Fit a small, strongly regularized ranking of candidate-match reliability."""
    positives = int(correctness.sum())
    if positives < 3 or len(features) - positives < 3:
        return EdgeReliability(None, None, None)
    center = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-4)
    x = (features - center) / scale
    y = correctness.float() - correctness.float().mean()
    cov = x.T @ x / len(x)
    cov.diagonal().add_(ridge)
    weights = torch.linalg.solve(cov, x.T @ y / len(x))
    return EdgeReliability(center, scale, weights)
