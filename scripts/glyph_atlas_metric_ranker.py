"""Rerank a union of spike-derived atlas candidates with held-fold OCF labels."""

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AtlasCandidates:
    indices: torch.Tensor
    valid: torch.Tensor
    features: torch.Tensor
    baseline_z: torch.Tensor
    feature_names: tuple[str, ...]
    atlas_size: int
    rank_depth: int
    baseline: str

    def take_rows(self, rows):
        """Select a slice or vector of query rows without changing feature definitions."""
        return AtlasCandidates(self.indices[rows], self.valid[rows], self.features[rows], self.baseline_z[rows],
                               self.feature_names, self.atlas_size, self.rank_depth, self.baseline)


def _schema(candidates):
    return candidates.feature_names, candidates.atlas_size, candidates.rank_depth, candidates.baseline


def concat_atlas_candidates(parts):
    """Concatenate query folds using identical atlas coordinates and features."""
    if not parts:
        raise ValueError("Provide at least one candidate batch.")
    first = parts[0]
    if any(_schema(part) != _schema(first) or part.indices.shape[1] != first.indices.shape[1] for part in parts):
        raise ValueError("Candidate batches must use the same atlas, features and padded width.")
    target = first.features.device
    return AtlasCandidates(torch.cat([part.indices.to(target) for part in parts]),
                           torch.cat([part.valid.to(target) for part in parts]),
                           torch.cat([part.features.to(target) for part in parts]),
                           torch.cat([part.baseline_z.to(target) for part in parts]),
                           first.feature_names, first.atlas_size, first.rank_depth, first.baseline)


def _deduplicate(nominations, atlas_size):
    sorted_ids, positions = nominations.sort(dim=1, stable=True)
    unique = sorted_ids < atlas_size
    unique[:, 1:] &= sorted_ids[:, 1:] != sorted_ids[:, :-1]
    width = nominations.shape[1]
    ordered = positions.masked_fill(~unique, width).sort(dim=1).values
    valid = ordered < width
    indices = nominations.gather(1, ordered.masked_fill(~valid, 0)).masked_fill(~valid, 0)
    return indices, valid


def _row_features(matrix, chosen, valid, top_score, top_index):
    available = torch.isfinite(matrix)
    observed = matrix.masked_fill(~available, 0)
    size = available.sum(dim=1, keepdim=True).clamp_min(1)
    mean = observed.sum(dim=1, keepdim=True) / size
    centered = (observed - mean).masked_fill(~available, 0)
    scale = (centered.square().sum(dim=1, keepdim=True) / size).sqrt().clamp_min(1e-5)
    present = valid & available.gather(1, chosen)
    values = ((observed.gather(1, chosen) - mean) / scale).clamp(-12, 12)
    values = values.masked_fill(~present, 0)
    leader = ((top_score[:, :1] - mean) / scale).clamp(-12, 12)
    margin = (leader - values).clamp(0, 12).masked_fill(~present, 0)

    depth = top_index.shape[1]
    ranks = torch.arange(1, depth + 1, device=matrix.device)
    matches = chosen[:, :, None] == top_index[:, None, :]
    rank = torch.where(matches, ranks, depth + 1).amin(dim=2)
    support = (rank <= depth) & present
    reciprocal = rank.float().rsqrt().masked_fill(~support, 0)
    if depth > 1:
        leader_gap = ((top_score[:, :1] - top_score[:, 1:2]) / scale).clamp(0, 12)
        leader_gap = leader_gap.masked_fill(~torch.isfinite(top_score[:, 1:2]), 0)
    else:
        leader_gap = torch.zeros_like(mean)
    return (values, margin, reciprocal, leader_gap.expand_as(values), present.float()), rank


@torch.no_grad()
def pack_atlas_candidates(
    scores: Mapping[str, torch.Tensor],
    nominate: Mapping[str, int],
    *,
    baseline: str = "v4",
    rank_depth: int = 32,
) -> AtlasCandidates:
    """Combine explicitly nominated top lists and gather row-relative features.

    Each supplied dense matrix has the same [query, atlas] shape and larger
    values have higher preference. Missing and nonfinite scores cannot
    nominate a candidate. Padded columns have safe index 0 and ``valid=False``.
    """
    if not scores or not nominate or baseline not in scores or rank_depth < 1:
        raise ValueError("Provide score matrices, nominations, a baseline name and a positive rank depth.")
    if any(name not in scores or not isinstance(count, int) or count < 1 for name, count in nominate.items()):
        raise ValueError("Each nomination needs a known score name and a positive integer quota.")
    first = next(iter(scores.values()))
    if first.ndim != 2 or min(first.shape) < 1 or any(
        matrix.shape != first.shape or matrix.device != first.device for matrix in scores.values()
    ):
        raise ValueError("Use identically shaped, nonempty score matrices on one device.")
    atlas_size = first.shape[1]
    if max(nominate.values()) > atlas_size:
        raise ValueError("A top-k quota cannot exceed the atlas width.")
    depth = min(atlas_size, max(rank_depth, *nominate.values()))
    listings = {}
    for name, matrix in scores.items():
        matrix = matrix.float()
        values, indexes = matrix.masked_fill(~torch.isfinite(matrix), -torch.inf).topk(depth, dim=1)
        listings[name] = (values, indexes.masked_fill(~torch.isfinite(values), atlas_size))
    nominations = torch.cat([listings[name][1][:, :count] for name, count in nominate.items()], dim=1)
    indices, valid = _deduplicate(nominations, atlas_size)

    columns, names, agreeing_ranks = [], [], []
    for name in sorted(scores):
        components, ranks = _row_features(scores[name].float(), indices, valid, *listings[name])
        columns.extend(components)
        names.extend(f"{name}.{feature}" for feature in ("z", "distance_to_first", "reciprocal_rank",
                                                        "leader_margin", "has_value"))
        if name in nominate:
            agreeing_ranks.append(ranks)
        if name == baseline:
            baseline_z = components[0]
    ranks = torch.stack(agreeing_ranks)
    columns.extend(((ranks == 1).float().mean(dim=0), (ranks <= min(5, depth)).float().mean(dim=0),
                    (ranks <= depth).float().mean(dim=0)))
    names.extend(("agreement.first", "agreement.top_five", "agreement.ranked"))
    features = torch.stack(columns, dim=-1).masked_fill(~valid[:, :, None], 0)
    return AtlasCandidates(indices, valid, features, baseline_z, tuple(names), atlas_size, depth, baseline)


class _ExpectedOCF(nn.Module):
    def __init__(self, size, hidden):
        super().__init__()
        self.linear = nn.Linear(size, 1)
        self.residual = nn.Sequential(nn.Linear(size, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, features):
        return (self.linear(features) + 0.50 * self.residual(features)).squeeze(-1).sigmoid()


@dataclass
class AtlasMetricRanker:
    model: _ExpectedOCF
    mean: torch.Tensor
    scale: torch.Tensor
    candidate_schema: tuple
    diagnostics: dict

    @torch.no_grad()
    def predict_ocf(self, candidates):
        """Return predicted OCF in [0, 100], with NaN in unused padding."""
        if _schema(candidates) != self.candidate_schema:
            raise ValueError("Scoring candidates must match fitted feature definitions and atlas order.")
        features = ((candidates.features.to(self.mean) - self.mean) / self.scale).clamp(-8, 8)
        predicted = 100 * self.model(features).to(candidates.features.device)
        return predicted.masked_fill(~candidates.valid, torch.nan)

    @torch.no_grad()
    def rank(self, candidates, *, anchor_ocf=0.75):
        """Return expected OCF plus a small nominal baseline preference."""
        if anchor_ocf < 0:
            raise ValueError("Use a nonnegative baseline anchoring weight.")
        predicted = self.predict_ocf(candidates) + anchor_ocf * candidates.baseline_z
        return predicted.masked_fill(~candidates.valid, -torch.inf)

    @torch.no_grad()
    def select(self, candidates, *, anchor_ocf=0.75):
        """Return one atlas index per row, or -1 for a fully missing score row."""
        best = self.rank(candidates, anchor_ocf=anchor_ocf).argmax(dim=1, keepdim=True)
        return candidates.indices.gather(1, best)[:, 0].masked_fill(~candidates.valid.any(dim=1), -1)


def fit_atlas_metric_ranker(
    candidates: AtlasCandidates,
    ocf: torch.Tensor,
    *,
    seed: int = 211,
    epochs: int = 80,
    hidden: int = 32,
    batch_rows: int = 48,
    learning_rate: float = 0.003,
    weight_decay: float = 0.03,
    ranking_weight: float = 0.055,
) -> AtlasMetricRanker:
    """Train on cross-fitted candidate scores and their exact 0–100 OCF labels.

    Each query contributes equal weight regardless of its number of deduped
    or scored candidates. NaN targets and padding contribute no loss.
    Distinct outer held queries belong only in the prediction call.
    """
    if ocf.shape != candidates.valid.shape or min(epochs, hidden, batch_rows) < 1 or learning_rate <= 0:
        raise ValueError("Provide OCF labels matching the candidate pad and valid optimization settings.")
    if min(weight_decay, ranking_weight) < 0:
        raise ValueError("Regularization and ranking weights must be nonnegative.")
    features = candidates.features.detach().float()
    device = features.device
    targets = ocf.to(features).detach()
    valid = candidates.valid & torch.isfinite(targets)
    rows = valid.any(dim=1).nonzero()[:, 0]
    if not len(rows):
        raise ValueError("At least one query must have a finite OCF label for a valid candidate.")
    valid = valid[rows]
    targets = (targets[rows].masked_fill(~valid, 0) / 100).clamp(0, 1)
    features = features[rows]
    weights = valid.float() / valid.sum(dim=1, keepdim=True)
    mean = (weights[:, :, None] * features).sum(dim=(0, 1)) / len(rows)
    variance = (weights[:, :, None] * (features - mean).square()).sum(dim=(0, 1)) / len(rows)
    scale = variance.sqrt().clamp_min(0.15)
    features = ((features - mean) / scale).clamp(-8, 8)

    rng = torch.Generator(device=device).manual_seed(seed)
    model = _ExpectedOCF(features.shape[-1], hidden).to(device)
    with torch.no_grad():
        mean_target = (weights * targets).sum() / len(rows)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=rng)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(model.linear.weight)
        model.linear.bias.copy_(mean_target.clamp(0.01, 0.99).logit())
        nn.init.zeros_(model.residual[-1].weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0003)

    best = targets.masked_fill(~valid, -torch.inf).argmax(dim=1, keepdim=True)
    true_margin = (targets.gather(1, best) - targets).masked_fill(~valid, 0)
    pair_weight = true_margin / (true_margin + 0.10)
    pair_weight = pair_weight / pair_weight.sum(dim=1, keepdim=True).clamp_min(1)
    recent_loss = None
    for _ in range(epochs):
        order = torch.randperm(len(rows), device=device, generator=rng)
        recent_loss = features.new_zeros(())
        for row_batch in order.split(batch_rows):
            prediction = model(features[row_batch])
            row_error = F.smooth_l1_loss(prediction, targets[row_batch], beta=0.10, reduction="none")
            pointwise = (weights[row_batch] * row_error).sum(dim=1).mean()
            best_prediction = prediction.gather(1, best[row_batch])
            pair_error = F.softplus((prediction - best_prediction) / 0.12)
            pairwise = (pair_weight[row_batch] * pair_error).sum(dim=1).mean()
            loss = pointwise + ranking_weight * pairwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            recent_loss = recent_loss + loss.detach() * len(row_batch)
        scheduler.step()
    diagnostics = {"train_queries": len(rows), "labeled_candidates": int(valid.sum()), "epochs": epochs,
                   "seed": seed, "mean_target_ocf": float(100 * mean_target),
                   "final_training_loss": float(recent_loss / len(rows))}
    return AtlasMetricRanker(model.eval(), mean, scale, _schema(candidates), diagnostics)
