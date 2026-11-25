"""Learn a shared Task 2 embedding from supervised cross-recording episodes."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from glyph_retrieval import _reliable_axes, _whiten
from kaggle_diagnostics import feature_view, standardize


class ResidualProjection(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.adjustment = nn.Sequential(nn.Linear(dimension, 96), nn.LayerNorm(96), nn.GELU(),
                                        nn.Linear(96, dimension))
        self.scale = dimension ** -0.5

    def forward(self, response):
        anchor = F.normalize(response, dim=-1)
        adjustment = self.adjustment(anchor / self.scale)
        return F.normalize(anchor + self.scale * adjustment, dim=-1)


@dataclass
class EpisodicAssignment:
    view: str
    projection: torch.Tensor
    support: torch.Tensor
    network: ResidualProjection
    diagnostics: dict

    @torch.no_grad()
    def scores(self, counts):
        """Score a new recording from grouped spike counts and labeled support only."""
        normalized, _ = standardize(feature_view(counts, self.view))
        response = normalized.mean(dim=1) @ self.projection
        return {"frozen_prototype": F.normalize(response) @ F.normalize(self.support).T,
                "episodic_prototype": self.network(response) @ self.network(self.support).T}


def _projection(training, view, dimension):
    values, centers, covariances = [], [], []
    for counts in training.values():
        normalized, _ = standardize(feature_view(counts, view))
        centers.append(normalized.mean(dim=1))
        centered = (normalized - centers[-1][:, None]).flatten(0, 1)
        covariances.append(centered.T @ centered / (len(counts) * (counts.shape[1] - 1)))
        values.append(normalized)
    covariance = torch.stack(covariances).mean(dim=0)
    covariance.diagonal().add_(0.35)
    factor = torch.linalg.cholesky(covariance)
    centers = torch.stack(centers)
    whitened = _whiten(centers.flatten(0, 1), factor).reshape_as(centers)
    available = torch.ones(centers.shape[1], device=centers.device, dtype=torch.bool)
    axes = _reliable_axes(whitened, available)[:, -dimension:]
    projection = torch.linalg.solve_triangular(factor.T, axes, upper=True)
    return projection, torch.stack([data @ projection for data in values])


def _sinkhorn_cost(logits):
    transport = logits
    for _ in range(5):
        transport = transport - transport.logsumexp(dim=1, keepdim=True)
        transport = transport - transport.logsumexp(dim=0, keepdim=True)
    return -transport.diagonal().mean()


def _initialize(network, rng):
    with torch.no_grad():
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=rng)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(network.adjustment[-1].weight)


def fit_episodic_assignment(training, *, view="evoked-multiscale", dimension=96, epochs=14,
                            batch_size=160, sinkhorn_weight=0.10, seed=211):
    """Train leave-one-recording-out episodes among the labeled recordings.

    Each episode chooses one labeled recording as its query. The support
    prototypes come exclusively from the other recordings. Extra negatives
    are the nearby glyphs of a fixed training-only spike metric.
    """
    if len(training) < 2 or min(dimension, epochs, batch_size) < 1 or not 0 <= sinkhorn_weight <= 0.25:
        raise ValueError("Provide paired recordings and valid episodic settings.")
    first = next(iter(training.values()))
    if (first.ndim != 3 or len(first) < 3 or first.shape[1] != 4 or first.shape[2] % 10
            or any(value.shape != first.shape for value in training.values())):
        raise ValueError("The episodic fit needs aligned glyphs with four 25-ms-binned trials per recording.")
    device = first.device
    rng = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        projection, responses = _projection(training, view, dimension)
        centers = responses.mean(dim=2)
        support = centers.mean(dim=0)
        similarity = F.normalize(support) @ F.normalize(support).T
        similarity.fill_diagonal_(-torch.inf)
        neighbors = similarity.topk(min(8, len(support) - 1), dim=1).indices
        del similarity
        source = (centers.sum(dim=0) - centers) / (len(centers) - 1)

    network = ResidualProjection(projection.shape[-1]).to(device)
    _initialize(network, rng)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.0015, weight_decay=0.025)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0002)
    final_loss = 0.0
    for epoch in range(epochs):
        total_loss = support.new_zeros(())
        examples = 0
        for recording in range(len(responses)):
            shuffled = torch.randperm(len(support), generator=rng, device=device)
            for batch in shuffled.split(batch_size):
                hard_draw = torch.randint(neighbors.shape[1], (len(batch[::2]),), generator=rng, device=device)
                labels = torch.cat((batch, neighbors[batch[::2], hard_draw])).unique()
                ordering = torch.randperm(len(labels), device=device, generator=rng)
                labels = labels[ordering]
                full = centers[recording, labels]
                repeats = responses[recording, labels]
                smallest = torch.rand((len(labels), 4), device=device, generator=rng).argsort(dim=1)[:, :2]
                partial = repeats.gather(1, smallest[:, :, None].expand(-1, -1, repeats.shape[-1])).mean(dim=1)
                mix = 0.35 * (torch.rand((len(labels), 1), device=device, generator=rng) < 0.35)
                query = full.lerp(partial, mix)
                mask = torch.rand(query.shape, device=device, generator=rng) < 0.04
                query = query.masked_fill(mask, 0)

                optimizer.zero_grad(set_to_none=True)
                logits = 14 * (network(query) @ network(source[recording]).T)
                episode = logits[:, labels]
                aligned = torch.arange(len(labels), device=device)
                pointwise = F.cross_entropy(logits, labels)
                reverse = F.cross_entropy(episode.T, aligned)
                sinkhorn = _sinkhorn_cost(episode) if sinkhorn_weight else logits.new_zeros(())
                loss = (0.82 - sinkhorn_weight) * pointwise + 0.18 * reverse + sinkhorn_weight * sinkhorn
                loss.backward()
                optimizer.step()
                total_loss += loss.detach() * len(labels)
                examples += len(labels)
        scheduler.step()
        final_loss = float(total_loss / examples)
        if epoch == 0 or (epoch + 1) % 4 == 0 or epoch + 1 == epochs:
            print(f"Episodic epoch {epoch + 1}/{epochs}: {final_loss:.4f}", flush=True)
    return EpisodicAssignment(view, projection, support, network.eval(),
                              {"labeled_recordings": len(training), "dimensions": projection.shape[-1],
                               "fit_glyphs": len(support), "epochs": epochs, "sinkhorn_weight": sinkhorn_weight,
                               "final_loss": final_loss})
