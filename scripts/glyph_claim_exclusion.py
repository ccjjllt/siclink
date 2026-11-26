"""Apply calibrated soft glyph competition with leave-one-out claims."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class ClaimCalibrator:
    center: torch.Tensor
    scale: torch.Tensor
    weights: torch.Tensor
    bias: torch.Tensor

    @torch.no_grad()
    def predict(self, features):
        arguments = ((features - self.center) / self.scale).clamp(-6, 6)
        return (arguments @ self.weights + self.bias).sigmoid()


def fit_claim_calibrator(features, correct, *, ridge=0.0005):
    """Estimate credibility of v11 top-class claims on independent responses."""
    correct = correct.float()
    center = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(1e-3)
    arguments = ((features - center) / scale).clamp(-6, 6).detach()
    weights = torch.nn.Parameter(arguments.new_zeros(arguments.shape[1]))
    bias = torch.nn.Parameter(correct.mean().clamp(0.0001, 0.9999).logit())
    optimizer = torch.optim.LBFGS([weights, bias], max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(arguments @ weights + bias, correct)
        loss = loss + ridge * weights.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return ClaimCalibrator(center, scale, weights.detach(), bias.detach())


@torch.no_grad()
def exclusion_logits(logits, destinations, certainty, *, threshold=0.10, max_penalty=1.0):
    """Suppress glyphs credibly claimed by another row without requiring a match."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or destinations.shape != certainty.shape:
        raise ValueError("Exclusion requires exactly one row per labeled glyph in a full recording")
    if destinations.shape != logits.shape[:1]:
        raise ValueError("Provide one candidate identity and probability for every query")
    weight = -torch.log1p(-certainty.clamp(0, 0.9))
    weight = weight.masked_fill(certainty < threshold, 0)
    aggregate = torch.zeros_like(weight).scatter_add_(0, destinations, weight)
    penalties = aggregate.clamp_max(max_penalty)
    own_class = penalties[destinations] - (aggregate[destinations] - weight).clamp(0, max_penalty)
    result = logits - penalties
    result[torch.arange(len(result), device=result.device), destinations] += own_class
    return result, penalties
