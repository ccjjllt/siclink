"""Leave-one-block-out soft support for an unlabeled public glyph atlas."""

from dataclasses import dataclass
from math import log

import torch


@dataclass
class AtlasSupportConsensus:
    bias_by_block: dict
    diagnostics: dict

    def rerank(self, block, logits, *, strength=0.45):
        """Apply only peer-block evidence to any subset of the held block's rows."""
        bias = self.bias_by_block[block]
        if logits.ndim != 2 or logits.shape[1] != len(bias) or strength < 0:
            raise ValueError("Use the same atlas column order and a nonnegative strength")
        return logits + strength * bias.to(logits.device)


def _capacity_evidence(scores, temperature, capacity, iterations):
    """Allocate partial row evidence while limiting support from duplicate votes."""
    log_posterior = (scores.float() / temperature).log_softmax(dim=1)
    entropy = -(log_posterior.exp() * log_posterior).sum(dim=1)
    certainty = (1 - entropy / log(scores.shape[1])).clamp(0, 1)
    evidence_weight = (3 * certainty).clamp(0.02, 0.90)
    log_evidence = log_posterior + evidence_weight.log()[:, None]
    log_dustbin = (-evidence_weight).log1p()

    # A row can keep mass in a private dustbin, and a candidate can stay unused.
    # Alternate row normalization with the upper-capacity column projection.
    log_discount = scores.new_zeros(scores.shape[1], dtype=torch.float32)
    for _ in range(iterations):
        log_kept = (log_evidence + log_discount).logsumexp(dim=1)
        log_row_scale = -torch.logaddexp(log_kept, log_dustbin)
        log_column_mass = (log_evidence + log_row_scale[:, None]).logsumexp(dim=0)
        log_discount = -(log_column_mass - log(capacity)).clamp_min(0)
    occupancy = (log_column_mass + log_discount).exp().clamp_max_(capacity)

    # A uniform atlas fallback makes zero peer evidence finite. Clipping limits
    # the influence of an atlas hub that many unrelated queries already prefer.
    mean_mass = occupancy.mean().clamp_min(1e-8)
    evidence = ((occupancy + mean_mass) / (2 * mean_mass)).log().clamp(-0.60, 1.40)
    metrics = {
        "rows": len(scores),
        "mean_entropy_fraction": float((1 - certainty).mean()),
        "assigned_mass_per_row": float(occupancy.sum() / len(scores)),
        "maximum_candidate_occupancy": float(occupancy.max()),
        "candidates_near_capacity": int((occupancy > 0.98 * capacity).sum()),
    }
    return evidence, metrics


@torch.no_grad()
def fit_support_consensus(logits_by_block, *, temperature=0.55, capacity=1.0, iterations=24,
                          uniform_mix=0.50):
    """Estimate a bounded peer-only atlas correction for each recording block.

    Each input is a [query, atlas] score matrix from measured spikes. Row order
    and counts can differ across blocks. Only the public atlas columns must
    align. The method uses possible cross-block overlap as weak positive
    evidence and leaves unused candidates and unmatched queries feasible.
    """
    if len(logits_by_block) < 2 or temperature <= 0 or not 0 < capacity <= 1 or iterations < 1:
        raise ValueError("Use at least two blocks, positive settings and a capacity of at most one")
    if not 0 < uniform_mix <= 1:
        raise ValueError("Keep a positive uniform fallback in the support prior")
    first = next(iter(logits_by_block.values()))
    if first.ndim != 2 or first.shape[1] < 2:
        raise ValueError("The atlas must have at least two candidate columns")

    evidence, stats = {}, {}
    for block, scores in logits_by_block.items():
        if scores.ndim != 2 or not len(scores) or scores.shape[1] != first.shape[1]:
            raise ValueError("Provide nonempty score matrices with a shared atlas")
        if not bool(torch.isfinite(scores).all()):
            raise ValueError(f"Nonfinite atlas evidence in {block}")
        supported, stats[block] = _capacity_evidence(scores, temperature, capacity, iterations)
        evidence[block] = supported.to(first.device)

    adjustments = {}
    for block in evidence:
        peers = torch.stack([values for other, values in evidence.items() if other != block])
        # One unusually strong peer can still contribute, but a consistent
        # increase across multiple peers carries more weight.
        robust = 0.5 * peers.mean(dim=0) + 0.5 * peers.quantile(0.5, dim=0)
        lift = (robust - robust.logsumexp(dim=0) + log(len(robust))).exp()
        bias = temperature * ((1 - uniform_mix) * lift + uniform_mix).log()
        adjustments[block] = (bias - bias.mean()).clamp(-0.35, 0.65)
        stats[block].update(peer_blocks=len(peers), peer_bias_min=float(adjustments[block].min()),
                            peer_bias_max=float(adjustments[block].max()))

    return AtlasSupportConsensus(adjustments, stats)
