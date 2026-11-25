"""Decode one occurrence per glyph within each complete Task 1 recording."""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


def distinct_atlas_glyphs(logits):
    """Maximize the neural score with a different public candidate per row."""
    if logits.ndim != 2 or not 0 < logits.shape[0] <= logits.shape[1]:
        raise ValueError("Expected a complete query block with at least as many atlas columns as rows")
    rows, labels = linear_sum_assignment(logits.detach().float().cpu().numpy(), maximize=True)
    matches = np.empty(len(rows), dtype=np.int64)
    matches[rows] = labels
    return torch.as_tensor(matches, device=logits.device)


@torch.no_grad()
def shared_atlas_assignment(logits_by_block, *, strength=0.9, sweeps=2):
    """Coordinate distinct assignments with a bounded reward for peer support.

    The reward depends on which public atlas glyphs receive neural support in
    other blocks. Rows do not need to correspond and unmatched repertoires are
    possible. Each update maximizes the original score plus the fixed objective
    reward for agreement with the currently assigned peer blocks.
    """
    if len(logits_by_block) < 2 or strength < 0 or sweeps < 1:
        raise ValueError("Provide at least two blocks, a nonnegative reward and positive sweeps")
    width = next(iter(logits_by_block.values())).shape[-1]
    if any(logits.ndim != 2 or logits.shape[1] != width for logits in logits_by_block.values()):
        raise ValueError("All blocks must use the same public atlas columns")
    cpu_scores = {block: logits.detach().float().cpu() for block, logits in logits_by_block.items()}
    current = {block: distinct_atlas_glyphs(logits).tolist() for block, logits in cpu_scores.items()}
    support = Counter(label for labels in current.values() for label in labels)
    blocks = tuple(cpu_scores)
    for sweep in range(sweeps):
        for block in blocks if sweep % 2 == 0 else reversed(blocks):
            support.subtract(current[block])
            bias = torch.zeros(width)
            occupied = [label for label, count in support.items() if count > 0]
            bias[occupied] = torch.as_tensor([support[label] for label in occupied], dtype=bias.dtype)
            bias *= strength / (len(blocks) - 1)
            current[block] = distinct_atlas_glyphs(cpu_scores[block] + bias).tolist()
            support.update(current[block])
    return {block: torch.as_tensor(labels, device=logits_by_block[block].device)
            for block, labels in current.items()}
