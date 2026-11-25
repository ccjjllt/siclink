"""Construct a hard glyph by accepting sparse, posterior-supported stroke edits."""

import itertools

import numpy as np
from scipy import ndimage
import torch
from torch.nn import functional as F


def _align_to_anchor(masks, radius):
    height, width = masks.shape[-2:]
    blurred = F.avg_pool2d(masks.flatten(0, 1)[:, None].float(), 3, stride=1, padding=1)
    blurred = blurred[:, 0].reshape_as(masks)
    padding = (radius,) * 4
    spread = F.pad(blurred, padding)
    padded = F.pad(masks, padding)
    anchor = blurred[:, :1]
    anchor_energy = anchor.square().sum((-2, -1))
    offsets = sorted(itertools.product(range(-radius, radius + 1), repeat=2),
                     key=lambda shift: abs(shift[0]) + abs(shift[1]))
    best = blurred.new_full(masks.shape[:2], -1)
    aligned = masks.clone()
    for dy, dx in offsets:
        y, x = radius + dy, radius + dx
        candidate = spread[..., y:y + height, x:x + width]
        agreement = (2 * (candidate * anchor).sum((-2, -1)))
        agreement /= (anchor_energy + candidate.square().sum((-2, -1))).clamp_min(1)
        agreement -= 0.002 * (abs(dy) + abs(dx))
        replace = agreement > best
        aligned = torch.where(replace[..., None, None], padded[..., y:y + height, x:x + width], aligned)
        best = torch.maximum(best, agreement)
    return aligned, best.clamp(0, 1)


def _proposals(anchor, alternatives, support, weights, budget):
    if budget < 6:
        return []
    structure = np.ones((3, 3), dtype=bool)
    near = ndimage.distance_transform_edt(~anchor) <= 4
    boundary = anchor & ~ndimage.binary_erosion(anchor, structure=structure)
    proposals = []
    for alternate, weight in zip(alternatives, weights):
        if weight < 0.02:
            continue
        for addition, changed in ((True, alternate & ~anchor), (False, anchor & ~alternate)):
            components, _ = ndimage.label(changed, structure=structure)
            sizes = np.bincount(components.ravel())
            for identifier in np.flatnonzero((sizes >= 6) & (sizes <= budget)):
                if identifier == 0:
                    continue
                pixels = components == identifier
                evidence = support[pixels]
                if addition:
                    accepted = near[pixels].mean() >= 0.8 and evidence.mean() >= 0.54 and (evidence >= 0.46).mean() >= 0.8
                    utility = evidence.mean() - 0.5
                else:
                    accepted = boundary[pixels].any() and evidence.mean() <= 0.29 and (evidence <= 0.37).mean() >= 0.8
                    utility = 0.5 - evidence.mean()
                if accepted:
                    proposals.append((float(utility * sizes[identifier]**0.5), addition, pixels))
    return sorted(proposals, key=lambda proposal: proposal[0], reverse=True)


def _compose(anchor, proposals, budget, *, additions_only):
    output = anchor.copy()
    already = np.zeros_like(anchor)
    for _, addition, pixels in proposals:
        if additions_only and not addition:
            continue
        size = int(pixels.sum())
        if size > budget or np.any(pixels & already):
            continue
        output[pixels] = addition
        already |= pixels
        budget -= size
    return output


@torch.inference_mode()
def stroke_consensus_images(logits, images, *, top_k=12, temperature=0.55, radius=2,
                            edit_fraction=0.16, batch_size=32):
    """Return matched hard controls and sparse edits without any reference images."""
    if not 2 <= top_k <= len(images) or temperature <= 0 or not 0 <= radius <= 4 or not 0 <= edit_fraction < 1:
        raise ValueError("Use a valid gallery neighborhood, alignment radius, temperature and edit budget")
    if logits.ndim != 2 or logits.shape[1] != len(images) or images.shape[-2:] != (64, 64):
        raise ValueError("Expected [queries, glyphs] scores and aligned 64×64 gallery masks")
    gallery = images >= 128
    variants = {"v7_hard": [], "aligned_medoid": [], "aligned_hard_vote": [],
                "sparse_add": [], "sparse_edit": []}
    for scores in logits.split(batch_size):
        winner = scores.argmax(dim=1, keepdim=True)
        remaining = scores.clone().scatter_(1, winner, -torch.inf)
        values, indices = remaining.topk(top_k - 1, dim=1)
        indices = torch.cat((winner, indices), dim=1)
        values = torch.cat((scores.gather(1, winner), values), dim=1)
        masks = gallery[indices]
        anchor = masks[:, 0]
        aligned, agreement = _align_to_anchor(masks, radius)
        scale = scores.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        weights = (((values - values[:, :1]) / scale) / temperature).softmax(dim=1)
        weights *= 0.18 + 0.82 * ((agreement - 0.20) / 0.80).clamp(0, 1).sqrt()
        weights /= weights.sum(dim=1, keepdim=True)
        support = (aligned * weights[..., None, None]).sum(dim=1)

        flat = aligned.flatten(2).float()
        area = flat.sum(dim=2)
        similarity = 2 * (flat @ flat.transpose(1, 2)) / (area[:, :, None] + area[:, None, :]).clamp_min(1)
        medoid = (similarity @ weights[:, :, None]).squeeze(-1).argmax(dim=1)
        selected = masks[torch.arange(len(masks), device=masks.device), medoid]
        vote = support >= 0.5
        vote = torch.where(vote.any(dim=(-2, -1), keepdim=True), vote, anchor)

        additions, edits = [], []
        for base, candidates, agreement_map, sample_weights in zip(anchor.cpu().numpy(), aligned[:, 1:].cpu().numpy(),
                                                                  support.cpu().numpy(), weights[:, 1:].cpu().numpy()):
            budget = int(base.sum() * edit_fraction)
            proposals = _proposals(base, candidates, agreement_map, sample_weights, budget)
            additions.append(_compose(base, proposals, budget, additions_only=True))
            edits.append(_compose(base, proposals, budget, additions_only=False))
        variants["v7_hard"].append(anchor)
        variants["aligned_medoid"].append(selected)
        variants["aligned_hard_vote"].append(vote)
        variants["sparse_add"].append(torch.as_tensor(np.stack(additions), device=anchor.device))
        variants["sparse_edit"].append(torch.as_tensor(np.stack(edits), device=anchor.device))
    return {name: torch.cat(parts).to(torch.uint8) * 255 for name, parts in variants.items()}
