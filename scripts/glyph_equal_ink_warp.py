"""Cross-validated shape evidence from ink-matched candidate deformations."""

import torch

from glyph_local_warp import (BATCH_SIZE, LocalWarp, MAX_CENTROID_SHIFT, RENDER_SCALES, encode_evoked, landmarks,
                              observed_evoked, poisson_log_likelihood, warp)


def conditional_likelihood(rates, observed):
    """Score the spatial distribution of spikes given each 25 ms population total."""
    log_rate = rates.log().unflatten(1, (-1, 8))
    log_probability = log_rate - log_rate.logsumexp(dim=1, keepdim=True)
    return (observed.unflatten(1, (-1, 8)) * log_probability).sum(dim=(1, 2))


@torch.no_grad()
def score_raster(encoder, masks, counts, session):
    scores, poisson = [], []
    for batch, response in zip(masks.split(BATCH_SIZE), observed_evoked(counts).split(BATCH_SIZE)):
        rates = encode_evoked(encoder, batch[:, None].float(), session)
        scores.append(conditional_likelihood(rates, response))
        poisson.append(poisson_log_likelihood(rates, response))
    return {"conditional": torch.cat(scores), "poisson": torch.cat(poisson)}


@torch.no_grad()
def equal_ink_render(batch, field, xy, grid, ink, center):
    """Rasterize a warp at exactly the original foreground pixel count."""
    result = batch[:, 0].bool()
    selected = field.new_zeros(len(field))
    accepted = torch.zeros(len(field), dtype=torch.bool, device=field.device)
    ink = ink.to(torch.long)[:, None]
    for scale in RENDER_SCALES[:-1]:
        if bool(accepted.all()):
            break
        warped = warp(batch, scale * field, grid)[:, 0]
        # Resolve equal intensities using closest-to-original coverage before spatial order.
        pixel_score = warped + batch[:, 0] * 1e-7
        ranks = pixel_score.flatten(1).argsort(dim=1, descending=True, stable=True).argsort(dim=1)
        raster = (ranks < ink).reshape_as(warped)
        _, centroid = landmarks(raster.float(), xy)
        use = ((centroid - center).norm(dim=1) <= MAX_CENTROID_SHIFT) & ~accepted
        result = torch.where(use[:, None, None], raster, result)
        selected[use] = scale
        accepted |= use
    return LocalWarp(result, selected[:, None, None, None] * field, selected)
