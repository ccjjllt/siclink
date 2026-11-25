"""Choose sharp gallery glyphs with minimum posterior contour risk."""

import torch
from torch.nn import functional as F

from ocf_training import directional_fields


@torch.no_grad()
def gallery_similarity(images, *, batch_size=128):
    """Approximate pairwise OCF of the labeled gallery without registration.

    At each scale, retain signed directional derivatives on a 16×16 grid and
    local orientation covariance on a coarser grid. Convert signature agreement
    and relative contour energy to an overlap-to-union ratio.
    """
    if images.ndim != 3 or images.shape[-2:] != (64, 64) or not len(images):
        raise ValueError("Expected a nonempty [glyphs, 64, 64] image gallery")
    if batch_size <= 0:
        raise ValueError("Use a positive gallery batch size")
    limit = 128 if images.max().item() > 1 else 0.5
    first, second = torch.triu_indices(4, 4, device=images.device)
    signatures, energies = [], []
    for batch in images.split(batch_size):
        target = (batch >= limit).float().unsqueeze(1)
        signature, scale_energies = [], []
        for sigma in (1, 2, 4):
            fields = directional_fields(target, sigma)
            signed = F.avg_pool2d(fields, 4).flatten(1)
            covariance = fields[:, first] * fields[:, second]
            orientation = F.avg_pool2d(covariance, 8).flatten(1)
            signature.extend((0.6 * F.normalize(signed), 0.8 * F.normalize(orientation)))
            scale_energies.append(fields.square().mean(dim=(1, 2, 3)) * sigma**2)
        signatures.append(F.normalize(torch.cat(signature, dim=1)))
        energies.append(torch.stack(scale_energies).sum(dim=0))

    signature = torch.cat(signatures)
    energy = torch.cat(energies).clamp_min(1e-8)
    scores = signature.new_empty((len(signature), len(signature)))
    for start in range(0, len(signature), batch_size):
        rows = slice(start, start + batch_size)
        agreement = (signature[rows] @ signature.T).clamp(0, 1)
        lower = torch.minimum(energy[rows, None], energy)
        upper = torch.maximum(energy[rows, None], energy)
        ratio = lower / upper
        shared = agreement * ratio
        scores[rows] = shared / (1 + ratio - shared)
    return scores.fill_diagonal_(1)


@torch.no_grad()
def fit_temperature(calibration):
    """Choose a scalar by NLL on independently held labeled spike queries."""
    if not calibration:
        raise ValueError("Temperature calibration needs held labeled queries")
    # Geometric spacing accommodates both under- and overconfident logits.
    candidates = tuple(0.10 * 1.23**index for index in range(18))
    total_rows = sum(len(labels) for _, labels in calibration)
    nll = []
    for temperature in candidates:
        loss = 0.0
        for scores, labels in calibration:
            scaled = scores.float() / temperature
            correct = scaled.gather(1, labels[:, None]).squeeze(1)
            loss += (scaled.logsumexp(dim=1) - correct).sum()
        nll.append(loss / total_rows)
    nll = torch.stack(nll)
    best = int(nll.argmin())
    return {"temperature": candidates[best], "nll": float(nll[best]), "n": total_rows}


@torch.no_grad()
def glyph_posterior(logits, temperature, *, balance=False, iterations=24):
    """Normalize class scores, optionally adding the complete-block class prior."""
    if logits.ndim != 2 or not logits.numel() or temperature <= 0 or iterations < 1:
        raise ValueError("Provide nonempty [queries, glyphs] logits and positive settings")
    log_mass = logits.float() / temperature
    if balance:
        if logits.shape[0] != logits.shape[1]:
            raise ValueError("Balanced posterior requires a verified complete glyph block")
        # Scale rows and columns in log space so low-probability candidates
        # remain usable even with relatively sharp calibrated distributions.
        for _ in range(iterations):
            log_mass -= log_mass.logsumexp(dim=1, keepdim=True)
            log_mass -= log_mass.logsumexp(dim=0, keepdim=True)
    return log_mass.softmax(dim=1)


@torch.no_grad()
def glyph_medoid(posterior, similarity):
    """Choose a real gallery glyph with highest expected contour overlap."""
    if posterior.ndim != 2 or similarity.shape != (posterior.shape[1], posterior.shape[1]):
        raise ValueError("The posterior and both similarity axes must use the same glyph order")
    return (posterior @ similarity).argmax(dim=1)
