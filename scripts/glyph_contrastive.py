"""Learn a shared embedding for spike responses and glyph contours."""

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from kaggle_gallery_validation import image_descriptors


def response_features(trials):
    """Return square-root transformed evoked totals and 50 ms bins per unit."""
    evoked = trials.unflatten(-1, (-1, 10))[..., 2:]
    windows = evoked.unflatten(-1, (4, 2)).sum(dim=-1)
    combined = torch.cat((windows.sum(dim=-1, keepdim=True), windows), dim=-1)
    return combined.clamp_min_(0).sqrt_().flatten(-2)


@dataclass
class CanonicalProjection:
    response_center: torch.Tensor
    glyph_center: torch.Tensor
    response_weights: torch.Tensor
    glyph_weights: torch.Tensor

    def neural(self, features):
        return F.normalize((features - self.response_center) @ self.response_weights, dim=-1)

    def visual(self, descriptors):
        return F.normalize((descriptors - self.glyph_center) @ self.glyph_weights, dim=-1)


def _fit_cca(responses, descriptors, dimension):
    n_sessions, n_glyphs, _ = responses.shape
    response_center = responses.mean(dim=(0, 1))
    glyph_center = descriptors.mean(dim=0)
    centered_responses = responses - response_center
    centered_glyphs = descriptors - glyph_center
    rows = centered_responses.flatten(0, 1)
    response_cov = rows.T @ rows / (n_sessions * n_glyphs)
    glyph_cov = centered_glyphs.T @ centered_glyphs / n_glyphs
    cross_cov = centered_responses.mean(dim=0).T @ centered_glyphs / n_glyphs
    response_cov.diagonal().add_(0.30 * response_cov.diagonal().mean().clamp_min(1e-5))
    glyph_cov.diagonal().add_(0.45 * glyph_cov.diagonal().mean().clamp_min(1e-5))
    response_factor = torch.linalg.cholesky(response_cov)
    glyph_factor = torch.linalg.cholesky(glyph_cov)
    whitened = torch.linalg.solve_triangular(response_factor, cross_cov, upper=False)
    whitened = torch.linalg.solve_triangular(glyph_factor, whitened.T.contiguous(), upper=False).T
    left, correlations, right = torch.linalg.svd(whitened, full_matrices=False)
    n_components = min(dimension, len(correlations), n_glyphs - 1)
    scale = correlations[:n_components].clamp_min(0).sqrt()
    response_weights = torch.linalg.solve_triangular(response_factor.T, left[:, :n_components], upper=True)
    glyph_weights = torch.linalg.solve_triangular(glyph_factor.T, right[:n_components].T, upper=True)
    return CanonicalProjection(response_center, glyph_center, response_weights * scale, glyph_weights * scale)


class ContrastiveReadout(nn.Module):
    def __init__(self, neural_size, image_size, n_sessions, dimension):
        super().__init__()
        self.session_embedding = nn.Embedding(n_sessions, 8)
        self.neural = nn.Sequential(nn.Linear(neural_size + 8, 192), nn.LayerNorm(192), nn.GELU(),
                                    nn.Linear(192, dimension))
        self.visual = nn.Sequential(nn.Linear(image_size, 144), nn.LayerNorm(144), nn.GELU(),
                                     nn.Linear(144, dimension))
        self.strength = 1.0 / math.sqrt(dimension)

    def encode_neural(self, response, recording, anchor):
        inputs = torch.cat((response, self.session_embedding(recording)), dim=1)
        return F.normalize(anchor + self.strength * self.neural(inputs).float(), dim=-1)

    def encode_visual(self, descriptor, anchor):
        return F.normalize(anchor + self.strength * self.visual(descriptor).float(), dim=-1)


@dataclass
class GlyphContrastive:
    sessions: tuple[str, ...]
    normalization: dict[str, tuple[torch.Tensor, torch.Tensor]]
    projection: CanonicalProjection
    readout: ContrastiveReadout

    @torch.no_grad()
    def image_embeddings(self, images, *, descriptors=False, batch_size=1024):
        """Embed PNG-style [N,64,64] images or an atlas's precomputed contours."""
        device = self.projection.glyph_center.device
        embeddings = {"cca": [], "contrastive": []}
        for batch in images.split(batch_size):
            batch = batch.to(device)
            feature = batch.float() if descriptors else image_descriptors((batch >= 128).to(torch.uint8) * 255)
            anchor = self.projection.visual(feature)
            embeddings["cca"].append(anchor)
            embeddings["contrastive"].append(self.readout.encode_visual(feature, anchor))
        return {name: torch.cat(batches, dim=0) for name, batches in embeddings.items()}

    @torch.no_grad()
    def neural_embeddings(self, trials, session):
        """Embed a recording's held or test spike counts [N,repeats,units × 10]."""
        center, scale = self.normalization[session]
        trials = trials.to(device=center.device, dtype=torch.float32)
        features = ((response_features(trials) - center) / scale).mean(dim=1)
        anchor = self.projection.neural(features)
        session_index = torch.full((len(trials),), self.sessions.index(session), dtype=torch.long, device=center.device)
        contrastive = self.readout.encode_neural(features, session_index, anchor)
        return {"cca": anchor, "contrastive": contrastive}

    @torch.no_grad()
    def rank(self, trials, session, gallery_embeddings):
        observed = self.neural_embeddings(trials, session)
        return {name: observed[name] @ templates.T for name, templates in gallery_embeddings.items()}


def _training_style_views(images, aligned_images):
    original = (images >= 128).to(torch.uint8) * 255
    aligned = (aligned_images >= 128).to(torch.uint8) * 255
    original_features = image_descriptors(original)
    aligned_features = image_descriptors(aligned)
    agreement = (original_features * aligned_features).sum(dim=1)
    # Low-similarity matches contribute mostly their true training image.
    style_weight = ((agreement - 0.45) / 0.45).clamp(0, 1)
    mixed = torch.lerp(original_features, aligned_features, 0.65 * style_weight[:, None])
    shifted = torch.cat((torch.zeros_like(aligned[:, :1, :]), aligned[:, :-1, :]), dim=1)
    shifted = torch.cat((shifted[:, :, 1:], torch.zeros_like(shifted[:, :, :1])), dim=2)
    shifted_features = image_descriptors(shifted)
    aligned_features = torch.lerp(original_features, aligned_features, style_weight[:, None])
    shifted_features = torch.lerp(original_features, shifted_features, style_weight[:, None])
    return mixed, torch.stack((original_features, aligned_features, shifted_features))


def _sample_responses(blocks, prototypes, labels, recordings, rng):
    repeats = blocks.shape[2]
    device = blocks.device
    choices = torch.rand((len(labels), repeats), generator=rng, device=device).argsort(dim=1)
    sample_size = torch.randint(1, repeats + 1, (len(labels),), generator=rng, device=device)
    keep_all = torch.rand(len(labels), generator=rng, device=device) < 0.65
    count = torch.where(keep_all, repeats, sample_size)
    weight = (choices < count[:, None]).to(torch.float32) / count[:, None]
    sampled = (blocks[recordings, labels] * weight[:, :, None]).sum(dim=1)
    # A small mixture with other labeled sessions preserves the glyph while
    # reducing dependence on recording-specific patterns.
    use_pool = (torch.rand(len(labels), generator=rng, device=device) < 0.30).to(torch.float32) * 0.15
    return torch.lerp(sampled, prototypes[labels], use_pool[:, None])


def _initialize(model, rng):
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=rng)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Embedding):
                layer.weight.normal_(mean=0, std=0.10, generator=rng)
        nn.init.zeros_(model.neural[-1].weight)
        nn.init.zeros_(model.visual[-1].weight)


def fit_glyph_contrastive(
    train_blocks: Mapping[str, torch.Tensor],
    images: torch.Tensor,
    aligned_images: torch.Tensor | None = None,
    *,
    epochs: int = 30,
    batch_size: int = 192,
    dimension: int = 96,
    seed: int = 211,
    verbose: bool = False,
) -> GlyphContrastive:
    """Fit from paired training glyphs only, using every training image as a candidate.

    Each block is raw grouped counts ``[glyph, repeat, unit × 10]`` for one
    mouse. Both image stacks must follow the same fit-glyph row order. If
    supplied, aligned images must be matched using the fit images alone.
    """
    if not train_blocks or images.ndim != 3 or images.shape[1:] != (64, 64) or len(images) < 2:
        raise ValueError("Provide training-recording blocks and at least two aligned 64 × 64 images.")
    first = next(iter(train_blocks.values()))
    if (first.ndim != 3 or first.shape[0] != len(images) or min(first.shape[1:]) < 1 or first.shape[2] % 10
            or any(block.shape != first.shape for block in train_blocks.values())):
        raise ValueError("Training counts must share shape [glyph, repeat, unit × 10].")
    if min(batch_size, dimension) < 1 or epochs < 0:
        raise ValueError("Invalid CCA and contrastive settings.")
    device = first.device
    sessions = tuple(train_blocks)
    images = images.to(device)
    aligned = images if aligned_images is None else aligned_images.to(device)
    if aligned.shape != images.shape:
        raise ValueError("Aligned glyphs must match the training image rows.")
    rng = torch.Generator(device=device).manual_seed(seed)

    normalization = {}
    normalized = []
    with torch.no_grad():
        for session, counts in train_blocks.items():
            features = response_features(counts.to(device=device, dtype=torch.float32))
            trials = features.flatten(0, 1)
            center = trials.mean(dim=0)
            scale = (trials.var(dim=0, unbiased=False) + 0.05).sqrt()
            normalization[session] = (center, scale)
            normalized.append((features - center) / scale)
        blocks = torch.stack(normalized)
        del normalized
        prototype = blocks.mean(dim=2)
        pooled = prototype.mean(dim=0)
        mixed, views = _training_style_views(images, aligned)
        projection = _fit_cca(prototype, mixed, dimension)
        visual_anchors = projection.visual(views)
        n_components = projection.response_weights.shape[1]

        # Similar glyphs receive a small soft-label mass alongside the exact
        # pairing, so nearly identical public shapes need not be pushed apart.
        geometry = mixed @ mixed.T
        geometry.fill_diagonal_(-1)
        neighbor_score, neighbors = geometry.topk(min(6, len(images) - 1), dim=1)
        neighbor_weight = (neighbor_score / 0.08).softmax(dim=1)
        del geometry
    network = ContrastiveReadout(blocks.shape[-1], mixed.shape[1], len(sessions), n_components).to(device)
    _initialize(network, rng)
    if not epochs:
        return GlyphContrastive(sessions, normalization, projection, network.eval())

    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=cuda)
    all_glyphs = torch.arange(len(images), device=device)
    for epoch in range(epochs):
        order = torch.randperm(len(images), device=device, generator=rng)
        epoch_loss = images.new_zeros((), dtype=torch.float32)
        for batch in order.split(batch_size):
            # One distinct glyph per column keeps the reverse objective's
            # positive labels unique while sampling a recording independently.
            recording = torch.randint(len(sessions), (len(batch),), device=device, generator=rng)
            response = _sample_responses(blocks, pooled, batch, recording, rng)
            units = response.unflatten(-1, (-1, 5))
            mask = torch.rand((*units.shape[:-1], 1), device=device, generator=rng) >= 0.06
            response = (units * mask / 0.94).flatten(-2)
            anchor = projection.neural(response)
            style = torch.randint(len(views), (len(images),), device=device, generator=rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cuda):
                neural = network.encode_neural(response, recording, anchor)
                visual = network.encode_visual(views[style, all_glyphs], visual_anchors[style, all_glyphs])
                logits = neural @ visual.T / 0.08
            logits = logits.float()
            logprob = logits.log_softmax(dim=1)
            exact = F.nll_loss(logprob, batch)
            nearby = -(logprob.gather(1, neighbors[batch]) * neighbor_weight[batch]).sum(1).mean()
            reverse = F.cross_entropy(logits[:, batch].T, torch.arange(len(batch), device=device))
            loss = 0.70 * exact + 0.10 * nearby + 0.20 * reverse
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.detach() * len(batch)
        scheduler.step()
        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs):
            print(f"Contrastive epoch {epoch + 1}/{epochs}: {(epoch_loss / len(images)).item():.4f}", flush=True)
    return GlyphContrastive(sessions, normalization, projection, network.eval())
