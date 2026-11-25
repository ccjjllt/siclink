"""Predict evoked spike rates with an image-space kernel on glyph contours."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ocf_training import directional_fields


def descriptors(images):
    images = images.float()
    if images.max() > 1:
        images = images / 255
    canvas = images[:, None]
    views = [F.avg_pool2d(canvas, 4).flatten(1)]
    for scale in (1, 2, 4):
        fields = directional_fields(canvas, scale)
        views.append(F.avg_pool2d(fields, 4).flatten(1))
        views.append(F.avg_pool2d(fields.square(), 8).sqrt().flatten(1))
    return views


def image_kernel(query, train):
    inner = query @ train.T
    distances = (query.square().sum(1, keepdim=True) + train.square().sum(1) - 2 * inner).clamp_min_(0)
    return 0.5 * inner + 0.35 * (-distances).exp() + 0.15 * (-0.25 * distances).exp()


@dataclass
class KernelGlyphEncoder:
    session_index: dict[str, int]
    training_rates: torch.Tensor
    feature_mean: list[torch.Tensor]
    feature_scale: list[torch.Tensor]
    anchors: torch.Tensor
    kernel_mean: torch.Tensor
    weights: torch.Tensor
    baseline: torch.Tensor

    @torch.no_grad()
    def expected_counts(self, images, session=None, *, batch_size=128):
        recording = len(self.session_index) if session is None else self.session_index[session]
        predictions = []
        for batch in images.to(self.anchors.device).split(batch_size):
            features = torch.cat([(view - center) / scale for view, center, scale in
                                  zip(descriptors(batch), self.feature_mean, self.feature_scale)], dim=1)
            kernel = image_kernel(features, self.anchors)
            centered = kernel - self.kernel_mean - kernel.mean(1, keepdim=True) + self.kernel_mean.mean()
            stabilized = centered @ self.weights[recording] + self.baseline[recording]
            predictions.append((stabilized.clamp_min(0).square() - 0.10).clamp_min_(1e-5))
        return torch.cat(predictions)


@torch.no_grad()
def fit_kernel_glyph_encoder(session_counts, images, *, ridge=0.4):
    """Fit across labeled glyphs. Each recording retains its own firing-rate offsets."""
    if not session_counts or not len(images) or ridge <= 0:
        raise ValueError("Provide aligned training glyphs, recordings and positive ridge regularization")
    sessions = {session: index for index, session in enumerate(session_counts)}
    device = next(iter(session_counts.values())).device
    blocks = [descriptors(batch) for batch in images.to(device).split(128)]
    features = [torch.cat(view) for view in zip(*blocks)]
    means = [view.mean(0) for view in features]
    centered = [view - mean for view, mean in zip(features, means)]
    scales = [(view.square().sum(1).mean() * len(centered)).sqrt().clamp_min_(1e-4) for view in centered]
    anchors = torch.cat([view / scale for view, scale in zip(centered, scales)], dim=1)
    gram = image_kernel(anchors, anchors)
    kernel_mean = gram.mean(0)
    gram = gram - kernel_mean - kernel_mean[:, None] + kernel_mean.mean()
    gram.diagonal().add_(ridge)

    rate = torch.stack([counts.mean(1) for counts in session_counts.values()])
    all_rates = torch.cat((rate, rate.mean(0, keepdim=True)))
    stabilized = (all_rates + 0.10).sqrt().transpose(0, 1).flatten(1)
    baseline = stabilized.mean(0)
    weights = torch.cholesky_solve(stabilized - baseline, torch.linalg.cholesky(gram))
    weights = weights.reshape(len(images), len(all_rates), rate.shape[-1]).transpose(0, 1).contiguous()
    return KernelGlyphEncoder(sessions, rate.mean(1), means, scales, anchors, kernel_mean, weights,
                              baseline.reshape(len(all_rates), -1))
