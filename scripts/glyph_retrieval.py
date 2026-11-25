"""Closed-set spike decoders for SiClink Tasks 2 and 3."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SessionModel:
    mean: torch.Tensor
    scale: torch.Tensor
    centers: torch.Tensor


@dataclass
class RetrievalModel:
    sessions: dict
    available: torch.Tensor
    center: torch.Tensor
    image_center: torch.Tensor
    noise_factor: torch.Tensor
    reliable_axes: torch.Tensor
    network: nn.Module
    n_bins: int


def _variance_stabilize(counts, *, n_bins):
    """Keep timing and retain sums that are less sensitive to spike jitter."""
    if n_bins != 10 or counts.shape[-1] % n_bins:
        raise ValueError("Retrieval needs 10 adjacent 25 ms bins per unit")
    counts = counts.float()
    counts = counts.reshape(*counts.shape[:-1], -1, n_bins)
    pools = [counts[..., :2], counts[..., 2:8].reshape(*counts.shape[:-1], 3, 2).sum(dim=-1),
             counts[..., 8:].sum(dim=-1, keepdim=True), counts[..., 2:].sum(dim=-1, keepdim=True),
             counts.sum(dim=-1, keepdim=True)]
    return torch.cat([counts, *pools], dim=-1).sqrt().flatten(-2)


def _moments(features):
    flat = features.flatten(0, -2)
    scale = (flat.var(dim=0, unbiased=False) + 0.05).sqrt()
    return flat.mean(dim=0), scale


def _standardize(features, mean, scale):
    return ((features - mean) / scale).clamp(-8, 8)


def _prototypes(features, labels, classes):
    counts = torch.bincount(labels, minlength=classes)
    centers = features.new_zeros((classes, features.shape[1]))
    centers.index_add_(0, labels, features)
    centers /= counts.clamp_min(1)[:, None]
    residual = features - centers[labels]
    degrees = max(len(features) - int((counts > 0).sum()), 1)
    return centers, counts, residual.T @ residual / degrees


def _whiten(features, noise_factor):
    return torch.linalg.solve_triangular(noise_factor, features.T.contiguous(), upper=False).T


def _reliable_axes(session_centers, available):
    """Find between-glyph directions that recur across independently recorded sessions."""
    sessions, classes, dimensions = session_centers.shape
    if sessions < 2:
        return torch.eye(dimensions, device=session_centers.device)

    centers = session_centers[:, available]
    classes = centers.shape[1]
    centers = centers - centers.mean(dim=1, keepdim=True)
    total = centers.sum(dim=0)
    within = centers.flatten(0, 1)
    signal = (total.T @ total - within.T @ within) / (sessions * (sessions - 1) * classes)
    values, axes = torch.linalg.eigh(signal)
    values, axes = values[-256:], axes[:, -256:]
    strength = values.clamp_min(0)
    return axes * (strength / (strength + 0.10)).sqrt()


def _image_prior(images, center, available):
    """Predict a glyph's spike template from related training glyphs, excluding itself."""
    selected = images[available].float()
    selected = (selected >= 128) if selected.max() > 1 else (selected >= 0.5)
    selected = selected.float().unsqueeze(1)
    coarse = F.adaptive_avg_pool2d(selected, (8, 8)).flatten(1)
    fine = F.adaptive_avg_pool2d(selected, (16, 16)).flatten(1)
    features = torch.cat((coarse, fine / 2), dim=1)
    gram = features @ features.T
    diagonal = gram.diagonal()
    distances = (diagonal[:, None] + diagonal[None, :] - 2 * gram).clamp_min(0)
    width = distances.mean().clamp_min(0.01) * 0.60
    kernel = torch.exp(-distances / width)
    kernel.diagonal().add_(0.35)
    factor = torch.linalg.cholesky(kernel)
    inverse = torch.cholesky_inverse(factor)
    selected_centers = center[available]
    fitted = inverse @ selected_centers
    prior = center.clone()
    prior[available] = selected_centers - fitted / inverse.diagonal()[:, None]
    return prior


class SpikeClassifier(nn.Module):
    def __init__(self, dimensions, classes):
        super().__init__()
        width = 512
        self.input = nn.Linear(dimensions, width)
        self.residual = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                      nn.Linear(width, width))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 256), nn.GELU(),
                                    nn.LayerNorm(256), nn.Linear(256, classes, bias=False))

    def forward(self, features):
        hidden = F.gelu(self.input(features))
        return self.output(hidden + 0.5 * self.residual(hidden))


def _mix_repeats(features, labels, rows, sorted_rows, counts, start, generator):
    """Pool four responses to one glyph from randomized training sessions."""
    mixed_rows = rows[:len(rows) // 3]
    target = labels[mixed_rows]
    mixed = features[mixed_rows].clone()
    for _ in range(3):
        offset = (torch.rand(len(target), device=features.device, generator=generator) * counts[target]).long()
        mixed += features[sorted_rows[start[target] + offset]]
    return mixed.mul_(0.25), target


def _train_classifier(features, labels, classes, available, epochs, seed, verbose):
    if epochs <= 0:
        return None
    device = features.device
    generator = torch.Generator(device=device).manual_seed(seed)
    network = SpikeClassifier(features.shape[1], classes).to(device)
    with torch.no_grad():
        for layer in network.modules():
            if isinstance(layer, nn.Linear):
                bound = layer.in_features ** -0.5
                layer.weight.uniform_(-bound, bound, generator=generator)
                if layer.bias is not None:
                    layer.bias.zero_()

    counts = torch.bincount(labels, minlength=classes)
    start = counts.cumsum(0) - counts
    sorted_rows = labels.argsort()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.025)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00012)
    batch_size = 512
    for epoch in range(epochs):
        permutation = torch.randperm(len(features), device=device, generator=generator)
        for rows in permutation.split(batch_size):
            pooled, pooled_labels = _mix_repeats(features, labels, rows, sorted_rows, counts, start, generator)
            inputs = torch.cat((features[rows], pooled))
            target = torch.cat((labels[rows], pooled_labels))
            mask = torch.rand(inputs.shape, device=device, generator=generator) >= 0.08
            inputs *= mask
            inputs += torch.randn(inputs.shape, device=device, generator=generator) * 0.06
            optimizer.zero_grad(set_to_none=True)
            logits = network(inputs).masked_fill(~available, -1e4)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
        schedule.step()
        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs):
            print(f"{device} glyph classifier {epoch + 1}/{epochs} loss={loss.item():.4f}", flush=True)
    return network.eval()


def fit_retrieval(train_blocks, num_glyphs, *, device, seed=41, epochs=16, images=None, verbose=True,
                  n_bins=10):
    """Fit from one mouse's (session, raw [N,D] spike counts, glyph indices) blocks.

    Each unit's ten sequential 25ms bins should be adjacent in the feature array.
    Optional images are the [num_glyphs, 64, 64] training glyphs in index order.
    """
    sessions, covariances, labels_all, features_all, counts_all = {}, [], [], [], []
    for name, counts, labels in train_blocks:
        features = _variance_stabilize(torch.as_tensor(counts, device=device), n_bins=n_bins)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        mean, scale = _moments(features)
        features = _standardize(features, mean, scale)
        centers, class_counts, covariance = _prototypes(features, labels, num_glyphs)
        sessions[name] = SessionModel(mean, scale, centers)
        covariances.append(covariance)
        labels_all.append(labels)
        features_all.append(features)
        counts_all.append(class_counts)

    counts_all = torch.stack(counts_all)
    class_counts = counts_all.sum(0)
    available = class_counts > 0
    centers = torch.stack([session.centers for session in sessions.values()])
    center = (centers * counts_all[..., None]).sum(0) / class_counts.clamp_min(1)[:, None]
    covariance = torch.stack(covariances).mean(0)
    covariance.diagonal().add_(0.30)
    noise_factor = torch.linalg.cholesky(covariance)
    white_centers = _whiten(centers.flatten(0, 1), noise_factor).reshape(centers.shape)
    reliable_axes = _reliable_axes(white_centers, (counts_all > 0).all(0))

    image_center = None
    if images is not None:
        image_center = _image_prior(torch.as_tensor(images, device=device), center, available)

    network = _train_classifier(torch.cat(features_all), torch.cat(labels_all), num_glyphs, available,
                                epochs, seed, verbose)
    return RetrievalModel(sessions, available, center, image_center, noise_factor, reliable_axes, network, n_bins)


def _template_logits(query, centers):
    return query @ centers.T - 0.5 * centers.square().sum(dim=1)


def _scaled_logits(scores, available):
    available_scores = scores[:, available]
    average = available_scores.mean(dim=1, keepdim=True)
    scale = (available_scores - average).square().mean().sqrt().clamp_min(0.01)
    return ((scores - average) / scale).masked_fill(~available, -1e4)


@torch.inference_mode()
def retrieval_score_components(model, query, session, *, task):
    """Return normalized scores for each decoder. Query has shape [N, repeats, D]."""
    features = _variance_stabilize(torch.as_tensor(query, device=model.center.device), n_bins=model.n_bins)
    local = model.sessions.get(session) if task == "task3" else None
    mean, scale = (local.mean, local.scale) if local is not None else _moments(features)
    features = _standardize(features, mean, scale)
    pooled = features.mean(dim=1)
    center = model.center
    if local is not None:
        center = 0.72 * local.centers + 0.28 * center

    white_query = _whiten(pooled, model.noise_factor)
    white_center = _whiten(center, model.noise_factor)
    reliable_query = white_query @ model.reliable_axes
    reliable_center = white_center @ model.reliable_axes
    scores = {"lda": _template_logits(white_query, white_center),
              "reliable": _template_logits(reliable_query, reliable_center)}
    if model.image_center is not None:
        shrunk_center = 0.82 * center + 0.18 * model.image_center
        scores["image_prior"] = _template_logits(white_query, _whiten(shrunk_center, model.noise_factor))
    if model.network is not None:
        chunks = []
        for trial_group in features.split(512):
            repeats = trial_group.shape[1]
            if repeats == 1:
                logits = model.network(trial_group[:, 0])
            else:
                pooled_logits = model.network(trial_group.mean(dim=1))
                trial_logits = model.network(trial_group.flatten(0, 1)).reshape(len(trial_group), repeats, -1)
                logits = 0.70 * pooled_logits + 0.30 * trial_logits.mean(dim=1)
            chunks.append(logits)
        scores["classifier"] = torch.cat(chunks)
    return {name: _scaled_logits(score, model.available) for name, score in scores.items()}


DEFAULT_WEIGHTS = {
    "task2": {"lda": 0.25, "reliable": 0.35, "image_prior": 0.15, "classifier": 0.25},
    "task3": {"lda": 0.35, "reliable": 0.25, "image_prior": 0.15, "classifier": 0.25},
}


def combine_logits(scores, task, weights=None):
    weights = DEFAULT_WEIGHTS[task] if weights is None else weights
    total = sum(weights.get(name, 0) for name in scores)
    if total <= 0:
        raise ValueError("Choose at least one fitted decoder with a positive weight")
    return sum(score * (weights.get(name, 0) / total) for name, score in scores.items())


def assign_glyphs(logits):
    """Honor the known one-of-each-glyph prior for a complete recording block."""
    from scipy.optimize import linear_sum_assignment

    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("One-to-one assignment requires a complete glyph block")
    rows, columns = linear_sum_assignment(logits.detach().cpu().numpy(), maximize=True)
    matched = torch.empty(len(rows), dtype=torch.long, device=logits.device)
    matched[torch.from_numpy(rows).to(logits.device)] = torch.from_numpy(columns).to(logits.device)
    return matched


def predict_retrieval(model, query, session, *, task, return_logits=False, assignment=False, weights=None):
    scores = retrieval_score_components(model, query, session, task=task)
    logits = combine_logits(scores, task, weights)
    if return_logits:
        return logits
    return assign_glyphs(logits) if assignment else logits.argmax(dim=1)


@torch.inference_mode()
def posterior_images(logits, images, *, temperature=0.38, top_k=48):
    """Render predicted common strokes when several glyphs have comparable scores."""
    masks = (images >= 128).float().flatten(1)
    foreground = masks.mean().item()
    estimates = []
    for chunk in logits.split(128):
        score, glyphs = chunk.topk(min(top_k, masks.shape[0]), dim=1)
        weights = (score / temperature).softmax(dim=1)
        pixels = (masks[glyphs] * weights.unsqueeze(-1)).sum(dim=1)
        threshold = torch.quantile(pixels, 1 - foreground, dim=1, keepdim=True)
        threshold = (0.25 * 0.40 + 0.75 * threshold).clamp(0.12, 0.50)
        estimates.append(((pixels - threshold) / 0.14 + 0.5).clamp(0, 1))
    return (torch.cat(estimates).reshape(-1, *images.shape[-2:]) * 255).round().to(torch.uint8)
