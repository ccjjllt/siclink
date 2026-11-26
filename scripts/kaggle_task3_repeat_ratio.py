"""Compare full-gallery Task 3 discriminative scores with matched repeat noise.

The two classifiers share whitened training representations, leave-one-out
template anchors, initialization, and real-example minibatches. One arm also
samples centered residuals from other glyphs with comparable fitted firing
rates in the same recording. Only original training trials supply either arm.
One measured repeat per glyph is reserved before every fit and never augmented.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from glyph_contour_retrieval import fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_encoding import fit_glyph_encoder
from glyph_gain_likelihood import fit_gain_model
from glyph_mbr import contour_signatures
from glyph_retrieval import _template_logits
from kaggle_diagnostics import official_scorer
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_part_discovery import SESSIONS, project_counts, standardize_rows
from kaggle_task3_part_recheck import score_v11


LOCAL_FRACTION = 0.20
AUGMENTATION_FRACTION = 0.40
ACTIVITY_BINS = 8
BLENDS = (0.20, 0.35)


@dataclass
class SessionView:
    origin: torch.Tensor
    trials: torch.Tensor
    templates: torch.Tensor
    leave_one_out: torch.Tensor
    residuals: torch.Tensor
    donor_order: torch.Tensor
    donor_position: torch.Tensor
    bins: torch.Tensor
    bin_start: torch.Tensor
    bin_size: torch.Tensor


def whiten(features, factor):
    shape = features.shape
    flat = features.reshape(-1, shape[-1]).T.contiguous()
    return torch.linalg.solve_triangular(factor, flat, upper=False).T.reshape(shape).contiguous()


@torch.no_grad()
def training_views(contour, training, dimensions):
    projected, origin, covariance = {}, {}, []
    for session, counts in training.items():
        values = project_counts(contour, counts, session)[..., -dimensions:]
        origin[session] = values.flatten(0, 1).mean(dim=0)
        projected[session] = values - origin[session]
        residuals = (values - values.mean(dim=1, keepdim=True)).flatten(0, 1)
        covariance.append(residuals.T @ residuals / (len(values) * (values.shape[1] - 1)))
    noise = torch.stack(covariance).mean(dim=0)
    pooled = 0.80 * noise + 0.20 * torch.diag(noise.diagonal())
    pooled.diagonal().add_(0.08 * noise.diagonal().mean().clamp_min(1e-5))
    factor = torch.linalg.cholesky(pooled)
    measured = {name: whiten(values, factor) for name, values in projected.items()}
    centers = {name: values.mean(dim=1) for name, values in measured.items()}
    shared = torch.stack(tuple(centers.values())).mean(dim=0)
    repetitions = next(iter(measured.values())).shape[1]
    sessions = len(centers)
    trial_variance = torch.stack([(values - centers[name][:, None]).square().mean(dim=(0, 1))
                                  for name, values in measured.items()]).mean(dim=0)
    trial_variance = trial_variance * repetitions / (repetitions - 1)
    observed = shared.var(dim=0, unbiased=False).clamp_min(1e-6)
    shrink = (1 - trial_variance / (observed * repetitions * sessions)).clamp(0.15, 0.98)
    leave_one_out_weight = LOCAL_FRACTION + (1 - LOCAL_FRACTION) / sessions

    views = {}
    for session, values in measured.items():
        center = centers[session]
        estimate = center.lerp(shared, 1 - LOCAL_FRACTION) * shrink
        residuals = values - center[:, None]
        leave_one_out = estimate[:, None] - (leave_one_out_weight / (repetitions - 1)) * residuals * shrink
        residuals = residuals * (repetitions / (repetitions - 1)) ** 0.5
        # Balanced bins based on actual fit-only evoked count, with no empty donor strata.
        activity = training[session].unflatten(-1, (-1, 10))[..., 2:].sum(dim=(-2, -1)).float().mean(dim=1)
        donor_order = activity.argsort(stable=True)
        position = donor_order.argsort()
        bins = (ACTIVITY_BINS * position).div(len(position), rounding_mode="floor")
        bin_size = torch.bincount(bins, minlength=ACTIVITY_BINS)
        views[session] = SessionView(origin[session], values, estimate, leave_one_out, residuals,
                                     donor_order, position, bins, bin_size.cumsum(0) - bin_size, bin_size)
    return factor, views, {"mean_prototype_shrinkage": float(shrink.mean()),
                           "mean_present_repeat_variance": float(trial_variance.mean())}


class DensityRatio(nn.Module):
    """A learned residual on a frozen, repeat-whitened prototype likelihood."""

    def __init__(self, dimensions, classes):
        super().__init__()
        self.features = nn.Sequential(nn.LayerNorm(dimensions), nn.Linear(dimensions, 96),
                                      nn.SiLU(), nn.Linear(96, dimensions))
        self.class_head = nn.Sequential(nn.Linear(dimensions, 48, bias=False), nn.SiLU(), nn.Linear(48, classes))
        for layer in (self.features[-1], self.class_head[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, observations, templates, labels=None, leave_one_out=None):
        embedded = observations + 0.25 * self.features(observations)
        logits = _template_logits(embedded, templates)
        if leave_one_out is not None:
            positive = (embedded * leave_one_out).sum(dim=1) - 0.5 * leave_one_out.square().sum(dim=1)
            logits = logits.scatter(1, labels[:, None], positive[:, None])
        return logits + 0.35 * self.class_head(embedded)


def matched_residual(view, labels, generator):
    # Sample uniformly within the aligned count stratum, excluding the same glyph.
    group = view.bins[labels]
    start = view.bin_start[group]
    available = view.bin_size[group] - 1
    ordinal = (torch.rand(len(labels), generator=generator, device=labels.device) * available).long()
    ordinal += (ordinal >= view.donor_position[labels] - start).long()
    donor = view.donor_order[start + ordinal]
    repeat = torch.randint(view.residuals.shape[1], (len(labels),), generator=generator, device=labels.device)
    return view.residuals[donor, repeat]


def fit_ratio(views, args, *, augmented):
    torch.manual_seed(args.seed + 17)
    trials = next(iter(views.values())).trials
    device = trials.device
    order = torch.Generator(device=device).manual_seed(args.seed + 18)
    noise = torch.Generator(device=device).manual_seed(args.seed + 19)
    classes, repeats, dimensions = trials.shape
    model = DensityRatio(dimensions, classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.03)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.00015)
    final_loss = None
    for epoch in range(args.epochs):
        order_sessions = torch.randperm(len(views), generator=order, device=device).tolist()
        batches = tuple(views.values())
        epoch_loss = 0.0
        for i in order_sessions:
            view = batches[i]
            for rows in torch.randperm(classes * repeats, generator=order, device=device).split(512):
                labels = rows.div(repeats, rounding_mode="floor")
                repeat = rows.remainder(repeats)
                real = view.trials[labels, repeat]
                anchor = view.leave_one_out[labels, repeat]
                logits = model(real, view.templates, labels, anchor)
                loss = F.cross_entropy(logits, labels)
                if augmented:
                    synthetic = anchor + matched_residual(view, labels, noise)
                    added = model(synthetic, view.templates, labels, anchor)
                    loss = (1 - AUGMENTATION_FRACTION) * loss + AUGMENTATION_FRACTION * F.cross_entropy(added, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(labels)
        final_loss = epoch_loss / (classes * repeats * len(views))
        schedule.step()
        if epoch == 0 or (epoch + 1) % 4 == 0 or epoch + 1 == args.epochs:
            print(f"{'matched' if augmented else 'real-only'} epoch {epoch + 1}/{args.epochs}: {final_loss:.4f}",
                  flush=True)
    return model.eval(), final_loss


@torch.no_grad()
def rank_gallery(v11, features, templates, real, matched):
    scores = {"v11": v11, "analytic": _template_logits(features, templates),
              "real": real(features, templates), "matched": matched(features, templates)}
    anchor = standardize_rows(v11)
    for name in ("real", "matched"):
        correction = standardize_rows(scores[name])
        for weight in BLENDS:
            scores[f"{name}_{weight:.2f}"] = anchor.lerp(correction, weight)
    return scores


def row_metrics(scores, guesses, labels):
    return {name: {"top1": float((guesses[name] == labels).float().mean()),
                   "top5": float((values.topk(5, dim=1).indices == labels[:, None]).any(dim=1).float().mean()),
                   "row_standardized_nll": float(F.cross_entropy(standardize_rows(values), labels)),
                   "changed_vs_v11": float((guesses[name] != guesses["v11"]).float().mean())}
            for name, values in scores.items()}


def exact_ocf(guesses, sample, labels, masks, scorer):
    matched_rows = labels.argsort()[sample]
    predictions = torch.stack([guess[matched_rows] for guess in guesses.values()])
    values, n_unique = unique_pair_ocf(predictions, sample.expand_as(predictions), masks, scorer,
                                      torch.device("cpu"), 256)
    return dict(zip(guesses, values.numpy())), n_unique


def summarize(observations):
    values = {name: np.stack(blocks) for name, blocks in observations.items()}
    summary = {}
    for name, scores in values.items():
        references = {"v11": "v11"}
        if name.startswith("matched"):
            references["real_only"] = name.replace("matched", "real", 1)
        comparisons = {}
        for label, reference in references.items():
            delta = (scores - values[reference]).mean(axis=0)
            comparisons[label] = {"delta": float(delta.mean()),
                                  "paired_se": float(delta.std(ddof=1) / np.sqrt(len(delta)))}
        summary[name] = {"ocf": float(scores.mean()), "scored_rows": scores.size, "comparisons": comparisons}
    return summary


def extend(accumulated, observed):
    for name, values in observed.items():
        accumulated.setdefault(name, []).append(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--sessions", nargs="+", choices=SESSIONS, default=SESSIONS)
    parser.add_argument("--ocf-images", type=int, default=64)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int, default=1807)
    parser.add_argument("--glyph-seed", type=int, default=1901)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_repeat_ratio.json"))
    args = parser.parse_args()
    if args.ocf_images < 2 or not 1 <= args.dimensions <= 192 or min(args.epochs, args.encoder_epochs) < 1:
        parser.error("Provide a valid paired count, latent dimension and epoch counts.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle CPU with the competition data mounted.")
    torch.set_num_threads(4)
    device = torch.device("cpu")
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    if args.ocf_images > len(images) or len(images) < 2 * ACTIVITY_BINS:
        parser.error("The gallery must support the paired sample and count-matched donors.")
    masks = images >= 128
    signature, energy = contour_signatures(images)
    scorer = official_scorer(root)
    rng = np.random.default_rng(args.glyph_seed)
    sample = torch.as_tensor(rng.choice(len(images), args.ocf_images, replace=False))
    output = {"seed": args.seed, "repeat_seed": args.repeat_seed, "glyph_seed": args.glyph_seed,
              "latent_dimensions": args.dimensions, "classifier_epochs": args.epochs,
              "encoder_epochs": args.encoder_epochs, "gallery_size": len(images),
              "ocf_images_per_block": len(sample), "activity_strata": ACTIVITY_BINS,
              "matched_trial_loss_fraction": AUGMENTATION_FRACTION, "v11_blend_weights": BLENDS,
              "v11_refit": {"mice1": "contour + timed Poisson", "mice2": "contour + time-window gain"},
              "protocol": "Fit on three repeats. Leave one independent repeat per glyph for final ranking. "
                          "Align only training responses by labeled glyph; shuffle every outer block before scoring. "
                          "Within a batch the real anchor excludes its true training trial for both models. "
                          "The augmented arm adds other-glyph residuals from its fit-only count stratum and session.",
              "folds": [], "per_mouse": {}, "training": {}}
    all_scores = {}
    for mouse in dict.fromkeys(args.mice):
        recordings = grouped_blocks(root, catalog, mouse, device, len(images))
        training, held = {}, {}
        for i, (name, counts) in enumerate(recordings.items()):
            if counts.shape[1] != 4:
                raise ValueError("Expected four labeled trials per glyph in each recording.")
            training[name], held[name] = split_repeats(counts, args.repeat_seed + i)
        del recordings

        print(f"{mouse}: build exact v11 and repeat-whitened reference", flush=True)
        torch.manual_seed(args.seed)
        contour = fit_contour_retrieval(training, signature, energy, task="task3", seed=args.seed)
        encoder = fit_glyph_encoder(training, images, epochs=args.encoder_epochs, seed=args.seed)
        gain = fit_gain_model(training, "timed", temporal=True) if mouse == "mice2" else None
        factor, views, diagnostics = training_views(contour, training, args.dimensions)
        del training
        real, real_loss = fit_ratio(views, args, augmented=False)
        matched, matched_loss = fit_ratio(views, args, augmented=True)
        output["training"][mouse] = {**diagnostics, "real_training_loss": real_loss,
                                     "matched_training_loss": matched_loss}

        mouse_scores = {}
        for session in dict.fromkeys(args.sessions):
            block_started = time.monotonic()
            order = torch.as_tensor(rng.permutation(len(images)))
            query = held[session][order]
            v11 = score_v11(mouse, contour, encoder, gain, query, images, session)
            test = project_counts(contour, query, session).mean(dim=1)[:, -args.dimensions:]
            features = whiten(test - views[session].origin, factor)
            scores = rank_gallery(v11, features, views[session].templates, real, matched)
            predictions = {name: values.argmax(dim=1) for name, values in scores.items()}
            paired, n_unique = exact_ocf(predictions, sample, order, masks, scorer)
            metrics = row_metrics(scores, predictions, order)
            fold = {"mouse": mouse, "session": session, "unique_exact_pairs": n_unique, "full_gallery": metrics,
                    "paired_ocf": summarize({name: [values] for name, values in paired.items()}),
                    "seconds": time.monotonic() - block_started}
            output["folds"].append(fold)
            extend(mouse_scores, paired)
            extend(all_scores, paired)
            output["per_mouse"][mouse] = summarize(mouse_scores)
            output["summary"] = summarize(all_scores)
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(fold), flush=True)
            del scores, query, v11
        del contour, encoder, gain, views, held, real, matched
    print(json.dumps({"summary": output["summary"], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
