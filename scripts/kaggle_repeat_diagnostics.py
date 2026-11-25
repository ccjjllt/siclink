"""Measure signal and decoder scaling with labeled SiClink training repeats."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_glyphs
from kaggle_baseline import fit_ridge
from kaggle_diagnostics import feature_view, official_scorer, standardize


VIEWS = ("evoked-total", "evoked-multiscale", "early-late", "original")
WITHIN_REPEATS = (1, 2, 4)


@dataclass
class SessionDiagnostic:
    logits: dict[int, torch.Tensor]
    pixels: dict[int, torch.Tensor]
    response: torch.Tensor
    split_half: dict
    forecast: dict[str, float]


def agreement(left, right):
    """Flatten a cross-glyph correlation after removing each coordinate's mean."""
    left = (left - left.mean(0)).flatten()
    right = (right - right.mean(0)).flatten()
    return float((left @ right) / (left.norm() * right.norm()).clamp_min(1e-12))


def precision_weights(covariance_factor, centers):
    return torch.cholesky_solve(centers.T.contiguous(), covariance_factor).T


def template_logits(query, centers, weights):
    return query @ weights.T - 0.5 * (centers * weights).sum(1)


def rank_glyphs(logits):
    predicted = logits.argmax(1)
    ranks = (logits > logits.diagonal()[:, None]).sum(1) + 1
    correct = predicted == torch.arange(len(predicted), device=logits.device)
    return predicted, correct, ranks


def binary_images(estimate, foreground):
    threshold = 0.25 * 0.45 + 0.75 * torch.quantile(estimate, 1 - foreground, dim=1, keepdim=True)
    return estimate >= threshold


def score_block(estimate, logits, reference, foreground, score, selected):
    predicted, correct, ranks = rank_glyphs(logits)
    image = binary_images(estimate, foreground)
    dice = 2 * (image & reference).sum(1) / (image.sum(1) + reference.sum(1)).clamp_min(1)
    result = {"top1_count": int(correct.sum()), "top5_count": int((ranks <= 5).sum()),
              "rank_sum": int(ranks.sum()), "n": len(ranks), "dice_sum": float(dice.sum())}
    if score is not None:
        targets = reference[selected].reshape(-1, 64, 64).cpu().numpy()
        ridge = image[selected].reshape(-1, 64, 64).cpu().numpy()
        gallery = reference[predicted[selected]].reshape(-1, 64, 64).cpu().numpy()
        device = str(estimate.device)
        result["ridge_ocfs"] = score(ridge, targets, device=device).tolist()
        result["gallery_ocfs"] = score(gallery, targets, device=device).tolist()
    return result


def aggregate_score(parts):
    total = sum(part["n"] for part in parts)
    result = {"n": total, "top1": sum(part["top1_count"] for part in parts) / total,
              "top5": sum(part["top5_count"] for part in parts) / total,
              "mean_rank": sum(part["rank_sum"] for part in parts) / total,
              "ridge_dice": sum(part["dice_sum"] for part in parts) / total}
    for name in ("ridge_ocfs", "gallery_ocfs"):
        measured = [value for part in parts for value in part.get(name, [])]
        if measured:
            result[name.removesuffix("s")] = sum(measured) / len(measured)
            result["ocf_n"] = len(measured)
    return result


def session_diagnostic(counts, fit, held, masks, descriptors, repeat_order):
    fitting, stats = standardize(counts[fit])
    query, _ = standardize(counts[held], stats)
    query = query.gather(1, repeat_order[:, :, None].expand_as(query))

    centers = fitting.mean(1)
    errors = (fitting - centers[:, None]).flatten(0, 1)
    residual = errors.T @ errors / (len(fitting) * (fitting.shape[1] - 1))
    residual.diagonal().add_(0.35)
    factor = torch.linalg.cholesky(residual)

    visual_mean = descriptors[fit].mean(0)
    visual_train = descriptors[fit] - visual_mean
    visual_scale = visual_train.square().mean().sqrt().clamp_min(1e-4)
    forward_weights, forward_bias = fit_ridge(visual_train / visual_scale, centers)
    predicted_centers = ((descriptors[held] - visual_mean) / visual_scale) @ forward_weights + forward_bias
    template_weights = precision_weights(factor, predicted_centers)
    readout, intercept = fit_ridge(centers, masks[fit].float())

    logits = {}
    pixels = {}
    for repeats in WITHIN_REPEATS:
        response = query[:, :repeats].mean(1)
        logits[repeats] = template_logits(response, predicted_centers, template_weights)
        pixels[repeats] = response @ readout + intercept

    split_half = {}
    for repeats in (1, 2):
        left = query[:, :repeats].mean(1)
        right = query[:, 2:2 + repeats].mean(1)
        right_logits = template_logits(right, predicted_centers, template_weights)
        # The independent right-hand neural responses form a labeled gallery only for this diagnostic.
        repeat_match = template_logits(left, right, precision_weights(factor, right))
        _, correct, ranks = rank_glyphs(repeat_match)
        split_half[str(repeats)] = {
            "spike_response_correlation": agreement(left, right),
            "gallery_logit_correlation": agreement(logits[repeats], right_logits),
            "ridge_pixel_correlation": agreement(pixels[repeats], right @ readout + intercept),
            "held_spike_template_top1": float(correct.float().mean()),
            "held_spike_template_mean_rank": float(ranks.float().mean()),
            "n": len(correct),
        }

    half_reliability = max(0, min(1, split_half["2"]["spike_response_correlation"]))
    forecast = {}
    for count in (4, 8, 12, 16, 20):
        scale = count / 2
        forecast[str(count)] = scale * half_reliability / (1 + (scale - 1) * half_reliability)
    return SessionDiagnostic(logits, pixels, query.mean(1), split_half, forecast)


@torch.no_grad()
def run_mouse(mouse, device, root, catalog, score, args):
    masks = load_images(root, device) >= 128
    fit, held = split_glyphs(len(masks), args.held_glyphs, args.seed, device)
    descriptors = F.avg_pool2d(masks[:, None].float(), 4).flatten(1)
    masks = masks.flatten(1)
    target = masks[held]
    foreground = float(masks[fit].float().mean())
    recordings = grouped_blocks(root, catalog, mouse, device, len(masks))
    rng = torch.Generator().manual_seed(args.seed)
    selected = torch.randperm(len(held), generator=rng)[:args.ocf_images].to(device)
    repeat_order = {name: torch.rand(len(held), 4, generator=rng).argsort(1).to(device)
                    for name in recordings}
    reports = []

    for view in args.views:
        sessions = {name: session_diagnostic(feature_view(counts, view), fit, held, masks,
                                             descriptors, repeat_order[name])
                    for name, counts in recordings.items()}
        within = {}
        for repeats in WITHIN_REPEATS:
            parts = [score_block(item.pixels[repeats], item.logits[repeats], target, foreground, score, selected)
                     for item in sessions.values()]
            within[str(repeats)] = aggregate_score(parts)

        pooled = {}
        for session_count in range(2, len(sessions) + 1):
            parts = []
            for selected_sessions in combinations(sessions.values(), session_count):
                logits = torch.stack([item.logits[4] for item in selected_sessions]).mean(0)
                pixels = torch.stack([item.pixels[4] for item in selected_sessions]).mean(0)
                parts.append(score_block(pixels, logits, target, foreground, score, selected))
            pooled[str(4 * session_count)] = aggregate_score(parts)

        cross_session = {}
        for (first, left), (second, right) in combinations(sessions.items(), 2):
            cross_session[f"{first}/{second}"] = {
                "spike_response_correlation": agreement(left.response, right.response),
                "gallery_logit_correlation": agreement(left.logits[4], right.logits[4]),
                "ridge_pixel_correlation": agreement(left.pixels[4], right.pixels[4]),
            }

        result = {"mouse": mouse, "view": view, "device": str(device),
                  "within_session": within, "cross_session_same_glyph_proxy": pooled,
                  "disjoint_halves": {name: item.split_half for name, item in sessions.items()},
                  "cross_session_agreement": cross_session,
                  "independent_repeat_reliability_forecast": {name: item.forecast for name, item in sessions.items()}}
        reports.append(result)
        compact = {key: result[key] for key in ("mouse", "view", "within_session", "cross_session_same_glyph_proxy")}
        print(json.dumps(compact), flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", nargs="+", default=VIEWS[:2], choices=VIEWS)
    parser.add_argument("--held-glyphs", type=int, default=320)
    parser.add_argument("--ocf-images", type=int, default=8, help="Sampled glyphs per session or session combination; 0 skips OCF")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/repeat_diagnostics.json"))
    args = parser.parse_args()
    if not 2 <= args.held_glyphs < 2748 or not 0 <= args.ocf_images <= args.held_glyphs:
        parser.error("Require 2–2747 held glyphs and 0–held glyphs sampled for OCF")

    started = time.monotonic()
    torch.set_num_threads(2)
    root = locate_data()
    catalog = load_catalog(root)
    score = official_scorer(root) if args.ocf_images else None
    devices = [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())] or [torch.device("cpu")]
    result = {
        "seed": args.seed, "held_glyphs": args.held_glyphs, "gallery_chance_top1": 1 / args.held_glyphs,
        "ocf_images_per_group": args.ocf_images,
        "protocol": {
            "within_session": "1/2/4 trials from one labeled training recording; query glyphs held out across sessions. OCF samples reuse the same glyphs for paired comparisons.",
            "gallery": "Closed diagnostic gallery of held training images; only fitted images/responses train either ridge",
            "disjoint_halves": "Per-glyph shuffled nonoverlapping trials; right neural response is an oracle labeled template",
            "cross_session_same_glyph_proxy": "Same known glyph pooled across 2/3/4 distinct training sessions via independently normalized session scores and pixels. Training session day labels are unavailable; session5 and competition test are unused.",
            "forecast": "Spearman–Brown extrapolation from disjoint two-repeat halves assumes independent repeat noise and fixed response tuning. Values describe centered neural repeatability, not OCF or true Task 1 performance at 20 repeats.",
        },
        "results": [],
    }
    with ThreadPoolExecutor(max_workers=min(2, len(devices))) as executor:
        jobs = [executor.submit(run_mouse, mouse, devices[index % len(devices)], root, catalog, score, args)
                for index, mouse in enumerate(("mice1", "mice2"))]
        for job in as_completed(jobs):
            result["results"].extend(job.result())
            result["results"].sort(key=lambda item: (item["mouse"], item["view"]))
            result["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {result['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
