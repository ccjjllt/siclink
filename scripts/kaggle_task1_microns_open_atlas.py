"""Replicate the published MICrONS core comparison in the full Task 1 public atlas."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_kernel_encoding import image_kernel
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_task1_bipartite_validation import image_proxy, session_predictions, split_whole
from kaggle_task1_crossfit_candidate import fit_crossfit_head
from kaggle_task1_ids_retrieval import split_glyphs as ids_split
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE
from kaggle_task1_microns_transfer import (BLACK_LEVEL, FNN_SOURCE, FNN_VERSION, FRAMES, OFFICIAL_MD5, PUBLIC_ARCHIVE,
                                          SEED as CLOSED_SEED, STREAM, core_descriptors, load_released_core)
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, fit_v8, nested_ranker, rendered_bank, summarize


ATLAS_CHUNK = 384
CORE_BATCH = 16
HELD_COUNT = 320
RIDGE = 0.4
V14 = "v14"
ARMS = ("random_core", "published_core")
SEED = 86711
OUTPUT = Path("/kaggle/working/task1_microns_open_atlas.json")


def outer_split(size, seed, device):
    cpu = torch.device("cpu")
    _, bipartite, _ = split_whole(size, 36467, cpu)
    _, ids = ids_split(size, cpu)
    earlier = np.union1d(bipartite.numpy(), ids.numpy())
    closed_eligible = np.setdiff1d(np.arange(size), earlier)
    closed = np.sort(np.random.default_rng(CLOSED_SEED).choice(closed_eligible, size=HELD_COUNT, replace=False))
    eligible = np.setdiff1d(closed_eligible, closed)
    held = np.sort(np.random.default_rng(seed).choice(eligible, size=HELD_COUNT, replace=False))
    fit = np.setdiff1d(np.arange(size), held)
    past = {"task1_microns_closed_v2": closed, "task1_bipartite_v1": bipartite.numpy(), "task1_ids_v1": ids.numpy()}
    overlap = {name: int(np.intersect1d(held, cohort).size) for name, cohort in past.items()}
    if any(overlap.values()):
        raise ValueError("The open-atlas cohort must use fresh whole glyphs.")
    evidence = {"excluded_prior_glyphs": size - len(eligible), "overlap_with_earlier_evaluation_cohorts": overlap}
    return torch.as_tensor(fit, device=device), torch.as_tensor(held, device=device), evidence


@dataclass(frozen=True)
class KernelReadout:
    anchors: torch.Tensor
    feature_centers: tuple[torch.Tensor, ...]
    feature_scales: tuple[torch.Tensor, ...]
    kernel_center: torch.Tensor
    kernel_grand: torch.Tensor
    weights: torch.Tensor
    readout_mean: torch.Tensor

    def rates(self, views):
        vector = normalized_features(views, self.feature_centers, self.feature_scales)
        kernel = image_kernel(vector, self.anchors)
        centered = kernel - self.kernel_center - kernel.mean(1, keepdim=True) + self.kernel_grand
        return ((centered @ self.weights + self.readout_mean).clamp_min(0).square() - 0.1).clamp_min_(1e-5)


def normalized_features(views, centers, scales):
    return torch.cat([(view - center) / scale for view, center, scale in zip(views, centers, scales, strict=True)], dim=1)


@torch.no_grad()
def fit_kernel_readout(core, images, stabilized):
    views = core_descriptors(core, images, batch_size=CORE_BATCH, offload_core=False)
    return fit_kernel_readout_from_views(views, stabilized)


@torch.no_grad()
def fit_kernel_readout_from_views(views, stabilized):
    centers = tuple(view.mean(0) for view in views)
    scales = tuple((view - center).square().sum(1).mean().sqrt().clamp_min(1e-5) * len(views) ** 0.5
                   for view, center in zip(views, centers, strict=True))
    anchors = normalized_features(views, centers, scales)
    kernel = image_kernel(anchors, anchors)
    column_mean = kernel.mean(0)
    grand_mean = column_mean.mean()
    centered = kernel - column_mean[None] - column_mean[:, None] + grand_mean
    centered.diagonal().add_(RIDGE)
    mean = stabilized.mean(0)
    factor = torch.linalg.cholesky(centered)
    weights = torch.cholesky_solve(stabilized - mean, factor)
    return KernelReadout(anchors, centers, scales, column_mean, grand_mean, weights, mean)


@torch.no_grad()
def stream_atlas(core, fitted, atlas, n_features, chunk):
    # All templates are exact FP32. Store only rates on host and move one day's templates back when scoring.
    rates = torch.empty((len(SESSIONS), len(atlas), n_features), dtype=torch.float32)
    milestones = []
    started = time.monotonic()
    for offset in range(0, len(atlas), chunk):
        block = atlas[offset:offset + chunk]
        views = core_descriptors(core, block, batch_size=CORE_BATCH, offload_core=False)
        expected = fitted.rates(views).reshape(len(block), len(SESSIONS), n_features).transpose(0, 1)
        if not bool(expected.isfinite().all()):
            raise ValueError("The chunked kernel readout returned nonfinite spike rates.")
        stop = offset + len(block)
        rates[:, offset:stop].copy_(expected.cpu())
        if (offset // chunk) % 20 == 0 or stop == len(atlas):
            elapsed = time.monotonic() - started
            milestones.append({"glyphs": stop, "seconds": elapsed})
            print(f"Forwarded {stop:,}/{len(atlas):,} atlas glyphs after {elapsed:.1f} s", flush=True)
    return rates, milestones


def publish(report, output, started, stage):
    report.update(seconds=time.monotonic() - started, stage=stage)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": stage, "seconds": report["seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--atlas-chunk", type=int, default=ATLAS_CHUNK)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 16 <= args.ocf_images <= HELD_COUNT or args.atlas_chunk < CORE_BATCH:
        parser.error("Select a valid OCF size and a reasonably sized streaming block.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run this labeled experiment in a private Kaggle T4 session.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, held, independence = outer_split(len(images), args.seed, device)
    groups = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    if tuple(groups) != SESSIONS or any(value.shape[1] != 4 for value in groups.values()):
        raise ValueError("Require four labeled days and four measured repeats per glyph.")
    training = {day: value[fit] for day, value in groups.items()}
    queries = {day: value[held] for day, value in groups.items()}
    del groups
    fitted_images = images[fit]
    references = images[held]
    training_rates = torch.stack([counts.mean(1) for counts in training.values()])
    n_features = training_rates.shape[-1]
    stabilized = (training_rates + 0.1).sqrt().transpose(0, 1).flatten(1)
    del training_rates

    cores, checksum = load_released_core()
    report = {"seed": args.seed, "mouse": "mice1", "fit_glyphs": len(fit), "held_glyphs": len(held),
              "measured_repeats_per_query": 4, "ocf_images_per_recording": args.ocf_images, **independence,
              "outer_v14_anchor_ocf": ANCHOR, "original_nomination_counts": NOMINATIONS,
              "kernel_ridge": RIDGE, "core_stream": STREAM, "feature_frames_at_30hz": list(FRAMES),
              "dark_prefix_frames": 3, "common_black_level": BLACK_LEVEL, "core_dtypes": "float32",
              "core_inference_batch": CORE_BATCH, "atlas_chunk": args.atlas_chunk, "seed_of_same_architecture_random_core": CLOSED_SEED,
              "published_core": {"source": FNN_SOURCE, "source_version": FNN_VERSION, "source_license": "MIT",
                                 "checkpoint_url": PUBLIC_ARCHIVE, "archive_sha256": checksum, "archive_expected_md5": OFFICIAL_MD5,
                                 "checkpoint_license": "CC BY 4.0 per the published MICrONS Minnie project listing"},
              "protocol": "All fits exclude 320 whole glyphs from every day. Train equal kernel readouts on the real labeled "
                          "training images in both published and same-architecture random arms. Extract fixed frozen FNN features "
                          "from the same 27k display-aligned public atlas used by v14. Replace only its timed and total-rate Poisson "
                          "sources while retaining its uncertainty, contours, prior, ranker, output geometry and anchor. Compare fixed "
                          "edges and new nominations with exactly the original per-row number of distinct candidates. Real held images "
                          "are used only for the high-margin nearest-outline identity proxy and paired OCF after prediction. "
                          "This is open-atlas inference from held measured spikes with no formal test access.",
              "sessions": {}, "cores": {}}
    publish(report, args.output, started, "source_verified")

    scorer = official_scorer(root)
    stock_split = split_calibration(len(fit), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    _, noise, nested, columns = nested_ranker(training, fitted_images, stock_split, scorer, MODEL_SEED)
    ranker, folds = fit_crossfit_head(training, fitted_images, scorer, columns)
    model = fit_v8(training, fitted_images, MODEL_SEED)
    del training
    if not torch.equal(columns, model.codepoints.cpu()):
        raise ValueError("Public atlas columns must agree with the frozen v14 calibration folds.")
    report["v14"] = {"near_fold": nested, "calibration_folds": folds, "ranker": ranker.diagnostics,
                    "outer_fit": model.report, "atlas_glyphs": len(model.atlas.images)}
    report["atlas_column_order_verified"] = True
    publish(report, args.output, started, "v14_ready")

    banks = {}
    for arm in ARMS:
        began = time.monotonic()
        head = cores.pop(arm)
        readout = fit_kernel_readout(head, fitted_images, stabilized)
        fit_seconds = time.monotonic() - began
        bank, milestones = stream_atlas(head, readout, model.atlas.images, n_features, args.atlas_chunk)
        banks[arm] = bank
        report["cores"][arm] = {"fit_seconds": fit_seconds, "streaming_progress": milestones,
                                "kernel_dimension": readout.anchors.shape[1], "atlas_rate_shape": list(bank.shape),
                                "atlas_rate_mean": float(bank.mean()), "atlas_device_after_encoding": str(bank.device),
                                "single_psd_solve": True}
        del head, readout, bank
        publish(report, args.output, started, arm + "_atlas_ready")
    del stabilized, fitted_images

    proxy, trusted = image_proxy(model, references)
    sampled = torch.as_tensor(np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_images, replace=False), device=device)
    rendered = rendered_bank(model)
    exact = PairedOCF(scorer, rendered, (references[sampled] >= 128).cpu().numpy(), device, 12)
    ocf = {}
    for index, (day, queries_in_day) in enumerate(queries.items()):
        alternatives = {arm: bank[index].to(device) for arm, bank in banks.items()}
        choices, response = session_predictions(model, ranker, noise[day], queries_in_day, day, alternatives, proxy, trusted)
        scores = {arm: exact.evaluate(column[sampled, None])[:, 0] for arm, column in choices.items()}
        response["paired_exact_ocf_vs_v14"] = {arm: summarize([values], [scores[V14]]) for arm, values in scores.items()}
        response["published_vs_same_architecture_random"] = {mode: summarize([scores["published_core" + mode]], [scores["random_core" + mode]])
                                                               for mode in ("_fixed_graph", "_equal_budget")}
        report["sessions"][day] = response
        for arm, values in scores.items():
            ocf.setdefault(arm, []).append(values)
        report["exact_ocf_vs_v14_clustered_by_glyph"] = {arm: summarize(values, ocf[V14]) for arm, values in ocf.items()}
        report["published_vs_same_architecture_random_across_days"] = {mode: summarize(ocf["published_core" + mode], ocf["random_core" + mode])
                                                                       for mode in ("_fixed_graph", "_equal_budget")}
        report["unique_exact_pairs"] = len(exact.values)
        publish(report, args.output, started, day)
    publish(report, args.output, started, "complete")


if __name__ == "__main__":
    main()
