"""Measure whether published mouse V1 core features predict responses to unseen glyphs."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F

from glyph_bipartite_encoding import repeat_noise
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits
from glyph_kernel_encoding import descriptors, image_kernel
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_task1_bipartite_validation import evaluate_pairs, held_response_halves, mechanistic, split_whole
from kaggle_task1_ids_retrieval import split_glyphs as ids_split
from kaggle_v8_metric_validation import summarize


FNN_VERSION = "v2.1.0"
FNN_SOURCE = "https://github.com/cajal/fnn"
PUBLIC_ARCHIVE = ("https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/"
                  "foundation_model/foundational_model_weights_and_metadata_v1.zip")
OFFICIAL_MD5 = "58fcac4b31ad2902c81e339432cec787"
SEED = 83291
FRAMES = (2, 4, 7)
STREAM = 0
BLACK_LEVEL = 0.02
OCF_GLYPHS = 96
OUTPUT = Path("/kaggle/working/task1_microns_transfer.json")


def load_released_core():
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                    f"git+{FNN_SOURCE}.git@{FNN_VERSION}"], check=True)
    from fnn.microns.build import network

    with tempfile.TemporaryDirectory(prefix="microns-core-", dir="/kaggle/working") as directory:
        archive = Path(directory) / "published.zip"
        with urlopen(PUBLIC_ARCHIVE, timeout=90) as source, archive.open("wb") as destination:
            shutil.copyfileobj(source, destination, 1024 * 1024)
        with archive.open("rb") as data:
            if hashlib.file_digest(data, "md5").hexdigest() != OFFICIAL_MD5:
                raise ValueError("MICrONS archive differs from the published checkpoint checksum.")
        with archive.open("rb") as data:
            sha256 = hashlib.file_digest(data, "sha256").hexdigest()
        with ZipFile(archive) as released, released.open("params_core.pt") as member:
            checkpoint = torch.load(member, map_location="cpu", weights_only=True)
    torch.manual_seed(SEED)
    random_core = network(units=1).core
    torch.manual_seed(SEED)
    trained_core = network(units=1).core
    if not all(name.startswith("core.") for name in checkpoint):
        raise ValueError("Expected the released shared core without scan-specific weights.")
    trained_core.load_state_dict({name.removeprefix("core."): value for name, value in checkpoint.items()})
    return {"contour": None, "random_core": random_core, "published_core": trained_core}, sha256


@torch.inference_mode()
def core_descriptors(core, images, batch_size=16, *, offload_core=True):
    core = core.to(images.device).eval()
    collected = []
    for batch in images.split(batch_size):
        # Express the same square image on the center of the public core's 128×192 input grid.
        glyph = F.interpolate(batch[:, None].float() / 255, (96, 96), mode="bilinear", align_corners=False)
        glyph = F.pad(glyph, (48, 48, 16, 16)) * (1 - BLACK_LEVEL) + BLACK_LEVEL
        # FNN divides attention vectors by their norms. A tiny common black level
        # gives the untrained, zero-bias core defined states during the blank prefix.
        dark = torch.full_like(glyph, BLACK_LEVEL)
        modulation = glyph.new_zeros((len(batch), core.modulations))
        core.reset()
        features = []
        for frame in range(-2, max(FRAMES) + 1):
            state = core(dark if frame <= 0 else glyph, modulation, stream=STREAM)
            if frame in FRAMES:
                features.append(F.adaptive_avg_pool2d(state, (8, 8)).flatten(1))
        batch_features = torch.stack(features, dim=1)
        if not bool(batch_features.isfinite().all()):
            raise ValueError("FNN produced a nonfinite feature for the fixed glyph clip.")
        collected.append(batch_features)
    core.reset()
    if offload_core:
        core.cpu()
    return torch.cat(collected).unbind(1)


@torch.no_grad()
def normalized_views(views, fit):
    features = []
    for view in views:
        center = view[fit].mean(0)
        scale = (view[fit] - center).square().sum(1).mean().sqrt().clamp_min(1e-5)
        features.append((view - center) / (scale * len(views) ** 0.5))
    return torch.cat(features, dim=1)


@torch.no_grad()
def template_rates(train_kernel, held_kernel, stabilized, ridge=0.4):
    center = train_kernel.mean(0)
    grand = center.mean()
    gram = train_kernel - center[None] - center[:, None] + grand
    gram.diagonal().add_(ridge)
    held = held_kernel - center[None] - held_kernel.mean(1, keepdim=True) + grand
    mean = stabilized.mean(0)
    weights = torch.cholesky_solve(stabilized - mean, torch.linalg.cholesky(gram))
    return ((held @ weights + mean).clamp_min(0).square() - 0.1).clamp_min_(1e-5)


def publish(report, started, stage):
    report.update(stage=stage, seconds=time.monotonic() - started)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{stage} after {report['seconds']:.1f} s", flush=True)


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the transfer experiment on Kaggle with labels and an enabled GPU.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    _, bipartite_held, _ = split_whole(len(images), 36467, torch.device("cpu"))
    _, ids_held = ids_split(len(images), torch.device("cpu"))
    eligible = np.setdiff1d(np.arange(len(images)), np.union1d(bipartite_held.numpy(), ids_held.numpy()))
    rng = np.random.default_rng(SEED)
    hold = np.sort(rng.choice(eligible, size=320, replace=False))
    fit = torch.as_tensor(np.setdiff1d(np.arange(len(images)), hold), device=device)
    held = torch.as_tensor(hold, device=device)
    groups = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    observed = {day: counts[held] for day, counts in groups.items()}
    training = {day: counts[fit] for day, counts in groups.items()}
    rates = torch.stack([counts.mean(1) for counts in training.values()])
    sessions, _, flat_dimensions = rates.shape
    stabilized = (rates + 0.1).sqrt().transpose(0, 1).flatten(1)
    noise = repeat_noise(training)
    del groups, training, rates

    cores, archive_sha = load_released_core()
    report = {"seed": SEED, "mouse": "mice1", "fit_whole_glyphs": len(fit), "held_whole_glyphs": len(held),
              "ocf_glyphs_per_session": OCF_GLYPHS, "stream": STREAM, "positive_frames_at_30hz": list(FRAMES),
              "dark_frames": 3, "common_black_level": BLACK_LEVEL,
              "code": {"url": FNN_SOURCE, "version": FNN_VERSION, "license": "MIT"},
              "checkpoint": {"url": PUBLIC_ARCHIVE, "sha256": archive_sha, "official_md5": OFFICIAL_MD5,
                             "params": "params_core.pt", "license": "CC BY 4.0 per the BossDB project listing",
                             "project": "https://bossdb.org/project/microns-minnie"},
              "protocol": "Known-gallery diagnostic with 320 real held glyph images and no formal test inputs. One frozen "
                          "natural-video pretrained stream and a same-shape untrained stream see identical fixed centered glyph "
                          "clips and zero modulation. Each is compared alone and with 50% contour kernel. The exact fixed ridge, "
                          "training glyphs and repetitions are shared. This is a core transfer probe with stylized display input, "
                          "without the original mouse's readout, eye or behavior calibration. Retrieval sees only held spikes and "
                          "these predeclared gallery images. It measures within-gallery identification, before open-atlas coverage.",
              "arms": {}}
    publish(report, started, "source_verified")

    pair_ids, report["near_pairs"] = evaluate_pairs(images[held], SEED + 1)
    halves = held_response_halves(observed, SEED + 2)
    ocf_rows = torch.as_tensor(rng.choice(len(held), size=OCF_GLYPHS, replace=False), device=device)
    held_images = images[held]
    exact = PairedOCF(official_scorer(root), held_images, (held_images[ocf_rows] >= 128).cpu().numpy(), device, 12)
    contour_views = zip(*(descriptors(batch) for batch in images.split(128)))
    contour_features = normalized_views([torch.cat(view) for view in contour_views], fit)
    contour_kernel = image_kernel(contour_features, contour_features[fit])
    del contour_features
    ocf_scores = {}
    for name, core in cores.items():
        if core is None:
            options = {name: contour_kernel}
        else:
            features = normalized_views(core_descriptors(core, images), fit)
            transfer_kernel = image_kernel(features, features[fit])
            options = {name: transfer_kernel, name + "_with_contour": (transfer_kernel + contour_kernel) * 0.5}
            del features
        for arm, kernel in options.items():
            rates = template_rates(kernel[fit], kernel[held], stabilized).reshape(len(held), sessions, flat_dimensions).transpose(0, 1)
            near, _ = mechanistic(rates, halves, noise, pair_ids)
            details, scores = {}, []
            for index, (day, spikes) in enumerate(observed.items()):
                raw = count_template_logits(spikes, rates[index], "timed")
                ranked = raw.topk(16, dim=1).indices
                hit = ranked == torch.arange(len(held), device=device)[:, None]
                measured = exact.evaluate(ranked[ocf_rows, :1])[:, 0]
                scores.append(measured)
                details[day] = {"top_1": float(hit[:, 0].float().mean()),
                                "top_5": float(hit[:, :5].any(1).float().mean()),
                                "top_16": float(hit.any(1).float().mean())}
            ocf_scores[arm] = scores
            report["arms"][arm] = {"held_gallery": details, "held_near_pair_prediction": near,
                                   "exact_ocf_vs_contour": summarize(scores, ocf_scores["contour"])}
            if arm.startswith("published_core"):
                random_arm = arm.replace("published_core", "random_core")
                report["arms"][arm]["exact_ocf_vs_same_shape_random"] = summarize(scores, ocf_scores[random_arm])
            publish(report, started, arm)
    publish(report, started, "complete")


if __name__ == "__main__":
    main()
