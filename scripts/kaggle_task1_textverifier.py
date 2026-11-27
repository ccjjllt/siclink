"""Test frozen stroke-sensitive features for Task 1 on labeled glyphs using a remote CUDA GPU."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from urllib.request import urlopen

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from glyph_bipartite_encoding import neighbors, timing_view
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import build_font_atlas, standard_error, standardized
from kaggle_task1_hanzi_latent_likelihood import SEED as LIKELIHOOD_SEED, previous_cohorts, split as likelihood_split
from kaggle_task1_native_hanzi import (
    OCR_CHECKSUM, OCR_REVISION, PC_COUNT, RIDGE, SEED as OCR_SEED,
    feature_extractors, fit_response_ridge, normalized_glyph,
)
from kaggle_task1_native_hanzi_open_atlas import visual_proxy


SEED = 223267
CALIBRATION, OUTER, ATLAS_SIZE = 256, 320, 27_550
DEPTHS = (1, 5, 16, 32)
OCR = "incumbent_ocr"
GLOBAL, SPATIAL = "textverifier_global", "textverifier_spatial"
FUSION = "equal_ocr_textverifier_spatial"
REVISION = "adc233a117cbb039842b245b5acf84e0143a5f11"
CHECKSUM = "ee6860f93debc240bfdef5572cd2ce9359c84e0c8e2552a4047919ba59780ef3"
SOURCE = f"https://huggingface.co/ATH-MaaS/TextVerifier-CJK-CLIP/resolve/{REVISION}/textverifier_stage3.pth"
PILOT_BANK = "task1_textverifier_pilot_bank.npz"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full", "throughput", "prepare-pilot"), default="pilot")
    parser.add_argument("--data-root", type=Path, help="Extracted competition directory containing catalog.json")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_textverifier.json"))
    parser.add_argument("--checkpoint", type=Path,
                        help="Pinned TextVerifier checkpoint file or an input directory containing it")
    parser.add_argument("--mouse", choices=("mice1", "mice2"), default="mice2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pilot-fit-glyphs", type=int, default=0,
                        help="Limit the pilot ridge/PCA fit to a seeded subset of the fitting glyphs")
    parser.add_argument("--pilot-bank", type=Path, help="Precomputed public pilot glyphs or their input directory")
    parser.add_argument("--throughput-images", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--ocf-glyphs", type=int, default=0, help="Optional paired score and Top-4 oracle sample in full mode")
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.ocf_glyphs <= OUTER:
        parser.error("Use a positive batch size and at most 320 exact OCF glyphs.")
    if args.ocf_glyphs and args.mode != "full":
        parser.error("Exact OCF scoring is available in full mode.")
    if args.pilot_fit_glyphs and (args.mode != "pilot" or not 256 <= args.pilot_fit_glyphs <= 2172):
        parser.error("The optional fit subset applies to pilot mode and contains 256 to 2,172 glyphs.")
    if args.pilot_bank is not None and args.mode != "pilot":
        parser.error("A prepared bank applies to pilot mode.")
    return args


def publish(report, output, started, stage):
    report.update(stage=stage, seconds=time.monotonic() - started)
    staging = output.with_suffix(output.suffix + ".tmp")
    staging.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(output)
    print(f"{stage} after {report['seconds']:.1f}s", flush=True)


def load_vision(checkpoint, device):
    import timm

    digest = sha256()
    with checkpoint.open("rb") as handle:
        is_zip = handle.read(4) == b"PK\x03\x04"
        handle.seek(0)
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != CHECKSUM:
        raise ValueError("TextVerifier checkpoint SHA-256 does not match the pinned release.")
    state = torch.load(checkpoint, map_location="cpu", mmap=is_zip, weights_only=True)
    state = state.get("model_state_dict", state)
    vision = timm.create_model("vit_large_patch16_224", pretrained=False, img_size=128,
                               patch_size=16, num_classes=0, global_pool="avg")
    projector = nn.Sequential()
    projector.add_module("fc1", nn.Linear(1024, 1024, bias=False))
    projector.add_module("ln", nn.LayerNorm(1024))
    projector.add_module("act", nn.GELU())
    projector.add_module("fc2", nn.Linear(1024, 1024, bias=False))
    for module, prefix in ((vision, "image_encoder.backbone."), (projector, "image_projector.")):
        module.load_state_dict({name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)},
                               strict=True)
    return vision.to(device, dtype=torch.float16).eval(), projector.to(device, dtype=torch.float16).eval()


def textverifier(checkpoint, device):
    if checkpoint is not None:
        if checkpoint.is_dir():
            root = checkpoint
            checkpoint = next(root.rglob("textverifier_stage3.pth"), None)
            if checkpoint is None:
                raise FileNotFoundError(f"TextVerifier checkpoint is missing under {root}")
        return load_vision(checkpoint, device)
    with TemporaryDirectory(prefix="siclink-textverifier-") as work:
        path = Path(work) / "textverifier_stage3.pth"
        with urlopen(SOURCE, timeout=120) as response, path.open("wb") as target:
            shutil.copyfileobj(response, target, length=4 * 1024 * 1024)
        return load_vision(path, device)


def verifier_features(images, vision, projector):
    # Competition rasters are high-ink on black. The verifier was fit on ordinary RGB text crops.
    glyphs = 1 - images[:, None].to(device=next(vision.parameters()).device, dtype=torch.float32) / 255
    rgb = F.interpolate(glyphs, size=(128, 128), mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        tokens = vision.forward_features(rgb.mul(2).sub(1))
        projected = projector(vision.forward_head(tokens, pre_logits=True))
    patches = tokens[:, vision.num_prefix_tokens:].float().transpose(1, 2).reshape(len(images), 1024, 8, 8)
    return {GLOBAL: F.normalize(projected.float()), SPATIAL: F.avg_pool2d(patches, 2).flatten(1)}


def ocr_features(images, ocr, device):
    normalized = np.stack([normalized_glyph(image) for image in images.numpy()])
    values = torch.from_numpy(normalized)[:, None].to(device=device, dtype=torch.float32).div_(255)
    return ocr(values.contiguous(memory_format=torch.channels_last)).flatten(1)


def fit_axes(fitting):
    """Match the incumbent whitened PCA while allowing the atlas to stream in batches."""
    if not bool(fitting.isfinite().all()):
        raise RuntimeError("Vision features are nonfinite before fitting PCA.")
    center = fitting.mean(0)
    centered = fitting - center
    torch.manual_seed(OCR_SEED + 1)
    _, _, basis = torch.pca_lowrank(centered, q=PC_COUNT + 16, center=False, niter=3)
    basis = basis[:, :PC_COUNT]
    coordinates = centered @ basis
    scale = coordinates.std(0).clamp_min(1e-4)
    return center, basis / scale, coordinates / scale


def coordinates(fit_images, candidate_images, extract, batch_size):
    chunks = [extract(batch) for batch in fit_images.split(batch_size)]
    axes = {}
    for arm in tuple(chunks[0]):
        axes[arm] = fit_axes(torch.cat([chunk.pop(arm) for chunk in chunks]))
    fitted = {arm: parameters[2] for arm, parameters in axes.items()}
    banks = {name: {} for name in candidate_images}
    for name, images in candidate_images.items():
        coded = {arm: [] for arm in axes}
        for batch in images.split(batch_size):
            for arm, features in extract(batch).items():
                center, basis, _ = axes[arm]
                coded[arm].append(F.normalize((features - center) @ basis))
        banks[name] = {arm: torch.cat(parts).T.contiguous() for arm, parts in coded.items()}
    return fitted, banks


def cohorts(size):
    earlier = previous_cohorts(size)
    earlier["mice1_ocr_likelihood"] = likelihood_split(size, np.random.default_rng(LIKELIHOOD_SEED), earlier)[1]
    earlier["mice2_ocr_likelihood"] = likelihood_split(size, np.random.default_rng(223071), earlier)[1]
    eligible = np.setdiff1d(np.arange(size), np.unique(np.concatenate(tuple(earlier.values()))))
    rng = np.random.default_rng(SEED)
    outer = np.sort(rng.choice(eligible, OUTER, replace=False))
    development = np.setdiff1d(np.arange(size), outer)
    calibration = np.sort(rng.choice(development, CALIBRATION, replace=False))
    fit = np.setdiff1d(development, calibration)
    overlap = {name: int(np.intersect1d(outer, previous).size) for name, previous in earlier.items()}
    return torch.from_numpy(fit), torch.from_numpy(calibration), torch.from_numpy(outer), overlap


def pair_index(references, proxy, trusted):
    close, _ = neighbors(references, depth=32)
    used_rows, used_codes, pairs = set(), set(), []
    codes, trusted = proxy.tolist(), trusted.tolist()
    for left in np.random.default_rng(SEED + 1).permutation(len(references)):
        if left in used_rows or not trusted[left] or codes[left] in used_codes:
            continue
        right = next((row for row in close[left].tolist()
                      if row not in used_rows and trusted[row] and codes[row] not in used_codes
                      and codes[row] != codes[left]), None)
        if right is not None:
            pairs.append((left, right))
            used_rows.update((left, right))
            used_codes.update((codes[left], codes[right]))
    if not pairs:
        raise ValueError("No distinct trusted contour proxies for held calibration pairs.")
    return torch.tensor(pairs)


def outcome(values, baseline):
    difference = values - baseline
    return {"mean": float(values.mean()), "paired_delta_vs_ocr": float(difference.mean()),
            "paired_se_vs_ocr": standard_error(difference), "paired_count": len(values)}


def pair_preferences(scores, pairs):
    left, right = pairs.to(scores.device).T
    margin = scores[left, left] - scores[left, right] + scores[right, right] - scores[right, left]
    return (margin.gt(0).float() + 0.5 * margin.eq(0)).cpu().numpy()


def pair_summary(scores, pairs):
    correct = {arm: pair_preferences(values, pairs) for arm, values in scores.items()}
    return {arm: outcome(values, correct[OCR]) for arm, values in correct.items()}, correct


def rank_hits(scores, proxy):
    matches = scores.topk(DEPTHS[-1], dim=1).indices == proxy[:, None]
    return matches.cumsum(dim=1)[:, [depth - 1 for depth in DEPTHS]].gt(0).float().cpu().numpy()


def rank_summary(hits, trusted):
    return {
        arm: {str(depth): {"all": outcome(values[:, i], hits[OCR][:, i]),
                           "trusted": outcome(values[trusted, i], hits[OCR][trusted, i]) if trusted.any() else None}
              for i, depth in enumerate(DEPTHS)}
        for arm, values in hits.items()
    }


def neural_predictions(counts, fit, held, fitted, device):
    fit_counts = timing_view(counts[fit].to(device).unflatten(-1, (-1, 10))).mean(1)
    held_counts = timing_view(counts[held].to(device).unflatten(-1, (-1, 10))).mean(1)
    ridge, center, scale = fit_response_ridge(fit_counts)
    query = (held_counts - center) / scale
    return {arm: F.normalize(query @ (ridge @ design)) for arm, design in fitted.items()}


def score_bank(predictions, bank):
    scores = {arm: standardized(predicted @ bank[arm]) for arm, predicted in predictions.items()}
    scores[FUSION] = 0.5 * (scores[OCR] + scores[SPATIAL])
    return scores


def read_pilot_bank(path, images, held):
    if path.is_dir():
        root = path
        path = next(root.rglob(PILOT_BANK), None)
        if path is None:
            raise FileNotFoundError(f"The prepared pilot bank is missing under {root}")
    with np.load(path, allow_pickle=False) as stored:
        public = stored["public_images"]
        pairs = stored["pairs"]
        metadata = json.loads(stored["metadata"].item())
        glyphs = stored["calibration_indices"]
    if (not np.array_equal(glyphs, held.numpy()) or metadata["atlas_size"] != ATLAS_SIZE
            or metadata["training_raster_sha256"] != sha256(images.numpy().tobytes()).hexdigest()
            or public.shape != (len(held), 64, 64) or public.dtype != np.uint8
            or pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.size == 0 or pairs.dtype.kind not in "iu"
            or pairs.min() < 0 or pairs.max() >= len(held) or len(np.unique(pairs)) != pairs.size):
        raise ValueError("Prepared glyphs and disjoint pairs must match this pilot cohort.")
    return torch.from_numpy(public), torch.from_numpy(pairs).long(), metadata


@torch.inference_mode()
def main():
    args = arguments()
    prepare = args.mode == "prepare-pilot"
    if not prepare and not torch.cuda.is_available():
        raise RuntimeError("Use a Kaggle or Colab CUDA GPU for this experiment.")
    torch.set_num_threads(4)
    started = time.monotonic()
    cpu = torch.device("cpu")
    device = cpu if prepare else torch.device("cuda:0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"mode": args.mode, "seed": SEED, "device": str(device), "batch_size": args.batch_size,
              "torch_version": torch.__version__,
              "textverifier": {"revision": REVISION, "sha256": CHECKSUM, "precision": "float16 autocast"},
              "formal_queries_loaded": False}
    if not prepare:
        report["gpu"] = torch.cuda.get_device_name(device)
        publish(report, args.output, started, "checkpoint_started")
        vision, projector = textverifier(args.checkpoint, device)
        report["timm_version"] = version("timm")
        publish(report, args.output, started, "checkpoint_loaded")
    if args.mode == "throughput":
        generator = torch.Generator().manual_seed(SEED)
        images = torch.randint(2, (args.throughput_images, 64, 64), dtype=torch.uint8, generator=generator).mul_(255)
        verifier_features(images[:args.batch_size], vision, projector)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        encoded = time.monotonic()
        for batch in images.split(args.batch_size):
            verifier_features(batch, vision, projector)
        torch.cuda.synchronize()
        duration = time.monotonic() - encoded
        report.update(synthetic_glyphs=len(images), encoding_seconds=duration, glyphs_per_second=len(images) / duration,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        publish(report, args.output, started, "complete")
        return

    root = args.data_root.resolve(strict=True) if args.data_root else locate_data()
    images = load_images(root, cpu)
    if images.shape != (2748, 64, 64) or images.dtype != torch.uint8:
        raise ValueError("Expected all 2,748 labeled 64×64 uint8 training rasters.")
    fit, calibration, outer, overlap = cohorts(len(images))
    held = outer if args.mode == "full" else calibration
    rng = np.random.default_rng(SEED + 2)
    style = rng.choice(len(fit), size=128, replace=False)
    references = images[held]
    if args.pilot_bank is None:
        atlas, font, codepoints = build_font_atlas(images[fit[style]], device, return_codepoints=True)
        if len(atlas.images) != ATLAS_SIZE:
            raise ValueError("Expected all 27,550 ordered Noto Sans candidates.")
        proxy, trusted, agreement = visual_proxy(atlas, references.to(device))
        trusted_rows = trusted.cpu().numpy()
        atlas_info = {"font": font, "atlas_size": len(atlas.images), "atlas_device": str(device),
                      "training_raster_sha256": sha256(images.numpy().tobytes()).hexdigest(),
                      "codepoint_order_sha256": sha256(codepoints.cpu().contiguous().numpy().tobytes()).hexdigest(),
                      "trusted_proxy_glyphs": int(trusted.sum()), "mean_proxy_contour_agreement": agreement}
        if args.mode == "full":
            banks = {"public": atlas.images.cpu()}
        else:
            pairs = pair_index(references, proxy, trusted)
            banks = {"public": atlas.images[proxy].cpu(), "true": references}
    else:
        public, pairs, atlas_info = read_pilot_bank(args.pilot_bank, images, held)
        banks = {"public": public, "true": references}
    report.update(atlas_info)
    if args.mode != "full":
        report["paired_calibration_gallery_indices"] = held[pairs].tolist()
    if prepare:
        prepared = args.output.with_name(PILOT_BANK)
        np.savez_compressed(prepared, public_images=banks["public"].numpy(), pairs=pairs.numpy(),
                            calibration_indices=held.numpy(), metadata=json.dumps(atlas_info))
        report.update(pilot_bank=prepared.name, calibration_glyphs=len(held), glyph_pairs=len(pairs))
        publish(report, args.output, started, "complete")
        return
    if args.pilot_fit_glyphs:
        positions = np.sort(rng.choice(len(fit), size=args.pilot_fit_glyphs, replace=False))
        fit = fit[torch.from_numpy(positions)]
    report.update(mouse=args.mouse, fit_glyphs=len(fit), calibration_glyphs=calibration.tolist(),
                  outer_glyphs=outer.tolist(), outer_overlap_with_prior_ocr_cohorts=overlap,
                  pca_components=PC_COUNT, ridge_per_glyph=RIDGE,
                  repeat_conditions={"same_day": 4, "known_correspondence_pooled": 16},
                  protocol="Standalone feature readouts share a fit-only renderer, PCA dimension and ridge penalty. "
                           "The v20 ranker is not used. Pilot pairs have distinct trusted contour matches and use "
                           "a two-query assignment. Outer references are used only for proxy recall and optional OCF.",
                  ocr={"revision": OCR_REVISION, "sha256": OCR_CHECKSUM})
    publish(report, args.output, started, "features_started")
    fitted, coded = coordinates(images[fit], banks, lambda batch: verifier_features(batch, vision, projector), args.batch_size)
    del vision, projector
    ocr = feature_extractors()["frozen_hanzi"].to(device)
    ocr_fit, ocr_coded = coordinates(images[fit], banks, lambda batch: {OCR: ocr_features(batch, ocr, device)},
                                     args.batch_size)
    fitted.update(ocr_fit)
    for bank, values in coded.items():
        values.update(ocr_coded[bank])
    del ocr, banks
    publish(report, args.output, started, "features_ready")

    groups = grouped_blocks(root, load_catalog(root), args.mouse, cpu, len(images))
    if tuple(groups) != ("session1", "session2", "session3", "session4") or any(
        counts.shape[:2] != (len(images), 4) for counts in groups.values()
    ):
        raise ValueError("Expected four complete labeled recordings with four repeats per glyph.")
    report["recordings"] = {}
    totals, day_values, day_hits = {}, {}, {}
    if args.ocf_glyphs:
        sample = np.sort(rng.choice(len(held), args.ocf_glyphs, replace=False))
        sample_device = torch.as_tensor(sample, device=device)
        exact = PairedOCF(official_scorer(root), atlas.images, (references[sample] >= 128).numpy(), device, 12)
        report["ocf_gallery_indices"] = held[sample].tolist()

    def measure(scores):
        if args.mode == "pilot":
            return pair_summary(scores, pairs)
        hits = {arm: rank_hits(logits, proxy) for arm, logits in scores.items()}
        measurements = {"full_atlas_proxy_recall": rank_summary(hits, trusted_rows)}
        values = {}
        if args.ocf_glyphs:
            values = {arm: exact.evaluate(logits[sample_device].topk(4, dim=1).indices) for arm, logits in scores.items()}
            measurements["official_ocf"] = {arm: {"top1": outcome(rows[:, 0], values[OCR][:, 0]),
                                                  "top4_oracle": outcome(rows.max(1), values[OCR].max(1))}
                                           for arm, rows in values.items()}
        return measurements, (hits, values)

    for day, counts in groups.items():
        report["recordings"][day] = {}
        predictions = neural_predictions(counts, fit, held, fitted, device)
        for name, bank in coded.items():
            scores = score_bank(predictions, bank)
            report["recordings"][day][name], measured = measure(scores)
            if args.mode == "full":
                hits, measured = measured
                day_hits.setdefault(name, []).append(hits)
            day_values.setdefault(name, []).append(measured)
            if name not in totals:
                totals[name] = scores
            else:
                for arm, matrix in scores.items():
                    totals[name][arm].add_(matrix)
        publish(report, args.output, started, f"{day}_scored")
    report["pooled_16_known_correspondence"] = {name: measure(scores)[0] for name, scores in totals.items()}
    report["same_day_four_repeat"] = {}
    for name, observations in day_values.items():
        if args.mode == "pilot":
            pooled = {arm: np.mean([day[arm] for day in observations], axis=0) for arm in observations[0]}
            report["same_day_four_repeat"][name] = {arm: outcome(values, pooled[OCR]) for arm, values in pooled.items()}
            continue
        hits = {arm: np.mean([day[arm] for day in day_hits[name]], axis=0) for arm in day_hits[name][0]}
        report["same_day_four_repeat"][name] = {"full_atlas_proxy_recall": rank_summary(hits, trusted_rows)}
        if args.ocf_glyphs:
            measured = {arm: np.stack([day[arm] for day in observations]) for arm in observations[0]}
            report["same_day_four_repeat"][name]["official_ocf"] = {
                arm: {"top1": outcome(values[:, :, 0].mean(0), measured[OCR][:, :, 0].mean(0)),
                      "top4_oracle": outcome(values.max(2).mean(0), measured[OCR].max(2).mean(0))}
                for arm, values in measured.items()}
    if args.ocf_glyphs:
        report["unique_exact_pairs_scored"] = len(exact.values)
    publish(report, args.output, started, "complete")


if __name__ == "__main__":
    main()
