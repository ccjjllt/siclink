"""Compare native single-Hanzi features and contour features for Task 1 retrieval."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.request import urlopen

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v2

from glyph_bipartite_encoding import timing_view
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import build_font_atlas, image_descriptors, standard_error, standardized
from kaggle_task1_component_raster import picked_scope as component_scope
from kaggle_task1_context_prediction import fit_indices as context_scope
from kaggle_task1_microns_repeat_recheck import fresh_cohort as microns_scope


SEED = 158873
HELD_COUNT, DISTRACTORS, OCF_COUNT = 320, 1024, 80
PC_COUNT, RIDGE = 128, 0.12
OCR_REVISION = "99fe1300705f0d2446c057e041d99d3a2cff2419"
OCR_CHECKSUM = "151f32b03c49dbed77f3b8076035d28ce050d70a1a102e12581274a44cb10460"
OCR_SOURCE = f"https://huggingface.co/Ismantic/Handwritten/resolve/{OCR_REVISION}/best.pt"
STAGE = 13
OUTPUT = Path("/kaggle/working/task1_native_hanzi.json")


def split_glyphs(size, rng):
    prior = (component_scope(size)[1], context_scope(size)[1], microns_scope(size, torch.device("cpu"))[1])
    eligible = np.setdiff1d(np.arange(size), np.unique(np.concatenate(prior)))
    held = np.sort(rng.choice(eligible, HELD_COUNT, replace=False))
    return torch.from_numpy(np.setdiff1d(np.arange(size), held)), torch.from_numpy(held), size - len(eligible)


def normalized_glyph(image):
    # The released model crops pixels darker than 220 on a white background,
    # fits the long axis to 56px, and inverts to high foreground in the loader.
    rows, columns = np.nonzero(image > 35)
    canvas = np.zeros((64, 64), dtype=np.uint8)
    if len(rows):
        glyph = image[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
        height, width = glyph.shape
        scale = 56 / max(height, width)
        height, width = max(1, round(height * scale)), max(1, round(width * scale))
        pixels = Image.fromarray(255 - glyph).resize((width, height), Image.Resampling.BILINEAR)
        top, left = (64 - height) // 2, (64 - width) // 2
        canvas[top:top + height, left:left + width] = 255 - np.asarray(pixels)
    return canvas


def mobilenet():
    model = mobilenet_v2(weights=None, num_classes=3755)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    nn.init.kaiming_normal_(model.features[0][0].weight, mode="fan_out")
    model.features[18] = nn.Sequential(nn.Conv2d(320, 576, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(576), nn.ReLU6(inplace=True))
    model.classifier[1] = nn.Linear(576, 3755)
    return model.eval()


def feature_extractors():
    with urlopen(OCR_SOURCE, timeout=45) as response:
        payload = response.read()
    if sha256(payload).hexdigest() != OCR_CHECKSUM:
        raise ValueError("The OCR checkpoint does not match its published SHA-256.")
    checkpoint = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    if checkpoint["cfg"]["model"] != "mobilenet_v2":
        raise ValueError("Unexpected OCR checkpoint architecture.")
    torch.manual_seed(SEED)
    random = mobilenet()
    frozen = mobilenet()
    frozen.load_state_dict(checkpoint["model"], strict=True)
    return {"random_mobilenet": random.features[:STAGE + 1].to(memory_format=torch.channels_last),
            "frozen_hanzi": frozen.features[:STAGE + 1].to(memory_format=torch.channels_last)}


def descriptors(images, extractors):
    normalized = torch.from_numpy(np.stack([normalized_glyph(image) for image in images.numpy()]))
    features = {"contour": image_descriptors(normalized)}
    for name, model in extractors.items():
        spatial = [model(batch[:, None].float().div_(255).contiguous(memory_format=torch.channels_last))
                   for batch in normalized.split(64)]
        maps = torch.cat(spatial)
        if maps.shape[1:] != (96, 4, 4) or not bool(maps.isfinite().all()):
            raise RuntimeError(f"Unexpected spatial activations for {name}: {maps.shape}.")
        features[name] = maps.flatten(1)
    return features


def shared_bank(atlas, fit_images, held_images, rng):
    """Freeze one Noto bank before fitting or observing any held spike score."""
    signatures = image_descriptors((held_images >= 128).to(torch.uint8) * 255)
    similarity = signatures @ atlas.contours.T
    rows, matched = linear_sum_assignment(similarity.numpy(), maximize=True)
    if not np.array_equal(rows, np.arange(len(held_images))):
        raise RuntimeError("Every held glyph needs a distinct public match.")
    matched = torch.from_numpy(matched)
    pair_similarity = similarity[torch.arange(len(matched)), matched]
    best, best_indices = similarity.topk(2, dim=1)
    confident = (matched == best_indices[:, 0]) & (best[:, 0] - best[:, 1] >= 0.02)

    # Choose distractors round-robin among visual neighbors of fit glyphs.
    selected = rng.choice(len(fit_images), 512, replace=False)
    fit_signatures = image_descriptors((fit_images[selected] >= 128).to(torch.uint8) * 255)
    close = fit_signatures @ atlas.contours.T
    close[:, matched] = -torch.inf
    ordered = close.topk(32, dim=1).indices.T.flatten().tolist()
    extra = torch.tensor(list(dict.fromkeys(ordered))[:DISTRACTORS])
    if len(extra) != DISTRACTORS:
        raise RuntimeError("The controlled experiment requires 1,024 distinct distractors.")
    return torch.cat((matched, extra)), confident, float(pair_similarity.mean())


def fitted_pca(features, fit_count):
    train = features[:fit_count]
    center = train.mean(0)
    scale = (train - center).square().mean().sqrt().clamp_min(torch.finfo(train.dtype).tiny)
    normalized = (features - center) / scale
    torch.manual_seed(SEED + 1)
    _, _, basis = torch.pca_lowrank(normalized[:fit_count], q=PC_COUNT + 16, center=False, niter=3)
    coordinates = normalized @ basis[:, :PC_COUNT]
    deviation = coordinates[:fit_count].std(0).clamp_min(1e-4)
    return coordinates / deviation


def fit_response_ridge(observation):
    center, scale = observation.mean(0), observation.std(0).clamp_min(0.04)
    design = (observation - center) / scale
    gram = design.T @ design
    gram.diagonal().add_(RIDGE * len(observation))
    ridge_map = torch.cholesky_solve(design.T, torch.linalg.cholesky(gram))
    return ridge_map, center, scale


def response_views(groups, fit, held, rng):
    views = {}
    for day, counts in groups.items():
        if counts.shape[1] != 4:
            raise ValueError(f"Expected four labeled training repeats in {day}.")
        values = timing_view(counts.unflatten(-1, (-1, 10)))
        ridge_map, center, scale = fit_response_ridge(values[fit].mean(1))
        random_trial = torch.from_numpy(rng.integers(4, size=len(held)))
        means = (values[held, random_trial], values[held].mean(1))
        views[day] = (ridge_map, *((mean - center) / scale for mean in means))
    return views


def score_banks(coordinates, views, fit_count):
    train, exact, public = coordinates.split((fit_count, HELD_COUNT, HELD_COUNT + DISTRACTORS))
    banks = {"true_320": F.normalize(exact), "noto_320": F.normalize(public[:HELD_COUNT]),
             "noto_1344": F.normalize(public)}
    logits = {}
    for name, candidates in banks.items():
        single, one_per_day, four_per_day = {}, [], []
        for day, (ridge_map, one, four) in views.items():
            projection = ridge_map @ train
            one_per_day.append(standardized(F.normalize(one @ projection) @ candidates.T))
            day_score = standardized(F.normalize(four @ projection) @ candidates.T)
            single[day] = day_score
            four_per_day.append(day_score)
        logits[name] = {"single_day_4": single, "coindexed_4": {"pooled": torch.stack(one_per_day).sum(0)},
                        "coindexed_16": {"pooled": torch.stack(four_per_day).sum(0)}}
    return logits


def rank_metrics(logits, subset):
    positions = torch.arange(HELD_COUNT)
    rank = 1 + (logits > logits[positions, positions, None]).sum(1)
    measured = rank[subset]
    return np.array([float((measured <= depth).float().mean()) for depth in (1, 5, 32)])


def report_ranks(arms, confident):
    outcome = {}
    all_rows = torch.ones_like(confident)
    for bank, conditions in arms["contour"].items():
        outcome[bank] = {}
        for condition in conditions:
            result = {}
            for arm, banks in arms.items():
                logits = banks[bank][condition].values()
                all_recall = np.stack([rank_metrics(scores, all_rows) for scores in logits])
                stable_recall = np.stack([rank_metrics(scores, confident) for scores in logits])
                result[arm] = {"recall_top_1_5_32": all_recall.mean(0).tolist(),
                               "confident_proxy_recall_top_1_5_32": stable_recall.mean(0).tolist()}
            outcome[bank][condition] = result
    return outcome


def add_exact_ocf(outcome, arms, sample, day_allocation, endpoint, condition):
    values = {}
    day_order = torch.arange(len(sample))
    for arm, banks in arms.items():
        scores = banks["noto_1344"][condition].values()
        picked = torch.stack([logits.argmax(1)[sample] for logits in scores])
        selected = picked[day_allocation, day_order] if condition == "single_day_4" else picked[0]
        values[arm] = endpoint.evaluate(selected[:, None])[:, 0]
    for arm, output in values.items():
        difference = output - values["contour"]
        outcome[arm].update(ocf=float(output.mean()), paired_delta_vs_contour=float(difference.mean()),
                            paired_se_vs_contour=standard_error(difference), ocf_glyphs=len(output))
    contrast = values["frozen_hanzi"] - values["random_mobilenet"]
    outcome["frozen_vs_random_ocf"] = {"paired_delta": float(contrast.mean()), "paired_se": standard_error(contrast)}


def publish(result, started, stage):
    result.update(stage=stage, seconds=time.monotonic() - started)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{stage} after {result['seconds']:.1f}s", flush=True)


@torch.inference_mode()
def main():
    if not Path("/kaggle/input").is_dir() or not OUTPUT.parent.is_dir():
        raise RuntimeError("Run this experiment on Kaggle CPU with the official training input.")
    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED)
    started = time.monotonic()
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    fit, held, previous = split_glyphs(len(images), rng)
    fit_images, held_images = images[fit], images[held]
    extractors = feature_extractors()
    print(f"Verified OCR weights after {time.monotonic() - started:.1f}s", flush=True)
    atlas, font = build_font_atlas(fit_images[rng.choice(len(fit), 128, replace=False)], device)
    columns, confident, match_similarity = shared_bank(atlas, fit_images, held_images, rng)
    public_images = atlas.images[columns]
    del atlas
    print(f"Fixed Noto bank of {len(public_images)} after {time.monotonic() - started:.1f}s", flush=True)
    started_features = time.monotonic()
    visual = descriptors(torch.cat((fit_images, held_images, public_images)), extractors)
    feature_seconds = time.monotonic() - started_features
    print(f"Encoded three matched views in {feature_seconds:.1f}s", flush=True)
    groups = grouped_blocks(root, load_catalog(root), "mice1", device, len(images), bin_ms=25)
    views = response_views(groups, fit, held, rng)
    del groups
    arms = {name: score_banks(fitted_pca(feature, len(fit)), views, len(fit)) for name, feature in visual.items()}
    sample = torch.as_tensor(np.sort(rng.choice(len(held), OCF_COUNT, replace=False)))
    day_allocation = torch.as_tensor(rng.permutation(np.repeat(np.arange(len(views)), OCF_COUNT // len(views))))
    result = {"seed": SEED, "mouse": "mice1", "device": str(device), "fit_glyphs": len(fit), "held_glyphs": len(held),
              "excluded_previous_scoring_glyphs": previous, "ocf_glyphs": len(sample), "recordings": list(views),
              "ocr": {"checkpoint": OCR_SOURCE, "revision": OCR_REVISION, "sha256": OCR_CHECKSUM,
                      "license": "Apache-2.0", "features": "MobileNetV2 stage 13, 96x4x4",
                      "feature_input": "released 56px centered crop on 64px canvas, foreground high, [0,1]"},
              "font": font, "match_similarity": match_similarity, "confident_proxy_glyphs": int(confident.sum()),
              "pca_components": PC_COUNT, "ridge_per_glyph": RIDGE, "feature_seconds": feature_seconds,
              "exact_same_day_sample_sizes": dict(zip(views, torch.bincount(day_allocation).tolist(), strict=True)),
              "protocol": "Fit PCA and spike readouts on official fit glyphs only. A known held raster finds a matched "
                          "Noto proxy for the fixed diagnostic bank, and distractors are drawn from fit-image neighbors. "
                          "All arms output the same original Noto bank geometry. A separate closed bank uses exact "
                          "held-image features as diagnostic gallery entries. Known co-indexed trials are cross-day "
                          "high-repeat proxies. External OCR training codepoints can overlap independent SiClink glyphs. "
                          "Evaluation reads only official training spikes and images.",
              "results": report_ranks(arms, confident)}
    publish(result, started, "rank_metrics")
    metric = PairedOCF(official_scorer(root), public_images, (held_images[sample] >= 128).numpy(), device, 12)
    hard_results = result["results"]["noto_1344"]
    for condition in ("coindexed_16", "single_day_4"):
        add_exact_ocf(hard_results[condition], arms, sample, day_allocation, metric, condition)
        result["exact_ocf_pairs"] = len(metric.values)
        publish(result, started, condition)
    publish(result, started, "complete")
    print(json.dumps(hard_results), flush=True)


if __name__ == "__main__":
    main()
