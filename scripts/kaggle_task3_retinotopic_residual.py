"""Learn layout and OCR detail from real, independent single-trial spikes."""

import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_mbr import contour_signatures, glyph_posterior
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import standardized
from kaggle_posterior_utility_validation import fit_neural, v11_logits
from kaggle_task1_native_hanzi import OCR_CHECKSUM, OCR_REVISION, OCR_SOURCE, feature_extractors
from kaggle_task3_count_occupancy import paired_delta, posterior_metrics
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_mice2_full_balance_replication import cohorts
from kaggle_task3_mice2_utility_candidate import MOUSE, SHORTLIST, TABLE_SHA, validated_utility
from kaggle_task3_rankband_ocf_validation import NeuralRows
from kaggle_task3_relational_consensus import UTILITY_WEIGHT, choose_images
from kaggle_task3_v20_candidate_headroom import DAYS, GLYPHS_PER_DAY, held_repeats
from kaggle_task3_v20_exact_utility import held_panels, original_panel, paired_summary


SEED = 363313
PATCHES, CONTOURS, DETAIL = 4, 5, 8
EPOCHS, BATCH, WIDTH = 7, 256, 24
NEIGHBORS, CALIBRATION_GLYPHS = 16, 128
SHAPE_WEIGHT, DETAIL_WEIGHT, HARD_WEIGHT = 0.30, 0.5, 0.2
V20, COARSE = "v20", "retinotopic_coarse_only"
DIRECT, RESIDUAL, SHUFFLED = "unconditional_ocr", "conditional_ocr_residual", "shuffled_residual"
OUTPUT = Path("/kaggle/working/task3_retinotopic_residual.json")


class PatchReadout(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(channels, WIDTH, 1), nn.GroupNorm(4, WIDTH), nn.SiLU())
        self.layout = nn.Sequential(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.SiLU(), nn.Conv2d(WIDTH, CONTOURS, 1))
        self.detail = nn.Sequential(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.SiLU(), nn.Conv2d(WIDTH, DETAIL, 1))

    def forward(self, values):
        spatial = self.stem(values)
        return self.layout(spatial).permute(0, 2, 3, 1), self.detail(spatial).permute(0, 2, 3, 1)


def fit_standardize(values, fit, floor):
    center = values[fit].mean(0)
    scale = values[fit].std(0, unbiased=False).clamp_min(floor)
    return (values - center) / scale


@torch.no_grad()
def image_targets(images, fit):
    # Keep displayed geometry fixed for both the visual features and receptive fields.
    pixels = images[:, None].float() / 255
    dx = F.pad(pixels.diff(dim=-1), (0, 1, 0, 0))
    dy = F.pad(pixels.diff(dim=-2), (0, 0, 0, 1))
    lines = torch.cat((pixels, dx.clamp_min(0), (-dx).clamp_min(0), dy.clamp_min(0), (-dy).clamp_min(0)), 1)
    coarse = fit_standardize(F.avg_pool2d(lines, 64 // PATCHES).permute(0, 2, 3, 1), fit, 0.02)

    ocr = feature_extractors()["frozen_hanzi"]
    maps = torch.cat([ocr(batch.contiguous(memory_format=torch.channels_last)).permute(0, 2, 3, 1)
                      for batch in pixels.split(64)])
    if maps.shape != (len(images), PATCHES, PATCHES, 96) or not torch.isfinite(maps).all():
        raise ValueError("Expected finite OCR features at sixteen spatial positions.")
    maps -= maps[fit].mean(0)
    layout = coarse.flatten(1)
    predictors = layout[fit]
    gram = predictors.T @ predictors
    gram.diagonal().add_(0.05 * len(fit))
    mapping = torch.linalg.solve(gram, predictors.T @ maps[fit].flatten(1))
    residual = maps - (layout @ mapping).reshape_as(maps)
    samples = residual[fit].flatten(0, 2)
    _, axes = torch.linalg.eigh(samples.T @ samples / len(samples))
    axes = axes[:, -DETAIL:]
    unconditional = fit_standardize(maps @ axes, fit, 1e-4)
    conditional = fit_standardize(residual @ axes, fit, 1e-4)
    neighbors = torch.cdist(layout, layout).fill_diagonal_(torch.inf).topk(NEIGHBORS, largest=False).indices
    fraction_remaining = float(residual[fit].square().sum() / maps[fit].square().sum())
    return coarse, unconditional, conditional, neighbors, fraction_remaining


def within_shape_permutation(coarse, fit):
    """Rotate target labels within disjoint local groups of nearby layout codes."""
    rng = torch.Generator().manual_seed(SEED + 3)
    remaining = torch.randperm(len(fit), generator=rng)
    layout = coarse[fit].flatten(1)
    distances = torch.cdist(layout, layout)
    permutation = torch.arange(len(coarse))
    while len(remaining):
        width = min(24, len(remaining))
        if len(remaining) == width + 1:
            width += 1
        first = remaining[0]
        positions = distances[first, remaining].topk(width, largest=False).indices
        members = remaining[positions]
        order = members[torch.randperm(width, generator=rng)]
        permutation[fit[order]] = fit[order.roll(1)]
        keep = torch.ones(len(remaining), dtype=torch.bool)
        keep[positions] = False
        remaining = remaining[keep]
    if (permutation[fit] == fit).any() or not torch.equal(permutation.sort().values, torch.arange(len(coarse))):
        raise RuntimeError("The matched shape control must permute every fitted label exactly once.")
    return permutation


def delayed_counts(counts):
    return counts.unflatten(-1, (-1, 10))[..., 2:].sqrt()


@torch.no_grad()
def project_recording(support, query, coarse, fit, calibrate):
    fit_spikes = delayed_counts(support[fit])
    rows = fit_spikes.flatten(0, 1)
    center, scale = rows.mean(0), (rows.var(0, unbiased=False) + 0.05).sqrt()
    fit_spikes.sub_(center).div_(scale).clamp_(-8, 8)
    responses = fit_spikes.mean((1, 3))
    del fit_spikes, rows
    design = coarse[fit].flatten(1)
    gram = design.T @ design
    gram.diagonal().add_(0.25 * len(fit))
    fields = torch.linalg.solve(gram, design.T @ responses)
    weights = F.normalize(torch.cat((fields.clamp_min(0), (-fields).clamp_min(0)), 0), dim=1).T.contiguous()

    def project(counts):
        groups = []
        for batch in counts.split(512):
            delayed = ((delayed_counts(batch) - center) / scale).clamp_(-8, 8)
            pooled = delayed.transpose(1, 2) @ weights
            pooled = pooled.reshape(len(batch), 8, 2, PATCHES, PATCHES, CONTOURS)
            groups.append(pooled.permute(0, 1, 2, 5, 3, 4).flatten(1, 3))
        return torch.cat(groups)

    train = project(support[fit].flatten(0, 1))
    validation = project(support[calibrate].flatten(0, 1))
    held = project(query[:, 0])
    center = train.mean(0)
    scale = train.std(0, unbiased=False).clamp_min(0.1)
    return tuple(((part - center) / scale).clamp_(-8, 8) for part in (train, validation, held))


def fit_readout(spikes, labels, coarse, detail, hard, name):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED + 1)
    model = PatchReadout(spikes.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.02)
    for epoch in range(EPOCHS):
        total = 0.0
        for rows in torch.randperm(len(spikes), generator=rng).split(BATCH):
            glyph = labels[rows]
            negative = hard[glyph, torch.randint(NEIGHBORS, (len(rows),), generator=rng)]
            layout, parts = model(spikes[rows])
            layout_loss = F.mse_loss(layout, coarse[glyph])
            true_distance = (parts - detail[glyph]).square().flatten(1).mean(1)
            wrong_distance = (parts - detail[negative]).square().flatten(1).mean(1)
            loss = layout_loss + true_distance.mean() + HARD_WEIGHT * F.softplus(0.10 + true_distance - wrong_distance).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        print(f"{name} epoch {epoch + 1}/{EPOCHS}: {total / len(spikes):.4f}", flush=True)
    return model.eval()


@torch.no_grad()
def predict_patches(model, feature):
    layout, detail = zip(*(model(batch) for batch in feature.split(BATCH)), strict=True)
    return torch.cat(layout), torch.cat(detail)


def distance_logits(predicted, gallery, scale):
    weighted = (gallery * scale).flatten(1)
    score = 2 * (predicted * scale).flatten(1) @ weighted.T - weighted.square().sum(1)
    return standardized(score)


@torch.no_grad()
def measure_readout(model, inputs, calib, coarse, detail):
    validation_x, validation_glyph = calib
    calibration_prediction = predict_patches(model, validation_x)
    validation = []
    weight = []
    for predicted, gallery in zip(calibration_prediction, (coarse, detail), strict=True):
        actual = gallery[validation_glyph]
        squared = (predicted - actual).square()
        site_error = squared.mean((0, 3), keepdim=True)
        accuracy = 1 - float(squared.mean() / actual.square().mean().clamp_min(1e-7))
        reliability = (0.25 + site_error).reciprocal()
        scale = (reliability / reliability.mean()).clamp(0.5, 2).sqrt()
        validation.append({"r2_against_zero": accuracy, "site_weights": scale.reshape(-1).tolist()})
        weight.append(scale)
    result = {}
    for day, feature in inputs.items():
        predictions = predict_patches(model, feature)
        result[day] = tuple(distance_logits(predicted, target, scale)
                            for predicted, target, scale in zip(predictions, (coarse, detail), weight, strict=True))
    return result, validation


def ranks(scores, truth, selection, neighbors):
    same = scores.gather(1, truth[:, None])
    near = scores.gather(1, neighbors[truth])
    hard_difference = (same[selection] - near[selection])
    top = scores[selection].topk(32).indices == truth[selection, None]
    return {**{f"top{k}": float(top[:, :k].any(1).float().mean()) for k in (1, 7, 16, 32)},
            "near_shape_top1": float((hard_difference >= 0).all(1).float().mean()),
            "near_shape_pair_accuracy": float((hard_difference > 0).float().mean()),
            "near_shape_nll": float(F.softplus(torch.logsumexp(-hard_difference, dim=1)).mean())}


def publish(report, started, stage):
    report.update(stage=stage, seconds=time.monotonic() - started)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{stage} after {report['seconds']:.1f}s", flush=True)


def main():
    torch.set_num_threads(4)
    started = time.monotonic()
    cpu = torch.device("cpu")
    root = locate_data()
    images = load_images(root, cpu)
    utility, temperatures, _ = validated_utility(images, cpu)
    visual_fit, held_glyphs, temperature_fit, _ = cohorts(len(images), cpu)
    references, orders = held_panels(held_glyphs, torch.cat((visual_fit, temperature_fit)), GLYPHS_PER_DAY)
    all_glyphs = torch.arange(len(images))
    eligible = all_glyphs[~torch.isin(all_glyphs, held_glyphs)]
    rng = torch.Generator().manual_seed(SEED + 2)
    calibrate, fit = eligible[torch.randperm(len(eligible), generator=rng)].split((CALIBRATION_GLYPHS,
                                                                                len(eligible) - CALIBRATION_GLYPHS))
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, cpu, len(images), bin_ms=25)
    support, query, overlap = held_repeats(recordings, held_glyphs)
    del recordings
    report = {"mouse": MOUSE, "seed": SEED, "gallery": len(images), "readout_fit_glyphs": len(fit),
              "calibration_glyphs": len(calibrate), "shape_held_glyphs": len(held_glyphs), "training_days": list(support),
              "scored_days": list(DAYS), "paired_glyphs_per_day": GLYPHS_PER_DAY, "trial_overlap": overlap,
              "architecture": {"patches_per_side": PATCHES, "bin_ms": 25, "range_ms": [50, 250], "width": WIDTH,
                               "epochs": EPOCHS, "batch_size": BATCH, "ocr_dimensions_per_patch": DETAIL},
              "ocr": {"source": OCR_SOURCE, "revision": OCR_REVISION, "sha256": OCR_CHECKSUM},
              "posterior_shape_weight": SHAPE_WEIGHT, "equalized_detail_weight": DETAIL_WEIGHT,
              "neighbor_count": NEIGHBORS, "candidate_budget": SHORTLIST, "fixed_utility_weight": UTILITY_WEIGHT,
              "utility_sha256": TABLE_SHA, "protocol": "Signed, directional receptive weights are learned per day "
              "from shape-fit spike counts and glyphs. A shared spatial CNN learns from measured single repetitions. "
              "Its architecture, initialization, losses, negatives and budget are matched across raw OCR, coarse-"
              "conditioned OCR residual, and residual targets shuffled within local coarse groups. Held scores see "
              "true gallery images and independent spikes, using true unshuffled residuals for both residual arms. "
              "An independent labeled calibration glyph subset determines per-patch reliability. All original v20 "
              "candidate actions, anchor logits, prior temperatures and utility are fixed; only visual posterior "
              "evidence changes. Exact endpoints reuse the predeclared headroom panel. No formal test input is opened.",
              "readouts": {}, "sessions": {}}
    publish(report, started, "split_ready")

    incumbent = fit_neural(support, images, *contour_signatures(images), MOUSE, 211)
    base = {day: v11_logits(incumbent, query[day][orders[day]], day, images) for day in DAYS}
    del incumbent
    publish(report, started, "v20_ready")
    coarse, direct, residual, neighbors, remaining_variance = image_targets(images, fit)
    report["ocr"]["residual_variance_fraction"] = remaining_variance
    permutation = within_shape_permutation(coarse, fit)
    train, validation, held = {}, {}, {}
    for day, counts in support.items():
        train[day], validation[day], held[day] = project_recording(counts, query[day], coarse, fit, calibrate)
    repeats = next(iter(support.values())).shape[1]
    del support, query
    train_x = torch.cat(tuple(train.values()))
    calibration_x = torch.cat(tuple(validation.values()))
    train_label = fit.repeat_interleave(repeats).repeat(len(train))
    calibration_label = calibrate.repeat_interleave(repeats).repeat(len(train))
    del train, validation
    report["trained_single_trials"] = len(train_x)
    report["calibration_single_trials"] = len(calibration_x)
    publish(report, started, "spatial_design_ready")

    # Hard contrasts are matched to fit glyphs and exclude all score/calibration glyphs.
    fitting_layout = coarse[fit].flatten(1)
    near_fit = torch.cdist(fitting_layout, fitting_layout).fill_diagonal_(torch.inf)
    hard = torch.empty((len(images), NEIGHBORS), dtype=torch.long)
    hard[fit] = fit[near_fit.topk(NEIGHBORS, largest=False).indices]
    del near_fit
    inputs = {day: held[day][orders[day]] for day in DAYS}
    results = {day: {} for day in DAYS}
    for name, train_target, gallery_target in ((DIRECT, direct, direct), (RESIDUAL, residual, residual),
                                               (SHUFFLED, residual[permutation], residual)):
        model = fit_readout(train_x, train_label, coarse, train_target, hard, name)
        by_day, calibration_stats = measure_readout(model, inputs, (calibration_x, calibration_label),
                                                     coarse, gallery_target)
        report["readouts"][name] = {"parameters": sum(value.numel() for value in model.parameters()),
                                     "calibration": calibration_stats}
        for day, (layout, detail) in by_day.items():
            results[day][name] = standardized((1 - DETAIL_WEIGHT) * layout + DETAIL_WEIGHT * detail)
            if name == RESIDUAL:
                results[day][COARSE] = layout
        publish(report, started, f"{name}_ready")

    scorer = official_scorer(root)
    endpoint_values = {}
    for day in DAYS:
        order, truth, logits = orders[day], references[day], base[day]
        rows = order.argsort()[truth]
        held_mask = torch.isin(order, held_glyphs)
        temperature = temperatures[day]["temperature"]
        actions, original_mass, original_choice = original_panel(logits[rows], temperature, utility)
        posteriors = {V20: original_mass}
        metrics = {V20: {"held_glyph_ranks": ranks(logits, order, held_mask, neighbors),
                         "full_gallery": posterior_metrics(glyph_posterior(logits, temperature), order)[0]}}
        for name, standalone in results[day].items():
            fused = logits + SHAPE_WEIGHT * standalone
            mass = glyph_posterior(fused, temperature)
            posteriors[name] = mass[rows]
            metrics[name] = {
                "held_glyph_ranks": ranks(fused, order, held_mask, neighbors),
                "standalone_held_glyph_ranks": ranks(standalone, order, held_mask, neighbors),
                "full_gallery": posterior_metrics(mass, order)[0],
                "held_glyph_posterior": posterior_metrics(mass[held_mask], order[held_mask])[0],
            }
        choices = choose_images(NeuralRows(logits[rows], logits[rows], actions), posteriors, utility)
        if not torch.equal(original_choice, choices[V20]):
            raise RuntimeError("The v20 baseline must reproduce the exact seven-action decision.")
        selected = torch.stack([choices[name] for name in posteriors])
        pairs, unique = unique_pair_ocf(selected, truth.expand_as(selected), images >= 128, scorer, cpu, 96)
        values = dict(zip(posteriors, pairs, strict=True))
        endpoint_values[day] = values
        report["sessions"][day] = {"scored_glyph_indices": truth.tolist(), "ranks": metrics, "changed_from_v20":
                                   {name: int((choices[name] != original_choice).sum()) for name in posteriors},
                                   "ocf": paired_summary(values), "unique_exact_pairs": unique}
        publish(report, started, f"{day}_exact_ocf")
        print(json.dumps(report["sessions"][day]["ocf"]), flush=True)
    combined = {arm: torch.cat([values[arm] for values in endpoint_values.values()])
                for arm in next(iter(endpoint_values.values()))}
    report["pooled_ocf"] = paired_summary(combined)
    report["residual_ocf_vs_control"] = {arm: paired_delta(combined[RESIDUAL], combined[arm])
                                         for arm in (COARSE, DIRECT, SHUFFLED)}
    publish(report, started, "complete")
    print(json.dumps(report["pooled_ocf"]), flush=True)


if __name__ == "__main__":
    main()
