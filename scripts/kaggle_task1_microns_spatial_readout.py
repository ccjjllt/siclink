"""Test unit-specific spatial sampling of a frozen cortex model on unseen SiClink glyphs."""

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_bipartite_encoding import repeat_noise
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits
from glyph_kernel_encoding import image_kernel
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_task1_bipartite_validation import evaluate_pairs, held_response_halves, mechanistic
from kaggle_task1_mice2_incumbent_crossfit import independent_split as mice2_prior_split
from kaggle_task1_microns_open_atlas import SEED as OPEN_SEED, outer_split as open_split
from kaggle_task1_microns_repeat_recheck import fresh_cohort as repeat_split
from kaggle_task1_microns_transfer import (BLACK_LEVEL, FRAMES, PUBLIC_ARCHIVE, STREAM, core_descriptors,
                                          load_released_core, normalized_views, template_rates)
from kaggle_v8_metric_validation import SESSIONS, summarize


SEED = 93131
MICE = ("mice1", "mice2")
HELD_GLYPHS = 320
OCF_GLYPHS = 128
FIELD_WIDTH = 0.28
RIDGE_PER_GLYPH = 0.12
GRID_SIZE = 8
OUTPUT = Path("/kaggle/working/task1_microns_spatial_readout.json")


def new_whole_glyphs(size, device):
    cpu = torch.device("cpu")
    old = {"mice1_open_atlas": open_split(size, OPEN_SEED, cpu)[1].numpy(),
           "mice1_repeat_recheck": repeat_split(size, cpu)[1].numpy(),
           "mice2_incumbent_crossfit": mice2_prior_split(size, 19391, cpu)[1].numpy()}
    excluded = np.unique(np.concatenate(tuple(old.values())))
    eligible = np.setdiff1d(np.arange(size), excluded)
    rng = np.random.default_rng(SEED)
    held = np.sort(rng.choice(eligible, HELD_GLYPHS, replace=False))
    fit = np.setdiff1d(np.arange(size), held)
    cohorts = {name: int(np.intersect1d(rows, held).size) for name, rows in old.items()}
    return (torch.as_tensor(fit, device=device), torch.as_tensor(held, device=device),
            {"excluded_previous_glyphs": len(excluded), "held_overlap": cohorts})


def residuals(value, ink):
    centered = value - value.mean(0)
    direction = F.normalize(ink - ink.mean(), dim=0)
    return centered - direction[:, None] * (direction @ centered)[None]


@torch.no_grad()
def spatial_fields(maps, fit, responses, ink, seed):
    """Use agreement across two independent training glyph sets to locate each unit's field."""
    rates = (responses[..., 2:].sum(-1) + 0.1).sqrt()
    rates = (rates - rates.mean(1, keepdim=True)) / rates.std(1, keepdim=True).clamp_min(0.05)
    target = rates.mean(0)
    rng = torch.Generator(device=maps.device).manual_seed(seed)
    pieces = torch.randperm(len(fit), device=maps.device, generator=rng).tensor_split(2)
    half_maps = []
    for rows in pieces:
        selected = fit[rows]
        pixel = residuals(maps[selected].flatten(1), ink[selected])
        neural = residuals(target[rows], ink[selected])
        association = F.normalize(neural, dim=0).T @ F.normalize(pixel, dim=0)
        half_maps.append(association.reshape(target.shape[1], *maps.shape[1:]))
    reproducible = (half_maps[0] * half_maps[1]).mean(dim=(1, 2))
    smooth = F.avg_pool2d(reproducible.reshape(-1, 1, GRID_SIZE, GRID_SIZE), 3, stride=1, padding=1,
                          count_include_pad=False).flatten(1)
    coordinate = torch.linspace(-1, 1, GRID_SIZE, device=maps.device)
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    locations = torch.stack((y, x), dim=-1).reshape(-1, 2)
    strongest = smooth.argmax(1)
    squared_distance = (locations[None] - locations[strongest, None]).square().sum(-1)
    localized = (-squared_distance / (2 * FIELD_WIDTH ** 2)).softmax(dim=1)
    unit_means = responses.sum(-1).mean((0, 1))
    donors = torch.empty_like(strongest)
    for group in unit_means.argsort().tensor_split(4):
        shuffled_order = group[torch.randperm(len(group), device=maps.device, generator=rng)]
        donors[shuffled_order] = shuffled_order.roll(1)
    pools = {"global_average": torch.full_like(localized, 1 / localized.shape[1]),
             "unit_field": localized, "rate_stratified_shuffled_field": localized[donors]}
    diagnostics = {"units": len(strongest), "occupied_spatial_cells": int(strongest.unique().numel()),
                   "positive_repeatable_field_fraction": float((smooth.amax(1) > 0).float().mean()),
                   "donors_with_different_locations": float((strongest != strongest[donors]).float().mean())}
    return pools, diagnostics


@dataclass(frozen=True)
class SpatialReadout:
    masks: torch.Tensor
    weights: torch.Tensor
    intercept: torch.Tensor
    sessions: int
    bins: int

    @torch.no_grad()
    def rates(self, maps):
        units = []
        for part in torch.arange(len(self.masks), device=maps.device).split(16):
            pooled = torch.einsum("gfcp,up->ugfc", maps, self.masks[part]).flatten(2)
            predicted = pooled @ self.weights[part] + self.intercept[part]
            units.append((predicted.clamp_min(0).square() - 0.1).clamp_min_(1e-5))
        return torch.cat(units).unflatten(-1, (self.sessions, self.bins)).permute(2, 1, 0, 3).flatten(2)


@torch.no_grad()
def fit_spatial_readout(fit_maps, masks, responses):
    weights, intercepts = [], []
    values = (responses + 0.1).sqrt().permute(2, 1, 0, 3).flatten(2)
    for unit in torch.arange(len(masks), device=fit_maps.device).split(16):
        fitted = torch.einsum("gfcp,up->ugfc", fit_maps, masks[unit]).flatten(2)
        center = fitted.mean(1, keepdim=True)
        scale = fitted.std(1, keepdim=True).clamp_min(1e-4)
        fitted = (fitted - center) / scale
        mean = values[unit].mean(1, keepdim=True)
        gram = fitted.transpose(1, 2) @ fitted
        gram.diagonal(dim1=1, dim2=2).add_(RIDGE_PER_GLYPH * len(fit_maps))
        right = fitted.transpose(1, 2) @ (values[unit] - mean)
        # Notebook CPU avoids the T4 MAGMA batched solve error for these small systems.
        normalized = torch.cholesky_solve(right.cpu(), torch.linalg.cholesky(gram.cpu())).to(fit_maps.device)
        weights.append(normalized / scale.transpose(1, 2))
        intercepts.append(mean - center @ weights[-1])
    return SpatialReadout(masks, torch.cat(weights), torch.cat(intercepts), responses.shape[0], responses.shape[-1])


def measure_rates(arms, queries, images, near_pairs, halves, noise, scorer, sample):
    exact = PairedOCF(scorer, images, (images[sample] >= 128).cpu().numpy(), images.device, 12)
    actual, treatments = {}, {}
    labels = torch.arange(len(images), device=images.device)[:, None]
    for name, rates in arms.items():
        near, _ = mechanistic(rates, halves, noise, near_pairs)
        details, results = {}, []
        for day_index, (day, query) in enumerate(queries.items()):
            score = count_template_logits(query, rates[day_index], "timed")
            shortlist = score.topk(16, dim=1).indices
            correct = shortlist == labels
            measured = exact.evaluate(shortlist[sample, :1])[:, 0]
            results.append(measured)
            details[day] = {f"top_{width}": float(correct[:, :width].any(1).float().mean()) for width in (1, 5, 16)}
        actual[name] = results
        treatments[name] = {"gallery_identity": details, "near_shape": near}
    for name, result in treatments.items():
        result["paired_exact_ocf_vs_published_kernel"] = summarize(actual[name], actual["published_kernel"])
    for control in ("global_average", "rate_stratified_shuffled_field"):
        treatments["unit_field"]["paired_exact_ocf_vs_" + control] = summarize(actual["unit_field"], actual[control])
    return treatments


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use a private Kaggle T4 with competition labels and access to the public MICrONS checkpoint.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, held, cohorts = new_whole_glyphs(len(images), device)
    catalog = load_catalog(root)
    scorer = official_scorer(root)
    cores, digest = load_released_core()
    views = core_descriptors(cores["published_core"], images)
    del cores
    channels = views[0].shape[1] // GRID_SIZE ** 2
    maps = torch.stack(views, dim=1).unflatten(-1, (channels, GRID_SIZE ** 2))
    features = normalized_views(views, fit)
    gram = image_kernel(features[fit], features[fit])
    cross_gram = image_kernel(features[held], features[fit])
    del features, views
    ink = images.float().mean((1, 2)) / 255
    near_pairs, pair_summary = evaluate_pairs(images[held], SEED + 1)
    sample = torch.as_tensor(np.random.default_rng(SEED + 2).choice(len(held), OCF_GLYPHS, replace=False), device=device)
    report = {"fit_glyphs": len(fit), "held_glyphs": len(held), "ocf_glyphs_per_recording": len(sample),
              "cohorts": cohorts, "spatial_field_width": FIELD_WIDTH, "relative_local_ridge": RIDGE_PER_GLYPH,
              "microns": {"source": PUBLIC_ARCHIVE, "checkpoint_sha256": digest, "stream": STREAM,
                          "frames_at_30_hz": FRAMES, "black_level": BLACK_LEVEL}, "near_pairs": pair_summary,
              "protocol": "One shared frozen image core. Estimate each unit's location from the signed repeatability "
                          "of image-channel associations across two disjoint sets of fitting glyphs after removing ink. "
                          "Matched ridge readouts use real unit-specific, rate-stratified deranged or global spatial pooling. "
                          "A full-image published-core kernel is the prior closed-gallery reference. Score true held glyphs "
                          "with a predeclared known gallery, four observed repeats, and exact OCF. No formal query data.",
              "mouse": {}}
    for mouse_index, name in enumerate(MICE):
        blocks = grouped_blocks(root, catalog, name, device, len(images))
        if tuple(blocks) != SESSIONS or any(trajectories.shape[1] != 4 for trajectories in blocks.values()):
            raise ValueError("Every mouse must have four labeled sessions and four repeats per fitting glyph.")
        training = {day: value[fit] for day, value in blocks.items()}
        queries = {day: value[held] for day, value in blocks.items()}
        del blocks
        response = torch.stack([value.mean(1) for value in training.values()]).unflatten(-1, (-1, 10))
        stabilized = (response + 0.1).sqrt().transpose(0, 1).flatten(1)
        kernel = template_rates(gram, cross_gram, stabilized).reshape(len(held), len(SESSIONS), -1).transpose(0, 1)
        pools, fields = spatial_fields(maps, fit, response, ink, SEED + 10 + mouse_index)
        arms = {"published_kernel": kernel}
        for arm, masks in pools.items():
            arms[arm] = fit_spatial_readout(maps[fit], masks, response).rates(maps[held])
        halves = held_response_halves(queries, SEED + 20 + mouse_index)
        measured = measure_rates(arms, queries, images[held], near_pairs, halves, repeat_noise(training), scorer, sample)
        report["mouse"][name] = {"fields": fields, "arms": measured}
        report.update(stage=name, seconds=time.monotonic() - started)
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"stage": name, "seconds": report["seconds"]}), flush=True)
    report.update(stage="complete", seconds=time.monotonic() - started)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
