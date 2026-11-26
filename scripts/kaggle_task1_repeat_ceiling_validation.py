"""Measure four-, eight-, and sixteen-repeat Task 1 ceilings on unseen training glyphs."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_multiview_alignment import fit_multiview
from kaggle_diagnostics import official_scorer
from kaggle_task3_metric_validation import unique_pair_ocf


VIEWS = {"evoked_total": "rate", "timed": "timed"}
REPEATS_PER_DAY = (1, 2, 4)
DEPTHS = (1, 4, 16, 32)


def split_cohorts(count, seeds):
    # Reproduce the first repeat-scaling split and the independent retinal-shift fold.
    old_repeat = np.random.default_rng(211).permutation(count)[:320]
    old_shift = np.random.default_rng(2651).permutation(count)[-320:]
    previous = np.union1d(old_repeat, old_shift)
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(count)
        held = order[~np.isin(order, previous)][:320]
        if len(held) != 320:
            raise ValueError("The independent cohorts need at least 320 fresh labeled glyphs each.")
        available = order[~np.isin(order, held)]
        calibration = available[:192]
        fitting = available[192:]
        yield fitting, calibration, held
        previous = np.union1d(previous, held)


@torch.no_grad()
def per_trial_logits(counts, rates, view):
    glyphs, repeats, features = counts.shape
    return count_template_logits(counts.reshape(-1, 1, features), rates, view).reshape(glyphs, repeats, len(rates))


@torch.no_grad()
def fit_inverse_temperature(logits):
    """Calibrate score magnitudes independently for each day and count view."""
    total = logits.sum(dim=1)
    centered = total - total.mean(dim=1, keepdim=True)
    labels = centered.diagonal()
    levels = torch.logspace(-2, 1, 37, device=logits.device)
    errors = torch.stack([(torch.logsumexp(level * centered, dim=1) - level * labels).mean() for level in levels])
    best = int(errors.argmin())
    return levels[best], {"inverse_temperature": float(levels[best]), "nll_4_repeats": float(errors[best]),
                          "at_grid_boundary": best in (0, len(levels) - 1)}


def held_queries(groups, held, generator):
    shuffled = {}
    for session, counts in groups.items():
        values = counts[held]
        if values.shape[1] != 4:
            raise ValueError("Expected exactly four labeled repeats in each training recording.")
        order = torch.randperm(len(values), device=held.device, generator=generator)
        repeat_order = torch.rand(values.shape[:2], device=held.device, generator=generator).argsort(dim=1)
        values = values.gather(1, repeat_order[:, :, None].expand_as(values))
        shuffled[session] = values[order], order.argsort()
    return shuffled


@torch.no_grad()
def forward_day_scores(encoder, training_groups, calibration, queries, images):
    daily, details = {name: {} for name in VIEWS}, {}
    calibration_images, gallery_images = images
    for session, (values, inverse) in queries.items():
        rates_calibration = encoder.expected_counts(calibration_images, session)
        rates_gallery = encoder.expected_counts(gallery_images, session)
        details[session] = {}
        for name, view in VIEWS.items():
            cal = per_trial_logits(training_groups[session][calibration], rates_calibration, view)
            scale, details[session][name] = fit_inverse_temperature(cal)
            ranked = per_trial_logits(values, rates_gallery, view)
            # Co-indexing across days is used only in the labeled assay.
            daily[name][session] = (scale * (ranked - ranked.mean(dim=-1, keepdim=True)))[inverse]
    return daily, details


def repeat_curves(daily):
    sessions = tuple(daily)
    balanced = {4 * count: [sum(daily[name][:, :count].sum(dim=1) for name in sessions)] for count in REPEATS_PER_DAY}
    whole_day = {4 * days: [sum(daily[name].sum(dim=1) for name in picked)
                             for picked in combinations(sessions, days)] for days in (1, 2, 4)}
    return {**{f"balanced_{repeats}": values for repeats, values in balanced.items()},
            **{f"whole_day_{repeats}": values for repeats, values in whole_day.items() if repeats < 16}}


def per_glyph_identity(matrix):
    actual = matrix.diagonal()[:, None]
    better = (matrix > actual).sum(dim=1)
    tied = (matrix == actual).sum(dim=1).clamp_min(1)
    midpoint = better + 0.5 * (1 + tied)
    hits = [(limit - better).div(tied).clamp(0, 1) for limit in (1, 4, 8, 16, 32)]
    return torch.stack([*hits, midpoint, midpoint.reciprocal()], dim=-1)


def summarize_identity(method, anchor):
    data = torch.stack([per_glyph_identity(candidate) for candidate in method]).mean(dim=0)
    delta = data[:, 0] - anchor[:, 0]
    names = ("top1", "top4", "top8", "top16", "top32", "mean_rank", "mean_reciprocal_rank")
    return {**dict(zip(names, data.mean(dim=0).tolist())), "paired_delta_top1": float(delta.mean()),
            "paired_glyph_top1_se": float(delta.std() / len(data)**0.5), "day_combinations": len(method),
            "gallery_queries": len(data)}


@torch.no_grad()
def neural_correspondence(mouse, training, queries):
    output = {}
    for view, signal in (("evoked_total", "evoked-total"), ("timed", "early-late")):
        alignment = fit_multiview({(mouse, name): values for name, values in training.items()}, view=signal, dimension=64)
        reference = F.normalize(torch.stack([alignment.encode((mouse, name), rows[:, 2:])[inverse]
                                            for name, (rows, inverse) in queries.items()]).mean(dim=0))
        output[view] = {}
        for n in (1, 2):
            means = [alignment.encode((mouse, name), values[:, :n])[inverse]
                     for name, (values, inverse) in queries.items()]
            output[view][4 * n] = F.normalize(torch.stack(means).mean(dim=0)) @ reference.T
    return output


def paired_ocf_stats(values, baseline):
    delta = values - baseline
    return {"ocf": float(values.mean()), "paired_delta": float(delta.mean()),
            "paired_glyph_se": float(delta.std(ddof=1) / np.sqrt(len(delta))), "paired_glyphs": len(values)}


def exact_curves(curves, selected, masks, scorer):
    predictions, keys = [], []
    for view, repetition in curves.items():
        for name, matrices in repetition.items():
            predictions.extend(matrix[selected].argmax(1) for matrix in matrices)
            keys.extend((view, name) for _ in matrices)
    predictions = torch.stack(predictions)
    scores, unique = unique_pair_ocf(predictions, selected.expand_as(predictions), masks, scorer, masks.device, 384)
    by_method = {}
    for key, row in zip(keys, scores.cpu().numpy()):
        by_method.setdefault(key, []).append(row)
    values = {key: np.mean(rows, axis=0) for key, rows in by_method.items()}
    return {view: {name: paired_ocf_stats(values[view, name], values[view, "balanced_4"]) for name in repetition}
            for view, repetition in curves.items()}, unique


def oracle_curves(curves, selected, masks, scorer):
    keys = [(view, f"balanced_{count}") for view in curves for count in (4, 8, 16)]
    predictions = torch.stack([curves[view][count][0][selected].topk(max(DEPTHS), dim=1).indices for view, count in keys])
    references = selected[None, :, None].expand_as(predictions)
    values, unique = unique_pair_ocf(predictions, references, masks, scorer, masks.device, 384)
    identity, _ = unique_pair_ocf(selected, selected, masks, scorer, masks.device, 384)
    by_method = dict(zip(keys, values.cpu().numpy()))
    identity = identity.cpu().numpy()
    results = {view: {} for view in curves}
    for (view, name), exact in by_method.items():
        metrics = {}
        for depth in DEPTHS:
            oracle = exact[:, :depth].max(axis=1)
            earlier = by_method[view, "balanced_4"][:, :depth].max(axis=1)
            metrics[str(depth)] = {**paired_ocf_stats(oracle, earlier), "gap_to_exact_identity": float((identity - oracle).mean()),
                                   "headroom_vs_own_first": float((oracle - exact[:, 0]).mean())}
        results[view][name] = metrics
    return results, unique, float(identity.mean())


def compare_correspondence(neural, curves):
    report = {}
    for view, responses in neural.items():
        baseline = per_glyph_identity(responses[4])
        report[view] = {}
        for n, matrix in responses.items():
            source = per_glyph_identity(matrix)
            forward = per_glyph_identity(curves[view][f"balanced_{n}"][0])
            difference = source[:, 0] - forward[:, 0]
            report[view][str(n)] = {**summarize_identity([matrix], baseline),
                                    "paired_top1_gap_vs_forward": float(difference.mean()),
                                    "paired_top1_gap_se": float(difference.std() / len(source)**0.5),
                                    "neural_correct_forward_wrong": float((source[:, 0] * (1 - forward[:, 0])).mean())}
    return report


def run_fold(mouse, groups, images, splits, score, args, seed):
    started = time.monotonic()
    device = images.device
    fitting, calibration, held = (torch.as_tensor(rows, device=device) for rows in splits)
    source = {name: counts[fitting] for name, counts in groups.items()}
    model = fit_glyph_encoder(source, images[fitting], epochs=18, seed=211)
    generator = torch.Generator(device=device).manual_seed(seed + 173)
    queries = held_queries(groups, held, generator)
    daily, temperatures = forward_day_scores(model, groups, calibration, queries, (images[calibration], images[held]))
    curves = {view: repeat_curves(values) for view, values in daily.items()}
    measures = {view: {name: summarize_identity(variants, per_glyph_identity(methods["balanced_4"][0]))
                       for name, variants in methods.items()} for view, methods in curves.items()}
    masks = images[held] >= 128
    rows = torch.randperm(len(held), device=device, generator=generator)[:args.ocf_images]
    exact, count = exact_curves(curves, rows, masks, score)
    oracle, oracle_count, ceiling = oracle_curves(curves, rows[:args.oracle_images], masks, score)
    neural = neural_correspondence(mouse, source, queries)
    return {"mouse": mouse, "seed": seed, "fit_glyphs": len(fitting), "temperature_fit_glyphs": len(calibration),
            "held_glyphs": len(held), "temperatures": temperatures, "forward_identity": measures,
            "forward_exact_ocf": exact, "oracle_exact_ocf": oracle, "self_image_ocf": ceiling,
            "exact_ocf_unique_pairs": count, "oracle_ocf_unique_pairs": oracle_count,
            "disjoint_spike_reference": compare_correspondence(neural, curves), "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(3797, 4081))
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--oracle-images", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_repeat_ceiling_validation.json"))
    args = parser.parse_args()
    if (not 2 <= args.oracle_images <= args.ocf_images <= 320 or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Choose distinct cohort seeds and valid exact OCF sample sizes.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run this ceiling diagnostic on Kaggle with one GPU and the labeled competition inputs.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, torch.device("cuda:0"))
    score = official_scorer(root)
    splits = list(split_cohorts(len(images), args.seeds))
    output = {"seeds": args.seeds, "exact_ocf_images": args.ocf_images, "oracle_images": args.oracle_images,
              "protocol": "Fixed 320-glyph exact-image gallery with whole-glyph fit exclusions. Session score temperatures "
                          "are fit on a separate labeled glyph cohort. Balanced pools use 1/2/4 repeats from each of four "
                          "independently shuffled recordings; whole-day pools average every combination of days. Training "
                          "labels co-index sessions only for ceiling measurement; no actual test recordings are accessed. "
                          "GCCA measured-spike retrieval uses two disjoint references per day and one or two query trials "
                          "per day. It cannot furnish an independent 16-trial comparison. Timed GCCA uses two temporal bands.",
              "folds": []}
    for mouse in ("mice1", "mice2"):
        groups = grouped_blocks(root, catalog, mouse, images.device, len(images))
        if len(groups) != 4:
            raise ValueError("This closed-set ceiling needs the four labeled recordings.")
        for seed, split in zip(args.seeds, splits):
            print(f"{mouse}/{seed}: fit independent forward and alignment models", flush=True)
            result = run_fold(mouse, groups, images, split, score, args, seed)
            output["folds"].append(result)
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"mouse": mouse, "seed": seed, "forward_exact_ocf": result["forward_exact_ocf"],
                              "seconds": result["seconds"]}), flush=True)
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
