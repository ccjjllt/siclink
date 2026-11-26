"""Measure whether per-trial displacement likelihood helps identify repeated stimuli."""

import argparse
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import fit_glyph_encoder
from glyph_gain_likelihood import fit_gain_model, population_gain_logits
from kaggle_diagnostics import official_scorer
from kaggle_retinal_shift_validation import (CENTER, fit_spatial_prior, shifted_rates,
                                            shuffled_shift_rates)
from kaggle_task3_metric_validation import unique_pair_ocf


REPEATS = (1, 2, 4)
SHIFT_PIXELS = 3
SHIFTS = tuple(product((-SHIFT_PIXELS, 0, SHIFT_PIXELS), repeat=2))


@torch.no_grad()
def trial_logits(counts, rates, gain):
    """Complete rectangular scoring over glyph and shift, before combining repeats."""
    n_query, n_trials, n_features = counts.shape
    n_glyphs, n_shifts, _ = rates.shape
    observations = counts.reshape(n_query * n_trials, 1, n_features).unflatten(-1, (-1, 10))
    gallery = rates.flatten(0, 1).float().unflatten(-1, (-1, 10))
    logits = 0
    for (start, stop), alpha in zip(gain.windows, gain.alpha):
        spikes = observations[..., start:stop].flatten(-2)
        expected = gallery[..., start:stop].flatten(-2)
        logits = logits + population_gain_logits(spikes, expected, alpha)
    return logits.reshape(n_query, n_trials, n_glyphs, n_shifts)


def fold_rates(model, images, session, counts):
    rates = shifted_rates(model, images, session, SHIFTS, SHIFT_PIXELS)
    gain = fit_gain_model({session: counts}, "timed", temporal=True)
    return rates, gain


def fit_prior(model, images, fit, calibration):
    rows = torch.arange(len(images), device=images.device)
    likelihoods = []
    for name, counts in calibration.items():
        rates, gain = fold_rates(model, images, name, fit[name])
        likelihood = trial_logits(counts, rates, gain)
        likelihoods.append(likelihood[rows, :, rows])
    return fit_spatial_prior(likelihoods, SHIFTS)


def split_query(counts, permutation, rng):
    trial_order = np.argsort(rng.random(counts.shape[:2]), axis=1)
    trial_order = torch.as_tensor(trial_order, device=counts.device)
    ordered = counts.gather(1, trial_order[:, :, None].expand_as(counts))
    return ordered[permutation]


def score_variants(logits, shuffled, moment, prior, repeats):
    likelihood = logits[:, :repeats]
    shuffled = shuffled[:, :repeats]
    return {
        "zero": likelihood[..., CENTER].sum(1),
        "moment_rate": moment[:, :repeats, :, 0].sum(1),
        "trial_shift": (likelihood + prior).logsumexp(-1).sum(1),
        "shared_shift": (likelihood.sum(1) + prior).logsumexp(-1),
        "shuffled_shift": (shuffled + prior).logsumexp(-1).sum(1),
    }


def retrieval(scores, labels):
    order = scores.argsort(dim=1, descending=True, stable=True)
    rank = 1 + (order == labels[:, None]).int().argmax(dim=1)
    nll = scores.logsumexp(dim=1) - scores.gather(1, labels[:, None]).squeeze(1)
    values = torch.stack((rank == 1, rank <= 5, rank <= 16)).float().mean(dim=1)
    return {"top1": float(values[0]), "top5": float(values[1]), "top16": float(values[2]),
            "mean_reciprocal_rank": float(rank.float().reciprocal().mean()), "mean_nll": float(nll.mean())}


def summarize(rows):
    control = np.stack(rows["zero"])
    summary = {}
    for name, scores in rows.items():
        values = np.stack(scores)
        paired = (values - control).mean(axis=0)
        summary[name] = {"ocf": float(values.mean()), "paired_delta": float(paired.mean()),
                         "paired_glyph_se": float(paired.std(ddof=1) / np.sqrt(len(paired))),
                         "scored_rows": int(values.size)}
    return summary


def run_mouse(mouse, root, catalog, images, split, scorer, args, result, save):
    device = images.device
    fit_ids, calibration_ids, held_ids = split
    held_images = images[held_ids]
    masks = held_images >= 128
    groups = grouped_blocks(root, catalog, mouse, device, len(images))
    fitting = {name: counts[fit_ids] for name, counts in groups.items()}
    calibration = {name: counts[calibration_ids] for name, counts in groups.items()}
    held = {name: counts[held_ids] for name, counts in groups.items()}
    del groups
    encoder = fit_glyph_encoder(fitting, images[fit_ids], epochs=args.epochs, seed=args.seed)
    prior, info = fit_prior(encoder, images[calibration_ids], fitting, calibration)
    prior = prior.float()
    result["prior"] = {**info, "probabilities": prior.exp().tolist()}
    save()

    rng = np.random.default_rng(args.seed + 3)
    sample = torch.as_tensor(rng.choice(len(held_ids), args.ocf_images, replace=False), device=device)
    order = torch.as_tensor(rng.permutation(len(held_ids)), device=device)
    donor = torch.empty_like(order)
    donor[order] = order.roll(1)
    paired = {}
    for session, counts in held.items():
        perm = torch.as_tensor(rng.permutation(len(held_ids)), device=device)
        queries = split_query(counts, perm, rng)
        rates, gain = fold_rates(encoder, held_images, session, fitting[session])
        shuffled_rates, norm_error = shuffled_shift_rates(rates, donor)
        average_rate = (rates * prior.exp()[None, :, None]).sum(1, keepdim=True)
        exact = trial_logits(queries, rates, gain)
        shuffled = trial_logits(queries, shuffled_rates, gain)
        moment = trial_logits(queries, average_rate, gain)
        variants = {repeats: score_variants(exact, shuffled, moment, prior, repeats) for repeats in REPEATS}
        numbers = {repeats: {name: retrieval(values, perm) for name, values in matrices.items()}
                   for repeats, matrices in variants.items()}
        sample_rows = perm.argsort()[sample]
        picks = {name: values[sample_rows].argmax(1) for name, values in variants[4].items()}
        predictions = torch.stack(tuple(picks.values()))
        measured, pair_count = unique_pair_ocf(predictions, sample.expand_as(predictions), masks, scorer, device, 128)
        measured = dict(zip(picks, measured.cpu().numpy()))
        for name, values in measured.items():
            paired.setdefault(name, []).append(values)
        report = {"retrieval_by_repeat_count": numbers, "exact_ocf_unique_pairs": pair_count,
                  "exact_ocf": summarize({name: [values] for name, values in measured.items()}),
                  "max_shuffled_log_rate_norm_error": norm_error}
        result["sessions"][session] = report
        result["paired_four_repeat_ocf"] = summarize(paired)
        save()
        print(json.dumps({"mouse": mouse, "session": session, **report}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2651)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--calibration-glyphs", type=int, default=256)
    parser.add_argument("--held-glyphs", type=int, default=320)
    parser.add_argument("--ocf-images", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_shift_repeats.json"))
    args = parser.parse_args()
    if not 2 <= args.ocf_images <= args.held_glyphs or min(args.calibration_glyphs, args.epochs) < 1:
        parser.error("Choose valid split sizes and a positive number of epochs.")
    if not Path("/kaggle/input").is_dir() or (args.device != "cpu" and not torch.cuda.is_available()):
        raise RuntimeError("Use the requested accelerator in a Kaggle Notebook with official training data.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    device = torch.device(args.device)
    images = load_images(root, device)
    n_fit = len(images) - args.calibration_glyphs - args.held_glyphs
    if n_fit < 512:
        parser.error("Reserve at least 512 separate glyphs for the image-to-spike encoder.")
    order = torch.as_tensor(np.random.default_rng(args.seed).permutation(len(images)), device=device)
    split = order.split((n_fit, args.calibration_glyphs, args.held_glyphs))
    output = {"seed": args.seed, "device": args.device, "fit_glyphs_per_mouse": n_fit,
              "calibration_glyphs_per_mouse": args.calibration_glyphs, "held_glyphs_per_mouse": args.held_glyphs,
              "exact_ocf_images_per_session": args.ocf_images, "shifts_xy_pixels": SHIFTS,
              "protocol": "Closed-gallery mechanistic Task 1 proxy: candidate images are the held training glyphs, "
                          "without known correspondence at selection. The encoder and population gain fit only other "
                          "glyphs; spatial prior uses a separate calibration set. Query trial and glyph order are "
                          "independently shuffled before comparison. The formal test and its twenty repeats are unused. "
                          "The experiment compares likelihoods at fixed candidate images, without the v11 OCF ranker.",
              "mice": {}}

    def save():
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    scorer = official_scorer(root)
    for mouse in ("mice1", "mice2"):
        print(f"{mouse}: fit encoder on {n_fit} glyphs, freeze translation on {args.calibration_glyphs}", flush=True)
        result = {"sessions": {}}
        output["mice"][mouse] = result
        run_mouse(mouse, root, catalog, images, split, scorer, args, result, save)


if __name__ == "__main__":
    main()
