"""Measure whether fine co-spiking improves closed-gallery single-trial decoding."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_event_timing import event_contour_scores, fit_event_contours
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task3_metric_validation import unique_pair_ocf


def counts25(fine):
    counts = fine.unflatten(-1, (-1, 10, 5))[..., 2:, :].sum(dim=-1).float()
    groups = counts.unflatten(-1, (4, 2)).sum(dim=-1)
    return torch.cat((counts, groups, groups.sum(-1, keepdim=True)), dim=-1).sqrt_().flatten(-2)


def reliable_units(fit, count):
    tuning, scatter = [], []
    for fine in fit.values():
        evoked = fine.unflatten(-1, (-1, 50))[..., 10:].sum(dim=-1).float().sqrt()
        scatter.append(evoked.var(dim=1, unbiased=True).mean(dim=0))
        tuning.append(evoked.mean(dim=1).var(dim=0, unbiased=False))
    within = torch.stack(scatter).mean(0)
    signal = (torch.stack(tuning).mean(0) - within / next(iter(fit.values())).shape[1]).clamp_min(0)
    selected = (signal / (within + 0.10)).topk(count).indices.sort().values
    return selected, float((signal / (within + 0.10))[selected].median())


@torch.no_grad()
def fine_pairs(fine, units, *, jitter_seed=None):
    """Accumulate conditional within-bin coincidences without changing rates."""
    pairs = torch.triu_indices(len(units), len(units), offset=1, device=units.device)
    generator = None if jitter_seed is None else torch.Generator(device=units.device).manual_seed(jitter_seed)
    outputs = []
    for block in fine.split(256):
        phase = block.unflatten(-1, (-1, 10, 5))[..., units, 2:, :].float()
        if generator is not None:
            offset = torch.randint(5, phase.shape[:-1] + (1,), generator=generator, device=units.device)
            cycle = (torch.arange(5, device=units.device) + offset).remainder(5)
            phase = phase.gather(-1, cycle)
        total = phase.sum(dim=-1, keepdim=True)
        phase = (phase - total / 5) / (1 + total).sqrt()
        parts = []
        for window in (phase[..., :2, :], phase[..., 2:, :]):
            vector = window.flatten(-2)
            similarity = (vector @ vector.transpose(-1, -2))[..., pairs[0], pairs[1]]
            parts.append(similarity.sign() * similarity.abs().sqrt())
        outputs.append(torch.cat(parts, dim=-1))
    return torch.cat(outputs)


def statistics(current, baseline):
    current = np.asarray(current)
    paired = (current - np.asarray(baseline)).mean(axis=0)
    return {"ocf": float(current.mean()), "paired_delta": float(paired.mean()), "pairs": current.size,
            "paired_glyph_se": float(paired.std(ddof=1) / np.sqrt(paired.size))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", choices=("mice1", "mice2"), default="mice1")
    parser.add_argument("--units", type=int, default=28)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int, default=419)
    parser.add_argument("--ocf-images-per-session", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/spike_synchrony_validation.json"))
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use a Kaggle Notebook with the competition data mounted.")
    if min(args.units, args.ocf_images_per_session) < 2:
        parser.error("Use at least two selected units and two OCF glyphs per recording.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    device = torch.device("cpu")
    images = load_images(root, device)
    masks = images >= 128
    signatures, energy = contour_signatures(images)
    score = official_scorer(root)
    draws = np.random.default_rng(args.seed).choice(len(images), args.ocf_images_per_session, replace=False)
    selected = torch.as_tensor(draws, device=device)
    raw = grouped_blocks(root, catalog, args.mouse, device, len(masks), bin_ms=5)
    fit, held = {}, {}
    for index, name in enumerate(tuple(raw)):
        fit[name], held[name] = split_repeats(raw.pop(name).to(torch.int16), args.repeat_seed + index)
    selected_units, unit_snr = reliable_units(fit, args.units)
    print(f"{args.mouse}: selected {len(selected_units)} repeat-reliable units from fit trials", flush=True)
    coarse_fit = {name: counts25(fine) for name, fine in fit.items()}
    coarse_held = {name: counts25(fine) for name, fine in held.items()}
    options = ("counts25", "real_pairs", "jitter_pairs")
    decisions, retrieval = {}, {}
    labels = torch.arange(len(masks), device=device)
    for name in options:
        if name == "counts25":
            features, queries = coarse_fit, coarse_held
        else:
            jitter = name == "jitter_pairs"
            features, queries = {}, {}
            for index, session in enumerate(fit):
                seed = args.seed + 50 * index if jitter else None
                real = fine_pairs(fit[session], selected_units, jitter_seed=seed)
                query = fine_pairs(held[session], selected_units, jitter_seed=seed + 1 if jitter else None)
                features[session] = torch.cat((coarse_fit[session], real), dim=-1)
                queries[session] = torch.cat((coarse_held[session], query), dim=-1)
        model = fit_event_contours(features, signatures, energy)
        decisions[name], retrieval[name] = {}, {}
        for session, query in queries.items():
            logits = event_contour_scores(model, query, session)
            ranking = logits.topk(5, dim=1).indices
            decisions[name][session] = ranking[selected, 0]
            retrieval[name][session] = {"top1": float((ranking[:, 0] == labels).float().mean()),
                                        "top5": float((ranking == labels[:, None]).any(dim=1).float().mean())}
        print(json.dumps({"method": name, "complete_gallery": retrieval[name]}), flush=True)
        del model, features, queries

    output = {"mouse": args.mouse, "device": str(device), "seed": args.seed, "repeat_seed": args.repeat_seed,
              "selected_units": args.units, "fit_median_selected_unit_snr": unit_snr,
              "fine_bins_ms": 5, "coarse_bins_ms": 25, "windows_ms": [[50, 100], [100, 250]],
              "ocf_images_per_session": len(selected), "complete_gallery": retrieval, "blocks": {},
              "protocol": "One labeled trial per glyph/session is reserved before both unit selection and model fit. "
              "Pair features measure fine coincidences of differently identified units after subtracting their 25 ms "
              "mean. Independent within-bin cyclic shifts are a timing channel control that preserve exact 25 ms rates. "
              "The count-only reference is the Task 3 contour branch."}
    aggregated = {name: [] for name in options}
    for session in fit:
        chosen = torch.stack([decisions[name][session] for name in options])
        exact, _ = unique_pair_ocf(chosen, selected.expand_as(chosen), masks, score, device, 128)
        measured = dict(zip(options, exact.numpy()))
        comparison = {name: statistics([values], [measured["counts25"]]) for name, values in measured.items()}
        output["blocks"][session] = comparison
        for name, values in measured.items():
            aggregated[name].append(values)
        output["summary"] = {name: statistics(values, aggregated["counts25"]) for name, values in aggregated.items()}
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": session, "metrics": comparison}), flush=True)


if __name__ == "__main__":
    main()
