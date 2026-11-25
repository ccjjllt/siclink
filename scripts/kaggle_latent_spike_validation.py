"""Evaluate unit-specific latent spike likelihood on held Task 3 repeats."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_contour_retrieval import blend_contour_poisson, contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_gain_likelihood import fit_gain_model, gain_template_logits
from glyph_latent_spike_likelihood import fit_latent_spikes, latent_spike_logits
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task3_metric_validation import unique_pair_ocf


def summary(values, baseline):
    values = np.asarray(values)
    delta = (values - np.asarray(baseline)).mean(axis=0)
    return {"ocf": float(values.mean()), "paired_delta": float(delta.mean()), "n": values.size,
            "paired_glyph_se": float(delta.std(ddof=1) / np.sqrt(delta.size))}


def classify(scores, labels):
    true = scores.diagonal()
    rank = 1 + (scores > true[:, None]).sum(dim=1)
    return {"top1": float((scores.argmax(dim=1) == labels).float().mean()),
            "top5": float((rank <= 5).float().mean()), "mean_rank": float(rank.float().mean()),
            "categorical_nll": float((scores.logsumexp(dim=1) - true).mean())}


@torch.no_grad()
def evaluate(mouse, session, query, rates, models, selected, images, scorer):
    contour, temporal, latent, shuffled = models
    ordinary = count_template_logits(query, rates, "timed")
    population = gain_template_logits(query, rates, temporal)
    unit_specific = latent_spike_logits(query, rates, latent.loadings)
    shuffled_spikes = latent_spike_logits(query, rates, shuffled)
    shape = contour_retrieval_scores(contour, query, session)["contour_ridge"]
    exact_v8 = blend_contour_poisson(shape, ordinary if mouse == "mice1" else population)
    methods = {"v8": exact_v8, "contour_temporal_gain": blend_contour_poisson(shape, population),
               "contour_latent_spikes": blend_contour_poisson(shape, unit_specific),
               "contour_shuffled_spikes": blend_contour_poisson(shape, shuffled_spikes)}
    chosen = torch.stack([value.argmax(dim=1)[selected] for value in methods.values()])
    exact, _ = unique_pair_ocf(chosen, selected.expand_as(chosen), images, scorer, images.device, 256)
    measured = dict(zip(methods, exact.cpu().numpy()))
    labels = torch.arange(len(images), device=images.device)
    classification = {name: classify(value, labels) for name, value in
                      {"timed_poisson": ordinary, "temporal_gamma": population,
                       "latent_spikes": unit_specific, "shuffled_spikes": shuffled_spikes, **methods}.items()}
    true_gain = {name: float((value.diagonal() - ordinary.diagonal()).mean()) for name, value in
                 {"latent_spikes": unit_specific, "shuffled_spikes": shuffled_spikes}.items()}
    return {"ocf": {name: summary([score], [measured["v8"]]) for name, score in measured.items()},
            "classification": classification, "matched_log_density_delta_vs_poisson": true_gain}, measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int, default=419)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--ocf-images-per-session", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/latent_spike_validation.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the fitted latent spike experiment on Kaggle GPU.")
    if args.ocf_images_per_session < 2:
        parser.error("Choose at least two independently sampled glyphs.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    device = torch.device("cuda:0")
    images = load_images(root, device)
    masks = images >= 128
    signature, energy = contour_signatures(images)
    scorer = official_scorer(root)
    draw = np.random.default_rng(args.seed).choice(len(images), args.ocf_images_per_session, replace=False)
    selected = torch.as_tensor(draw, device=device)
    output = {"seed": args.seed, "repeat_seed": args.repeat_seed, "rank": args.rank,
              "ocf_glyphs_per_recording": len(selected), "mice": {},
              "protocol": "One of four labeled repeats is withheld before all fits, for every glyph and recording. "
              "Exact OCF uses a common sampled glyph subset; classification uses the full gallery. "
              "A fixed permutation preserves low-rank shared gain magnitude but breaks cell identity."}
    for mouse in dict.fromkeys(args.mice):
        raw = grouped_blocks(root, catalog, mouse, device, len(images))
        folds = {session: split_repeats(counts, args.repeat_seed + index)
                 for index, (session, counts) in enumerate(raw.items())}
        del raw
        training = {session: pair[0] for session, pair in folds.items()}
        latent = fit_latent_spikes(training, rank=args.rank, seed=args.seed)
        rng = torch.Generator(device=device).manual_seed(args.seed + 61)
        permutation = torch.randperm(latent.loadings[0].shape[0], generator=rng, device=device)
        shuffled = tuple(matrix[permutation] for matrix in latent.loadings)
        temporal = fit_gain_model(training, "timed", temporal=True)
        contour = fit_contour_retrieval(training, signature, energy, task="task3", seed=args.seed)
        common = fit_glyph_encoder(training, images, epochs=18, seed=args.seed)
        models = contour, temporal, latent, shuffled
        results = {"latent_fit": latent.diagnostics, "gamma_alpha": temporal.alpha.tolist(), "sessions": {}}
        output["mice"][mouse] = results
        collected = {}
        for session, (_, query) in folds.items():
            rates = common.expected_counts(images, session)
            block, measured = evaluate(mouse, session, query, rates, models, selected, masks, scorer)
            results["sessions"][session] = block
            for name, value in measured.items():
                collected.setdefault(name, []).append(value)
            results["paired_summary"] = {name: summary(values, collected["v8"]) for name, values in collected.items()}
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"mouse": mouse, "session": session, **block}), flush=True)
        del folds, training, models, common, contour


if __name__ == "__main__":
    main()
