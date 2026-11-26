"""Test a calibrated single-trial posterior with exact-OCF-trained image utility."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_contour_retrieval import blend_contour_poisson, contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_exact_utility import fit_exact_utility, gallery_geometry, gallery_utility, pair_features, select_pairs
from glyph_gain_likelihood import fit_gain_model, gain_template_logits
from glyph_mbr import contour_signatures, fit_temperature, similarity_from_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task3_metric_validation import nested_repeats, unique_pair_ocf


def measure_utility(geometry, model, action, reference, measured, proxy):
    fitted = model.predict(pair_features(geometry, action, reference))
    approximate = proxy[action, reference] * 100
    zero = torch.full_like(fitted, float(measured.mean()))
    return {name: {"mae": float((values - measured).abs().mean()),
                   "rmse": float((values - measured).square().mean().sqrt()),
                   "correlation": float(torch.corrcoef(torch.stack((values, measured)))[0, 1]) if name != "mean" else None}
            for name, values in (("mean", zero), ("contour_proxy", approximate), ("fitted_exact", fitted))}


def fit_image_utility(images, signature, energy, scorer, args):
    masks = images >= 128
    device = images.device
    exact_scorer = lambda a, r: unique_pair_ocf(a, r, masks, scorer, device, 384)
    geometry = gallery_geometry(masks, signature, energy)
    proxy = similarity_from_signatures(signature, energy)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    hold, fit = torch.randperm(len(images), generator=generator, device=device).split((320, len(images) - 320))
    train_a, train_r = select_pairs(fit, proxy, args.utility_fit_pairs, generator)
    fit_labels, fit_unique = exact_scorer(train_a, train_r)
    model = fit_exact_utility(geometry, train_a, train_r, fit_labels, seed=args.seed)
    held_a, held_r = select_pairs(hold, proxy, args.utility_held_pairs, generator)
    held_labels, held_unique = exact_scorer(held_a, held_r)
    with torch.no_grad():
        diagnostics = measure_utility(geometry, model, held_a, held_r, held_labels, proxy)
    return proxy * 100, gallery_utility(model, geometry), {"fit_glyphs": len(fit), "held_glyphs": len(hold),
                                                           "unique_fit_pairs": fit_unique, "unique_held_pairs": held_unique,
                                                           "pair_prediction": diagnostics}


def fit_neural(training, images, signature, energy, mouse, seed):
    encoder = fit_glyph_encoder(training, images, epochs=18, seed=seed)
    contour = fit_contour_retrieval(training, signature, energy, task="task3", seed=seed)
    gain = fit_gain_model(training, "timed", temporal=True) if mouse == "mice2" else None
    return encoder, contour, gain


@torch.no_grad()
def v11_logits(model, counts, session, images):
    encoder, contour, gain = model
    rates = encoder.expected_counts(images, session)
    raw = contour_retrieval_scores(contour, counts, session)["contour_ridge"]
    likelihood = count_template_logits(counts, rates, "timed") if gain is None else gain_template_logits(counts, rates, gain)
    return blend_contour_poisson(raw, likelihood)


@torch.no_grad()
def choose(logits, approximate, exact, temperature):
    posterior = (logits / temperature).softmax(dim=1)
    proxy_mean = posterior @ approximate.T
    exact_mean = posterior @ exact.T
    excess = exact_mean - exact.mean(dim=1)
    norm = lambda x: F.layer_norm(x, (x.shape[-1],), eps=1e-8)
    anchor = 0.80 * norm(logits)
    options = {"v11": logits, "approximate_mbr": proxy_mean, "exact_mbr": exact_mean,
               "v11_approximate": anchor + 0.20 * norm(proxy_mean), "v11_exact": anchor + 0.20 * norm(exact_mean),
               "v11_centered_exact": anchor + 0.20 * norm(excess)}
    return {name: score.argmax(dim=1) for name, score in options.items()}, float(posterior.topk(32).values.sum(1).mean())


def summarize(measurements, baseline):
    values = np.asarray(measurements)
    delta = (values - np.asarray(baseline)).mean(axis=0)
    return {"ocf": float(values.mean()), "delta_vs_v11": float(delta.mean()), "paired_n": values.size,
            "paired_glyph_se": float(delta.std(ddof=1) / np.sqrt(len(delta)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int, default=419)
    parser.add_argument("--utility-fit-pairs", type=int, default=6400)
    parser.add_argument("--utility-held-pairs", type=int, default=1024)
    parser.add_argument("--calibration-glyphs", type=int, default=192)
    parser.add_argument("--ocf-images-per-session", type=int, default=96)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/posterior_utility_validation.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Fit and validate utility in a Kaggle GPU Notebook.")
    if min(args.calibration_glyphs, args.ocf_images_per_session, args.utility_fit_pairs, args.utility_held_pairs) < 2:
        parser.error("Use at least two independent glyphs or pairs in each fit and evaluation split.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    device = torch.device("cuda:0")
    images = load_images(root, device)
    masks = images >= 128
    signature, energy = contour_signatures(images)
    scorer = official_scorer(root)
    if args.calibration_glyphs + args.ocf_images_per_session > len(images):
        parser.error("The spike temperature and score glyph subsets must be disjoint.")
    draw = np.random.default_rng(args.seed + 141).permutation(len(images))
    calibrated = torch.as_tensor(draw[:args.calibration_glyphs], device=device)
    scored = torch.as_tensor(draw[args.calibration_glyphs:args.calibration_glyphs + args.ocf_images_per_session],
                             device=device)
    approximate, exact, geometry = fit_image_utility(images, signature, energy, scorer, args)
    print(json.dumps({"shape_utility": geometry}), flush=True)
    output = {"seed": args.seed, "repeat_seed": args.repeat_seed, "image_utility": geometry,
              "protocol": "Only labeled gallery images provide the exact utility targets. The held recorded trial is "
              "reserved before every spike fit; posterior temperature is fit on a separate inner repeat and separate "
              "glyph subset. Exact and proxy utility use the same calibrated posterior and matched 80/20 blends.",
              "mice": {}}
    for mouse in dict.fromkeys(args.mice):
        raw = grouped_blocks(root, catalog, mouse, device, len(images))
        fit, calibration, outer_fit, outer_query = nested_repeats(raw, args.repeat_seed)
        del raw
        inner = fit_neural(fit, images, signature, energy, mouse, args.seed)
        posterior_fit = fit_temperature([(v11_logits(inner, counts[calibrated], name, images), calibrated)
                                         for name, counts in calibration.items()])
        del inner, fit, calibration
        outer = fit_neural(outer_fit, images, signature, energy, mouse, args.seed)
        del outer_fit
        reads = {}
        result = {"temperature_fit": posterior_fit, "sessions": {}}
        output["mice"][mouse] = result
        for session, counts in outer_query.items():
            logits = v11_logits(outer, counts[scored], session, images)
            choices, top_mass = choose(logits, approximate, exact, posterior_fit["temperature"])
            guesses = torch.stack(tuple(choices.values()))
            ocf, _ = unique_pair_ocf(guesses, scored.expand_as(guesses), masks, scorer, device, 384)
            records = dict(zip(choices, ocf.cpu().numpy()))
            summary = {name: summarize([values], [records["v11"]]) for name, values in records.items()}
            result["sessions"][session] = {"metrics": summary, "mean_top32_posterior_mass": top_mass}
            for name, values in records.items():
                reads.setdefault(name, []).append(values)
            result["summary"] = {name: summarize(values, reads["v11"]) for name, values in reads.items()}
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"mouse": mouse, "session": session, "ocf": summary}), flush=True)
        del outer, outer_query


if __name__ == "__main__":
    main()
