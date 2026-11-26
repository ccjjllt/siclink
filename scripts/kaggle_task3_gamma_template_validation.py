"""Validate a conjugate class-rate posterior on untouched Task 3 repeats."""

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
from glyph_gain_likelihood import fit_gain_model, gain_template_logits
from glyph_gamma_template import (PRIOR_EXPOSURES, build_gamma_prior, gamma_template_logits, gamma_templates,
                                  paired_count_log_probability)
from glyph_mbr import contour_signatures, fit_temperature
from kaggle_diagnostics import official_scorer
from kaggle_task3_metric_validation import nested_repeats, unique_pair_ocf


BLEND_WEIGHT = 0.25
SHORTLIST = 16
FULL_METHODS = ("v11", "matched_poisson", "gamma_predictive", "poisson_blend", "gamma_blend")


def normalize(scores):
    return F.layer_norm(scores, (scores.shape[-1],), eps=1e-8)


def fit_components(training, images, signature, energy, mouse, args):
    encoder = fit_glyph_encoder(training, images, epochs=args.encoder_epochs, seed=args.seed)
    rates = {name: encoder.expected_counts(images, name) for name in training}
    priors = {name: build_gamma_prior(rates, training, name) for name in training}
    contour = fit_contour_retrieval(training, signature, energy, task="task3", seed=args.seed)
    gain = fit_gain_model(training, "timed", temporal=True) if mouse == "mice2" else None
    return rates, priors, contour, gain


@torch.no_grad()
def score_block(counts, templates, rates, contour, gain, session):
    forward = (count_template_logits(counts, rates, "timed") if gain is None
               else gain_template_logits(counts, rates, gain))
    direction = contour_retrieval_scores(contour, counts, session)["contour_ridge"]
    v11 = blend_contour_poisson(direction, forward)
    poisson, gamma = gamma_template_logits(counts[:, 0], templates)
    anchor = (1 - BLEND_WEIGHT) * normalize(v11)
    return {"v11": v11, "matched_poisson": poisson, "gamma_predictive": gamma,
            "poisson_blend": anchor + BLEND_WEIGHT * normalize(poisson),
            "gamma_blend": anchor + BLEND_WEIGHT * normalize(gamma)}


def fit_exposure(priors, training, held, glyphs):
    scores = {}
    for exposure in PRIOR_EXPOSURES:
        probabilities = []
        for name, trial in held.items():
            templates = gamma_templates(priors[name][glyphs], training[name][glyphs], exposure)
            probabilities.append(torch.stack(paired_count_log_probability(trial[glyphs, 0], templates)))
        gamma, poisson = torch.cat(probabilities, dim=1).mean(dim=1).tolist()
        scores[exposure] = {"gamma_true_count_log_p": gamma, "matched_poisson_true_count_log_p": poisson}
    selected = max(scores, key=lambda exposure: scores[exposure]["gamma_true_count_log_p"])
    return selected, scores


def fit_temperatures(rates, priors, contour, gain, fit, inner, chosen, exposure, args):
    samples = {name: [] for name in FULL_METHODS}
    for session, counts in inner.items():
        templates = gamma_templates(priors[session], fit[session], exposure)
        frames = [score_block(batch, templates, rates[session], contour, gain, session)
                  for batch in counts[chosen].split(args.query_batch)]
        for name in FULL_METHODS:
            samples[name].append((normalize(torch.cat([frame[name] for frame in frames])), chosen))
    return {name: fit_temperature(pairs) for name, pairs in samples.items()}


def identify(scores, labels, temperatures):
    ranks = {name: values.topk(SHORTLIST, dim=1).indices for name, values in scores.items()}
    ranks["v11"] = scores["v11"].argsort(dim=1, descending=True, stable=True)[:, :SHORTLIST]
    policy = ranks["v11"]
    for name in ("poisson_blend", "gamma_blend"):
        reordering = scores[name].gather(1, policy).argsort(dim=1, descending=True, stable=True)
        ranks[f"{name}_top16"] = policy.gather(1, reordering)

    positions = torch.arange(1, SHORTLIST + 1, device=labels.device)
    stats = {}
    for name, indices in ranks.items():
        hit = (indices == labels[:, None]).float()
        metrics = {"top1": hit[:, 0], "top5": hit[:, :5].sum(dim=1), "top16": hit.sum(dim=1),
                   "mrr_at16": (hit / positions).sum(dim=1)}
        if name in temperatures:
            posterior = normalize(scores[name]) / temperatures[name]["temperature"]
            metrics["true_glyph_nll"] = posterior.logsumexp(dim=1) - posterior.gather(1, labels[:, None])[:, 0]
        stats[name] = {metric: value.cpu().numpy() for metric, value in metrics.items()}
    return stats, {name: order[:, 0].cpu().numpy() for name, order in ranks.items()}


def paired(values, control):
    values, control = np.stack(values), np.stack(control)
    delta = (values - control).mean(axis=0)
    return {"mean": float(values.mean()), "paired_delta_vs_v11": float(delta.mean()),
            "paired_glyph_se": float(delta.std(ddof=1) / np.sqrt(len(delta))), "n": values.size}


def summarize(ocf, identity):
    output = {name: {"ocf": paired(values, ocf["v11"]),
                     "identity": {metric: paired(rows, identity["v11"][metric])
                                  for metric, rows in identity[name].items()}}
              for name, values in ocf.items()}
    controls = {"gamma_predictive": "matched_poisson", "gamma_blend": "poisson_blend",
                "gamma_blend_top16": "poisson_blend_top16"}
    for name, control in controls.items():
        difference = paired(ocf[name], ocf[control])
        output[name]["ocf"].update(paired_delta_vs_matched_poisson=difference["paired_delta_vs_v11"],
                                   paired_se_vs_matched_poisson=difference["paired_glyph_se"])
    return output


def evaluate_session(counts, templates, rates, contour, gain, session, temperatures, held, masks, scorer, args):
    labels = torch.arange(len(counts), device=counts.device)
    collected, choices = {}, {}
    for batch, target in zip(counts.split(args.query_batch), labels.split(args.query_batch)):
        logits = score_block(batch, templates, rates, contour, gain, session)
        identity, selected = identify(logits, target, temperatures)
        for name, metrics in identity.items():
            destination = collected.setdefault(name, {})
            for metric, value in metrics.items():
                destination.setdefault(metric, []).append(value)
        for name, prediction in selected.items():
            choices.setdefault(name, []).append(prediction)
    full_metrics = {name: {metric: np.concatenate(batches) for metric, batches in results.items()}
                    for name, results in collected.items()}
    indices = torch.as_tensor(np.stack([np.concatenate(parts) for parts in choices.values()]), device=counts.device)
    images = indices[:, held]
    exact, pairs = unique_pair_ocf(images, held.expand_as(images), masks, scorer, counts.device, 384)
    paired_ocf = dict(zip(choices, exact.cpu().numpy()))
    gamma, matched = paired_count_log_probability(counts[:, 0], templates)
    count_logp = {"gamma_predictive": float(gamma.mean()), "matched_poisson": float(matched.mean()),
                  "paired_gain_nats": float((gamma - matched).mean())}
    return paired_ocf, full_metrics, pairs, count_logp


def run_mouse(mouse, data, glyphs, args, output, save):
    root, catalog, images, masks, signature, energy, scorer = data
    calibration, temperature_ids, evaluated = glyphs
    raw = grouped_blocks(root, catalog, mouse, images.device, len(images))
    inner_fit, inner_held, outer_fit, outer_query = nested_repeats(raw, args.repeat_seed)
    del raw
    print(f"{mouse}: fit two-repeat inner encoder and class-rate prior", flush=True)
    rates, priors, contour, gain = fit_components(inner_fit, images, signature, energy, mouse, args)
    exposure, choices = fit_exposure(priors, inner_fit, inner_held, calibration)
    temps = fit_temperatures(rates, priors, contour, gain, inner_fit, inner_held, temperature_ids, exposure, args)
    del rates, priors, contour, gain, inner_fit, inner_held
    record = {"exposure": exposure, "inner_exposure_comparison": choices, "inner_temperature_fit": temps,
              "sessions": {}}
    output["mice"][mouse] = record
    save()
    print(json.dumps({"mouse": mouse, "exposure": exposure, "temperatures": temps}), flush=True)

    print(f"{mouse}: fit three-repeat outer encoder and class-rate prior", flush=True)
    rates, priors, contour, gain = fit_components(outer_fit, images, signature, energy, mouse, args)
    ocf, identities = {}, {}
    for session, counts in outer_query.items():
        templates = gamma_templates(priors[session], outer_fit[session], exposure)
        exact, identity, pairs, count_logp = evaluate_session(counts, templates, rates[session], contour, gain,
                                                              session, temps, evaluated, masks, scorer, args)
        for name, values in exact.items():
            ocf.setdefault(name, []).append(values)
        for name, metrics in identity.items():
            pooled = identities.setdefault(name, {})
            for metric, values in metrics.items():
                pooled.setdefault(metric, []).append(values)
        metrics = summarize({name: [values] for name, values in exact.items()},
                            {name: {metric: [values] for metric, values in items.items()}
                             for name, items in identity.items()})
        record["sessions"][session] = {"unique_exact_pairs": pairs, "metrics": metrics,
                                       "true_glyph_count_log_p": count_logp}
        record["summary"] = summarize(ocf, identities)
        save()
        print(json.dumps({"mouse": mouse, "session": session, "metrics": metrics, "count_log_p": count_logp}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int, default=1429)
    parser.add_argument("--glyph-seed", type=int, default=2339)
    parser.add_argument("--exposure-calibration-glyphs", type=int, default=384)
    parser.add_argument("--temperature-calibration-glyphs", type=int, default=192)
    parser.add_argument("--ocf-images", type=int, default=96)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--query-batch", type=int, default=256)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_gamma_template_validation.json"))
    args = parser.parse_args()
    sizes = (args.exposure_calibration_glyphs, args.temperature_calibration_glyphs, args.ocf_images)
    if min(sizes) < 2 or min(args.query_batch, args.encoder_epochs) < 1:
        parser.error("Use nonempty batches and at least two glyphs per split.")
    if not Path("/kaggle/input").is_dir() or (args.device == "cuda:0" and not torch.cuda.is_available()):
        raise RuntimeError("Use a Kaggle Notebook with the requested accelerator.")

    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, args.device)
    masks = images >= 128
    signature, energy = contour_signatures(images)
    scorer = official_scorer(root)
    if sum(sizes) > len(images):
        parser.error("The disjoint calibration and scored glyph cohorts must fit inside the gallery.")
    order = torch.as_tensor(np.random.default_rng(args.glyph_seed).permutation(len(images)), device=images.device)
    folds = order[:sum(sizes)].split(sizes)
    output = {"device": args.device, "model_seed": args.seed, "repeat_seed": args.repeat_seed, "glyph_seed": args.glyph_seed,
              "encoder_epochs": args.encoder_epochs, "split_sizes": sizes, "prior_exposures": PRIOR_EXPOSURES,
              "gallery_glyphs": len(images), "blend_weight": BLEND_WEIGHT, "shortlist": SHORTLIST,
              "v11": {"mice1": "equal row-standardized contour and timed forward Poisson",
                      "mice2": "equal row-standardized contour and timed forward population gain"},
              "protocol": "Reserve one repeat per glyph and session before any model fit. Choose the effective Gamma "
                          "prior exposure and class-posterior temperatures on disjoint cohorts within a further "
                          "independent inner repeat. Refitted outer models use all three available repeats. The full "
                          "gallery identity metrics include all outer rows. Exact OCF uses a fixed glyph cohort and "
                          "compares unconditional and matched v11-top16 decisions. No official test records enter.",
              "mice": {}}

    def save():
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    data = root, catalog, images, masks, signature, energy, scorer
    for mouse in dict.fromkeys(args.mice):
        run_mouse(mouse, data, folds, args, output, save)
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
