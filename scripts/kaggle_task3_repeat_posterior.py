"""Test calibrated classifier evidence while holding the visual decision fixed.

An inner held repeat calibrates both full-gallery posteriors. Two posterior
sources and two equal-size action lists form a matched comparison. Both use the
same v11 anchor and prior-centered image utility. A separate outer repeat
supplies only measured inputs until all glyph decisions have been made.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_contour_retrieval import fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import fit_glyph_encoder
from glyph_gain_likelihood import fit_gain_model
from glyph_mbr import contour_signatures, fit_temperature, glyph_posterior, similarity_from_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task3_metric_validation import nested_repeats, unique_pair_ocf
from kaggle_task3_part_discovery import SESSIONS, project_counts, standardize_rows
from kaggle_task3_part_recheck import score_v11
from kaggle_task3_repeat_ratio import fit_ratio, training_views, whiten


CLASSIFIER_WEIGHT = 0.35
UTILITY_WEIGHT = 0.20
POOL_SIZE = 16
RETAIN = 12
BASE_POSTERIOR_BASE_POOL = "v11_posterior_v11_pool"
CLASS_POSTERIOR_BASE_POOL = "classifier_posterior_v11_pool"
BASE_POSTERIOR_NEW_POOL = "v11_posterior_nominated_pool"
CLASS_POSTERIOR_NEW_POOL = "classifier_posterior_nominated_pool"
CONTRASTS = {CLASS_POSTERIOR_BASE_POOL: (BASE_POSTERIOR_BASE_POOL,),
             BASE_POSTERIOR_NEW_POOL: (BASE_POSTERIOR_BASE_POOL,),
             CLASS_POSTERIOR_NEW_POOL: (BASE_POSTERIOR_NEW_POOL, CLASS_POSTERIOR_BASE_POOL)}


@dataclass
class ScoreModels:
    contour: object
    encoder: object
    gain: object
    classifier: object
    factor: torch.Tensor
    centers: dict
    templates: dict


def fit_models(training, images, signature, energy, mouse, args):
    torch.manual_seed(args.seed)
    contour = fit_contour_retrieval(training, signature, energy, task="task3", seed=args.seed)
    encoder = fit_glyph_encoder(training, images, epochs=args.encoder_epochs, seed=args.seed)
    gain = fit_gain_model(training, "timed", temporal=True) if mouse == "mice2" else None
    factor, views, details = training_views(contour, training, args.dimensions)
    classifier, loss = fit_ratio(views, args, augmented=True)
    models = ScoreModels(contour, encoder, gain, classifier, factor,
                         {name: view.origin for name, view in views.items()},
                         {name: view.templates for name, view in views.items()})
    return models, {**details, "classifier_loss": loss}


@torch.no_grad()
def score_sources(models, query, images, session, mouse, dimensions):
    v11 = score_v11(mouse, models.contour, models.encoder, models.gain, query, images, session)
    projected = project_counts(models.contour, query, session).mean(dim=1)[:, -dimensions:]
    features = whiten(projected - models.centers[session], models.factor)
    novel = models.classifier(features, models.templates[session])
    base = standardize_rows(v11)
    mixed = base.lerp(standardize_rows(novel), CLASSIFIER_WEIGHT)
    return v11, base, standardize_rows(mixed)


@torch.no_grad()
def expected_utility(posterior, slots, utility):
    """Use full gallery mass with image-utility rows only for supported actions."""
    return torch.cat([torch.einsum("bg,bkg->bk", mass, utility[actions])
                      for mass, actions in zip(posterior.split(64), slots.split(64))])


def normalize_from_v11_pool(values):
    reference = values[:, :POOL_SIZE]
    center = reference.mean(dim=1, keepdim=True)
    scale = reference.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (values - center) / scale


@torch.no_grad()
def decide(v11, base, mixed, temperatures, utility):
    original_pool = v11.topk(POOL_SIZE, dim=1).indices
    new_score = mixed.clone()
    new_score.scatter_(1, original_pool, -torch.inf)
    novel = new_score.topk(POOL_SIZE - RETAIN, dim=1).indices
    # Use the original v11 slots for normalization even when actions are added.
    slots = torch.cat((original_pool, novel), dim=1)
    anchored = (1 - UTILITY_WEIGHT) * normalize_from_v11_pool(base.gather(1, slots))
    pooled = torch.cat((torch.arange(RETAIN), torch.arange(POOL_SIZE, slots.shape[1])))
    common = {"v11": original_pool, "nominated": slots[:, pooled]}
    choices = {"v11": v11.argmax(dim=1)}
    posteriors = {"v11": glyph_posterior(base, temperatures["v11"]["temperature"]),
                  "classifier": glyph_posterior(mixed, temperatures["classifier"]["temperature"])}
    for source, posterior in posteriors.items():
        agreement = expected_utility(posterior, slots, utility)
        scores = anchored + UTILITY_WEIGHT * normalize_from_v11_pool(agreement)
        leader = scores[:, :POOL_SIZE].argmax(dim=1, keepdim=True)
        choices[f"{source}_posterior_v11_pool"] = common["v11"].gather(1, leader)[:, 0]
        other = scores[:, pooled].argmax(dim=1, keepdim=True)
        choices[f"{source}_posterior_nominated_pool"] = common["nominated"].gather(1, other)[:, 0]
    return choices, common, posteriors


def diagnostics(choices, pools, posteriors, labels):
    reference = labels[:, None]
    metrics = {"decision_top1": {name: float((guess == labels).float().mean()) for name, guess in choices.items()},
               "changed_vs_v11": {name: float((guess != choices["v11"]).float().mean()) for name, guess in choices.items()},
               "pools": {}, "posteriors": {}}
    for name, pool in pools.items():
        matches = pool == reference
        metrics["pools"][name] = {"top16_recall": float(matches.any(dim=1).float().mean()),
                                  "tail_recall": float(matches[:, RETAIN:].any(dim=1).float().mean())}
    for name, mass in posteriors.items():
        expected = mass.gather(1, reference)[:, 0].clamp_min(1e-12)
        selection = mass.topk(32, dim=1)
        metrics["posteriors"][name] = {"mean_true_mass": float(expected.mean()),
                                       "median_true_mass": float(expected.median()),
                                       "nll": float(-expected.log().mean()),
                                       "mean_top32_mass": float(selection.values.sum(dim=1).mean()),
                                       "top5_recall": float((selection.indices[:, :5] == reference).any(1).float().mean())}
    return metrics


def score_exact(choices, pools, labels, sample, masks, scorer, depth):
    rows = labels.argsort()[sample]
    selected = torch.stack([label[rows] for label in choices.values()])
    oracle = torch.stack([pool[rows[:depth]] for pool in pools.values()])
    select_target = sample.expand_as(selected)
    oracle_target = sample[None, :depth, None].expand_as(oracle)
    values, pairs = unique_pair_ocf(torch.cat((selected.flatten(), oracle.flatten())),
                                    torch.cat((select_target.flatten(), oracle_target.flatten())),
                                    masks, scorer, torch.device("cpu"), 256)
    top1, best = values.split((selected.numel(), oracle.numel()))
    top1 = top1.reshape(selected.shape).numpy()
    best = best.reshape(oracle.shape).amax(dim=2).numpy()
    return dict(zip(choices, top1)), dict(zip(pools, best)), pairs


def summarize(observations):
    values = {name: np.stack(blocks) for name, blocks in observations.items()}
    summary = {}
    for name, scores in values.items():
        comparisons = {}
        for reference in ("v11", *CONTRASTS.get(name, ())):
            delta = (scores - values[reference]).mean(axis=0)
            comparisons[reference] = {"delta": float(delta.mean()),
                                      "paired_se": float(delta.std(ddof=1) / np.sqrt(len(delta)))}
        summary[name] = {"ocf": float(scores.mean()), "scored_rows": scores.size, "comparisons": comparisons}
    return summary


def extend(target, observed):
    for name, values in observed.items():
        target.setdefault(name, []).append(values)


def gallery_summary(folds):
    aggregate = {}
    for signal in ("pools", "posteriors"):
        units = [fold["gallery"][signal] for fold in folds]
        aggregate[signal] = {name: {metric: float(np.mean([unit[name][metric] for unit in units]))
                                    for metric in metrics} for name, metrics in units[0].items()}
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--sessions", nargs="+", choices=SESSIONS, default=SESSIONS)
    parser.add_argument("--ocf-images", type=int, default=64)
    parser.add_argument("--oracle-images", type=int, default=8)
    parser.add_argument("--calibration-images", type=int, default=256)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int, default=2419)
    parser.add_argument("--glyph-seed", type=int, default=2729)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_repeat_posterior.json"))
    args = parser.parse_args()
    if not 2 <= args.oracle_images <= args.ocf_images or args.calibration_images < 2:
        parser.error("Choose valid calibration and OCF sample sizes.")
    if not 1 <= args.dimensions <= 192 or min(args.epochs, args.encoder_epochs) < 1:
        parser.error("Choose valid model dimensions and finishing epochs.")
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle CPU with the competition data mounted.")
    torch.set_num_threads(4)
    device = torch.device("cpu")
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    masks = images >= 128
    if args.ocf_images + args.calibration_images > len(images):
        parser.error("The calibration and scored glyph subsets must fit the gallery without overlap.")
    signature, energy = contour_signatures(images)
    utility = similarity_from_signatures(signature, energy)
    utility.sub_(utility.mean(dim=1, keepdim=True))
    scorer = official_scorer(root)
    rng = np.random.default_rng(args.glyph_seed)
    identifiers = torch.as_tensor(rng.permutation(len(images)))
    calibration_ids = identifiers[:args.calibration_images]
    outer_ids = identifiers[args.calibration_images:args.calibration_images + args.ocf_images]
    output = {"seed": args.seed, "repeat_seed": args.repeat_seed, "glyph_seed": args.glyph_seed,
              "gallery_size": len(images), "ocf_images_per_block": len(outer_ids), "oracle_images_per_block": args.oracle_images,
              "calibration_images_per_block": len(calibration_ids), "classifier_weight": CLASSIFIER_WEIGHT,
              "centered_contour_utility_weight": UTILITY_WEIGHT, "candidate_budget": POOL_SIZE, "retained_v11": RETAIN,
              "encoder_epochs": args.encoder_epochs, "classifier_epochs": args.epochs, "dimensions": args.dimensions,
              "protocol": "Two repeats fit the inner scales. An independent third repeat calibrates each posterior by known-class NLL. "
                          "Refit both scoring sources on all three non-outer repeats. Disjoint glyph samples calibrate vs score OCF. "
                          "Classifier scores change only posterior probabilities and nominations. Both decision pools and posteriors use "
                          "the same v11 anchor, centered image contour utility, and normalization on the original v11 pool.",
              "mice": {}, "folds": []}
    total_values, total_oracle = {}, {}
    for mouse in dict.fromkeys(args.mice):
        recordings = grouped_blocks(root, catalog, mouse, device, len(images))
        inner_fit, inner_queries, outer_fit, held = nested_repeats(recordings, args.repeat_seed)
        del recordings
        print(f"{mouse}: fit inner models on two repeats and calibrate both posterior scales", flush=True)
        inner, inner_training = fit_models(inner_fit, images, signature, energy, mouse, args)
        samples = {"v11": [], "classifier": []}
        for session in dict.fromkeys(args.sessions):
            _, base, mixed = score_sources(inner, inner_queries[session][calibration_ids], images, session,
                                          mouse, args.dimensions)
            samples["v11"].append((base, calibration_ids))
            samples["classifier"].append((mixed, calibration_ids))
        temperatures = {name: fit_temperature(scores) for name, scores in samples.items()}
        print(json.dumps({"mouse": mouse, "temperatures": temperatures}), flush=True)
        del inner, inner_fit, inner_queries, samples

        print(f"{mouse}: refit for independent outer repeats", flush=True)
        outer, outer_training = fit_models(outer_fit, images, signature, energy, mouse, args)
        del outer_fit
        result = {"temperatures": temperatures, "inner_fit": inner_training, "outer_fit": outer_training}
        output["mice"][mouse] = result
        mouse_values, mouse_oracle = {}, {}
        for session in dict.fromkeys(args.sessions):
            elapsed = time.monotonic()
            order = torch.as_tensor(rng.permutation(len(images)))
            v11, base, mixed = score_sources(outer, held[session][order], images, session, mouse, args.dimensions)
            choices, pools, posteriors = decide(v11, base, mixed, temperatures, utility)
            measures = diagnostics(choices, pools, posteriors, order)
            exact, oracle, pairs = score_exact(choices, pools, order, outer_ids, masks, scorer, args.oracle_images)
            fold = {"mouse": mouse, "session": session, "gallery": measures, "unique_exact_pairs": pairs,
                    "paired_ocf": summarize({name: [values] for name, values in exact.items()}),
                    "oracle_ocf": summarize({name: [values] for name, values in oracle.items()}),
                    "seconds": time.monotonic() - elapsed}
            output["folds"].append(fold)
            for group, observed in ((mouse_values, exact), (mouse_oracle, oracle), (total_values, exact), (total_oracle, oracle)):
                extend(group, observed)
            matching = [fold for fold in output["folds"] if fold["mouse"] == mouse]
            result["summary"] = {"ocf": summarize(mouse_values), "oracle": summarize(mouse_oracle),
                                 "gallery": gallery_summary(matching)}
            output["summary"] = {"ocf": summarize(total_values), "oracle": summarize(total_oracle),
                                 "gallery": gallery_summary(output["folds"])}
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(fold), flush=True)
            del v11, base, mixed, posteriors
        del outer, held
    print(json.dumps({"summary": output["summary"], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
