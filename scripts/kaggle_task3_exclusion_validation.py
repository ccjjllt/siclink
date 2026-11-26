"""Validate case-specific soft one-of-each constraints on complete Task 3 blocks."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_claim_exclusion import exclusion_logits, fit_claim_calibrator
from glyph_contour_retrieval import blend_contour_poisson, contour_retrieval_scores
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits
from glyph_gain_likelihood import gain_template_logits
from glyph_mbr import contour_signatures, fit_temperature, glyph_posterior, similarity_from_signatures
from glyph_retrieval import assign_glyphs
from kaggle_diagnostics import official_scorer
from kaggle_posterior_utility_validation import fit_image_utility, fit_neural, summarize
from kaggle_task3_metric_validation import nested_repeats, unique_pair_ocf


CLAIM_THRESHOLD = 0.10
CLAIM_CAP = 1.0


def normalize(values):
    return F.layer_norm(values, (values.shape[-1],), eps=1e-8)


@torch.no_grad()
def policy_inputs(model, counts, session, images):
    encoder, contour, gain = model
    rates = encoder.expected_counts(images, session)
    direction = contour_retrieval_scores(contour, counts, session)["contour_ridge"]
    likelihood = count_template_logits(counts, rates, "timed") if gain is None else gain_template_logits(counts, rates, gain)
    v11 = blend_contour_poisson(direction, likelihood)
    first = v11.argmax(dim=1)
    top = v11.topk(2, dim=1).values
    components = tuple(normalize(scores) for scores in (direction, likelihood))
    margin = [scores.amax(dim=1) - scores.gather(1, first[:, None])[:, 0] for scores in components]
    features = torch.stack((top[:, 0], top[:, 0] - top[:, 1], *margin), dim=1)
    return v11, first, features


@torch.no_grad()
def claim_features(logits, prediction, features, temperature):
    energy = logits / temperature
    top_probability = energy.gather(1, prediction[:, None]) - energy.logsumexp(dim=1, keepdim=True)
    return torch.cat((features, top_probability), dim=1)


def credibility_report(probability, correct):
    claimed = probability >= CLAIM_THRESHOLD
    return {"queries": len(probability), "v11_top1": float(correct.float().mean()),
            "credible_claims": int(claimed.sum()), "credible_fraction": float(claimed.float().mean()),
            "claimed_mean_probability": float(probability[claimed].mean()) if bool(claimed.any()) else None,
            "claimed_precision": float(correct[claimed].float().mean()) if bool(claimed.any()) else None,
            "brier": float((probability - correct.float()).square().mean())}


def fit_claims(model, calibration, images, fit_ids, check_ids):
    chosen = torch.cat((fit_ids, check_ids))
    cutoff = len(fit_ids)
    blocks = [policy_inputs(model, counts[chosen], session, images) for session, counts in calibration.items()]
    temperature = fit_temperature([(logits[:cutoff], fit_ids) for logits, _, _ in blocks])
    feature = [claim_features(*block, temperature["temperature"]) for block in blocks]
    correct = [prediction == chosen for _, prediction, _ in blocks]
    fit_features = torch.cat([rows[:cutoff] for rows in feature])
    fit_targets = torch.cat([rows[:cutoff] for rows in correct])
    model = fit_claim_calibrator(fit_features, fit_targets)
    check_prob = model.predict(torch.cat([rows[cutoff:] for rows in feature]))
    check_true = torch.cat([rows[cutoff:] for rows in correct])
    return temperature, model, credibility_report(check_prob, check_true)


@torch.no_grad()
def block_choices(logits, first, probability, temperature, utility, class_shuffle, sampled):
    base = logits / temperature
    independent = base.softmax(dim=1)
    sinkhorn = glyph_posterior(logits, temperature, balance=True)
    corrected, charges = exclusion_logits(base, first, probability, threshold=CLAIM_THRESHOLD, max_penalty=CLAIM_CAP)
    shuffled, _ = exclusion_logits(base, class_shuffle[first], probability,
                                    threshold=CLAIM_THRESHOLD, max_penalty=CLAIM_CAP)
    variants = {"independent": independent, "sinkhorn": sinkhorn, "credible_exclusion": corrected.softmax(dim=1),
                "shuffled_exclusion": shuffled.softmax(dim=1)}
    anchor = 0.80 * normalize(logits[sampled])
    uniform = utility.mean(dim=1)
    hard = assign_glyphs(logits)
    choices = {"v11": first[sampled], "hungarian": hard[sampled]}
    for name, posterior in variants.items():
        value = posterior[sampled] @ utility.T - uniform
        choices[name] = (anchor + 0.20 * normalize(value)).argmax(dim=1)
    return choices, variants, charges, hard


def posterior_diagnostics(variants, labels):
    return {name: {"nll": float(-values.gather(1, labels[:, None]).clamp_min(1e-30).log().mean()),
                   "top1": float((values.argmax(dim=1) == labels).float().mean()),
                   "mean_top32_mass": float(values.topk(32, dim=1).values.sum(1).mean())}
            for name, values in variants.items()}


def paired_report(measurements):
    report = {name: summarize(values, measurements["v11"]) for name, values in measurements.items()}
    for name in ("sinkhorn", "credible_exclusion", "shuffled_exclusion"):
        comparison = summarize(measurements[name], measurements["independent"])
        report[name].update(delta_vs_independent=comparison["delta_vs_v11"],
                            paired_se_vs_independent=comparison["paired_glyph_se"])
    return report


def run_mouse(mouse, args, data, folds, output, save):
    root, catalog, images, masks, signature, energy, scorer, utility = data
    fit_ids, check_ids, held_ids = folds
    raw = grouped_blocks(root, catalog, mouse, images.device, len(images))
    inner_fit, inner_query, outer_fit, outer_query = nested_repeats(raw, args.repeat_seed)
    del raw
    print(f"{mouse}: fit nested v11 and calibrate probability of a credible claim", flush=True)
    inner = fit_neural(inner_fit, images, signature, energy, mouse, args.seed)
    temperature, calibrator, credibility = fit_claims(inner, inner_query, images, fit_ids, check_ids)
    del inner_fit, inner, inner_query
    record = {"temperature_fit": temperature, "inner_credibility_check": credibility, "sessions": {}}
    output["mice"][mouse] = record
    print(json.dumps({"mouse": mouse, "temperature": temperature, "calibration_check": credibility}), flush=True)
    save()
    outer = fit_neural(outer_fit, images, signature, energy, mouse, args.seed)
    del outer_fit
    measurements = {}
    device = images.device
    random = np.random.default_rng(args.order_seed)
    for session, counts in outer_query.items():
        row_order = torch.as_tensor(random.permutation(len(images)), device=device)
        class_shuffle = torch.as_tensor(random.permutation(len(images)), device=device)
        held_positions = row_order.argsort()[held_ids]
        logits, first, features = policy_inputs(outer, counts[row_order], session, images)
        estimated = calibrator.predict(claim_features(logits, first, features, temperature["temperature"]))
        choices, variants, charges, hard = block_choices(logits, first, estimated, temperature["temperature"], utility,
                                                         class_shuffle, held_positions)
        # Ground-truth row indices only enter the retrospective checks and exact scorer.
        diagnostic = credibility_report(estimated, first == row_order)
        diagnostic.update(classes_with_claim=int((charges > 0).sum()), max_class_charge=float(charges.max()))
        posterior = posterior_diagnostics(variants, row_order)
        posterior["hungarian"] = {"top1": float((hard == row_order).float().mean())}
        del logits, variants, charges
        predicted = torch.stack(tuple(choices.values()))
        exact, pairs = unique_pair_ocf(predicted, held_ids.expand_as(predicted), masks, scorer, device, 384)
        values = dict(zip(choices, exact.cpu().numpy()))
        row = paired_report({name: [value] for name, value in values.items()})
        for name, value in values.items():
            measurements.setdefault(name, []).append(value)
            row[name]["fraction_different_from_independent"] = float((choices[name] != choices["independent"]).float().mean())
        report = {"credible_claims": diagnostic, "posterior": posterior, "unique_exact_pairs": pairs, "ocf": row}
        record["sessions"][session] = report
        record["summary"] = paired_report(measurements)
        save()
        print(json.dumps({"mouse": mouse, "session": session, "result": report}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--seed", type=int, default=211, help="v11 train seed and utility image-pair seed")
    parser.add_argument("--repeat-seed", type=int, default=419)
    parser.add_argument("--glyph-seed", type=int, default=1531)
    parser.add_argument("--order-seed", type=int, default=413)
    parser.add_argument("--calibration-glyphs", type=int, default=2048)
    parser.add_argument("--credibility-check-glyphs", type=int, default=384)
    parser.add_argument("--ocf-images-per-session", type=int, default=128)
    parser.add_argument("--utility-fit-pairs", type=int, default=6400)
    parser.add_argument("--utility-held-pairs", type=int, default=1024)
    parser.add_argument("--utility", choices=("exact-fitted", "proxy"), default="exact-fitted")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_exclusion_validation.json"))
    args = parser.parse_args()
    if min(args.calibration_glyphs, args.credibility_check_glyphs, args.ocf_images_per_session) < 2:
        parser.error("Choose at least two glyphs per disjoint calibration, check, and sampled OCF fold.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the full-block Task 3 validation on Kaggle GPU.")

    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, "cuda:0")
    masks = images >= 128
    scorer = official_scorer(root)
    sizes = (args.calibration_glyphs, args.credibility_check_glyphs, args.ocf_images_per_session)
    if sum(sizes) > len(images):
        parser.error("Disjoint neural calibration, check and held glyph subsets must fit the labeled gallery.")
    order = torch.as_tensor(np.random.default_rng(args.glyph_seed).permutation(len(images)), device=images.device)
    folds = order[:sum(sizes)].split(sizes)
    signature, energy = contour_signatures(images)
    if args.utility == "exact-fitted":
        proxy, utility, geometry = fit_image_utility(images, signature, energy, scorer, args)
        del proxy
    else:
        utility = 100 * similarity_from_signatures(signature, energy)
        geometry = {"type": "unregistered directional contour proxy"}
    print(json.dumps({"utility": args.utility, "diagnostics": geometry}), flush=True)
    output = {"model_seed": args.seed, "repeat_seed": args.repeat_seed, "glyph_seed": args.glyph_seed,
              "order_seed": args.order_seed, "split_sizes": sizes, "utility": args.utility, "geometry": geometry,
              "claim_threshold": CLAIM_THRESHOLD, "claim_log_penalty_cap": CLAIM_CAP,
              "v11": {"mice1": "contour + timed Poisson", "mice2": "contour + temporal gain"},
              "protocol": "All matching uses complete randomized recording blocks. After reserving the outer trial, "
                          "two repeats fit the inner policy and a different repeat calibrates claims of exact identity. "
                          "Held validation OCF, calibration fit and calibration check use disjoint glyph cohorts. "
                          "The gallery utility uses only known image pairs. Image actions remain independent.", "mice": {}}

    def save():
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    data = root, catalog, images, masks, signature, energy, scorer, utility
    for mouse in dict.fromkeys(args.mice):
        run_mouse(mouse, args, data, folds, output, save)
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
