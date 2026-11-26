"""Compare matched subfield encoders on measured near-glyph differences and open Task 1 retrieval."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_atlas_metric_ranker import describe_atlas_candidates, pack_atlas_candidates
from glyph_bipartite_encoding import ARCHITECTURES, INK_TOLERANCE, PAIR_WEIGHT, fit_near_encoder, neighbors, timing_view, training_inputs
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import _image_tensor, count_template_logits
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_gallery_validation import image_descriptors, standard_error, standardized
from kaggle_mice2_v9_metric_validation import outer_split
from kaggle_task1_crossfit_candidate import fit_crossfit_head
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE
from kaggle_task1_peer_transport import choose_split as last_peer_split
from kaggle_task1_v11_repeat_weight_transfer import matched_pool, normalized_sources, weighted_fusion
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, nested_ranker, rendered_bank, summarize


OUTER_GLYPHS = 320
MOUSE = "mice1"
V14 = "v14"
DEPTHS = (1, 5, 16, 32)
PAIR_HISTORY = (961, 2209, 613, 3307, 8723, 5573, 6379)
OUTPUT = Path("/kaggle/working/task1_bipartite_validation.json")


def split_whole(size, seed, device):
    cpu = torch.device("cpu")
    obsolete = [outer_split(size, old, PAIR_HISTORY[:index], cpu)[1].numpy() for index, old in enumerate(PAIR_HISTORY[-2:], len(PAIR_HISTORY) - 2)]
    obsolete.append(last_peer_split(size, cpu)[1].numpy())
    excluded = np.unique(np.concatenate(obsolete))
    eligible = np.setdiff1d(np.arange(size), excluded)
    held = np.sort(np.random.default_rng(seed).choice(eligible, size=OUTER_GLYPHS, replace=False))
    fit = np.setdiff1d(np.arange(size), held)
    return torch.as_tensor(fit, device=device), torch.as_tensor(held, device=device), len(excluded)


@torch.no_grad()
def evaluate_pairs(images, seed):
    nearest, ink = neighbors(images)
    rows = nearest.cpu().tolist()
    candidates = np.random.default_rng(seed).permutation(len(rows))
    used, pairs = set(), []
    for first in candidates:
        if first in used:
            continue
        second = next((near for near in rows[first] if near not in used), None)
        if second is not None:
            used.update((first, second))
            pairs.append((first, second))
    if len(pairs) < 100:
        raise ValueError("Expected at least 100 disjoint real-glyph near pairs.")
    pair_index = torch.as_tensor(pairs, device=images.device)
    length = (ink[pair_index[:, 0]] - ink[pair_index[:, 1]]).abs()
    pair_summary = {"n_disjoint_pairs": len(pairs), "mean_absolute_ink_difference": float(length.mean()),
                    "fraction_within_ink_tolerance": float((length <= INK_TOLERANCE).float().mean())}
    return pair_index, pair_summary


def held_response_halves(queries, seed):
    generator = torch.Generator(device=next(iter(queries.values())).device).manual_seed(seed)
    result = {}
    for day, observed in queries.items():
        reordered = torch.rand(observed.shape[:2], generator=generator, device=observed.device).argsort(dim=1)
        indexed = observed.gather(1, reordered[:, :, None].expand_as(observed))
        result[day] = indexed.unflatten(1, (2, 2)).mean(dim=2).unflatten(-1, (-1, 10))
    return result


@torch.no_grad()
def rates_all(model, images, batch=96):
    predictions = []
    for sample in images.split(batch):
        value = _image_tensor(sample, images.device)
        with torch.autocast(device_type="cuda"):
            drive = model.visual_drive(model.image_features(value))
        rate = F.softplus(model._shared_and_sessions(drive.float())).add(1e-6)
        predictions.append(rate.flatten(-2).to(torch.float16))
    return torch.cat(predictions, dim=1)


@torch.no_grad()
def mechanistic(rates, observed, scales, pairs):
    first, second = pairs.unbind(dim=1)
    results, raw = {}, {}
    for day_index, (day, halves) in enumerate(observed.items()):
        expected = rates[day_index].float().unflatten(-1, (-1, 10))
        whitened = timing_view(expected) / scales[day_index]
        measured = timing_view(halves) / scales[day_index]
        prediction = whitened[first] - whitened[second]
        difference = measured[first] - measured[second]
        cosine = F.cosine_similarity(prediction[:, None], difference, dim=-1)
        aligned = (prediction[:, None] * difference).sum(dim=-1) / prediction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        ceiling = (difference[:, 0] * difference[:, 1]).sum(dim=-1)
        swapped = halves[first, :, :, 2:] - halves[second, :, :, 2:]
        model_direction = (expected[first, :, 2:].clamp_min(1e-6).log()
                           - expected[second, :, 2:].clamp_min(1e-6).log())
        preference = (swapped * model_direction[:, None]).sum(dim=(-1, -2))
        paired = {"signed_cosine": cosine.mean(1), "signed_projection": aligned.mean(1),
                  "correct_two_choice": (preference > 0).float().mean(1), "joint_log_ratio_nats_per_trial_pair": preference.mean(1),
                  "both_independent_halves_correct": (preference > 0).all(1).float()}
        raw[day] = {key: value.cpu().numpy() for key, value in paired.items()}
        results[day] = {key: {"mean": float(value.mean()), "pair_se": standard_error(value)} for key, value in raw[day].items()}
        results[day]["actual_cross_half_delta_inner_product"] = float(ceiling.mean())
    return results, raw


def compare_mechanisms(raw):
    candidate = raw["bipartite_real"]
    result = {}
    for control, by_day in raw.items():
        if control == "bipartite_real":
            continue
        entries = {}
        for metric in candidate[SESSIONS[0]]:
            change = np.mean([candidate[day][metric] - by_day[day][metric] for day in SESSIONS], axis=0)
            entries[metric] = {"delta": float(change.mean()), "paired_se": standard_error(change), "independent_pairs": len(change)}
        result[control] = entries
    return result


@torch.no_grad()
def replace_poisson(model, stock, query, rates):
    new_poisson = {f"poisson_{form}": count_template_logits(query, rates, form) for form in ("rate", "timed")}
    updated = {**stock, **new_poisson}
    return {**updated, **weighted_fusion(model, normalized_sources(updated), 1.0)}


@torch.no_grad()
def image_proxy(model, images):
    reference = image_descriptors((images >= 128).to(torch.uint8) * 255)
    paired = reference @ model.atlas.contours.T
    leading, proxy = paired.topk(2, dim=1)
    trusted = (leading[:, 0] >= 0.65) & (leading[:, 0] - leading[:, 1] >= 0.02)
    return proxy[:, 0], trusted


def fraction(found, trusted):
    return {"all": float(found.float().mean()), "high_margin": float(found[trusted].float().mean()) if bool(trusted.any()) else None}


@torch.no_grad()
def recall(scores, candidates, proxy, trusted):
    take = scores["v8"].topk(max(DEPTHS), dim=1).indices
    target = proxy[:, None]
    result = {f"top_{depth}": fraction((take[:, :depth] == target).any(dim=1), trusted) for depth in DEPTHS}
    result["exact_matched_union"] = fraction(((candidates.indices == target) & candidates.valid).any(dim=1), trusted)
    return result


@torch.no_grad()
def session_predictions(model, ranker, noise, query, day, alternate_rates, proxy, trusted):
    stock = atlas_scores(model, noise, query, day)
    original = pack_atlas_candidates(stock, NOMINATIONS, baseline="v8")
    chosen = {V14: ranker.select(original, anchor_ocf=ANCHOR)}
    detail = {V14: {"recall": recall(stock, original, proxy, trusted)}, "median_row_candidate_count": float(original.valid.sum(1).float().median()),
              "high_margin_proxy_glyphs": int(trusted.sum())}
    for arm, rates in alternate_rates.items():
        updated = replace_poisson(model, stock, query, rates)
        proposed = pack_atlas_candidates(updated, NOMINATIONS, baseline=original.baseline)
        ids, support, novel = matched_pool(original, proposed)
        fixed = describe_atlas_candidates(updated, NOMINATIONS, original.baseline, original.rank_depth, original.indices, original.valid)
        rebuilt = describe_atlas_candidates(updated, NOMINATIONS, original.baseline, original.rank_depth, ids, support)
        chosen[arm + "_fixed_graph"] = ranker.select(fixed, anchor_ocf=ANCHOR)
        chosen[arm + "_equal_budget"] = ranker.select(rebuilt, anchor_ocf=ANCHOR)
        detail[arm] = {"recall": recall(updated, rebuilt, proxy, trusted), "mean_novel_candidates": float(novel.float().mean()),
                       "rows_with_novel_candidates": int((novel > 0).sum()), "equal_candidate_count_each_row": True,
                       "changes_only_rescoring": int((chosen[arm + "_fixed_graph"] != chosen[V14]).sum()),
                       "changes_allowing_renomination": int((chosen[arm + "_equal_budget"] != chosen[V14]).sum())}
    return chosen, detail


def publish(report, file, start, stage):
    report["seconds"] = time.monotonic() - start
    report["stage"] = stage
    file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": stage, "seconds": report["seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=36467)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run this experiment on one Kaggle GPU with the labeled competition data.")
    if not 16 <= args.ocf_images <= OUTER_GLYPHS or args.epochs < 1:
        parser.error("Select a valid subset of the common OCF cohort and a positive training budget.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, held, excluded = split_whole(len(images), args.seed, device)
    groups = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if tuple(groups) != SESSIONS or any(value.shape[1] != 4 for value in groups.values()):
        raise ValueError("Expected four days and four measured labeled repeats for every glyph.")
    training = {day: value[fit] for day, value in groups.items()}
    queries = {day: value[held] for day, value in groups.items()}
    del groups
    true_training, true_held = images[fit], images[held]
    inputs = training_inputs(training, true_training)
    pair_ids, pair_report = evaluate_pairs(true_held, args.seed + 1)
    halves = held_response_halves(queries, args.seed + 2)
    report = {"mouse": MOUSE, "seed": args.seed, "model_seed": MODEL_SEED, "labeled_fit_glyphs": len(fit),
              "whole_held_glyphs": len(held), "older_near_pair_or_peer_evaluation_glyphs_excluded": excluded,
              "pair_weight": PAIR_WEIGHT, "pair_ink_tolerance": INK_TOLERANCE, "held_pairs": pair_report,
              "near_encoding": {}, "same_row_budget": NOMINATIONS, "frozen_v14_anchor_ocf": ANCHOR,
              "labeled_repeat_count": 4, "ocf_glyphs_per_day": args.ocf_images,
              "protocol": "Whole glyphs outside every fit. The registered low-frequency and translation-tolerant high-frequency "
                          "fields are compared to an equal-parameter, equal-initialization control with orthogonally mixed inputs. "
                          "The shuffled arm keeps pairs and norms while randomizing their measured response direction. "
                          "Separate held response halves assess signed real-glyph contrast and jointly assigned two-choice likelihood. "
                          "True held images are only diagnostic pair/proxy references and final OCF labels. Open decisions see only "
                          "held spikes. Retain the old uncertainty, contour, public prior and v14 ranker. The fixed and exact equal-" 
                          "budget experiments replace the old two Poisson views. Atlas identity recall targets an image-nearest public "
                          "codepoint proxy because the competition does not release identity strings. No formal tests are accessed.",
              "sessions": {}}
    fitted, mechanism_data = {}, {}
    for arm in ARCHITECTURES:
        print(f"Fit {arm} on {len(fit):,} real training shapes", flush=True)
        head, diagnostics = fit_near_encoder(inputs, arm, epochs=args.epochs, seed=MODEL_SEED)
        fitted[arm] = head
        detail, mechanism_data[arm] = mechanistic(rates_all(head, true_held), halves, inputs.noise, pair_ids)
        report["near_encoding"][arm] = {"fit": diagnostics, "held": detail}
        publish(report, args.output, started, arm + "_mechanism")
    if len({value["fit"]["parameters"] for value in report["near_encoding"].values()}) != 1:
        raise ValueError("All three new arms must have identical trainable parameter counts.")
    scales = inputs.noise
    del inputs

    scorer = official_scorer(root)
    stock_split = split_calibration(len(fit), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    _, noise, nested, order = nested_ranker(training, true_training, stock_split, scorer, MODEL_SEED)
    ranker, folds = fit_crossfit_head(training, true_training, scorer, order)
    model = fit_v8(training, true_training, MODEL_SEED)
    del training
    if not torch.equal(model.codepoints.cpu(), order):
        raise ValueError("Every model must use the identical publicly rendered atlas ordering.")
    detail, mechanism_data[V14] = mechanistic(rates_all(model.encoder, true_held), halves, scales, pair_ids)
    report["near_encoding"][V14] = {"held": detail, "calibration_folds": folds, "nested_encoder": nested,
                                   "outer_encoder": model.report, "ranker": ranker.diagnostics}
    report["matched_mechanistic_comparisons_across_days"] = compare_mechanisms(mechanism_data)
    publish(report, args.output, started, "stock_ready")

    candidate_rates = {arm: rates_all(head, model.atlas.images) for arm, head in fitted.items()}
    rendered = rendered_bank(model)
    target_proxy, trusted_proxy = image_proxy(model, true_held)
    score_rows = torch.as_tensor(np.random.default_rng(args.seed + 3).choice(len(held), args.ocf_images, replace=False), device=device)
    exact = PairedOCF(scorer, rendered, (true_held[score_rows] >= 128).cpu().numpy(), device, 12)
    scores_by_arm = {}
    for index, (day, query) in enumerate(queries.items()):
        per_day_rates = {arm: bank[index].float() for arm, bank in candidate_rates.items()}
        chosen, detail = session_predictions(model, ranker, noise[day], query, day, per_day_rates, target_proxy, trusted_proxy)
        exact_scores = {name: exact.evaluate(indices[score_rows, None])[:, 0] for name, indices in chosen.items()}
        detail["exact_paired_ocf"] = {name: summarize([value], [exact_scores[V14]]) for name, value in exact_scores.items()}
        report["sessions"][day] = detail
        for name, values in exact_scores.items():
            scores_by_arm.setdefault(name, []).append(values)
        report["exact_ocf_across_days_glyph_clustered"] = {name: summarize(values, scores_by_arm[V14]) for name, values in scores_by_arm.items()}
        report["exact_bipartite_real_vs_matched_controls"] = {control + mode: summarize(scores_by_arm["bipartite_real" + mode], scores_by_arm[control + mode])
                                                                for control in ("mixed_real", "bipartite_shuffled")
                                                                for mode in ("_fixed_graph", "_equal_budget")}
        report["unique_exact_evaluation_pairs"] = len(exact.values)
        publish(report, args.output, started, day)
    publish(report, args.output, started, "complete")


if __name__ == "__main__":
    main()
