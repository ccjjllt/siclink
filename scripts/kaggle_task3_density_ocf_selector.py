"""Learn to convert Task 3 density nominations into exact-OCF decisions.

Both equal-budget action sets include the unchanged v11 leader. Fit expected
OCF differences against that leader on inner held repeats, with a separate
glyph cohort for posterior temperatures. Use glyph-group cross-fitted training
predictions to set the fallback threshold, then refit the spike models using
three training repetitions and evaluate on untouched outer repeats.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_contour_retrieval import _agreement
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_local_ocf_ranker import directional_tiles, local_evidence
from glyph_mbr import contour_signatures, fit_temperature
from kaggle_diagnostics import official_scorer
from kaggle_task3_metric_validation import nested_repeats, unique_pair_ocf
from kaggle_task3_part_discovery import SESSIONS, project_counts
from kaggle_task3_repeat_posterior import fit_models, score_sources


ORIGINAL = 7
FRESH = 3
BEYOND = 16
NOVEL_START = ORIGINAL
GATE_LEVELS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
PAIRED_CONTROLS = {"learned_union_positive": ("learned_original_positive", "probability_union"),
                   "learned_union_calibrated": ("learned_original_calibrated", "probability_union"),
                   "nominated": ("original",)}


@dataclass
class CandidateRows:
    labels: torch.Tensor
    features: torch.Tensor
    mixed_probability: torch.Tensor
    scope_original: torch.Tensor
    scope_nominated: torch.Tensor


def relative(values):
    return values - values[:, :1]


def pool_columns(device):
    original = torch.arange(ORIGINAL, device=device)
    nominated = torch.cat((original[:ORIGINAL - FRESH], torch.arange(ORIGINAL, ORIGINAL + FRESH, device=device)))
    return original, nominated


def inverse_rank(matrix, slots):
    order = matrix.argsort(dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(1, matrix.shape[1] + 1, device=matrix.device).expand_as(order))
    return ranks.gather(1, slots).float().log()


@torch.no_grad()
def build_candidates(models, query, images, signature, tiles, ink, energy, session, mouse, temperatures, dimensions):
    v11, base, mixed = score_sources(models, query, images, session, mouse, dimensions)
    anchor = v11.argmax(dim=1, keepdim=True)
    next_v11 = v11.clone().scatter_(1, anchor, -torch.inf).topk(BEYOND - 1, dim=1).indices
    v11_pool = torch.cat((anchor, next_v11), dim=1)
    discoveries = mixed.clone().scatter_(1, v11_pool, -torch.inf).topk(FRESH, dim=1).indices
    slots = torch.cat((v11_pool[:, :ORIGINAL], discoveries), dim=1)
    old, new = pool_columns(query.device)

    base_logprob = (base / temperatures["v11"]["temperature"]).log_softmax(dim=1).gather(1, slots)
    mix_logprob = (mixed / temperatures["classifier"]["temperature"]).log_softmax(dim=1).gather(1, slots)
    probability = mix_logprob.exp()
    v11_score = base.gather(1, slots)
    mix_score = mixed.gather(1, slots)
    base_rank = inverse_rank(base, slots)
    mix_rank = inverse_rank(mixed, slots)

    contour = models.contour
    projected = project_counts(contour, query, session).mean(dim=1)
    estimated = 0.80 * contour.readout.predict(projected) + 0.20 * contour.local[session].predict(projected)
    glyph_strength = _agreement(estimated, contour).gather(1, slots)
    regional = local_evidence(estimated[:, :-1], tiles, slots)
    # Six whole-region reductions and six central-tile means preserve direction and missing strokes.
    tile_scores = regional[..., :96].unflatten(-1, (6, 16))[..., (5, 6, 9, 10)].mean(dim=-1)
    region_delta = relative(torch.cat((regional[..., -6:], tile_scores), dim=-1))
    shape_to_anchor = (signature[slots] * signature[anchor]).sum(dim=-1) - 1.0
    is_novel = torch.zeros_like(v11_score)
    is_novel[:, NOVEL_START:] = 1
    non_anchor = (slots != anchor).float()
    anchor_certainty = probability[:, :1].clamp_min(1e-6).log() + 6
    anchor_margin = v11_score[:, :1] - v11_score[:, 1:2]
    score_delta = relative(mix_score)
    visual_delta = relative(glyph_strength)
    features = torch.stack((non_anchor, is_novel, relative(v11_score), score_delta,
                            relative(base_rank), relative(mix_rank), relative(base_logprob),
                            relative(mix_logprob), relative(probability.sqrt()), visual_delta,
                            relative(ink[slots]), relative(energy[slots].log()), shape_to_anchor,
                            anchor_margin * non_anchor, anchor_certainty * non_anchor,
                            anchor_margin * score_delta, anchor_certainty * visual_delta,
                            score_delta * visual_delta), dim=-1)
    features = torch.cat((features, region_delta), dim=-1)
    features[:, 0].zero_()
    return CandidateRows(slots, features, probability, old, new)


@dataclass
class DeltaSelector:
    scale: torch.Tensor
    weights: torch.Tensor

    def estimate(self, features):
        return (features / self.scale).clamp(-8, 8) @ self.weights


def fit_selector(features, delta):
    pairs = features[:, 1:].flatten(0, 1)
    target = delta[:, 1:].flatten()
    scale = pairs.square().mean(dim=0).sqrt().clamp_min(0.015)
    weights = (pairs / scale).clamp(-8, 8)
    count = len(weights)
    normal = weights.T @ weights / count
    normal.diagonal().add_(0.11)
    cross = (weights.T @ target / count)[:, None]
    beta = torch.cholesky_solve(cross, torch.linalg.cholesky(normal))[:, 0]
    return DeltaSelector(scale, beta)


def headroom(candidate_delta, columns):
    return candidate_delta[:, columns].amax(dim=1)


def pick(predicted, columns, level):
    gain, local = predicted[:, columns].max(dim=1)
    picked = columns[local]
    return picked.masked_fill(gain <= level, 0)


def calibrate(features, exact, glyph_count, novel_columns):
    delta = exact - exact[:, :1]
    device = features.device
    fold = torch.arange(len(exact), device=device).remainder(glyph_count).remainder(4)
    held = torch.zeros_like(exact)
    for index in range(4):
        validation = fold == index
        fit = fit_selector(features[~validation], delta[~validation])
        held[validation] = fit.estimate(features[validation])
    controls = []
    for threshold in GATE_LEVELS:
        selected = pick(held, novel_columns, threshold)
        measured = delta.gather(1, selected[:, None])[:, 0]
        controls.append({"threshold": threshold, "paired_delta": float(measured.mean()),
                         "switch_fraction": float((selected != 0).float().mean()),
                         "novel_fraction": float((selected >= NOVEL_START).float().mean())})
    # Keep exact v11 if independent inner predictions support none of the available switches.
    best = max(controls, key=lambda row: (row["paired_delta"], row["threshold"]))
    threshold = best["threshold"] if best["paired_delta"] > 0 else None
    fit = fit_selector(features, delta)
    return fit, threshold, {"pairs": int(exact.numel()), "oof_gates": controls, "chosen_threshold": threshold,
                            "oof_delta_rmse": float((held - delta).square().mean().sqrt()),
                            "oof_mean_calibration_ocf": float(exact[:, 0].mean()),
                            "oof_union_oracle_gain": float(headroom(delta, novel_columns).mean())}


def decide(candidates, selector, threshold):
    predicted = selector.estimate(candidates.features)
    original, nominated = candidates.scope_original, candidates.scope_nominated
    baseline = torch.zeros(len(predicted), dtype=torch.long, device=predicted.device)
    choices = {"v11": baseline,
               "probability_original": original[candidates.mixed_probability[:, original].argmax(dim=1)],
               "probability_union": nominated[candidates.mixed_probability[:, nominated].argmax(dim=1)],
               "learned_original_positive": pick(predicted, original, 0),
               "learned_union_positive": pick(predicted, nominated, 0),
               "learned_original_calibrated": baseline if threshold is None else pick(predicted, original, threshold),
               "learned_union_calibrated": baseline if threshold is None else pick(predicted, nominated, threshold)}
    return choices


def paired_summary(observations, baseline):
    values = np.stack(observations)
    delta = (values - np.stack(baseline)).mean(axis=0)
    return {"ocf": float(values.mean()), "delta": float(delta.mean()), "paired_se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
            "n": int(values.size)}


def summarize(observed):
    reference = "v11" if "v11" in observed else "original"
    return {name: {base: paired_summary(values, observed[base])
                   for base in (reference, *PAIRED_CONTROLS.get(name, ())) if base in observed}
            for name, values in observed.items()}


def evaluate(candidates, chosen, references, reorder, masks, scorer, depth):
    picked = torch.stack([candidates.labels.gather(1, positions[:, None])[:, 0] for positions in chosen.values()])
    picked = picked[:, reorder]
    references = references[reorder]
    oracle_ids = candidates.labels[reorder[:depth]]
    guessed = torch.cat((picked.flatten(), oracle_ids.flatten()))
    exact_ref = torch.cat((references.expand_as(picked).flatten(), references[:depth, None].expand_as(oracle_ids).flatten()))
    measured, pairs = unique_pair_ocf(guessed, exact_ref, masks, scorer, masks.device, 384)
    chosen_score, oracle = measured.split((picked.numel(), oracle_ids.numel()))
    oracle = oracle.reshape_as(oracle_ids)
    oracle = {"original": oracle[:, candidates.scope_original].amax(1).cpu().numpy(),
              "nominated": oracle[:, candidates.scope_nominated].amax(1).cpu().numpy()}
    selected = dict(zip(chosen, chosen_score.reshape(picked.shape).cpu().numpy()))
    diagnostics = {name: {"switch_fraction": float((positions != 0).float().mean()),
                          "novel_fraction": float((positions >= NOVEL_START).float().mean()),
                          "top1": float((picked[row] == references).float().mean())}
                   for row, (name, positions) in enumerate(chosen.items())}
    return selected, oracle, pairs, diagnostics


def record_results(target, observations):
    for name, value in observations.items():
        target.setdefault(name, []).append(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--sessions", nargs="+", choices=SESSIONS, default=SESSIONS)
    parser.add_argument("--calibration-images", type=int, default=80)
    parser.add_argument("--temperature-images", type=int, default=256)
    parser.add_argument("--ocf-images", type=int, default=64)
    parser.add_argument("--oracle-images", type=int, default=12)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int, default=3701)
    parser.add_argument("--glyph-seed", type=int, default=3727)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_density_ocf_selector.json"))
    args = parser.parse_args()
    if not 2 <= args.oracle_images <= args.ocf_images or min(args.calibration_images, args.temperature_images) < 4:
        parser.error("Choose valid held and calibration cohort sizes with four calibration folds.")
    if min(args.epochs, args.encoder_epochs) < 1 or not 1 <= args.dimensions <= 192:
        parser.error("Choose positive epoch counts and 1–192 stable neural dimensions.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the exact OCF selector on Kaggle GPU.")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    masks = images >= 128
    bank, energy = contour_signatures(images)
    tiles = directional_tiles(bank)
    ink = masks.float().mean(dim=(1, 2))
    scorer = official_scorer(root)
    rng = np.random.default_rng(args.glyph_seed)
    counts = (args.temperature_images, args.calibration_images, args.ocf_images)
    if sum(counts) > len(images):
        parser.error("Calibration and outer scored glyphs must be disjoint.")
    order = torch.as_tensor(rng.permutation(len(images))[:sum(counts)], device=device)
    temp_ids, inner_ids, outer_ids = order.split(counts)
    sessions = tuple(dict.fromkeys(args.sessions))
    output = {"seed": args.seed, "glyph_seed": args.glyph_seed, "repeat_seed": args.repeat_seed,
              "temperature_glyphs_per_block": len(temp_ids), "ranker_glyphs_per_block": len(inner_ids),
              "scored_glyphs_per_block": len(outer_ids), "oracle_glyphs_per_block": args.oracle_images,
              "pool": {"original": ORIGINAL, "nomination_slots": FRESH, "novel_beyond_rank": BEYOND},
              "gate_levels": GATE_LEVELS,
              "protocol": "Hold out one repeat before all fits. Fit inner spikes on two repeats, calibrate posterior and exact OCF "
                          "selector on a third repeat with disjoint glyph cohorts. Fit fallback on four glyph-group cross-fits. "
                          "Refit all spike models on the three earlier repeats. Score exact v11 and equal-budget pools on untouched repeats.",
              "mice": {}, "folds": []}
    total, oracle_total = {}, {}
    for mouse in dict.fromkeys(args.mice):
        recordings = grouped_blocks(root, catalog, mouse, device, len(images))
        first, calibration, final, held = nested_repeats(recordings, args.repeat_seed)
        del recordings
        print(f"{mouse}: fitting inner v11 and density models", flush=True)
        inner, fit_details = fit_models(first, images, bank, energy, mouse, args)
        inputs = {"v11": [], "classifier": []}
        for session in sessions:
            _, base, mixed = score_sources(inner, calibration[session][temp_ids], images, session, mouse, args.dimensions)
            inputs["v11"].append((base, temp_ids))
            inputs["classifier"].append((mixed, temp_ids))
        temperatures = {name: fit_temperature(rows) for name, rows in inputs.items()}
        packs = [build_candidates(inner, calibration[session][inner_ids], images, bank, tiles, ink, energy,
                                  session, mouse, temperatures, args.dimensions) for session in sessions]
        corpus = torch.cat([pool.labels for pool in packs])
        features = torch.cat([pool.features for pool in packs])
        exact, unique_pairs = unique_pair_ocf(corpus, inner_ids.repeat(len(sessions))[:, None].expand_as(corpus),
                                             masks, scorer, device, 384)
        selector, threshold, probe = calibrate(features, exact, len(inner_ids), packs[0].scope_nominated)
        diagnostics = {"temperatures": temperatures, "fit": fit_details, "selector": probe,
                       "unique_calibration_exact_pairs": unique_pairs}
        print(json.dumps({"mouse": mouse, "calibration": diagnostics}), flush=True)
        output["mice"][mouse] = diagnostics
        del inner, first, calibration, inputs, packs, corpus, features, exact

        print(f"{mouse}: refitting on three non-outer repeats", flush=True)
        outer, outer_fit = fit_models(final, images, bank, energy, mouse, args)
        del final
        diagnostics["outer_fit"] = outer_fit
        mouse_score, mouse_oracle = {}, {}
        for session in sessions:
            block_started = time.monotonic()
            shuffled = torch.as_tensor(rng.permutation(len(outer_ids)), device=device)
            labels = outer_ids[shuffled]
            candidates = build_candidates(outer, held[session][labels], images, bank, tiles, ink, energy,
                                          session, mouse, temperatures, args.dimensions)
            choices = decide(candidates, selector, threshold)
            measured, oracle, pairs, selection = evaluate(candidates, choices, labels, shuffled.argsort(),
                                                            masks, scorer, args.oracle_images)
            fold = {"mouse": mouse, "session": session, "exact_pairs": pairs, "selection": selection,
                    "recall": {name: float((candidates.labels[:, scope] == labels[:, None]).any(1).float().mean())
                               for name, scope in (("original", candidates.scope_original), ("nominated", candidates.scope_nominated))},
                    "ocf": summarize({name: [value] for name, value in measured.items()}),
                    "oracle": summarize({name: [value] for name, value in oracle.items()}),
                    "seconds": time.monotonic() - block_started}
            output["folds"].append(fold)
            for accumulator, observation in ((total, measured), (mouse_score, measured),
                                             (oracle_total, oracle), (mouse_oracle, oracle)):
                record_results(accumulator, observation)
            diagnostics["summary"] = {"ocf": summarize(mouse_score), "oracle": summarize(mouse_oracle)}
            output["summary"] = {"ocf": summarize(total), "oracle": summarize(oracle_total)}
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(fold), flush=True)
        del outer, held
    print(json.dumps({"summary": output["summary"], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
