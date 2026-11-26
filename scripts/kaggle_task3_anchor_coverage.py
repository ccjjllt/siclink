"""Bound the value of exact Task 3 repertoire constraints with controlled anchor coverage."""

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from glyph_contour_retrieval import blend_contour_poisson, contour_retrieval_scores
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits
from glyph_gain_likelihood import gain_template_logits
from glyph_mbr import contour_signatures, contour_utility
from kaggle_diagnostics import official_scorer
from kaggle_posterior_utility_validation import fit_neural
from kaggle_task3_measured_metric_validation import paired
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_mice2_ocf_recheck import new_cohorts, new_trial_split
from kaggle_task3_part_discovery import project_counts, standardize_rows
from kaggle_task3_rankband_ocf_validation import scale_actions
from kaggle_task3_repeat_posterior import CLASSIFIER_WEIGHT, fit_models, score_sources
from kaggle_task3_repeat_ratio import whiten
from kaggle_task3_shape_capacity_validation import REPEAT_SEED as CAPACITY_SEED, cohorts
from kaggle_task3_utility_nomination_recheck import independent_cohorts


SEED = 94549
REPEAT_SEED = 94553
TEMPERATURE = 0.5238909442826287
FRACTIONS = (0.05, 0.15, 0.30, 0.60)
EPS = 1e-20
V14 = "v14"
REFERENCE = "no_anchors_balanced"
OUTPUT = Path("/kaggle/working/task3_anchor_coverage.json")


def scoring_glyphs(size, count):
    cpu = torch.device("cpu")
    _, _, capacity = cohorts(size, cpu)
    _, _, recheck, _ = new_cohorts(size, cpu)
    nomination = independent_cohorts(size, cpu).outer
    used = torch.unique(torch.cat((capacity, recheck, nomination))).numpy()
    available = np.setdiff1d(np.arange(size), used)
    sample = np.random.default_rng(SEED).choice(available, count, replace=False)
    return torch.as_tensor(sample), len(used)


def novel_repeats(recordings, mouse):
    fake = {day: torch.arange(4).reshape(1, 4, 1).expand(len(counts), -1, -1) for day, counts in recordings.items()}
    previous_mice2 = new_trial_split(fake)[3] if mouse == "mice2" else {}
    masks = torch.arange(4)[None]
    fits, queries, history = {}, {}, {}
    for index, (day, counts) in enumerate(recordings.items()):
        old = np.random.default_rng(CAPACITY_SEED + index).integers(0, 4, len(counts))
        forbidden = masks == torch.as_tensor(old)[:, None]
        if day in previous_mice2:
            forbidden |= masks == previous_mice2[day].reshape(-1, 1)
        random = torch.from_numpy(np.random.default_rng(REPEAT_SEED + index).random((len(counts), 4)))
        selected = random.masked_fill(forbidden, 2).argmin(dim=1)
        support = masks.expand(len(counts), -1)[masks != selected[:, None]].reshape(len(counts), 3)
        fits[day] = counts.gather(1, support[:, :, None].expand(-1, -1, counts.shape[-1]))
        queries[day] = counts.gather(1, selected[:, None, None].expand(-1, -1, counts.shape[-1]))
        history[day] = {"same_as_prior_capacity_held": int((selected == torch.as_tensor(old)).sum()),
                        "same_as_prior_mice2_recheck_held": int((selected == previous_mice2[day].reshape(-1)).sum())
                        if day in previous_mice2 else None}
    return fits, queries, history


@torch.no_grad()
def panel(model, query, gallery, day, mouse, settings):
    if mouse == "mice1":
        encoder, contour, gain = model.encoder, model.contour, None
    else:
        encoder, contour, gain = model
    rates = encoder.expected_counts(gallery, day)
    logits, candidates = [], []
    for batch in query.split(128):
        directional = contour_retrieval_scores(contour, batch, day)["contour_ridge"]
        spike = count_template_logits(batch, rates, "timed") if gain is None else gain_template_logits(batch, rates, gain)
        v11 = blend_contour_poisson(directional, spike)
        anchor = v11.argmax(dim=1, keepdim=True)
        tail = v11.clone().scatter_(1, anchor, -torch.inf).topk(15, dim=1).indices[:, :6]
        candidates.append(torch.cat((anchor, tail), dim=1))
        if mouse == "mice1":
            projected = project_counts(contour, batch, day).mean(dim=1)[:, -settings.dimensions:]
            response = whiten(projected - model.centers[day], model.factor)
            novel = model.classifier(response, model.templates[day])
            mixed = standardize_rows(v11).lerp(standardize_rows(novel), CLASSIFIER_WEIGHT)
            logits.append(standardize_rows(mixed))
        else:
            logits.append(v11)
    scores, actions = torch.cat(logits), torch.cat(candidates)
    # Verify cached scoring on the same fit before any anchor is chosen.
    if mouse == "mice1":
        _, _, literal = score_sources(model, query[:16], gallery, day, mouse, settings.dimensions)
        if not torch.allclose(scores[:16], literal, atol=2e-5, rtol=1e-5):
            raise RuntimeError("Cached v14 neural scores differ from the original scoring function.")
    return scores, actions


def confounded_columns(ink, mass, rng):
    cut = ink.new_tensor((0.25, 0.5, 0.75))
    strata = 4 * torch.bucketize(ink, torch.quantile(ink, cut)) + torch.bucketize(mass, torch.quantile(mass, cut))
    shuffled = torch.arange(len(ink))
    for key in strata.unique():
        rows = torch.where(strata == key)[0]
        order = rows[torch.as_tensor(rng.permutation(len(rows)))]
        shuffled[order] = order.roll(1)
    return shuffled


def automated_anchors(logits, rows):
    assignment_rows, labels = linear_sum_assignment(-logits[rows].numpy())
    if not np.array_equal(assignment_rows, np.arange(len(rows))):
        raise RuntimeError("The automatic anchor fit must assign one distinct code to every selected row.")
    return torch.as_tensor(labels)


def empirical_claims(logits, mass, truth):
    predicted = logits.argmax(dim=1)
    probability = mass.gather(1, predicted[:, None])[:, 0]
    rows = probability.argsort(descending=True)
    winners, used = [], set()
    for row in rows[(probability[rows] >= 0.10)].tolist():
        label = int(predicted[row])
        if label not in used:
            winners.append(row)
            used.add(label)
    selected = torch.as_tensor(winners, dtype=torch.long)
    correct = (predicted[selected] == truth[selected]).sum()
    return {"softmax_threshold": 0.10, "unique_claim_count": len(winners), "claimed_fraction": len(winners) / len(logits),
            "correct_claim_count": int(correct), "measured_precision": float(correct / len(winners)) if winners else None,
            "correct_claim_coverage": int(correct) / len(logits)}


@torch.no_grad()
def balance_weights(weights, iterations=24):
    right = torch.ones(weights.shape[1])
    for _ in range(iterations):
        left = 1 / (weights @ right).clamp_min(EPS)
        right = 1 / (weights.T @ left).clamp_min(EPS)
        right /= right.mean().clamp_min(EPS)
    scaled = weights * right
    return scaled / scaled.sum(1, keepdim=True).clamp_min(EPS)


def true_metrics(probability, labels, available):
    atlas_to_local = torch.full((len(available),), -1, dtype=torch.long)
    atlas_to_local[available] = torch.arange(int(available.sum()))
    expected = atlas_to_local[labels]
    ranks = probability.topk(32, dim=1).indices
    possible = expected >= 0
    mass = probability.gather(1, expected[:, None].clamp_min(0))[:, 0].masked_fill(~possible, 0)
    nll = -mass.clamp_min(1e-12).log()
    return {"queries": len(probability), "true_label_available": float(possible.float().mean()),
            "top1": float((ranks[:, 0] == expected).float().mean()),
            "top32": float((ranks == expected[:, None]).any(1).float().mean()),
            "nll_fixed_floor": float(nll.mean()), "nll_conditional_on_available": float(nll[possible].mean()) if bool(possible.any()) else None}


def action_utility(actions, signatures, energies):
    indices = actions.flatten()
    flat = 100 * contour_utility(signatures[indices], energies[indices], signatures, energies)
    flat.scatter_(1, indices[:, None], 100.0)
    return flat.reshape(*actions.shape, len(signatures))


def action_choices(action, logits, probabilities, utility, available):
    evidence = torch.einsum("qk,qak->qa", probabilities, utility[:, :, available]) - utility.mean(dim=-1)
    score = 0.80 * scale_actions(logits, 0.30) + 0.20 * scale_actions(evidence, 0.50)
    return action.gather(1, score.argmax(1, keepdim=True))[:, 0]


def evaluated_probabilities(matrix, rows, cols, truth, scored, *, rebalance):
    remaining = torch.where(rows)[0]
    local = torch.searchsorted(remaining, scored)
    weights = matrix[rows][:, cols]
    probability = balance_weights(weights) if rebalance else weights / weights.sum(1, keepdim=True)
    p = probability[local]
    detail = {"all_remaining": true_metrics(probability, truth[remaining], cols),
              "fixed_scored_rows": true_metrics(p, truth[scored], cols),
              "mean_absolute_column_error": float((probability.sum(dim=0) - 1).abs().mean())}
    return p, detail


@torch.no_grad()
def analyze(logits, atlas_choices, truth, scored, signature, energy, ink, rng):
    count = len(logits)
    full = torch.ones(count, dtype=torch.bool)
    exponent = logits / TEMPERATURE
    weights = (exponent - exponent.amax(dim=1, keepdim=True)).exp().clamp_min_(EPS)
    independent = weights / weights.sum(dim=1, keepdim=True)
    leaders = logits.topk(2, dim=1).values
    order = (leaders[:, 0] - leaders[:, 1]).argsort(descending=True)
    protected = full.clone()
    protected[scored] = False
    ordered_anchors = order[protected[order]]
    wrong_columns = confounded_columns(ink, independent.sum(dim=0).contiguous(), rng)
    actions = atlas_choices[scored]
    action_logits = logits[scored].gather(1, actions)
    shape = action_utility(actions, signature, energy)
    choices = {V14: actions.gather(1, action_logits.argmax(dim=1, keepdim=True))[:, 0]}
    all_metrics = {"v14_independent_identity": {"all_remaining": true_metrics(independent, truth, full),
                                               "fixed_scored_rows": true_metrics(independent[scored], truth[scored], full)}}
    claims = empirical_claims(logits, independent, truth)
    choices["no_anchors_independent"] = action_choices(actions, action_logits, independent[scored], shape, full)
    probability, all_metrics[REFERENCE] = evaluated_probabilities(weights, full, full, truth, scored, rebalance=True)
    choices[REFERENCE] = action_choices(actions, action_logits, probability, shape, full)

    cohorts = {}
    for fraction in FRACTIONS:
        name = f"anchor{round(100 * fraction):02d}"
        anchor = ordered_anchors[:round(fraction * count)]
        rows = full.clone()
        rows[anchor] = False
        true_labels = truth[anchor]
        known = {"true": true_labels, "matched_shuffle": wrong_columns[true_labels],
                 "automatic": automated_anchors(logits, anchor)}
        if torch.equal(known["matched_shuffle"].sort().values, true_labels.sort().values):
            raise ValueError("The shuffled condition must actually remove a different set of columns.")
        cohorts[name] = {"anchor_rows": len(anchor), "anchor_fraction": len(anchor) / count, "variants": {}}
        # Compare to unanchored logits on exactly the same pool of unanchored observations.
        cohorts[name]["reference_independent_same_remaining_rows"] = true_metrics(independent[rows], truth[rows], full)
        for kind, columns in known.items():
            available = full.clone()
            available[columns] = False
            caption = f"{name}_{kind}"
            p, diagnostic = evaluated_probabilities(weights, rows, available, truth, scored, rebalance=True)
            choices[caption] = action_choices(actions, action_logits, p, shape, available)
            cohorts[name]["variants"][kind] = {"predicted_anchor_precision": float((columns == true_labels).float().mean()),
                                               "removed_column_overlap_with_true": float(torch.isin(columns, true_labels).float().mean()),
                                               **diagnostic}
            if kind == "true":
                p, prune = evaluated_probabilities(weights, rows, available, truth, scored, rebalance=False)
                choices[caption + "_prune_only"] = action_choices(actions, action_logits, p, shape, available)
                cohorts[name]["variants"]["true_prune_only"] = prune
    return choices, all_metrics, cohorts, claims


def score_choices(choices, actions, target, masks, scorer, oracle_size):
    prediction = torch.stack(tuple(choices.values()))
    oracle = actions[:oracle_size]
    pair_indices = torch.cat((prediction.flatten(), oracle.flatten()))
    labels = torch.cat((target.expand_as(prediction).flatten(), target[:oracle_size, None].expand_as(oracle).flatten()))
    score, pairs = unique_pair_ocf(pair_indices, labels, masks, scorer, masks.device, 384)
    measured, upper = score.split((prediction.numel(), oracle.numel()))
    values = dict(zip(choices, measured.reshape_as(prediction).numpy(), strict=True))
    best = upper.reshape_as(oracle).amax(dim=1).numpy()
    oracle_report = {**paired(best, values[V14][:oracle_size]), "oracle_glyphs": oracle_size}
    return values, oracle_report, pairs


def paired_report(observations):
    arrays = {name: np.stack(samples) for name, samples in observations.items()}
    result = {name: paired(values, arrays[V14]) for name, values in arrays.items()}
    for fraction in FRACTIONS:
        label = f"anchor{round(100 * fraction):02d}_true"
        result[label].update(vs_no_anchors_balanced=paired(arrays[label], arrays[REFERENCE]),
                             vs_true_prune_only=paired(arrays[label], arrays[label + "_prune_only"]))
    return result


def save(report, path, started, stage):
    report.update(seconds=time.monotonic() - started, stage=stage)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": stage, "seconds": report["seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--sessions", nargs="+", choices=("session1", "session2", "session3", "session4"), default=("session1", "session3"))
    parser.add_argument("--ocf-rows", type=int, default=128)
    parser.add_argument("--oracle-rows", type=int, default=32)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir() or not 2 <= args.oracle_rows <= args.ocf_rows <= 256:
        parser.error("Run on Kaggle CPU and use a small exact OCF cohort.")
    torch.set_num_threads(4)
    started = time.monotonic()
    cpu = torch.device("cpu")
    settings = SimpleNamespace(seed=211, encoder_epochs=18, epochs=8, dimensions=128)
    root = locate_data()
    catalog = load_catalog(root)
    gallery = load_images(root, cpu)
    masks = gallery >= 128
    scorer = official_scorer(root)
    signature, energy = contour_signatures(gallery)
    target, excluded = scoring_glyphs(len(gallery), args.ocf_rows)
    ink = masks.float().mean(dim=(1, 2))
    report = {"seed": SEED, "repeat_seed": REPEAT_SEED, "gallery_size": len(gallery), "ocf_glyphs_per_block": len(target),
              "older_eval_glyphs_excluded_from_ocf_sample": excluded, "anchor_fractions": FRACTIONS,
              "contour_utility_scale": "0 to 100, as in the fixed 0.50 OCF normalization floor",
              "frozen_temperature": TEMPERATURE, "frozen_temperature_source": "Task 3 v14 shape capacity prior labeled cohorts",
              "protocol": "Every block contains one independent held repeat of every known glyph, in a distinct random row order. "
                          "The protected exact-score queries are never locked. Each fraction removes the same high-confidence rows "
                          "for three controls, paired true columns, a prevalence-and-ink matched permutation of those columns, or "
                          "columns determined by unlabelled rectangular assignment. The true columns are diagnostic only. A separate "
                          "oracle pruning arm removes only rows and columns and omits column balancing, isolating elimination. "
                          "Remaining identity scores span the full residual gallery and NLL uses a fixed 1e-12 floor for excluded "
                          "true labels. Exact OCF uses the frozen v14 top-seven actions with a lightweight directional contour "
                          "utility proxy. Output images may repeat and may be chosen even when that glyph is an anchored identity. "
                          "The mouse2 v14 baseline itself is the full gallery MAP, which is always inside its diagnostic top seven. "
                          "No formal test inputs were used.", "mice": {}}
    combined = {}
    rng = np.random.default_rng(SEED + 1)
    save(report, args.output, started, "ready")
    for mouse in dict.fromkeys(args.mice):
        counts = grouped_blocks(root, catalog, mouse, cpu, len(gallery))
        support, held, history = novel_repeats(counts, mouse)
        del counts
        if mouse == "mice1":
            model, fit = fit_models(support, gallery, signature, energy, mouse, settings)
        else:
            model, fit = fit_neural(support, gallery, signature, energy, mouse, settings.seed), {"v14": "contour plus temporal group gain"}
        del support
        report["mice"][mouse] = {"fit": fit, "trial_exclusion_checks": history, "recordings": {}}
        save(report, args.output, started, mouse + "_ready")
        measured = {}
        for day in dict.fromkeys(args.sessions):
            truth = torch.as_tensor(rng.permutation(len(gallery)))
            protected = truth.argsort()[target]
            logits, actions = panel(model, held[day][truth], gallery, day, mouse, settings)
            choices, full_gallery, anchors, claims = analyze(logits, actions, truth, protected, signature, energy, ink, rng)
            values, image_oracle, unique = score_choices(choices, actions[protected], target, masks, scorer, args.oracle_rows)
            record = {"identity_without_anchors": full_gallery, "anchor_curves": anchors, "automatic_claims": claims,
                      "ocf": paired_report({name: [value] for name, value in values.items()}),
                      "top7_exact_oracle": image_oracle, "unique_exact_pairs": unique}
            report["mice"][mouse]["recordings"][day] = record
            for name, scores in values.items():
                measured.setdefault(name, []).append(scores)
                combined.setdefault(name, []).append(scores)
            report["mice"][mouse]["pooled_ocf"] = paired_report(measured)
            report["ocf_across_recordings_clustered_by_glyph"] = paired_report(combined)
            save(report, args.output, started, mouse + "_" + day)
        del model, held
    save(report, args.output, started, "complete")


if __name__ == "__main__":
    main()
