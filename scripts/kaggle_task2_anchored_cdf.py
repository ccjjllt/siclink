"""Test CDF assignment with a fixed reward for retaining an incumbent glyph."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_data import as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import blend_hierarchical_logits, fit_hierarchical_gallery
from glyph_retrieval import assign_glyphs, combine_logits, fit_retrieval, retrieval_score_components
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_task2_constrained_ceiling import fresh_samples as ceiling_cohorts
from kaggle_task2_native_hanzi import SEED as HANZI_SEED, scoring_glyphs as hanzi_cohorts
from kaggle_task2_peer_mouse import logit_scale, row_diagnostics
from kaggle_task2_rank_marginals import (CEILING_CROWDED, EPOCHS, MODEL_SEED, MOUSE, V9, evaluation_glyphs,
                                         normal_scores, responses, source_cdf, summarize, whiten)
from kaggle_task2_rank_marginals_recheck import PROVENANCE, evaluation_sample as prior_recheck, fit_marginal_templates
from kaggle_v8_metric_validation import SESSIONS


SEED, OCF_GLYPHS = 286249, 160
FOLLOWUP_SEED = 286271
STAY = 3.0
ADAPTED, FROZEN = "target_cdf_anchored", "frozen_source_cdf_anchored"
HALF_STAY = "target_cdf_anchored_stay_0p5"
OUTPUT = Path("/kaggle/working/task2_anchored_cdf.json")


def evaluation_sample(size, *, followup=False):
    primary, _ = evaluation_glyphs(size)
    repeated, _ = prior_recheck(size)
    ceiling, _, _ = ceiling_cohorts(size)
    hanzi, _ = hanzi_cohorts(size, np.random.default_rng(HANZI_SEED))
    older = {"rank_marginals_primary": primary.numpy(), "three_source_rank_recheck": repeated.numpy(),
             "exact_ceiling_uniform": np.concatenate(tuple(ceiling.values())), "exact_ceiling_crowded": np.asarray(CEILING_CROWDED),
             "native_hanzi_scoring": np.concatenate(tuple(hanzi.values()))}
    excluded = np.unique(np.concatenate(tuple(older.values())))
    eligible = np.setdiff1d(np.arange(size), excluded)
    if followup:
        older["anchored_stay_3"] = np.random.default_rng(SEED).choice(eligible, OCF_GLYPHS, replace=False)
        eligible = np.setdiff1d(eligible, older["anchored_stay_3"])
    if len(eligible) < OCF_GLYPHS:
        raise ValueError("The fixed previous cohorts must leave enough independent scored glyphs.")
    seed = FOLLOWUP_SEED if followup else SEED
    chosen = np.sort(np.random.default_rng(seed).choice(eligible, OCF_GLYPHS, replace=False))
    overlap = {name: int(np.intersect1d(chosen, rows).size) for name, rows in older.items()}
    return torch.from_numpy(chosen), {"eligible": len(eligible), "overlap_by_cohort": overlap}


def anchored_assignment(logits, incumbent, *, stay=STAY):
    """Optimize normalized rank evidence plus the same fixed reward at every incumbent edge."""
    if (incumbent.ndim != 1 or logits.shape != (len(incumbent), len(incumbent))
            or incumbent.unique().numel() != len(incumbent) or bool(((incumbent < 0) | (incumbent >= len(incumbent))).any())):
        raise ValueError("The initial assignment must contain one of every glyph in the square score matrix.")
    objective = logits.clone()
    objective[torch.arange(len(incumbent), device=incumbent.device), incumbent] += stay
    return assign_glyphs(objective)


def rank_scores(query, templates):
    _, gallery, factor, train_reference = templates
    raw = responses(query)
    penalty = 0.5 * gallery.square().sum(1)
    return {name: logit_scale(whiten(normal_scores(raw, reference), factor) @ gallery.T - penalty)
            for name, reference in ((ADAPTED, source_cdf(raw)), (FROZEN, train_reference))}


def publish(report, phase, started, output):
    report.update(stage=phase, seconds=time.monotonic() - started)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"stage": phase, "seconds": report["seconds"]}), flush=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("stay-3", "stay-1"), default="stay-3",
                        help="Use the original +3 control or the predeclared +1 follow-up with +0.5 sensitivity")
    args = parser.parse_args()
    followup = args.profile == "stay-1"
    output = OUTPUT.with_name("task2_anchored_cdf_followup.json") if followup else OUTPUT
    primary_stay = 1.0 if followup else STAY
    profiles = {ADAPTED: (ADAPTED, primary_stay), FROZEN: (FROZEN, primary_stay)}
    if followup:
        profiles[HALF_STAY] = (ADAPTED, 0.5)
    if not output.parent.is_dir():
        raise RuntimeError("Use a private Kaggle CPU notebook with the official training files.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    if len(images) != 2748:
        raise ValueError("Expected the full known gallery.")
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if tuple(recordings) != SESSIONS or any(counts.shape[:2] != (len(images), 4) for counts in recordings.values()):
        raise ValueError("Expected four complete labeled recordings, each with four measured repeats per glyph.")
    sample, cohort = evaluation_sample(len(images), followup=followup)
    rng = np.random.default_rng(SEED + 1)
    metric = PairedOCF(official_scorer(root), images, (images[sample] >= 128).numpy(), device, 12)
    report = {"mouse": MOUSE, "seed": FOLLOWUP_SEED if followup else SEED, "permutation_seed": SEED + 1,
              "v9_seed": MODEL_SEED, "v9_epochs": EPOCHS, "gallery_size": len(images),
              "baseline_archive_provenance": PROVENANCE, "labeled_source_days_per_fold": 3, "held_target_days": SESSIONS,
              "real_repeats": 4, "ocf_sample_indices": sample.tolist(), "cohort_independence": cohort,
              "profile": args.profile, "primary_arm": ADAPTED, "fixed_stay_score_units": primary_stay,
              "stay_score_units_by_arm": {name: reward for name, (_, reward) in profiles.items()},
              "formal_data_accessed": False, "recordings": {},
              "protocol": "Replay v9 on each shuffled held day using its other three labeled days. Treat the resulting "
                          "permutation as fixed incumbent. Use CDF rank-template evidence with a fixed reward at "
                          "the corresponding incumbent column before Hungarian assignment. Estimate target marginals on the "
                          "unlabeled full block. The frozen-source CDF control applies the same templates and stay reward. "
                          "The correction itself needs an incumbent permutation and CDF features, without v9 score values. "
                          "Paired OCF uses new matched glyphs only after assignments are complete."}
    publish(report, "inputs_ready", started, output)
    measured = {name: [] for name in (V9, *profiles)}
    for day, counts in recordings.items():
        sources = {other: group for other, group in recordings.items() if other != day}
        publish(report, f"{day}_fitting", started, output)
        order = torch.from_numpy(rng.permutation(len(images)))
        query = counts[order]
        torch.manual_seed(MODEL_SEED)
        with torch.enable_grad():
            retrieval = fit_retrieval(as_trial_blocks(sources), len(images), images=images, device=device,
                                      seed=MODEL_SEED, epochs=EPOCHS)
        hierarchy = fit_hierarchical_gallery(sources, view="total")
        validated = combine_logits(retrieval_score_components(retrieval, query, None, task="task2"), "task2")
        empirical = hierarchy.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
        v9 = blend_hierarchical_logits(validated, empirical)
        incumbent = assign_glyphs(v9)
        ranks = rank_scores(query, fit_marginal_templates(sources))
        choices = {V9: incumbent, **{name: anchored_assignment(ranks[source], incumbent, stay=reward)
                                     for name, (source, reward) in profiles.items()}}
        if any(chosen.unique().numel() != len(images) for chosen in choices.values()):
            raise ValueError("Every decision must use the complete known gallery exactly once.")
        details = {"source_days": list(sources),
                   "changed_vs_v9": {name: int((chosen != incumbent).sum()) for name, chosen in choices.items()},
                   "assigned_top1": {name: float((chosen == order).float().mean()) for name, chosen in choices.items()},
                   "identity_ranks_before_anchor": {name: row_diagnostics(scores, order) for name, scores in {V9: v9, **ranks}.items()}}
        report["recordings"][day] = details
        publish(report, f"{day}_assignments_ready", started, output)
        sample_rows = order.argsort()[sample]
        exact = {name: metric.evaluate(chosen[sample_rows, None])[:, 0] for name, chosen in choices.items()}
        for name, values in exact.items():
            measured[name].append(values)
        details["ocf_vs_v9"] = {name: summarize(values, exact[V9]) for name, values in exact.items()}
        details["target_minus_frozen"] = summarize(exact[ADAPTED], exact[FROZEN])
        if followup:
            details["sensitivity_minus_primary"] = summarize(exact[HALF_STAY], exact[ADAPTED])
        details["unique_exact_pairs_so_far"] = len(metric.values)
        publish(report, f"{day}_scored", started, output)

    glyph_ocf = {name: np.stack(values).mean(0) for name, values in measured.items()}
    report["ocf_vs_v9"] = {name: summarize(values, glyph_ocf[V9]) for name, values in glyph_ocf.items()}
    report["target_minus_frozen"] = summarize(glyph_ocf[ADAPTED], glyph_ocf[FROZEN])
    if followup:
        report["sensitivity_minus_primary"] = summarize(glyph_ocf[HALF_STAY], glyph_ocf[ADAPTED])
    report["ocf_rows_per_arm"] = len(sample) * len(SESSIONS)
    report["unique_exact_pairs"] = len(metric.values)
    publish(report, "complete", started, output)


if __name__ == "__main__":
    main()
