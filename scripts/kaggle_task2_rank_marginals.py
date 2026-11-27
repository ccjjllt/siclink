"""Test unlabeled per-unit response quantiles for Task 2 cross-day identity."""

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
from kaggle_gallery_validation import standard_error
from kaggle_task2_constrained_ceiling import fresh_samples as ceiling_cohorts
from kaggle_task2_peer_mouse import logit_scale, row_diagnostics


MOUSE = "mice2"
SEED, MODEL_SEED = 286157, 211
OCF_GLYPHS, EPOCHS = 112, 12
BLEND, TEMPERATURE = 0.25, 1 / 3
DAY_PAIRS = (("session1", "session3"), ("session2", "session4"))
V9, LINEAR, FROZEN, ADAPTED = "v9", "linear_marginal", "frozen_source_cdf", "target_cdf"
OUTPUT = Path("/kaggle/working/task2_rank_marginals.json")
# Nonuniform oracle rows from the already archived exact ceiling diagnostic.
CEILING_CROWDED = (1646, 2577, 2341, 1322, 414, 1287, 1373, 610, 2103, 807, 1860, 433, 2266, 591, 2734, 2698,
                   2423, 816, 2308, 2594, 2521, 2614, 1854, 1588, 2043, 1383, 1605, 2738, 1107, 1871, 1017, 2349)


def evaluation_glyphs(size):
    # The remaining ceiling cohort excludes its uniform rows and all earlier major Task 2 evaluations.
    _, remaining, _ = ceiling_cohorts(size)
    eligible = np.setdiff1d(remaining.numpy(), CEILING_CROWDED)
    chosen = np.sort(np.random.default_rng(SEED).choice(eligible, OCF_GLYPHS, replace=False))
    return torch.from_numpy(chosen), len(eligible)


def responses(counts):
    return (counts.unflatten(-1, (-1, 10))[..., 2:].sum(-1).mean(1) + 0.375).sqrt()


def linear(scores):
    return ((scores - scores.mean(dim=0)) / scores.std(dim=0).clamp_min(0.04)).clamp(-3, 3)


def source_cdf(values):
    return values.T.contiguous().sort(dim=1).values


def normal_scores(values, reference):
    ordered = values.T.contiguous()
    lower = torch.searchsorted(reference, ordered)
    upper = torch.searchsorted(reference, ordered, right=True)
    probability = ((lower + upper).float() / (2 * reference.shape[1])).clamp(0.001, 0.999)
    return torch.special.ndtri(probability).clamp(-3, 3).T.contiguous()


def whiten(values, factor):
    return torch.linalg.solve_triangular(factor, values.T.contiguous(), upper=False).T.contiguous()


def templates(sources):
    raw = tuple(responses(spikes) for spikes in sources.values())
    standardized = tuple(linear(day) for day in raw)
    ranks = tuple(normal_scores(day, source_cdf(day)) for day in raw)
    residual = (standardized[0] - standardized[1]) / 2**0.5
    covariance = residual.T @ residual / (len(residual) - 1)
    shrunk = 0.80 * covariance + 0.20 * torch.diag(covariance.diagonal())
    shrunk.diagonal().add_(0.10)
    factor = torch.linalg.cholesky(shrunk)
    return (whiten((standardized[0] + standardized[1]) / 2, factor),
            whiten((ranks[0] + ranks[1]) / 2, factor), factor, source_cdf(torch.cat(raw)))


def marginal_logits(observed, fitted):
    linear_templates, rank_templates, factor, train_reference = fitted
    held = responses(observed)
    transforms = {LINEAR: (linear(held), linear_templates), FROZEN: (normal_scores(held, train_reference), rank_templates),
                  ADAPTED: (normal_scores(held, source_cdf(held)), rank_templates)}
    return {name: logit_scale(whiten(query, factor) @ gallery.T - 0.5 * gallery.square().sum(1))
            for name, (query, gallery) in transforms.items()}


def summarize(values, base):
    difference = values - base
    return {"ocf": float(values.mean()), "paired_delta": float(difference.mean()),
            "paired_glyph_se": standard_error(difference), "distinct_glyphs": len(difference)}


def publish(report, phase, started):
    report.update(stage=phase, seconds=time.monotonic() - started)
    pending = OUTPUT.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(OUTPUT)
    print(json.dumps({"stage": phase, "seconds": report["seconds"]}), flush=True)


@torch.no_grad()
def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Use Kaggle CPU with official labeled training data.")
    torch.set_num_threads(4)
    device = torch.device("cpu")
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, device)
    if len(images) != 2748:
        raise ValueError("Expected the complete known 2,748-glyph gallery.")
    catalog = load_catalog(root)
    recordings = grouped_blocks(root, catalog, MOUSE, device, len(images))
    if any(recordings[day].shape[:2] != (len(images), 4) for days in DAY_PAIRS for day in days):
        raise ValueError("Each day must preserve four measured repeats per glyph.")
    sample, available = evaluation_glyphs(len(images))
    metric = PairedOCF(official_scorer(root), images, (images[sample] >= 128).numpy(), device, 12)
    rng = np.random.default_rng(SEED + 1)
    report = {"mouse": MOUSE, "seed": SEED, "model_seed": MODEL_SEED, "gallery_size": len(images),
              "fresh_eligible_ocf_glyphs": available, "ocf_glyphs": sample.tolist(), "labeled_source_pairs": DAY_PAIRS,
              "formal_test_accessed": False, "real_repeats": 4, "evoked_window_ms": [50, 250], "v9_fit_epochs": EPOCHS,
              "blend_weight": BLEND, "identity_nll_temperature": TEMPERATURE, "recordings": {},
              "protocol": "Fit the unchanged v9 hybrid and marginal classifiers on each of two complementary labeled source "
                          "pairs. Leave both other days out of every fitted template and precision. Use the same complete "
                          "one-to-one assignment and fixed 25% blend for each alternative. Unitwise normal ranks are computed "
                          "with tie-aware empirical CDF over each entire unlabelled recording. Compare to matched linear "
                          "per-unit moments and to applying a fixed pooled source CDF to the same held query, using the same "
                          "rank templates for both CDFs. Precision is fitted from source cross-day residuals and kept identical "
                          "across marginal arms. OCF and identity references do not enter any fitting or query decisions."}
    publish(report, "inputs_ready", started)
    scores = {name: [] for name in (V9, LINEAR, FROZEN, ADAPTED)}
    for index, source_days in enumerate(DAY_PAIRS):
        held_days = DAY_PAIRS[1 - index]
        sources = {day: recordings[day] for day in source_days}
        fitted = templates(sources)
        torch.manual_seed(MODEL_SEED)
        with torch.enable_grad():
            v9 = fit_retrieval(as_trial_blocks(sources), len(images), images=images, device=device,
                               seed=MODEL_SEED, epochs=EPOCHS)
        hierarchy = fit_hierarchical_gallery(sources, view="total")
        for day in held_days:
            order = torch.from_numpy(rng.permutation(len(images)))
            query = recordings[day][order]
            validated = combine_logits(retrieval_score_components(v9, query, None, task="task2"), "task2")
            empirical = hierarchy.logits(query, query_repeat_mix=0.5)["empirical_bayes_repeat_noise"]
            baseline = logit_scale(blend_hierarchical_logits(validated, empirical))
            alternatives = marginal_logits(query, fitted)
            decisions = {V9: baseline, **{name: baseline.lerp(extra, BLEND) for name, extra in alternatives.items()}}
            choices = {name: assign_glyphs(logits) for name, logits in decisions.items()}
            if any(choice.unique().numel() != len(images) for choice in choices.values()):
                raise ValueError("Every result must include all 2,748 glyphs exactly once.")
            # Held correspondence is used only after all masks have been selected.
            rows = order.argsort()[sample]
            measured = {name: metric.evaluate(choice[rows, None])[:, 0] for name, choice in choices.items()}
            for name, values in measured.items():
                scores[name].append(values)
            diagnostics = {}
            full_rows = torch.arange(len(order))
            for name, logits in decisions.items():
                density = logit_scale(logits) / TEMPERATURE
                diagnostics[name] = {**row_diagnostics(logits, order),
                                     "identity_nll_at_fixed_temperature": float((density.logsumexp(1) - density[full_rows, order]).mean()),
                                     "assigned_top1": float((choices[name] == order).float().mean()),
                                     "changed_vs_v9": int((choices[name] != choices[V9]).sum())}
            report["recordings"][day] = {"source_days": source_days, "diagnostics": diagnostics,
                                         "ocf_vs_v9": {name: summarize(values, measured[V9]) for name, values in measured.items()},
                                         "target_cdf_vs_linear": summarize(measured[ADAPTED], measured[LINEAR]),
                                         "target_cdf_vs_frozen": summarize(measured[ADAPTED], measured[FROZEN])}
            publish(report, f"{day}_scored", started)
    glyph_scores = {name: np.stack(values).mean(0) for name, values in scores.items()}
    report["ocf_vs_v9"] = {name: summarize(values, glyph_scores[V9]) for name, values in glyph_scores.items()}
    report["target_cdf_vs_linear"] = summarize(glyph_scores[ADAPTED], glyph_scores[LINEAR])
    report["target_cdf_vs_frozen"] = summarize(glyph_scores[ADAPTED], glyph_scores[FROZEN])
    report["exact_unique_pairs"] = len(metric.values)
    report["ocf_rows_per_arm"] = len(sample) * sum(len(days) for days in DAY_PAIRS)
    publish(report, "complete", started)


if __name__ == "__main__":
    main()
