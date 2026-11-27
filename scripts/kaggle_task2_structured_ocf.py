"""Learn a confidence-dependent contour blend with an exact-OCF assignment objective."""

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_contour_retrieval import contour_retrieval_scores, fit_contour_retrieval
from glyph_data import as_trial_blocks, grouped_blocks, load_catalog, load_images, locate_data
from glyph_hierarchical_gallery import fit_hierarchical_gallery
from glyph_mbr import contour_signatures
from glyph_retrieval import assign_glyphs, fit_retrieval
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_task2_anchored_cdf import evaluation_sample as anchored_cohort
from kaggle_task2_causal_motifs import evaluation_sample as motif_cohort
from kaggle_task2_cdf_proxy_cycles import evaluation_sample as proxy_cohort
from kaggle_task2_classifier_ensemble import evaluation_sample as ensemble_cohort
from kaggle_task2_constrained_ceiling import solve
from kaggle_task2_count_trajectory_recheck import evaluation_sample as trajectory_cohort
from kaggle_task2_day_shift import cdf_scores, hybrid_scores, sample_glyphs as shift_cohorts
from kaggle_task2_early_state_transport import evaluation_sample as early_state_cohort
from kaggle_task2_peer_mouse import logit_scale
from kaggle_task2_rank_marginals import BLEND, EPOCHS, MODEL_SEED, MOUSE, evaluation_glyphs, summarize
from kaggle_task2_rank_marginals_recheck import PROVENANCE, evaluation_sample as rank_recheck, fit_marginal_templates
from kaggle_task2_repeat_precision import evaluation_sample as precision_cohort
from kaggle_v8_metric_validation import SESSIONS


SEED, EVAL_COUNT, CALIBRATION_COUNT = 286663, 128, 40
HELD_DAYS, INNER_DAYS = ("session1", "session4"), ("session2", "session3")
WIDTH, SINKHORN_STEPS, ADAM_STEPS = 8, 16, 90
TEMPERATURE, MAX_CONTOUR, FIXED_WEIGHT, PENALTY = 0.55, 0.55, 0.10, 0.006
BASE, FIXED, ROWWISE, STRUCTURED = "stock_cdf", "fixed_contour", "rowwise_training", "balanced_training"
ARMS = (BASE, FIXED, ROWWISE, STRUCTURED)
OUTPUT = Path("/kaggle/working/task2_structured_ocf.json")


@dataclass(frozen=True)
class Graph:
    edges: torch.Tensor
    anchor: torch.Tensor
    contour: torch.Tensor
    confidence: torch.Tensor
    stock: torch.Tensor


def panels(size):
    first, _ = evaluation_glyphs(size)
    recheck, _ = rank_recheck(size)
    anchor, _ = anchored_cohort(size)
    followup, _ = anchored_cohort(size, followup=True)
    shift, oracle = shift_cohorts(size)
    ensemble, _ = ensemble_cohort(size)
    precision, _ = precision_cohort(size)
    proxy, _ = proxy_cohort(size)
    motifs, _ = motif_cohort(size)
    trajectory, _ = trajectory_cohort(size)
    early_state, _ = early_state_cohort(size)
    older = {"cdf_first": first.numpy(), "cdf_recheck": recheck.numpy(), "anchored": anchor.numpy(),
             "anchored_followup": followup.numpy(), "day_shift": shift.numpy(), "day_shift_oracle": oracle.numpy(),
             "classifier_ensemble": ensemble.numpy(), "repeat_precision": precision.numpy(), "proxy_cycles": proxy.numpy(),
             "causal_motifs": motifs.numpy(), "count_trajectory": trajectory.numpy(), "early_state": early_state.numpy()}
    eligible = np.setdiff1d(np.arange(size), np.unique(np.concatenate(tuple(older.values()))))
    drawn = np.random.default_rng(SEED).choice(eligible, EVAL_COUNT + 4 * CALIBRATION_COUNT, replace=False)
    scored = np.sort(drawn[:EVAL_COUNT])
    calibration = np.sort(drawn[EVAL_COUNT:].reshape(len(HELD_DAYS), len(INNER_DAYS), CALIBRATION_COUNT), axis=-1)
    overlaps = {name: int(np.intersect1d(scored, rows).size) for name, rows in older.items()}
    return torch.from_numpy(scored), torch.from_numpy(calibration), overlaps


def episode_graph(sources, query, images, signature, energy):
    torch.manual_seed(MODEL_SEED)
    with torch.enable_grad():
        retrieval = fit_retrieval(as_trial_blocks(sources), len(images), images=images, device=query.device,
                                  seed=MODEL_SEED, epochs=EPOCHS)
    hierarchy = fit_hierarchical_gallery(sources, view="total")
    hybrid = hybrid_scores(retrieval, hierarchy, query)
    fixed = cdf_scores(hybrid, query, fit_marginal_templates(sources))
    stock = assign_glyphs(fixed)
    anchor = logit_scale(fixed)
    contour_model = fit_contour_retrieval(sources, signature, energy, task="task2", seed=MODEL_SEED)
    contour = logit_scale(contour_retrieval_scores(contour_model, query, None)["contour_ridge"])

    leaders = anchor.topk(2, dim=1).values
    stock_edge = stock[:, None]
    confidence = torch.stack((leaders[:, 0] - leaders[:, 1],
                              leaders[:, 0] - anchor.gather(1, stock_edge)[:, 0],
                              contour.gather(1, stock_edge)[:, 0]), dim=1)
    confidence = ((confidence - confidence.mean(0)) / confidence.std(0, unbiased=False).clamp_min(0.05)).clamp(-3, 3)
    confidence = torch.cat((torch.ones((len(confidence), 1)), confidence), dim=1)
    alternatives = anchor.clone().scatter_(1, stock_edge, -torch.inf).topk(4, dim=1).indices
    core = torch.cat((stock_edge, alternatives), dim=1)
    additions = contour.clone().scatter_(1, core, -torch.inf).topk(3, dim=1).indices
    edges = torch.cat((core, additions), dim=1)
    return Graph(edges, anchor.gather(1, edges), contour.gather(1, edges), confidence, stock)


def reward_data(graph, order, calibration, images, scorer):
    rows = order.argsort()[calibration]
    metric = PairedOCF(scorer, images, (images[calibration] >= 128).numpy(), images.device, 12)
    utilities = torch.from_numpy(metric.evaluate(graph.edges[rows])).float()
    # Subtract the incumbent's utility. With row mass one this leaves the fitted optimum unchanged.
    reward = (utilities - utilities[:, :1]) / 100
    return (graph, rows, reward), {"labels": len(metric.values), "calibration_glyphs": len(rows),
                                  "incumbent_ocf": float(utilities[:, 0].mean()),
                                  "relaxed_eight_candidate_upper_bound": float(utilities.amax(1).mean())}


def probabilities(scores, edges, balanced):
    scaled = scores / TEMPERATURE
    probability = (scaled - scaled.amax(dim=1, keepdim=True)).exp().clamp_min(1e-18)
    if balanced:
        flat = edges.flatten()
        for _ in range(SINKHORN_STEPS):
            probability = probability / probability.sum(dim=1, keepdim=True)
            column_mass = probability.new_zeros(len(scores)).scatter_add(0, flat, probability.flatten())
            probability = probability / column_mass[edges].clamp_min(1e-12)
    return probability / probability.sum(dim=1, keepdim=True)


def contour_weight(graph, parameters):
    return MAX_CONTOUR * (graph.confidence @ parameters).sigmoid()


def fitted_scores(graph, parameters):
    return graph.anchor + contour_weight(graph, parameters)[:, None] * graph.contour


def fit_head(episodes, balanced):
    # Start both arms at the same fixed contour weight.
    initial = torch.tensor([np.log(FIXED_WEIGHT / (MAX_CONTOUR - FIXED_WEIGHT)), 0, 0, 0], dtype=torch.float32)
    parameters = initial.clone().requires_grad_()
    optimizer = torch.optim.Adam([parameters], lr=0.06)
    for _ in range(ADAM_STEPS):
        gain = torch.stack([(probabilities(fitted_scores(graph, parameters), graph.edges, balanced)[rows] * reward).sum(1).mean()
                            for graph, rows, reward in episodes]).mean()
        loss = -gain + PENALTY * (parameters - initial).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    parameters = parameters.detach()
    diagnostics = []
    with torch.no_grad():
        for graph, rows, reward in episodes:
            probability = probabilities(fitted_scores(graph, parameters), graph.edges, balanced)
            column_mass = probability.new_zeros(len(graph.edges)).scatter_add(0, graph.edges.flatten(), probability.flatten())
            diagnostics.append({"learned_soft_ocf_gain_vs_incumbent": float((probability[rows] * reward).sum(1).mean() * 100),
                                "mean_contour_weight": float(contour_weight(graph, parameters).mean()),
                                "column_mass_max_absolute_error": float((column_mass - 1).abs().max())})
    return parameters, {"parameters": parameters.tolist(), "training_diagnostics": diagnostics}


def publish(report, stage, started):
    report.update(stage=stage, seconds=time.monotonic() - started)
    pending = OUTPUT.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(OUTPUT)
    print(json.dumps({"stage": stage, "seconds": report["seconds"]}), flush=True)


@torch.no_grad()
def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Use a private Kaggle CPU notebook with authorized labeled recordings.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if len(images) != 2748 or tuple(recordings) != SESSIONS or any(values.shape[:2] != (len(images), 4)
                                                                  for values in recordings.values()):
        raise ValueError("Require four complete labeled recording blocks at four repeats per glyph.")
    sample, calibration, overlap = panels(len(images))
    scorer = official_scorer(root)
    evaluator = PairedOCF(scorer, images, (images[sample] >= 128).numpy(), device, 12)
    signature, energy = contour_signatures(images)
    rng = np.random.default_rng(SEED + 1)
    report = {"mouse": MOUSE, "seed": SEED, "held_days": HELD_DAYS, "gallery_size": len(images), "real_repeats": 4,
              "classifier_seed": MODEL_SEED, "classifier_epochs": EPOCHS, "fixed_cdf_weight": BLEND,
              "archive_provenance": PROVENANCE, "scored_glyph_indices": sample.tolist(), "overlap_by_earlier_cohort": overlap,
              "formal_test_accessed": False, "exact_ocf_calibration_glyphs_per_episode": CALIBRATION_COUNT,
              "candidate_edges_per_row": WIDTH, "fixed_contour_weight": FIXED_WEIGHT, "max_contour_weight": MAX_CONTOUR,
              "soft_temperature": TEMPERATURE, "sinkhorn_steps": SINKHORN_STEPS, "train_updates": ADAM_STEPS,
              "parameter_penalty": PENALTY, "primary_arm": STRUCTURED, "paired_reference": ROWWISE, "recordings": {},
              "protocol": "Fit two labeled out-of-day calibration episodes per held recording. Each calibration "
                          "neural model excludes both the held and calibration recording. Every row keeps the exact "
                          "v9+25% CDF assignment edge, four other top anchored edges, and three contour nominations. "
                          "Learn an identical four-parameter contour-weight policy either from expected exact OCF "
                          "under a row softmax or differentiating through full-gallery sparse Sinkhorn. Use exact "
                          "rewards on disjoint preselected calibration glyphs only. Both policies, the fixed contour "
                          "control, and the untouched anchor output all 2748 glyphs once. Outer neural models "
                          "use all three non-held recordings; no held labels enter decisions."}
    publish(report, "inputs_ready", started)
    exact = {name: [] for name in ARMS}
    total_calibration_pairs = 0
    for fold, held in enumerate(HELD_DAYS):
        episodes = []
        report["recordings"][held] = {"calibration": {}}
        detail = report["recordings"][held]
        for index, day in enumerate(INNER_DAYS):
            source = {name: values for name, values in recordings.items() if name not in (held, day)}
            order = torch.from_numpy(rng.permutation(len(images)))
            publish(report, f"{held}_calibrating_{day}", started)
            graph = episode_graph(source, recordings[day][order], images, signature, energy)
            episode, stats = reward_data(graph, order, calibration[fold, index], images, scorer)
            episodes.append(episode)
            total_calibration_pairs += stats["labels"]
            detail["calibration"][day] = {**stats, "source_days": tuple(source),
                                          "glyph_indices": calibration[fold, index].tolist()}
            publish(report, f"{held}_calibration_{day}_scored", started)
        with torch.enable_grad():
            row_head, row_report = fit_head(episodes, balanced=False)
            balanced_head, balanced_report = fit_head(episodes, balanced=True)
        detail["head_fit"] = {ROWWISE: row_report, STRUCTURED: balanced_report}
        del episodes, episode
        source = {name: values for name, values in recordings.items() if name != held}
        order = torch.from_numpy(rng.permutation(len(images)))
        publish(report, f"{held}_outer_fitting", started)
        graph = episode_graph(source, recordings[held][order], images, signature, energy)
        choices = {BASE: graph.stock,
                   FIXED: solve(graph.edges, graph.anchor + FIXED_WEIGHT * graph.contour),
                   ROWWISE: solve(graph.edges, fitted_scores(graph, row_head)),
                   STRUCTURED: solve(graph.edges, fitted_scores(graph, balanced_head))}
        if any(choice.unique().numel() != len(images) for choice in choices.values()):
            raise ValueError("Every final solution must assign the full known gallery exactly once.")
        detail["source_days"] = tuple(source)
        detail["full_gallery"] = {name: {"assigned_top1": float((choice == order).float().mean()),
                                                "changed_vs_stock": int((choice != graph.stock).sum()),
                                                "distinct_outputs": int(choice.unique().numel())} for name, choice in choices.items()}
        detail["full_gallery"][ROWWISE]["mean_contour_weight"] = float(contour_weight(graph, row_head).mean())
        detail["full_gallery"][STRUCTURED]["mean_contour_weight"] = float(contour_weight(graph, balanced_head).mean())
        publish(report, f"{held}_assignments_ready", started)
        rows = order.argsort()[sample]
        measured = {name: evaluator.evaluate(choice[rows, None])[:, 0] for name, choice in choices.items()}
        for name, values in measured.items():
            exact[name].append(values)
        detail["ocf_vs_stock"] = {name: summarize(values, measured[BASE]) for name, values in measured.items()}
        detail["structured_vs_rowwise"] = summarize(measured[STRUCTURED], measured[ROWWISE])
        publish(report, f"{held}_scored", started)
    means = {name: np.stack(values).mean(0) for name, values in exact.items()}
    report.update(ocf_vs_stock={name: summarize(values, means[BASE]) for name, values in means.items()},
                  structured_vs_rowwise=summarize(means[STRUCTURED], means[ROWWISE]),
                  structured_vs_fixed=summarize(means[STRUCTURED], means[FIXED]),
                  ocf_rows_per_arm=len(sample) * len(HELD_DAYS), unique_exact_evaluation_pairs=len(evaluator.values),
                  exact_calibration_pairs=total_calibration_pairs)
    publish(report, "complete", started)


if __name__ == "__main__":
    main()
