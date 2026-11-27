"""Test PP-OCR-aware exact-OCF reranking on a fixed held mice2 v20 pool."""

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import concat_atlas_candidates
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_predictive_likelihood import fit_predictive_likelihood
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import align_training_glyphs, standardized
from kaggle_task1_crossfit_metric_scale import fold_indices
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED, calibration_labels
from kaggle_task1_mice2_ppocrv6_v20 import SEED as PRIOR_SEED, balanced_panel, ppocr_logits
from kaggle_task1_mice2_repeat_matched_ranker import MOUSE, candidate_graph, model_of, stock_folds
from kaggle_task1_mice2_textverifier_recheck import fresh_cohort, tensor_hash
from kaggle_task1_microns_hanzi_recheck import fit_ocr_logits
from kaggle_task1_native_hanzi import OCR_CHECKSUM, OCR_REVISION, response_views
from kaggle_task1_textverifier import publish
from kaggle_task1_v11_repeat_weight_transfer import POISSON_AMPLIFICATION, normalized_sources, pooled_scores, weighted_fusion
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, rendered_bank, summarize


SEED = 224883
CALIBRATION_GLYPHS = 80
RIDGE = 0.25
HUBER = 12.0
RESIDUAL_LIMIT = 12.0
LABEL_BUDGET = {"v20": 5, "ppocr": 3, "hanzi": 3, "random_remaining": 2}
STOCK = "v20"
CONTROL = "no_pp_same_capacity"
REAL = "real_pp"
SHUFFLED = "shuffled_pp"
DAY4 = "same_day_four_repeats"
POOLED16 = "pooled_16_known_correspondence"
OUTPUT = Path("/kaggle/working/task1_mice2_ppocrv6_ranker.json")
COMMON = ("decoded_contour.z", "poisson_rate.z", "poisson_timed.z", "predictive_error.z", "agreement.top_five")


@dataclass
class Evidence:
    graph: object
    v20: torch.Tensor
    hanzi: torch.Tensor
    candidates: dict

    def take_rows(self, rows):
        return Evidence(self.graph.take_rows(rows), self.v20[rows], self.hanzi[rows],
                        {arm: features[rows] for arm, features in self.candidates.items()})

    def anchor(self):
        return self.v20.argmax(1, keepdim=True)

    def relative(self, arm):
        feature_ids = [self.graph.feature_names.index(name) for name in COMMON]
        common = self.graph.features[:, :, feature_ids]
        stock = self.v20.masked_fill(~self.graph.valid, 0)[:, :, None] / 15
        features = torch.cat((stock, common, self.hanzi[:, :, None], self.candidates[arm]), dim=2)
        anchor = features.gather(1, self.anchor()[:, :, None].expand(-1, -1, features.shape[2]))
        return features - anchor


@dataclass
class ResidualRanker:
    scale: torch.Tensor
    weight: torch.Tensor
    fit_weighted_mae: float

    def select(self, evidence, arm):
        relative = (evidence.relative(arm) / self.scale).clamp(-8, 8)
        residual = (relative @ self.weight).clamp(-RESIDUAL_LIMIT, RESIDUAL_LIMIT)
        column = (evidence.v20 + residual).argmax(1, keepdim=True)
        return evidence.graph.indices.gather(1, column)[:, 0]


def joined_evidence(parts):
    return Evidence(concat_atlas_candidates([part.graph for part in parts]),
                    torch.cat([part.v20 for part in parts]), torch.cat([part.hanzi for part in parts]),
                    {arm: torch.cat([part.candidates[arm] for part in parts]) for arm in parts[0].candidates})


def matching_features(scores, graph):
    scores = standardized(scores)
    top_values, top_ids = scores.topk(32, dim=1)
    candidate = scores.gather(1, graph.indices)
    ranks = torch.arange(1, 33).float()
    matches = graph.indices[:, :, None] == top_ids[:, None, :]
    support = (ranks.rsqrt() * matches).sum(2)
    margin = (top_values[:, :1] - top_values[:, 1:2]).clamp(0, 4)
    return torch.stack((candidate, support, candidate * margin), dim=2).masked_fill(~graph.valid[:, :, None], 0)


def shuffle_features(features, valid, random):
    # Move all components together and conserve the per-query marginal distribution.
    if not torch.equal(valid, torch.arange(valid.shape[1])[None] < valid.sum(1, keepdim=True)):
        raise ValueError("The fixed v20 list must be padded at the end of each row.")
    positions = torch.rand(valid.shape, generator=random).masked_fill(~valid, torch.inf).argsort(1)
    return features.gather(1, positions[:, :, None].expand_as(features)).masked_fill(~valid[:, :, None], 0)


@torch.inference_mode()
def make_evidence(model, stock, amplified, sources, hanzi, ppocr, v20, random):
    graph = candidate_graph(model, stock, amplified, sources, hanzi)
    scores = v20.rank(graph, anchor_ocf=ANCHOR)
    real = matching_features(ppocr, graph)
    control = matching_features(sources["ridge_contour_raw"], graph)
    shuffled = shuffle_features(real, graph.valid, random)
    incumbent_ocr = standardized(hanzi).gather(1, graph.indices).masked_fill(~graph.valid, 0)
    return Evidence(graph, scores, incumbent_ocr, {CONTROL: control, REAL: real, SHUFFLED: shuffled})


def label_slots(evidence, random):
    valid = evidence.graph.valid
    slots = torch.zeros_like(valid)
    rank_scores = {"v20": evidence.v20, "ppocr": evidence.candidates[REAL][:, :, 0], "hanzi": evidence.hanzi}
    for name, matrix in rank_scores.items():
        columns = matrix.masked_fill(~valid, -torch.inf).topk(LABEL_BUDGET[name], dim=1).indices
        slots.scatter_(1, columns, True)
    available = valid & ~slots
    order = torch.rand(valid.shape, generator=random).masked_fill(~available, -torch.inf)
    columns = order.topk(LABEL_BUDGET["random_remaining"], dim=1).indices
    slots.scatter_(1, columns, True)
    return replace(evidence.graph, valid=slots & valid)


@torch.inference_mode()
def train_head(evidence, labels, arm):
    anchor = evidence.anchor()
    stock_relative = (evidence.v20 - evidence.v20.gather(1, anchor)).masked_fill(~evidence.graph.valid, 0)
    valid = torch.isfinite(labels)
    anchor_labels = labels.gather(1, anchor)
    if not torch.isfinite(anchor_labels).all():
        raise ValueError("The exact label pool must include the frozen v20 decision for each calibration glyph.")
    label_weight = valid.float() / valid.sum(1, keepdim=True) / len(labels)
    features = evidence.relative(arm)
    scale = (label_weight[:, :, None] * features.square()).sum((0, 1)).sqrt().clamp_min(0.12)
    x = (features[valid] / scale).clamp(-8, 8)
    y = (labels - anchor_labels - stock_relative)[valid]
    sample = label_weight[valid]
    weight = x.new_zeros(x.shape[1])
    for _ in range(4):
        robust = sample * (HUBER / (y - x @ weight).abs().clamp_min(HUBER))
        gram = x.T @ (x * robust[:, None])
        gram.diagonal().add_(RIDGE)
        weight = torch.linalg.solve(gram, x.T @ (robust * y))
    mae = float((sample * (y - x @ weight).abs()).sum())
    return ResidualRanker(scale, weight, mae)


def compare(values):
    report = {arm: summarize([score], [values[STOCK]]) for arm, score in values.items()}
    report["real_pp_vs_no_pp"] = summarize([values[REAL]], [values[CONTROL]])
    report["real_pp_vs_shuffled_pp"] = summarize([values[REAL]], [values[SHUFFLED]])
    return report


def assess(evidence, heads, exact, codes, held_day, fresh):
    graph = evidence.graph
    chosen = {STOCK: graph.indices.gather(1, evidence.anchor())[:, 0]}
    chosen.update({arm: head.select(evidence, arm) for arm, head in heads.items()})
    values = {arm: exact.evaluate(indices[:, None])[:, 0] for arm, indices in chosen.items()}
    report = {"paired_ocf": compare(values), "fresh_32_paired_ocf": compare({arm: score[fresh] for arm, score in values.items()}),
              "changed_choices_vs_v20": {arm: int((indices != chosen[STOCK]).sum()) for arm, indices in chosen.items()},
              "ocf_by_glyph": {arm: score.tolist() for arm, score in values.items()},
              "chosen_codepoints_by_glyph": {arm: codes[indices].tolist() for arm, indices in chosen.items()},
              "mean_fixed_candidate_pool_size": float(graph.valid.sum(1).float().mean()),
              "fixed_candidate_pool_sha256": tensor_hash(graph.indices), "candidate_validity_sha256": tensor_hash(graph.valid)}
    if held_day is not None:
        report["day_panels"] = {day: compare({arm: score[held_day == index] for arm, score in values.items()})
                               for index, day in enumerate(SESSIONS)}
    return report


def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Use a private Kaggle CPU notebook with the official labeled data.")
    started = time.monotonic()
    torch.set_num_threads(4)
    cpu = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    shuffle = torch.Generator().manual_seed(SEED + 1)
    label_rng = torch.Generator().manual_seed(SEED + 2)
    root = locate_data()
    scorer = official_scorer(root)
    images = load_images(root, cpu)
    if images.shape != (2748, 64, 64) or images.dtype != torch.uint8:
        raise ValueError("Use the complete labeled grayscale glyph gallery.")
    fit, outer, independence = fresh_cohort(len(images))
    held, held_day, fresh, overlaps = balanced_panel(outer, len(images), np.random.default_rng(PRIOR_SEED))
    eligible = np.setdiff1d(outer.numpy(), held.numpy())
    selected = rng.choice(eligible, CALIBRATION_GLYPHS, replace=False)
    order = selected.argsort()
    calibration = torch.from_numpy(selected[order])
    calibration_day = torch.from_numpy(np.repeat(np.arange(len(SESSIONS)), CALIBRATION_GLYPHS // len(SESSIONS))[order])
    query = torch.cat((calibration, held))
    assignment = torch.cat((calibration_day, held_day))
    if query.unique().numel() != len(query) or np.intersect1d(query.numpy(), fit.numpy()).size:
        raise ValueError("Calibration and endpoint glyphs must be disjoint from each other and the full stem scope.")

    stock_report = next(Path("/kaggle/input").rglob("stock_report.json"), None)
    if stock_report is None:
        raise FileNotFoundError("Mount the private training-only v20 fourfold archive.")
    archive = json.loads(stock_report.read_text(encoding="utf-8"))
    if (archive["stage"] != "complete" or archive["mouse"] != MOUSE or archive["held_gallery_indices"] != outer.tolist()
            or archive["fit_gallery_indices_sha256"] != tensor_hash(fit) or archive["training_raster_sha256"] != tensor_hash(images)):
        raise ValueError("The v20 folds must match the exact labeled gallery and outer split.")
    noise, splits = fold_indices(len(fit), cpu)
    folds, hashes = stock_folds(stock_report.parent, splits, noise, archive["calibration_cache"]["key"], cpu)
    codes = folds[0][3]
    if any(not torch.equal(codes, fold[3]) for fold in folds):
        raise ValueError("The v20 calibration folds must agree on the atlas order.")
    v20 = model_of([fold[0] for fold in folds], [fold[1] for fold in folds])
    del folds
    inner, noise = fit[splits[0][1]], fit[noise]
    phases = {"calibration": (calibration, calibration_day), "held": (held, held_day)}
    report = {"mouse": MOUSE, "seed": SEED, "formal_queries_loaded": False, "device": str(cpu),
              "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "training_raster_sha256": tensor_hash(images),
              "stock_report_sha256": sha256(stock_report.read_bytes()).hexdigest(), "stock_fold_sha256": hashes,
              "fit_gallery_indices_sha256": tensor_hash(inner), "fit_glyph_count": len(inner), "noise_glyph_count": len(noise),
              "codepoint_order_sha256": tensor_hash(codes), "outer_cohort_prior_overlap": independence,
              "held_gallery_indices": held.tolist(), "calibration_gallery_indices": calibration.tolist(),
              "held_overlap_with_earlier_v20_cpu_experiments": overlaps, "fresh_cpu_gallery_indices": fresh.tolist(),
              "held_calibration_overlap": 0, "held_and_calibration_overlap_with_exploratory_ppocr": 0,
              "same_day_assignments": {name: {day: rows[days == i].tolist() for i, day in enumerate(SESSIONS)}
                                       for name, (rows, days) in phases.items()},
              "hanzi": {"revision": OCR_REVISION, "sha256": OCR_CHECKSUM}, "candidate_budget": NOMINATIONS,
              "calibration_label_budget": LABEL_BUDGET, "anchor_ocf": ANCHOR, "poisson_weight": POISSON_AMPLIFICATION,
              "head_settings": {"shared_features": ["frozen_v20_score", *COMMON, "stock_hanzi_z"],
                                "additional_features": ["candidate_z", "top32_reciprocal_sqrt_rank", "candidate_z_times_leader_gap"],
                                "ridge": RIDGE, "huber_delta": HUBER, "absolute_residual_limit": RESIDUAL_LIMIT},
              "protocol": "Use the original v20 candidate set and renderer for each glyph and endpoint. Train separate additive "
                          "OCF residual heads for day-four and coindexed pooled-16, from exact OCF on 80 calibration glyphs outside "
                          "the v20 stem and distinct from the 128 endpoint glyphs. All heads share exact candidate labels and "
                          "capacity. The no-PP head takes the current ridge-contour evidence in the additional feature slots; the "
                          "PP head uses global PP-OCRv6 scores. The matched negative control permutes those PP features jointly "
                          "within each glyph's candidates. No held reference raster is used before endpoint scoring."}
    publish(report, OUTPUT, started, "stem_started")
    counts = grouped_blocks(root, load_catalog(root), MOUSE, cpu, len(images))
    if tuple(counts) != SESSIONS or any(group.shape[:2] != (len(images), 4) for group in counts.values()):
        raise ValueError("Expected all four labeled recording sessions with four real repeats per glyph.")
    labeled = images[inner]
    model = fit_v8({day: block[inner] for day, block in counts.items()}, labeled, MODEL_SEED)
    if not torch.equal(model.codepoints, codes):
        raise ValueError("The refit stem must match the archived calibration codepoint order.")
    rendered = rendered_bank(model)
    report["atlas"] = {**model.report, "rendered_sha256": tensor_hash(rendered)}
    publish(report, OUTPUT, started, "ocr_started")
    views = response_views(counts, inner, query, rng)
    hanzi = fit_ocr_logits(labeled, model.atlas.images, views)
    publish(report, OUTPUT, started, "ppocr_started")
    ppocr, report["published_ocr"] = ppocr_logits(labeled, model.atlas.images, views)
    publish(report, OUTPUT, started, "ppocr_ready")
    noise_glyphs, _ = align_training_glyphs(model.atlas, images[noise])

    views_by_day, sources_by_day, order_by_day = [], {}, []
    for index, (day, block) in enumerate(counts.items()):
        likelihood = fit_predictive_likelihood(block[noise], model.encoder.expected_counts(noise_glyphs, day))
        stock = atlas_scores(model, likelihood, block[query], day)
        sources_by_day[day] = normalized_sources(stock)
        picked = torch.where(assignment == index)[0]
        sources = {name: value[picked] for name, value in sources_by_day[day].items()}
        stock = {name: value[picked] for name, value in stock.items()}
        amplified = {**stock, **weighted_fusion(model, sources, POISSON_AMPLIFICATION)}
        views_by_day.append(make_evidence(model, stock, amplified, sources, hanzi[day][picked], ppocr[day][picked], v20, shuffle))
        order_by_day.append(picked)
        publish(report, OUTPUT, started, f"{day}_candidates_ready")
    same_day = joined_evidence(views_by_day).take_rows(torch.cat(order_by_day).argsort())
    pooled = {name: torch.stack([sources_by_day[day][name] for day in SESSIONS]).mean(0) for name in sources_by_day[SESSIONS[0]]}
    stock, amplified = pooled_scores(model, pooled)
    sixteen = make_evidence(model, stock, amplified, pooled, torch.stack(tuple(hanzi.values())).mean(0),
                            torch.stack(tuple(ppocr.values())).mean(0), v20, shuffle)
    candidates = {DAY4: same_day, POOLED16: sixteen}
    calibration_sets = {endpoint: evidence.take_rows(slice(0, CALIBRATION_GLYPHS)) for endpoint, evidence in candidates.items()}
    labels_graph = concat_atlas_candidates([label_slots(evidence, label_rng) for evidence in calibration_sets.values()])
    label_started = time.monotonic()
    report["calibration_candidate_observations"] = int(labels_graph.valid.sum())
    publish(report, OUTPUT, started, "official_calibration_started")
    labels, measured = calibration_labels(labels_graph, rendered, images[calibration] >= 128, scorer)
    report["exact_calibration"] = {"unique_pairs": measured, "seconds": time.monotonic() - label_started}
    publish(report, OUTPUT, started, "official_calibration_complete")

    exact = PairedOCF(scorer, rendered, (images[held] >= 128).numpy(), cpu, 12)
    fresh_mask = np.isin(held.numpy(), fresh)
    report["head_fit"] = {}
    for (endpoint, evidence), labels_rows in zip(calibration_sets.items(), labels.split(CALIBRATION_GLYPHS), strict=True):
        heads = {arm: train_head(evidence, labels_rows, arm) for arm in evidence.candidates}
        report["head_fit"][endpoint] = {arm: {"weights": head.weight.tolist(), "scale": head.scale.tolist(),
                                               "weighted_fit_absolute_error": head.fit_weighted_mae}
                                        for arm, head in heads.items()}
        relevant_days = held_day.numpy() if endpoint == DAY4 else None
        report[endpoint] = assess(candidates[endpoint].take_rows(slice(CALIBRATION_GLYPHS, None)), heads, exact, codes,
                                  relevant_days, fresh_mask)
        publish(report, OUTPUT, started, f"{endpoint}_scored")
    report["unique_exact_endpoint_pairs"] = len(exact.values)
    publish(report, OUTPUT, started, "complete")


if __name__ == "__main__":
    main()
