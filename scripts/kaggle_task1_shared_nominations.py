"""Test cross-mouse visual nominations with the fixed v20 selector and renderer."""

from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_atlas_metric_ranker import concat_atlas_candidates, describe_atlas_candidates
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_predictive_likelihood import fit_predictive_likelihood
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import align_training_glyphs, standardized
from kaggle_task1_common_visual_readout import ACROSS, MOUSE, PEER, SHUFFLED, WITHIN, cohort_shared
from kaggle_task1_crossfit_metric_scale import fold_indices
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED, calibration_labels
from kaggle_task1_mice2_repeat_matched_cpu import HELD_GLYPHS as PREVIOUS_REPEAT_ROWS
from kaggle_task1_mice2_repeat_matched_ranker import (
    SEED as REPEAT_SEED, candidate_graph, fresh_cohort, model_of, publish, stock_folds, tensor_hash,
)
from kaggle_task1_microns_hanzi_recheck import with_hanzi
from kaggle_task1_native_hanzi import feature_extractors, response_views
from kaggle_task1_native_hanzi_open_atlas import image_coordinates, visual_proxy
from kaggle_task1_native_witness_utility import QUERY_GLYPHS as WITNESS_ROWS, SEED as WITNESS_SEED
from kaggle_task1_v11_repeat_weight_transfer import POISSON_AMPLIFICATION, normalized_sources, pooled_scores, weighted_fusion
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, rendered_bank, summarize


SEED, SCORED, ORACLE_ROWS, REPLACEMENTS = 224071, 112, 24, 4
V20, ARMS = "v20", (WITHIN, ACROSS, SHUFFLED)
OUTPUT = Path("/kaggle/working/task1_shared_nominations.json")


def evaluation_cohort(outer, rng):
    repeat = np.random.default_rng(REPEAT_SEED + 99).choice(len(outer), PREVIOUS_REPEAT_ROWS, replace=False)
    left = np.setdiff1d(np.arange(len(outer)), repeat)
    witness = np.random.default_rng(WITNESS_SEED).choice(left, WITNESS_ROWS, replace=False)
    remaining = np.setdiff1d(left, witness)
    selected = np.sort(rng.choice(remaining, SCORED, replace=False))
    return outer[torch.from_numpy(selected)], len(remaining)


def nominated_graph(stock, stock_scores, ranked, nomination):
    rows, cols = stock.valid.nonzero(as_tuple=True)
    available = nomination.clone()
    available[rows, stock.indices[rows, cols]] = -torch.inf
    new = available.topk(REPLACEMENTS, dim=1)
    if not bool(torch.isfinite(new.values).all()) or not bool((stock.valid.sum(1) > REPLACEMENTS).all()):
        raise ValueError("Each query needs four available proposals and at least one preserved candidate.")
    lowest = ranked.masked_fill(~stock.valid, torch.inf).topk(REPLACEMENTS, largest=False, dim=1).indices
    indices = stock.indices.clone().scatter_(1, lowest, new.indices)
    return describe_atlas_candidates(stock_scores, NOMINATIONS, stock.baseline, stock.rank_depth, indices, stock.valid)


def final_graphs(model, base, amplified, sources, ocr, incoming, ranker):
    stock = candidate_graph(model, base, amplified, sources, ocr)
    stock_scores = with_hanzi(model, amplified, sources, ocr)
    ranked = ranker.rank(stock, anchor_ocf=ANCHOR)
    for name in ARMS:
        incoming[name] = model.prior.variants(incoming[name])["usage_unseen"]
    return {V20: stock, **{name: nominated_graph(stock, stock_scores, ranked, incoming[name]) for name in ARMS}}


def paired_results(values):
    return {**{name: summarize([score], [values[V20]]) for name, score in values.items()},
            "joint_vs_within": summarize([values[ACROSS]], [values[WITHIN]]),
            "joint_vs_shuffled": summarize([values[ACROSS]], [values[SHUFFLED]])}


def assess(graphs, ranker, scorer, rendered, reference, proxy, trusted, oracle_rows, device):
    exact = PairedOCF(scorer, rendered, (reference >= 128).numpy(), device, 12)
    chosen = {name: ranker.select(graph, anchor_ocf=ANCHOR) for name, graph in graphs.items()}
    values = {name: exact.evaluate(indices[:, None])[:, 0] for name, indices in chosen.items()}
    ceilings, recall = {}, {}
    oracle = concat_atlas_candidates([graph.take_rows(oracle_rows) for graph in graphs.values()])
    exact_oracle, oracle_pairs = calibration_labels(oracle, rendered, reference[oracle_rows] >= 128, scorer)
    best = exact_oracle.masked_fill(~oracle.valid, -torch.inf).amax(1).numpy().reshape(len(graphs), len(oracle_rows))
    for (name, graph), observed in zip(graphs.items(), best, strict=True):
        matches = (graph.indices == proxy[:, None]) & graph.valid
        recall[name] = float(matches[trusted].any(1).float().mean()) if bool(trusted.any()) else None
        ceilings[name] = observed
    return {"selected_ocf": paired_results(values), "oracle_ocf": paired_results(ceilings),
            "trusted_proxy_coverage": recall, "mean_distinct_candidates": float(graphs[V20].valid.sum(1).float().mean()),
            "changed_vs_v20": {name: int((candidate != chosen[V20]).sum()) for name, candidate in chosen.items()},
            "unique_exact_endpoint_pairs": len(exact.values), "unique_exact_oracle_pairs": oracle_pairs}


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    rng = np.random.default_rng(SEED)
    root = locate_data()
    images = load_images(root, device)
    catalog = load_catalog(root)
    fit, outer, overlap = fresh_cohort(len(images))
    held, eligible = evaluation_cohort(outer, rng)
    oracle_rows = torch.from_numpy(np.sort(rng.choice(len(held), ORACLE_ROWS, replace=False)))
    stock_path = next(Path("/kaggle/input").rglob("stock_report.json"))
    archived = json.loads(stock_path.read_text())
    if (archived["stage"] != "complete" or archived["mouse"] != MOUSE
            or archived["held_gallery_indices"] != outer.tolist()
            or archived["fit_gallery_indices_sha256"] != tensor_hash(fit)
            or archived["training_raster_sha256"] != tensor_hash(images)):
        raise ValueError("Mount the v20 training-only fold archive for the exact held split.")
    noise, splits = fold_indices(len(fit), device)
    folds, hashes = stock_folds(stock_path.parent, splits, noise, archived["calibration_cache"]["key"], device)
    codepoints = folds[0][3]
    if any(not torch.equal(codepoints, fold[3]) for fold in folds):
        raise ValueError("The v20 ranking folds need the same atlas column order.")
    ranker = model_of([fold[0] for fold in folds], [fold[1] for fold in folds])
    del folds
    inner, noise = fit[splits[0][1]], fit[noise]
    counts = {mouse: grouped_blocks(root, catalog, mouse, device, len(images)) for mouse in (MOUSE, PEER)}
    if any(tuple(blocks) != SESSIONS or any(block.shape[1] != 4 for block in blocks.values())
           for blocks in counts.values()):
        raise ValueError("Use all four labeled recording days per mouse with four measured repeats each.")
    report = {"mouse": MOUSE, "peer_training_only": PEER, "seed": SEED, "formal_queries_loaded": False,
              "stock_fold_sha256": hashes, "stock_report_sha256": sha256(stock_path.read_bytes()).hexdigest(),
              "fit_glyph_count": len(inner), "eligible_unscored_in_reference_cohort": eligible,
              "outer_cohort_overlap": overlap, "held_gallery_indices": held.tolist(),
              "oracle_gallery_indices": held[oracle_rows].tolist(), "replacement_slots": REPLACEMENTS,
              "protocol": "Use the v20 candidate budget, score features, exact fold head and renderer. Replace the four lowest "
                          "ranked v20 candidates with proposals outside the original row from the within-mouse, true pair, or "
                          "shuffled pair shared visual heads. The new candidates are judged by the unchanged v20 scorer. "
                          "Four repeats use a balanced day per glyph. Sixteen repeats use known within-mouse labels across "
                          "days. Held peer observations and held references do not propose or choose any candidate."}
    publish(report, OUTPUT, started, "stem_started")
    model = fit_v8({day: block[inner] for day, block in counts[MOUSE].items()}, images[inner], MODEL_SEED)
    if not torch.equal(codepoints, model.codepoints):
        raise ValueError("The v20 stem and archived ranker must use the same public atlas.")
    rendered = rendered_bank(model)
    publish(report, OUTPUT, started, "shared_readouts_started")
    queries = {(MOUSE, day): data[held] for day, data in counts[MOUSE].items()}
    with torch.no_grad():
        features = feature_extractors()
        features.pop("random_mobilenet")
        coordinates = image_coordinates(images[inner], model.atlas.images, features)
        training = {(mouse, day): data[inner] for mouse, blocks in counts.items() for day, data in blocks.items()}
        shared, diagnostic = cohort_shared(training, queries, coordinates, torch.from_numpy(rng.permutation(len(inner))))
        labels, prototypes = coordinates["frozen_hanzi"]
        views = response_views(counts[MOUSE], inner, held, rng)
        ocr = {day: standardized(F.normalize(query @ (ridge @ labels)) @ prototypes)
               for day, (ridge, _, query) in views.items()}
    del training, coordinates
    report.update(atlas_columns=len(codepoints), shared_readouts=diagnostic, rendered_sha256=tensor_hash(rendered))
    publish(report, OUTPUT, started, "shared_readouts_ready")
    aligned, _ = align_training_glyphs(model.atlas, images[noise])
    sources, base_by_day, amplified_by_day = {}, {}, {}
    for day in SESSIONS:
        calibration = fit_predictive_likelihood(counts[MOUSE][day][noise], model.encoder.expected_counts(aligned, day))
        base = atlas_scores(model, calibration, queries[MOUSE, day], day)
        sources[day] = normalized_sources(base)
        base_by_day[day] = base
        amplified_by_day[day] = {**base, **weighted_fusion(model, sources[day], POISSON_AMPLIFICATION)}
        publish(report, OUTPUT, started, f"{day}_scored")
    del counts, queries

    glyph_rows = torch.arange(len(held))
    days = torch.from_numpy(rng.permutation(len(held)) % len(SESSIONS))

    def combine(values, pooled):
        stack = torch.stack(values)
        return stack.mean(0) if pooled else stack[days, glyph_rows]

    proxy, trusted, _ = visual_proxy(model.atlas, images[held])
    report["trusted_glyph_proxy_count"] = int(trusted.sum())
    scorer = official_scorer(root)
    for condition, pooled in (("same_day_four", False), ("known_correspondence_sixteen", True)):
        raw = {name: combine([sources[day][name] for day in SESSIONS], pooled) for name in sources[SESSIONS[0]]}
        ocr_score = combine([ocr[day] for day in SESSIONS], pooled)
        nominations = {name: combine([shared[name][MOUSE, day] for day in SESSIONS], pooled) for name in ARMS}
        if pooled:
            base, amplified = pooled_scores(model, raw)
        else:
            base, amplified = ({name: combine([recording[day][name] for day in SESSIONS], False)
                                for name in recording[SESSIONS[0]]} for recording in (base_by_day, amplified_by_day))
        graphs = final_graphs(model, base, amplified, raw, ocr_score, nominations, ranker)
        report[condition] = assess(graphs, ranker, scorer, rendered, images[held], proxy, trusted, oracle_rows, device)
        publish(report, OUTPUT, started, condition)
    publish(report, OUTPUT, started, "complete")


if __name__ == "__main__":
    main()
