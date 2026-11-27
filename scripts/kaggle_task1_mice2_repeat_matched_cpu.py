"""One-stem Kaggle CPU pilot of matched-repeat OCF calibration on held v20 lists."""

from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import concat_atlas_candidates, fit_atlas_metric_ranker
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_predictive_likelihood import fit_predictive_likelihood
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import align_training_glyphs
from kaggle_task1_crossfit_metric_scale import fold_indices
from kaggle_task1_metric_scale import MODEL_SEED, RANKER_EPOCHS, calibration_labels
from kaggle_task1_mice2_repeat_matched_ranker import (
    DAY_HEAD, MOUSE, PAIR_DAYS, POOLED_HEAD, SEED, V20, assess, comparisons, fresh_cohort, model_of,
    pooled_graph, publish, rank_view, save_calibration, stock_folds, tensor_hash,
)
from kaggle_task1_microns_hanzi_recheck import fit_ocr_logits
from kaggle_task1_native_hanzi import response_views
from kaggle_v8_metric_validation import SESSIONS, fit_v8, rendered_bank, summarize


OUTPUT = Path("/kaggle/working/task1_mice2_repeat_matched_cpu.json")
HELD_GLYPHS = 128


def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Use the private Kaggle CPU Notebook with the official training data.")
    torch.set_num_threads(4)
    device = torch.device("cpu")
    started = time.monotonic()
    root = locate_data()
    official = official_scorer(root)
    images = load_images(root, device)
    stock_report = next(Path("/kaggle/input").rglob("stock_report.json"), None)
    if stock_report is None:
        raise FileNotFoundError("The archived v20 calibration folds must be mounted as a private dataset.")
    archived = json.loads(stock_report.read_text(encoding="utf-8"))
    fit, all_held, independence = fresh_cohort(len(images))
    if (images.shape != (2748, 64, 64) or images.dtype != torch.uint8 or archived["stage"] != "complete"
            or archived["mouse"] != MOUSE or archived["held_gallery_indices"] != all_held.tolist()
            or archived["fit_gallery_indices_sha256"] != tensor_hash(fit)
            or archived["training_raster_sha256"] != tensor_hash(images)):
        raise ValueError("The archived v20 fit scope and labeled gallery must match the pilot.")
    noise_rows, splits = fold_indices(len(fit), device)
    archived_folds, hashes = stock_folds(stock_report.parent, splits, noise_rows, archived["calibration_cache"]["key"], device)
    codepoints = archived_folds[0][3]
    if any(not torch.equal(codepoints, values[3]) for values in archived_folds):
        raise ValueError("Archived v20 folds must use the same public atlas order.")
    heads = {V20: model_of([fold[0] for fold in archived_folds], [fold[1] for fold in archived_folds])}
    del archived_folds
    metric, inner = (fit[rows] for rows in splits[0])
    noise_rows = fit[noise_rows]
    rng = np.random.default_rng(SEED + 99)
    held = all_held[torch.from_numpy(np.sort(rng.choice(len(all_held), HELD_GLYPHS, replace=False)))]
    combined = torch.cat((metric, held))
    calibration_size = len(metric)
    script_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"device": str(device), "mouse": MOUSE, "seed": SEED, "script_sha256": script_hash,
              "stock_report_sha256": sha256(stock_report.read_bytes()).hexdigest(), "stock_fold_sha256": hashes,
              "stock_ranker": "v20 reference architecture and archives refit on CPU",
              "held_gallery_indices": held.tolist(), "reused_v20_reference_holdout": True,
              "held_unseen_glyphs": len(all_held), "paired_holdout_glyphs": len(held),
              "fit_glyphs": len(inner), "calibration_glyphs": metric.tolist(), "noise_glyphs": len(noise_rows),
              "cohort_independence": independence, "formal_queries_loaded": False, "conditions": {},
              "protocol": "One v20 stem fits the literal first inner split. The same fixed candidate list per query "
                          "and output renderer apply to every head. Both matched heads train on the same 96 calibration "
                          "glyphs, equal hidden size and 640 optimizer steps. Assigned single days give four observed "
                          "repeats, and four known-aligned days give 16. All 320 external glyphs are excluded from the "
                          "fit and calibration, with the same 128 prespecified glyphs evaluated in every condition."}
    publish(report, OUTPUT, started, "stem_started")
    counts = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if tuple(counts) != SESSIONS or any(value.shape[:2] != (len(images), 4) for value in counts.values()):
        raise ValueError("Use all four labeled recordings and four actual repeats per day.")
    model = fit_v8({day: block[inner] for day, block in counts.items()}, images[inner], MODEL_SEED)
    if not torch.equal(codepoints, model.codepoints):
        raise ValueError("The CPU atlas column order must match the v20 archive.")
    rendered = rendered_bank(model)
    report["atlas"] = {**model.report, "codepoint_order_sha256": tensor_hash(codepoints), "rendered_sha256": tensor_hash(rendered)}
    publish(report, OUTPUT, started, "ocr_started")
    ocr = fit_ocr_logits(images[inner], model.atlas.images,
                         response_views(counts, inner, combined, np.random.default_rng(SEED + 4)))
    matched_noise, _ = align_training_glyphs(model.atlas, images[noise_rows])
    day_graphs, sources = {}, {}
    for day, block in counts.items():
        predictive = fit_predictive_likelihood(block[noise_rows], model.encoder.expected_counts(matched_noise, day))
        day_graphs[day], sources[day] = rank_view(model, predictive, block[combined], day, ocr[day])
        publish(report, OUTPUT, started, f"{day}_list_ready")
    sixteen = pooled_graph(model, sources, ocr, SESSIONS)
    rows = torch.arange(calibration_size)
    balanced = rows + (rows % len(SESSIONS)) * len(combined)
    four = concat_atlas_candidates(list(day_graphs.values())).take_rows(balanced)
    training = [four, sixteen.take_rows(rows)]
    publish(report, OUTPUT, started, "exact_calibration_started")
    labels, measured = calibration_labels(concat_atlas_candidates(training), rendered, images[metric] >= 128, official)
    save_calibration(OUTPUT.with_name("repeat_matched_cpu_fold.pt"), script_hash, training, labels, measured)
    for arm, graph, exact_rows in zip((DAY_HEAD, POOLED_HEAD), training, labels.split(calibration_size), strict=True):
        heads[arm] = fit_atlas_metric_ranker(graph, exact_rows, seed=MODEL_SEED, epochs=RANKER_EPOCHS, batch_rows=12)
    report.update(exact_calibration_pairs=measured, head_fit={arm: head.diagnostics for arm, head in heads.items()})
    publish(report, OUTPUT, started, "head_fit_complete")

    endpoint = PairedOCF(official, rendered, (images[held] >= 128).numpy(), device, 12)
    evaluation = slice(calibration_size, None)
    graph = sixteen.take_rows(evaluation)
    values, changes = assess(graph, heads, endpoint)
    report["pooled_sixteen_known_correspondence"] = {"ocf": comparisons([values]), "changes_vs_v20": changes}
    publish(report, OUTPUT, started, "primary_complete")
    secondary = {1: [], 2: []}
    for days in [(day,) for day in SESSIONS] + list(PAIR_DAYS):
        view = day_graphs[days[0]] if len(days) == 1 else pooled_graph(model, sources, ocr, days)
        score, changes = assess(view.take_rows(evaluation), heads, endpoint)
        secondary[len(days)].append(score)
        report["conditions"]["_".join(days)] = {"ocf": comparisons([score]), "changes_vs_v20": changes}
        publish(report, OUTPUT, started, f"{'_'.join(days)}_scored")
    report["same_day_four_repeats"] = comparisons(secondary[1])
    report["pooled_eight_known_correspondence"] = comparisons(secondary[2])
    sample = np.sort(rng.choice(len(held), size=24, replace=False))
    oracle_graph = graph.take_rows(torch.from_numpy(sample))
    oracle, hits = calibration_labels(oracle_graph, rendered, images[held[sample]] >= 128, official)
    best = oracle.masked_fill(~oracle_graph.valid, -torch.inf).amax(1).numpy()
    report["fixed_pool_oracle"] = {"ocf": summarize([best], [values[V20][sample]]), "unique_exact_pairs": hits,
                                  "held_gallery_indices": held[sample].tolist()}
    report["unique_exact_endpoint_pairs"] = len(endpoint.values)
    publish(report, OUTPUT, started, "complete")


if __name__ == "__main__":
    main()
