"""Compare four- and 16-repeat calibration on fixed held mice2 Task 1 v20 lists."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import AtlasCandidates, concat_atlas_candidates, describe_atlas_candidates, fit_atlas_metric_ranker, pack_atlas_candidates
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_predictive_likelihood import fit_predictive_likelihood
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import align_training_glyphs
from kaggle_task1_crossfit_candidate import _load_fold
from kaggle_task1_crossfit_metric_scale import fold_indices
from kaggle_task1_metric_scale import ANCHOR, MATCHED_ROWS_PER_EPOCH, MODEL_SEED, RANKER_BATCH, RANKER_EPOCHS, calibration_labels
from kaggle_task1_mice2_textverifier_recheck import MOUSE, fresh_cohort, tensor_hash
from kaggle_task1_microns_hanzi_recheck import fit_ocr_logits, with_hanzi
from kaggle_task1_native_hanzi import OCR_CHECKSUM, OCR_REVISION, response_views
from kaggle_task1_textverifier import publish
from kaggle_task1_v11_repeat_weight_transfer import POISSON_AMPLIFICATION, matched_pool, normalized_sources, pooled_scores, weighted_fusion
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, rendered_bank, summarize


SEED = 223497
OUTPUT = Path("/kaggle/working/task1_mice2_repeat_matched_ranker.json")
V20 = "v20"
DAY_HEAD = "matched_four_repeat"
POOLED_HEAD = "matched_sixteen_repeat"
PAIR_DAYS = (SESSIONS[::2], SESSIONS[1::2])


def stock_folds(path, splits, noise, key, device):
    results, hashes = [], []
    for fold, (metric, inner) in enumerate(splits):
        snapshot = path / f"fold-{fold}.pt"
        scope = {"format": 1, "key": key, "fold": fold, "metric": metric.tolist(), "inner": inner.tolist(), "noise": noise.tolist()}
        results.append(_load_fold(snapshot, scope, device, SESSIONS))
        hashes.append(sha256(snapshot.read_bytes()).hexdigest())
    return results, hashes


def candidate_graph(model, baseline, amplified, sources, ocr):
    stock = pack_atlas_candidates(baseline, NOMINATIONS, baseline="v8")
    v20 = with_hanzi(model, amplified, sources, ocr)
    ids, valid, _ = matched_pool(stock, pack_atlas_candidates(v20, NOMINATIONS, baseline="v8"))
    return describe_atlas_candidates(v20, NOMINATIONS, stock.baseline, stock.rank_depth, ids, valid)


def pooled_graph(model, sources_by_day, ocr, days):
    sources = {name: torch.stack([sources_by_day[day][name] for day in days]).mean(0)
               for name in sources_by_day[days[0]]}
    base, amplified = pooled_scores(model, sources)
    return candidate_graph(model, base, amplified, sources, torch.stack([ocr[day] for day in days]).mean(0))


def rank_view(model, noise, counts, day, ocr):
    baseline = atlas_scores(model, noise, counts, day)
    sources = normalized_sources(baseline)
    amplified = {**baseline, **weighted_fusion(model, sources, POISSON_AMPLIFICATION)}
    return candidate_graph(model, baseline, amplified, sources, ocr), sources


def model_of(graphs, exact):
    return fit_atlas_metric_ranker(concat_atlas_candidates(graphs), torch.cat(exact), seed=MODEL_SEED,
                                   epochs=RANKER_EPOCHS, batch_rows=RANKER_BATCH, rows_per_epoch=MATCHED_ROWS_PER_EPOCH)


def assess(graph, heads, exact):
    selected = {arm: ranker.select(graph, anchor_ocf=ANCHOR) for arm, ranker in heads.items()}
    values = {arm: exact.evaluate(candidate[:, None])[:, 0] for arm, candidate in selected.items()}
    return values, {arm: int((candidate != selected[V20]).sum()) for arm, candidate in selected.items()}


def comparisons(blocks):
    values = {arm: [block[arm] for block in blocks] for arm in blocks[0]}
    return {**{arm: summarize(answers, values[V20]) for arm, answers in values.items()},
            "matched16_vs_matched4": summarize(values[POOLED_HEAD], values[DAY_HEAD])}


def save_calibration(path, scope, graphs, exact, count):
    temporary = path.with_suffix(".tmp")
    torch.save({"scope": scope, "graphs": [{name: value.cpu() if isinstance(value, torch.Tensor) else value
                                          for name, value in vars(graph).items()} for graph in graphs],
                "exact": exact.cpu(), "unique_pairs": count}, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--stock-report", type=Path, required=True)
    parser.add_argument("--stock-cache", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use the remote Colab T4 runtime.")
    torch.set_num_threads(4)
    cpu, device = torch.device("cpu"), torch.device("cuda:0")
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_cache.mkdir(parents=True, exist_ok=True)
    root = args.data_root.resolve(strict=True) if args.data_root else locate_data()
    scorer = official_scorer(root)
    images = load_images(root, cpu)
    if images.shape != (2748, 64, 64) or images.dtype != torch.uint8:
        raise ValueError("Use the complete official labeled image gallery.")
    fit, held, independence = fresh_cohort(len(images))
    archived = json.loads(args.stock_report.read_text(encoding="utf-8"))
    if (archived["stage"] != "complete" or archived["mouse"] != MOUSE or archived["held_gallery_indices"] != held.tolist()
            or archived["fit_gallery_indices_sha256"] != tensor_hash(fit)
            or archived["training_raster_sha256"] != tensor_hash(images)):
        raise ValueError("The archived v20 folds must use this exact train/held scope and labeled gallery.")
    noise_rows, splits = fold_indices(len(fit), device)
    old, hashes = stock_folds(args.stock_cache, splits, noise_rows, archived["calibration_cache"]["key"], device)
    codepoints = old[0][3]
    if any(not torch.equal(codepoints, values[3]) for values in old):
        raise ValueError("Archived v20 folds must agree on atlas order.")
    script_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    scope = sha256((script_hash + sha256(args.stock_report.read_bytes()).hexdigest() + "".join(hashes)).encode()).hexdigest()
    report = {"gpu": torch.cuda.get_device_name(device), "torch_version": torch.__version__, "mouse": MOUSE,
              "script_sha256": script_hash, "calibration_scope_sha256": scope, "stock_fold_sha256": hashes,
              "stock_report_sha256": sha256(args.stock_report.read_bytes()).hexdigest(), "seed": SEED,
              "held_gallery_indices": held.tolist(), "reused_v20_reference_holdout": True, "cohort_independence": independence,
              "codepoint_order_sha256": tensor_hash(codepoints), "nominations": NOMINATIONS, "anchor_ocf": ANCHOR,
              "ocr_revision": OCR_REVISION, "ocr_sha256": OCR_CHECKSUM, "formal_queries_loaded": False,
              "head_budget": {"hidden": 32, "optimizer_updates": RANKER_EPOCHS * MATCHED_ROWS_PER_EPOCH // RANKER_BATCH,
                              "matched_calibration_glyphs": MATCHED_ROWS_PER_EPOCH},
              "protocol": "Each matched head uses the same 384 calibration glyphs, one row per glyph. The four-repeat "
                          "condition uses a balanced rotating day allocation. The 16-repeat condition pools all four days "
                          "by known training glyph identity. Both use the v20 score schema, fixed ranking objective and equal "
                          "head capacity, training rows and steps. At inference each condition's v20 candidate list and "
                          "renderer are shared by all heads. Pooled eight and same-day four are secondary endpoints.",
              "folds": {}, "conditions": {}}
    publish(report, args.output, started, "training_inputs_started")
    heads = {V20: model_of([value[0] for value in old], [value[1] for value in old])}
    outer_noise = old[0][2]
    del old
    counts = grouped_blocks(root, load_catalog(root), MOUSE, cpu, len(images))
    if tuple(counts) != SESSIONS or any(value.shape[:2] != (len(images), 4) for value in counts.values()):
        raise ValueError("Use all four labeled recordings and four actual repeats per recording.")
    fitting = images[fit].to(device)
    train = {day: block[fit].to(device) for day, block in counts.items()}
    parts = {DAY_HEAD: [], POOLED_HEAD: []}
    labels = {arm: [] for arm in parts}

    for fold, (metric, inner) in enumerate(splits):
        path = args.calibration_cache / f"{scope}-fold-{fold}.pt"
        fold_started = time.monotonic()
        if path.is_file():
            cached = torch.load(path, map_location=device, weights_only=True)
            if cached["scope"] != scope:
                raise ValueError("Stored calibration scope differs.")
            graphs = [AtlasCandidates(**value) for value in cached["graphs"]]
            exact, measured = cached["exact"], cached["unique_pairs"]
        else:
            model = fit_v8({day: block[inner] for day, block in train.items()}, fitting[inner], MODEL_SEED)
            if not torch.equal(codepoints, model.codepoints.cpu()):
                raise ValueError("The fold's font columns differ from the stock policy.")
            noise_glyphs, _ = align_training_glyphs(model.atlas, fitting[noise_rows])
            noise = {day: fit_predictive_likelihood(train[day][noise_rows], model.encoder.expected_counts(noise_glyphs, day))
                     for day in SESSIONS}
            views = response_views(counts, fit[inner.cpu()], fit[metric.cpu()], np.random.default_rng(SEED + fold))
            ocr = fit_ocr_logits(fitting[inner].cpu(), model.atlas.images, views)
            per_day, sources = {}, {}
            for day, block in train.items():
                per_day[day], sources[day] = rank_view(model, noise[day], block[metric], day, ocr[day])
            rows = torch.arange(len(metric), device=device)
            balanced = rows + ((rows + fold) % len(SESSIONS)) * len(metric)
            four = concat_atlas_candidates([per_day[day] for day in SESSIONS]).take_rows(balanced)
            sixteen = pooled_graph(model, sources, ocr, SESSIONS)
            graphs = [four, sixteen]
            exact, measured = calibration_labels(concat_atlas_candidates(graphs), rendered_bank(model), fitting[metric] >= 128, scorer)
            save_calibration(path, scope, graphs, exact, measured)
            del model, sources, noise, views, ocr, per_day
        for arm, graph, exact_rows in zip(parts, graphs, exact.split(len(metric)), strict=True):
            parts[arm].append(graph)
            labels[arm].append(exact_rows)
        report["folds"][str(fold)] = {"seconds": time.monotonic() - fold_started, "calibration_glyphs": len(metric),
                                       "unique_exact_pairs": measured, "snapshot_sha256": sha256(path.read_bytes()).hexdigest()}
        publish(report, args.output, started, f"fold_{fold}_scored")

    for arm in parts:
        heads[arm] = model_of(parts[arm], labels[arm])
    report["head_fit"] = {arm: head.diagnostics for arm, head in heads.items()}
    publish(report, args.output, started, "outer_started")
    model = fit_v8(train, fitting, MODEL_SEED)
    del train
    if not torch.equal(codepoints, model.codepoints.cpu()):
        raise ValueError("The outer model's font columns differ from the stock policy.")
    rendered = rendered_bank(model)
    report["outer_model"] = {**model.report, "rendered_sha256": tensor_hash(rendered),
                              "exact_archived_renderer": tensor_hash(rendered) == archived["v20_atlas"]["rendered_bank_sha256"]}
    references = images[held] >= 128
    exact = PairedOCF(scorer, rendered, references.numpy(), device, 12)
    ocr = fit_ocr_logits(fitting.cpu(), model.atlas.images, response_views(counts, fit, held, np.random.default_rng(SEED + 4)))
    by_group, sources = {}, {}
    for day, block in counts.items():
        graph, sources[day] = rank_view(model, outer_noise[day], block[held].to(device), day, ocr[day])
        values, changes = assess(graph, heads, exact)
        by_group[day] = values
        report["conditions"][day] = {"ocf": comparisons([values]), "changes_vs_v20": changes}
    report["same_day_four_repeats"] = comparisons(list(by_group.values()))
    publish(report, args.output, started, "same_day_four_complete")

    eight = []
    for days in PAIR_DAYS:
        graph = pooled_graph(model, sources, ocr, days)
        values, changes = assess(graph, heads, exact)
        eight.append(values)
        report["conditions"]["_".join(days)] = {"ocf": comparisons([values]), "changes_vs_v20": changes}
    report["pooled_eight_known_correspondence"] = comparisons(eight)
    graph = pooled_graph(model, sources, ocr, SESSIONS)
    values, changes = assess(graph, heads, exact)
    report["pooled_sixteen_known_correspondence"] = {"ocf": comparisons([values]), "changes_vs_v20": changes}
    report["archived_v20_replay"] = {
        "same_day_delta": report["same_day_four_repeats"][V20]["ocf"] - archived["same_day_four_repeats"][V20]["ocf"],
        "pooled_sixteen_delta": float(values[V20].mean()) - archived["pooled_16_known_correspondence"]["paired_exact_ocf"][V20]["ocf"],
    }
    publish(report, args.output, started, "primary_complete")
    oracle_rows = np.sort(np.random.default_rng(SEED + 5).choice(len(held), size=64, replace=False))
    oracle_graph = graph.take_rows(torch.from_numpy(oracle_rows).to(device))
    oracle, oracle_pairs = calibration_labels(oracle_graph, rendered, references[oracle_rows].to(device), scorer)
    best = oracle.masked_fill(~oracle_graph.valid, -torch.inf).amax(1).cpu().numpy()
    report["fixed_pool_oracle"] = {"ocf": summarize([best], [values[V20][oracle_rows]]),
                                  "held_gallery_indices": held[oracle_rows].tolist(), "unique_exact_pairs": oracle_pairs}
    report["unique_exact_endpoint_pairs"] = len(exact.values)
    publish(report, args.output, started, "complete")


if __name__ == "__main__":
    main()
