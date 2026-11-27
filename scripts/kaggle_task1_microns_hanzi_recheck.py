"""Compare held mice1 v14, v16, and v16 with one fixed frozen-Hanzi score blend."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_atlas_metric_ranker import describe_atlas_candidates, pack_atlas_candidates
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import standardized
from kaggle_task1_bipartite_validation import image_proxy, recall, split_whole
from kaggle_task1_crossfit_candidate import fit_crossfit_policy
from kaggle_task1_ids_retrieval import split_glyphs as ids_holdout
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED
from kaggle_task1_microns_candidate import CORE_SHA256
from kaggle_task1_microns_open_atlas import (ATLAS_CHUNK, HELD_COUNT, RIDGE, SEED as MICRONS_OPEN_SEED,
                                             fit_kernel_readout, outer_split as microns_open_holdout, publish, stream_atlas)
from kaggle_task1_microns_repeat_recheck import accumulate, fresh_cohort as microns_repeat_holdout
from kaggle_task1_microns_transfer import (BLACK_LEVEL, FNN_VERSION, FRAMES, PUBLIC_ARCHIVE, SEED as MICRONS_CLOSED_SEED,
                                           STREAM, load_released_core)
from kaggle_task1_native_hanzi import (OCR_CHECKSUM, OCR_REVISION, OCR_SOURCE, PC_COUNT, RIDGE as OCR_RIDGE, SEED as OCR_SEED,
                                       STAGE, component_scope, context_scope, feature_extractors, response_views,
                                       split_glyphs as ocr_pilot_holdout)
from kaggle_task1_native_hanzi_open_atlas import image_coordinates, independent_split as ocr_full_holdout
from kaggle_task1_v11_repeat_weight_transfer import (POISSON_AMPLIFICATION, matched_pool, normalized_sources,
                                                      pooled_scores, score_selected, weighted_fusion)
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, rendered_bank, summarize


MOUSE = "mice1"
SEED = 222091
ATLAS_SIZE = 27_550
V14, V16 = "v14", "v16_microns_1p4"
FULL, FIXED, NOMINATE = "v16_hanzi", "v16_hanzi_fixed_pool", "v16_hanzi_nominations_only"
CONTOUR_OCR_WEIGHT = 0.5
OUTPUT = Path("/kaggle/working/task1_microns_hanzi_recheck.json")


def fresh_cohort(size):
    cpu = torch.device("cpu")
    bipartite = split_whole(size, 36467, cpu)[1]
    ids = ids_holdout(size, cpu)[1]
    closed_pool = np.setdiff1d(np.arange(size), np.union1d(bipartite, ids))
    prior = {"hanzi_pilot": ocr_pilot_holdout(size, np.random.default_rng(OCR_SEED))[1],
             "hanzi_full_atlas": ocr_full_holdout(size)[1],
             "microns_repeat": microns_repeat_holdout(size, cpu)[1],
             "microns_open_atlas": microns_open_holdout(size, MICRONS_OPEN_SEED, cpu)[1],
             "microns_closed": np.random.default_rng(MICRONS_CLOSED_SEED).choice(closed_pool, HELD_COUNT, replace=False),
             "component_raster": component_scope(size)[1], "context_prediction": context_scope(size)[1]}
    excluded = np.unique(np.concatenate(tuple(prior.values())))
    eligible = np.setdiff1d(np.arange(size), excluded)
    if len(eligible) < HELD_COUNT:
        raise ValueError(f"Only {len(eligible)} glyphs remain outside the fixed earlier cohorts.")
    held = np.sort(np.random.default_rng(SEED).choice(eligible, HELD_COUNT, replace=False))
    fit = np.setdiff1d(np.arange(size), held)
    overlaps = {name: int(np.intersect1d(held, old).size) for name, old in prior.items()}
    return torch.from_numpy(fit), torch.from_numpy(held), {"eligible": len(eligible), "overlap_by_cohort": overlaps}


def calibration_key(fit, held, labeled, groups, scorer, device, *, mouse=MOUSE):
    """Identify saved folds by the ordered fit data, calibration code and runtime."""
    settings = {"mouse": mouse, "seed": MODEL_SEED, "anchor": ANCHOR, "nominations": NOMINATIONS,
                "torch": str(torch.__version__), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "accelerator": torch.cuda.get_device_name(device), "threads": torch.get_num_threads(),
                "libraries": {name: version(name) for name in ("numpy", "scipy", "Pillow", "fonttools")}}
    digest = sha256(json.dumps(settings, sort_keys=True).encode())
    official = Path(scorer.__code__.co_filename).parent
    sources = sorted(Path(__file__).resolve().parent.glob("*.py"))
    scoring = [official.parent / "__init__.py", *sorted(official.glob("*.py"))]
    for section, paths in (("experiment", sources), ("organizer", scoring)):
        for path in paths:
            digest.update(f"{section}/{path.name}:".encode())
            digest.update(sha256(path.read_bytes()).digest())
    tensors = (("fit", fit), ("held", held), ("images", labeled))
    for name, values in tensors:
        array = values.contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}:".encode())
        digest.update(memoryview(array).cast("B"))
    for day, counts in groups.items():
        array = counts[fit].contiguous().numpy()
        digest.update(f"{day}:{array.dtype}:{array.shape}:".encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


@torch.no_grad()
def fit_ocr_atlas(labeled, atlas):
    frozen = feature_extractors()["frozen_hanzi"]
    coordinates, prototypes = image_coordinates(labeled.cpu(), atlas.cpu(), {"frozen_hanzi": frozen})["frozen_hanzi"]
    return coordinates, prototypes.to(atlas.device)


@torch.no_grad()
def fit_ocr_logits(labeled, atlas, views):
    coordinates, prototypes = fit_ocr_atlas(labeled, atlas)
    return {day: standardized(F.normalize(query @ (ridge_map @ coordinates)).to(atlas.device) @ prototypes)
            for day, (ridge_map, _, query) in views.items()}


@torch.no_grad()
def with_hanzi(model, v16, sources, ocr):
    decoded = standardized(sources["decoded_contour"])
    combined = (1 - CONTOUR_OCR_WEIGHT) * decoded + CONTOUR_OCR_WEIGHT * standardized(ocr)
    # Only the v4/v8 composites see the blend. Keep every score field expected by the stock ranker.
    mixed = {**sources, "decoded_contour": combined}
    return {**v16, **weighted_fusion(model, mixed, POISSON_AMPLIFICATION)}


@torch.no_grad()
def evaluate(exact, base, v16, fusion, ranker, proxy, trusted, inverse):
    stock = pack_atlas_candidates(base, NOMINATIONS, baseline="v8")
    v16_ids, v16_valid, _ = matched_pool(stock, pack_atlas_candidates(v16, NOMINATIONS, baseline="v8"))
    ocr_ids, ocr_valid, _ = matched_pool(stock, pack_atlas_candidates(fusion, NOMINATIONS, baseline="v8"))

    def describe(scores, indexes, valid):
        return describe_atlas_candidates(scores, NOMINATIONS, stock.baseline, stock.rank_depth, indexes, valid)

    graphs = {V14: stock, V16: describe(v16, v16_ids, v16_valid), FULL: describe(fusion, ocr_ids, ocr_valid),
              FIXED: describe(fusion, v16_ids, v16_valid), NOMINATE: describe(v16, ocr_ids, ocr_valid)}
    maps = {V14: base, V16: v16, FULL: fusion, FIXED: fusion, NOMINATE: v16}
    selected = {name: ranker.select(graph, anchor_ocf=ANCHOR) for name, graph in graphs.items()}
    budget = stock.valid.sum(1)
    if (any(not torch.equal(graph.valid.sum(1), budget) for graph in graphs.values())
            or any(bool((pick < 0).any()) for pick in selected.values())):
        raise ValueError("Every arm must keep the same number of distinct eligible candidates per glyph as v14.")
    common = ((ocr_ids[:, :, None] == v16_ids[:, None, :]) & v16_valid[:, None]).any(dim=2)
    novel = (~common & ocr_valid).sum(1)
    diagnostics = {"same_distinct_candidate_budget": True, "mean_candidate_count": float(budget.float().mean()),
                   "hanzi_mean_novel_candidates_vs_v16": float(novel.float().mean()),
                   "hanzi_rows_with_new_candidates_vs_v16": int((novel > 0).sum()),
                   "recall": {name: recall(maps[name], graph, proxy, trusted) for name, graph in graphs.items()},
                   "selected_changes_vs_v16": {name: int((choice != selected[V16]).sum()) for name, choice in selected.items()}}
    metrics = score_selected(exact, {name: choice[inverse] for name, choice in selected.items()})
    return metrics, diagnostics


def compare(values):
    return {"vs_v14": {name: summarize(scores, values[V14]) for name, scores in values.items()},
            "vs_v16": {name: summarize(values[name], values[V16]) for name in (FULL, FIXED, NOMINATE)},
            "full_vs_fixed_pool": summarize(values[FULL], values[FIXED]),
            "full_vs_nominations_only": summarize(values[FULL], values[NOMINATE])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Extracted competition directory containing catalog.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--calibration-cache", type=Path, help="Save and resume fit-only calibration folds in this directory")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use a remote Kaggle or Colab GPU for the exact held calibration and open-atlas scores.")
    started = time.monotonic()
    torch.set_num_threads(4)
    cpu, device = torch.device("cpu"), torch.device("cuda:0")
    root = args.data_root.resolve(strict=True) if args.data_root else locate_data()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scorer = official_scorer(root)
    images = load_images(root, cpu)
    if len(images) != 2748:
        raise ValueError("Expected the complete 2,748-glyph official training gallery.")
    fit, held, independence = fresh_cohort(len(images))
    groups = grouped_blocks(root, load_catalog(root), MOUSE, cpu, len(images))
    if tuple(groups) != SESSIONS or any(counts.shape[:2] != (len(images), 4) for counts in groups.values()):
        raise ValueError("Expected four labeled recordings with four real repeats for every glyph.")
    labeled_cpu = images[fit]
    cache_key, cache_dir = None, None
    if args.calibration_cache is not None:
        cache_key = calibration_key(fit, held, labeled_cpu, groups, scorer, device)
        cache_dir = args.calibration_cache.resolve() / cache_key
    views = response_views(groups, fit, held, np.random.default_rng(SEED + 1))
    training = {day: counts[fit].to(device) for day, counts in groups.items()}
    queries = {day: counts[held].to(device) for day, counts in groups.items()}
    del groups
    labeled, references = labeled_cpu.to(device), images[held].to(device)
    del labeled_cpu
    stabilized = (torch.stack([counts.mean(1) for counts in training.values()]) + 0.1).sqrt().transpose(0, 1).flatten(1)
    feature_count = next(iter(training.values())).shape[-1]
    cores, checkpoint = load_released_core(work_dir=None)
    if checkpoint != CORE_SHA256:
        raise ValueError("The pretrained core must use the same checkpoint as deployed v16.")
    core = cores["published_core"].float().requires_grad_(False)
    del cores
    report = {"mouse": MOUSE, "outer_seed": SEED, "v14_seed": MODEL_SEED, "fit_glyphs": len(fit), "held_glyphs": len(held),
              "held_gallery_indices": held.tolist(), "cohort_independence": independence,
              "repeat_conditions": {"single_day": 4, "known_correspondence_pooled": 16}, "formal_queries_loaded": False,
              "anchor_ocf": ANCHOR, "nominations": NOMINATIONS, "poisson_multiplier": POISSON_AMPLIFICATION,
              "fixed_ocr_share_in_decoded_contour": CONTOUR_OCR_WEIGHT, "sessions": {},
              "protocol": "Refit the v14 graph strictly outside the whole-glyph holdout, keeping the four folds and literal fold-zero "
                          "uncertainty. Refit v16 MICrONS rates on the same labeled support. All arms use the same 27,550 public glyphs "
                          "and v14 styled output images. Only the v4/v8 composites mix equally row-standardized decoded-contour and "
                          "frozen-Hanzi scores. Preserve v14 candidate counts and all other ranker fields. The primary comparison is "
                          "v16+Hanzi versus v16 under pooled 16 measured repeats with known held correspondence; four real repeats in "
                          "each recording are reported alongside. Held references only supply scores and retrospective atlas proxies.",
              "microns": {"version": FNN_VERSION, "archive": PUBLIC_ARCHIVE, "sha256": checkpoint, "frames": FRAMES,
                           "stream": STREAM, "black_level": BLACK_LEVEL, "kernel_ridge": RIDGE, "atlas_chunk": ATLAS_CHUNK},
              "ocr": {"source": OCR_SOURCE, "revision": OCR_REVISION, "sha256": OCR_CHECKSUM, "stage": STAGE,
                      "pca_components": PC_COUNT, "ridge_per_fit_glyph": OCR_RIDGE}}
    if cache_dir is not None:
        report["calibration_cache"] = {"key": cache_key, "directory": str(cache_dir)}
        print(f"Calibration snapshots: {cache_dir}/fold-*.pt", flush=True)
    publish(report, args.output, started, "microns_source_verified")

    ranker, folds, noise, codepoints = fit_crossfit_policy(training, labeled, scorer, cache_dir=cache_dir, cache_key=cache_key)
    model = fit_v8(training, labeled, MODEL_SEED)
    del training
    if len(codepoints) != ATLAS_SIZE or not torch.equal(codepoints, model.codepoints.cpu()):
        raise ValueError("All four folds and the outer model must use the exact public atlas columns.")
    if cache_dir is not None:
        font, unihan = model.report["atlas"]["sha256"], model.report["prior"]["unicode"]["sha256"]
        for fold in folds.values():
            source = fold["atlas_and_font_style"]
            if (source["atlas"]["sha256"], source["prior"]["unicode"]["sha256"]) != (font, unihan):
                raise ValueError("Calibration cache and outer model must have identical rendered-font and Unihan sources.")
    rendered = rendered_bank(model)
    report["v14_fit"] = {"noise": "literal calibration fold zero", "outer": model.report, "folds": folds,
                         "ranker": ranker.diagnostics, "atlas_size": ATLAS_SIZE,
                         "codepoint_order_sha256": sha256(codepoints.contiguous().numpy().tobytes()).hexdigest(),
                         "rendered_pixel_sha256": sha256(rendered.cpu().contiguous().numpy().tobytes()).hexdigest()}
    publish(report, args.output, started, "v14_ready")

    readout = fit_kernel_readout(core, labeled, stabilized)
    rate_bank, milestones = stream_atlas(core, readout, model.atlas.images, feature_count, ATLAS_CHUNK)
    del core, readout, stabilized
    report["microns"].update(atlas_rate_shape=list(rate_bank.shape), streaming_progress=milestones, kernel_solves=1)
    publish(report, args.output, started, "v16_microns_rates_ready")
    ocr = fit_ocr_logits(labeled, model.atlas.images, views)
    del labeled, views, images
    publish(report, args.output, started, "hanzi_scores_ready")

    proxy, trusted = image_proxy(model, references)
    report["trusted_proxy_glyphs"] = int(trusted.sum())
    exact = PairedOCF(scorer, rendered, (references >= 128).cpu().numpy(), device, 12)
    rng = np.random.default_rng(SEED + 2)
    total_stock, total_v16, measured = {}, {}, {}
    for day_index, (day, counts) in enumerate(queries.items()):
        order = torch.as_tensor(rng.permutation(len(held)), device=device)
        inverse = order.argsort()
        query = counts[order]
        base = atlas_scores(model, noise[day], query, day)
        rates = rate_bank[day_index].to(device)
        microns = {**base, **{f"poisson_{form}": count_template_logits(query, rates, form) for form in ("rate", "timed")}}
        sources = normalized_sources(microns)
        v16 = {**microns, **weighted_fusion(model, sources, POISSON_AMPLIFICATION)}
        fusion = with_hanzi(model, v16, sources, ocr[day][order])
        values, diagnostics = evaluate(exact, base, v16, fusion, ranker, proxy[order], trusted[order], inverse)
        for name, outcomes in values.items():
            measured.setdefault(name, []).append(outcomes)
        report["sessions"][day] = {**diagnostics, "paired_exact_ocf": compare({name: [outcome] for name, outcome in values.items()})}
        accumulate(total_stock, normalized_sources(base), inverse)
        accumulate(total_v16, sources, inverse)
        del base, microns, v16, fusion, sources, rates
        report["same_day_four_repeats"] = compare(measured)
        publish(report, args.output, started, day)

    del rate_bank, queries
    stock = {name: values.div_(len(SESSIONS)) for name, values in total_stock.items()}
    core_sources = {name: values.div_(len(SESSIONS)) for name, values in total_v16.items()}
    base, _ = pooled_scores(model, stock)
    _, v16 = pooled_scores(model, core_sources)
    fusion = with_hanzi(model, v16, core_sources, torch.stack(tuple(ocr.values())).mean(0))
    values, diagnostics = evaluate(exact, base, v16, fusion, ranker, proxy, trusted, torch.arange(len(held), device=device))
    report["pooled_16_known_correspondence"] = {**diagnostics, "paired_exact_ocf": compare({name: [value] for name, value in values.items()})}
    day_delta = (np.stack(measured[FULL]) - np.stack(measured[V16])).mean(0)
    report["hanzi_pooled_16_vs_mean_day_4_gain"] = summarize([values[FULL] - values[V16]], [day_delta])
    report["unique_exact_pairs_scored"] = len(exact.values)
    publish(report, args.output, started, "complete")


if __name__ == "__main__":
    main()
