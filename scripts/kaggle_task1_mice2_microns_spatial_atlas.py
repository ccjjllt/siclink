"""Compare global and unit-specific MICrONS readouts in the full mice2 Task 1 atlas."""

from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import describe_atlas_candidates, pack_atlas_candidates
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_task1_bipartite_validation import image_proxy, recall
from kaggle_task1_crossfit_candidate import fit_crossfit_head
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE
from kaggle_task1_mice2_incumbent_crossfit import independent_split
from kaggle_task1_microns_open_atlas import ATLAS_CHUNK, fit_kernel_readout_from_views, publish
from kaggle_task1_microns_repeat_recheck import accumulate
from kaggle_task1_microns_spatial_readout import FIELD_WIDTH, RIDGE_PER_GLYPH, fit_spatial_readout, new_whole_glyphs, spatial_fields
from kaggle_task1_microns_transfer import BLACK_LEVEL, FRAMES, PUBLIC_ARCHIVE, STREAM, core_descriptors, load_released_core
from kaggle_task1_v11_repeat_weight_transfer import POISSON_AMPLIFICATION, matched_pool, normalized_sources, pooled_scores, weighted_fusion
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, nested_ranker, rendered_bank, summarize


SEED = 94123
MOUSE = "mice2"
V14 = "v14"
POISSON = "original_poisson_1p4"
GLOBAL = "published_full_kernel"
LOCAL = "published_unit_fields"
OUTPUT = Path("/kaggle/working/task1_mice2_microns_spatial_atlas.json")


def fresh_glyphs(size, device):
    cpu = torch.device("cpu")
    earlier = {"closed_spatial": new_whole_glyphs(size, cpu)[1].numpy(),
               "mice2_incumbent_calibration": independent_split(size, 19391, cpu)[1].numpy()}
    excluded = np.unique(np.concatenate(tuple(earlier.values())))
    eligible = np.setdiff1d(np.arange(size), excluded)
    held = np.sort(np.random.default_rng(SEED).choice(eligible, 320, replace=False))
    overlap = {name: int(np.intersect1d(held, rows).size) for name, rows in earlier.items()}
    fit = np.setdiff1d(np.arange(size), held)
    return torch.as_tensor(fit, device=device), torch.as_tensor(held, device=device), overlap


@torch.no_grad()
def rate_banks(core, kernel, spatial, atlas, count_features):
    shape = (len(SESSIONS), len(atlas), count_features)
    banks = {name: torch.empty(shape, dtype=torch.float32) for name in (GLOBAL, LOCAL)}
    for offset in range(0, len(atlas), ATLAS_CHUNK):
        block = atlas[offset:offset + ATLAS_CHUNK]
        views = core_descriptors(core, block, offload_core=False)
        whole = kernel.rates(views).reshape(len(block), len(SESSIONS), count_features).transpose(0, 1)
        fields = spatial.rates(torch.stack(views, dim=1).unflatten(-1, (-1, 64)))
        for name, expected in ((GLOBAL, whole), (LOCAL, fields)):
            if not bool(expected.isfinite().all()):
                raise ValueError(f"Nonfinite response rates in {name}.")
            banks[name][:, offset:offset + len(block)].copy_(expected.cpu())
        if offset % (ATLAS_CHUNK * 20) == 0 or offset + len(block) == len(atlas):
            print(f"{offset + len(block)}/{len(atlas)} common MICrONS glyph maps", flush=True)
    return banks


def new_evidence(model, stock, query, rates):
    fitted = {f"poisson_{view}": count_template_logits(query, rates, view) for view in ("rate", "timed")}
    replaced = {**stock, **fitted}
    sources = normalized_sources(replaced)
    return {**replaced, **weighted_fusion(model, sources, POISSON_AMPLIFICATION)}, sources


@torch.no_grad()
def score_arms(dense, ranker, proxy, trusted, exact, original_order):
    incumbent = pack_atlas_candidates(dense[V14], NOMINATIONS, baseline="v8")
    details, outcome = {}, {}
    for name, scores in dense.items():
        if name == V14:
            graph = incumbent
        else:
            proposal = pack_atlas_candidates(scores, NOMINATIONS, baseline=incumbent.baseline)
            ids, valid, _ = matched_pool(incumbent, proposal)
            graph = describe_atlas_candidates(scores, NOMINATIONS, incumbent.baseline, incumbent.rank_depth, ids, valid)
        chosen = ranker.select(graph, anchor_ocf=ANCHOR)
        details[name] = recall(scores, graph, proxy, trusted)
        outcome[name] = exact.evaluate(chosen[original_order, None])[:, 0]
    return outcome, details


def compare(values):
    output = {name: summarize(score, values[V14]) for name, score in values.items()}
    output["spatial_vs_full_kernel"] = summarize(values[LOCAL], values[GLOBAL])
    output["spatial_vs_original_poisson_amplified"] = summarize(values[LOCAL], values[POISSON])
    output["full_kernel_vs_original_poisson_amplified"] = summarize(values[GLOBAL], values[POISSON])
    return output


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run on one private Kaggle T4 with labeled input and public checkpoint access.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, held, overlap = fresh_glyphs(len(images), device)
    groups = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if tuple(groups) != SESSIONS or any(block.shape[1] != 4 for block in groups.values()):
        raise ValueError("Expected all four mice2 sessions with four labeled repeats per glyph.")
    training = {day: block[fit] for day, block in groups.items()}
    query = {day: block[held] for day, block in groups.items()}
    del groups
    observed = torch.stack([block.mean(1) for block in training.values()])
    stabilized = (observed + 0.1).sqrt().transpose(0, 1).flatten(1)
    report = {"mouse": MOUSE, "outer_seed": SEED, "held_glyphs": len(held), "fit_glyphs": len(fit), "prior_overlap": overlap,
              "field_width": FIELD_WIDTH, "field_relative_ridge": RIDGE_PER_GLYPH, "anchor_ocf": ANCHOR,
              "poisson_multiplier": POISSON_AMPLIFICATION,
              "core": {"source": PUBLIC_ARCHIVE, "frames": FRAMES, "stream": STREAM, "black_level": BLACK_LEVEL},
              "protocol": "Fit exact mice2 four-fold production scorer on visible training glyphs, then hold it fixed. "
                          "Compare published full-glyph and unit-specific readouts with the same frozen MICrONS glyph maps "
                          "and 1.4× Poisson weight. Include the original encoder at 1.4× and v14 weighting. All arms "
                          "have the exact same per-row nomination budget. Four measured same-day repeats are primary. "
                          "Known same-glyph correspondence for the pooled 16-repeat diagnostic is not available on formal data.",
              "sessions": {}}
    factory, digest = load_released_core()
    core = factory["published_core"]
    del factory
    report["core"]["sha256"] = digest
    scorer = official_scorer(root)
    split = split_calibration(len(fit), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    _, noise, _, codes = nested_ranker(training, images[fit], split, scorer, MODEL_SEED)
    ranker, folds = fit_crossfit_head(training, images[fit], scorer, codes)
    model = fit_v8(training, images[fit], MODEL_SEED)
    del training
    if not torch.equal(codes, model.codepoints.cpu()):
        raise ValueError("The frozen scorer and every treatment require identical atlas columns.")
    report.update(atlas_glyphs=len(codes), calibration_folds=folds)
    publish(report, OUTPUT, started, "v14_ready")

    views = core_descriptors(core, images[fit], offload_core=False)
    kernel = fit_kernel_readout_from_views(views, stabilized)
    maps = torch.stack(views, dim=1).unflatten(-1, (-1, 64))
    del views
    ink = images[fit].float().mean((1, 2)) / 255
    positions = torch.arange(len(fit), device=device)
    responses = observed.unflatten(-1, (-1, 10))
    fields, report["spatial_fit"] = spatial_fields(maps, positions, responses, ink, SEED + 1)
    readout = fit_spatial_readout(maps, fields["unit_field"], responses)
    del maps, fields, responses, stabilized
    banks = rate_banks(core, kernel, readout, model.atlas.images, observed.shape[-1])
    del core, kernel, readout, observed
    publish(report, OUTPUT, started, "both_atlas_rates_ready")

    proxy, trusted = image_proxy(model, images[held])
    exact = PairedOCF(scorer, rendered_bank(model), (images[held] >= 128).cpu().numpy(), device, 12)
    totals = {name: {} for name in (V14, GLOBAL, LOCAL)}
    measured = {}
    rng = np.random.default_rng(SEED + 2)
    for position, (day, trials) in enumerate(query.items()):
        order = torch.as_tensor(rng.permutation(len(held)), device=device)
        reverse = order.argsort()
        observed = trials[order]
        stock = atlas_scores(model, noise[day], observed, day)
        original_sources = normalized_sources(stock)
        scores = {V14: stock, POISSON: {**stock, **weighted_fusion(model, original_sources, POISSON_AMPLIFICATION)}}
        accumulate(totals[V14], original_sources, reverse)
        for name, bank in banks.items():
            scores[name], sources = new_evidence(model, stock, observed, bank[position].to(device))
            accumulate(totals[name], sources, reverse)
        outcome, recall_by_policy = score_arms(scores, ranker, proxy[order], trusted[order], exact, reverse)
        for name, values in outcome.items():
            measured.setdefault(name, []).append(values)
        report["sessions"][day] = {"recall": recall_by_policy, "ocf": compare({name: [value] for name, value in outcome.items()})}
        report["four_repeat_ocf"] = compare(measured)
        del stock, scores, original_sources, sources
        publish(report, OUTPUT, started, day)

    pooled = {name: pooled_scores(model, {key: value / len(SESSIONS) for key, value in total.items()})
              for name, total in totals.items()}
    policies = {V14: pooled[V14][0], POISSON: pooled[V14][1], GLOBAL: pooled[GLOBAL][1], LOCAL: pooled[LOCAL][1]}
    outcome, pooled_recall = score_arms(policies, ranker, proxy, trusted, exact, torch.arange(len(held), device=device))
    report["known_correspondence_16"] = {"ocf": compare({name: [value] for name, value in outcome.items()}), "recall": pooled_recall}
    publish(report, OUTPUT, started, "complete")


if __name__ == "__main__":
    main()
