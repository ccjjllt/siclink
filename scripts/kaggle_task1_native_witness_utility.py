"""Select from fixed v20 glyphs using exact OCF against measured native training neighbors."""

from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_bipartite_encoding import timing_view
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_predictive_likelihood import fit_predictive_likelihood
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import align_training_glyphs, image_descriptors, standardized
from kaggle_task1_crossfit_metric_scale import fold_indices
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED
from kaggle_task1_mice2_repeat_matched_cpu import HELD_GLYPHS as PRIOR_PILOT_GLYPHS
from kaggle_task1_mice2_repeat_matched_ranker import (
    MOUSE, SEED as PRIOR_PILOT_SEED, fresh_cohort, model_of, pooled_graph, publish, rank_view, stock_folds, tensor_hash,
)
from kaggle_task1_microns_hanzi_recheck import fit_ocr_logits
from kaggle_task1_native_hanzi import response_views
from kaggle_v8_metric_validation import SESSIONS, fit_v8, rendered_bank, summarize


SEED = 223789
QUERY_GLYPHS = 48
WITNESSES = 2
UTILITY_SHARE = 0.20
OUTPUT = Path("/kaggle/working/task1_native_witness_utility.json")


@torch.no_grad()
def crossnobis_neighbors(counts, native_rows, query_rows, rng):
    native = timing_view(counts[native_rows].unflatten(-1, (-1, 10)))
    query = timing_view(counts[query_rows].unflatten(-1, (-1, 10)))
    centered = (native - native.mean(dim=1, keepdim=True)).flatten(0, 1)
    covariance = centered.T @ centered / (len(native_rows) * (native.shape[1] - 1))
    diagonal = covariance.diag().clamp_min(0.02)
    covariance.mul_(0.75).diagonal().add_(0.25 * diagonal + 1e-4)
    factor = torch.linalg.cholesky(covariance)

    def halves(values):
        order = torch.from_numpy(rng.random(values.shape[:2]).argsort(axis=1))
        ordered = values.gather(1, order[:, :, None].expand_as(values))
        means = ordered.unflatten(1, (2, 2)).mean(2)
        whitened = torch.linalg.solve_triangular(factor, means.flatten(0, 1).T, upper=False).T
        return whitened.reshape(means.shape).unbind(1)

    native_a, native_b = halves(native)
    query_a, query_b = halves(query)
    distance = (query_a * query_b).sum(1)[:, None] + (native_a * native_b).sum(1)[None, :]
    distance -= query_a @ native_b.T + query_b @ native_a.T
    return standardized(-distance)


@torch.no_grad()
def witness_utility(graph, rendered, native, witnesses, score):
    rows, columns = graph.valid.nonzero(as_tuple=True)
    candidate = graph.indices[rows, columns]
    witness_indices = torch.stack(tuple(witnesses.values()))[:, rows]
    keys = candidate[None, :, None] * len(native) + witness_indices
    unique, reverse = torch.unique(keys.flatten(), return_inverse=True)
    scored = []
    total = 0
    for batch in unique.split(384):
        prediction = rendered[batch.div(len(native), rounding_mode="floor")]
        reference = native[batch.remainder(len(native))]
        values = score((prediction >= 128).numpy(), (reference >= 128).numpy(), device="cpu", batch_size=12)
        scored.append(torch.as_tensor(values, dtype=torch.float32))
        total += len(batch)
        print(f"Official candidate/witness OCF: {total:,}/{len(unique):,}", flush=True)
    values = torch.cat(scored)[reverse].view_as(keys).mean(2)
    utilities = {}
    for name, value in zip(witnesses, values, strict=True):
        utility = torch.full_like(graph.baseline_z, -torch.inf)
        utility[rows, columns] = value
        utilities[name] = utility
    return utilities, len(unique)


def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Run in the private Kaggle CPU Notebook.")
    torch.set_num_threads(4)
    device = torch.device("cpu")
    started = time.monotonic()
    root = locate_data()
    scorer = official_scorer(root)
    images = load_images(root, device)
    stock = next(Path("/kaggle/input").rglob("stock_report.json"), None)
    if stock is None:
        raise FileNotFoundError("Mount the existing private v20 training-only fold input.")
    archived = json.loads(stock.read_text(encoding="utf-8"))
    fit, outer, independence = fresh_cohort(len(images))
    if (images.shape != (2748, 64, 64) or images.dtype != torch.uint8 or archived["stage"] != "complete"
            or archived["mouse"] != MOUSE or archived["held_gallery_indices"] != outer.tolist()
            or archived["fit_gallery_indices_sha256"] != tensor_hash(fit)
            or archived["training_raster_sha256"] != tensor_hash(images)):
        raise ValueError("The stock folds must match the exact fit scope and labeled gallery.")
    prior = np.random.default_rng(PRIOR_PILOT_SEED + 99).choice(len(outer), PRIOR_PILOT_GLYPHS, replace=False)
    eligible = np.setdiff1d(np.arange(len(outer)), prior)
    random = np.random.default_rng(SEED)
    held = outer[np.sort(random.choice(eligible, QUERY_GLYPHS, replace=False))]
    noise_rows, splits = fold_indices(len(fit), device)
    previous, hashes = stock_folds(stock.parent, splits, noise_rows, archived["calibration_cache"]["key"], device)
    codepoints = previous[0][3]
    if any(not torch.equal(codepoints, values[3]) for values in previous):
        raise ValueError("Stock folds must share a common atlas order.")
    stock_ranker = model_of([fold[0] for fold in previous], [fold[1] for fold in previous])
    del previous
    inner = fit[splits[0][1]]
    noise_rows = fit[noise_rows]
    report = {"device": str(device), "mouse": MOUSE, "seed": SEED, "repeat_proxy": 16,
              "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "stock_report_sha256": sha256(stock.read_bytes()).hexdigest(), "stock_fold_sha256": hashes,
              "training_raster_sha256": tensor_hash(images), "fit_gallery_indices_sha256": tensor_hash(inner),
              "codepoint_order_sha256": tensor_hash(codepoints), "fit_glyph_count": len(inner),
              "noise_fit_glyphs": len(noise_rows), "held_gallery_indices": held.tolist(), "held_count": len(held),
              "prior_cpu_scored_glyph_overlap": 0, "cohort_independence": independence, "formal_queries_loaded": False,
              "witnesses_per_query": WITNESSES, "utility_share": UTILITY_SHARE,
              "protocol": "Fit one v20 stem with the same training-only split as the previous CPU pilot. Hold the v20 "
                          "list and rendering fixed. For each candidate compute mean exact OCF against two native training "
                          "rasters retrieved either by pooled repeat-split noise-whitened distance or by visual similarity "
                          "to the v20 selection. Use either the utility alone or the same fixed 20% mixture with v20. "
                          "Outer reference rasters are accessed only for paired endpoint scoring."}
    publish(report, OUTPUT, started, "stem_started")
    counts = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if tuple(counts) != SESSIONS or any(value.shape[:2] != (len(images), 4) for value in counts.values()):
        raise ValueError("Expected four measured repeats in all four labeled days.")
    model = fit_v8({day: block[inner] for day, block in counts.items()}, images[inner], MODEL_SEED)
    if not torch.equal(codepoints, model.codepoints):
        raise ValueError("The v20 stem and stock folds must use identical atlas columns.")
    rendered = rendered_bank(model)
    native = images[inner]
    report["rendered_sha256"] = tensor_hash(rendered)
    publish(report, OUTPUT, started, "ocr_started")
    ocr = fit_ocr_logits(native, model.atlas.images, response_views(counts, inner, held, random))
    noise_images, _ = align_training_glyphs(model.atlas, images[noise_rows])
    sources, native_scores = {}, []
    for day, block in counts.items():
        predictive = fit_predictive_likelihood(block[noise_rows], model.encoder.expected_counts(noise_images, day))
        _, sources[day] = rank_view(model, predictive, block[held], day, ocr[day])
        native_scores.append(crossnobis_neighbors(block, inner, held, random))
        publish(report, OUTPUT, started, f"{day}_ready")
    graph = pooled_graph(model, sources, ocr, SESSIONS)
    base = stock_ranker.rank(graph, anchor_ocf=ANCHOR)
    incumbent = graph.indices.gather(1, base.argmax(1, keepdim=True))[:, 0]
    visual = image_descriptors(rendered[incumbent]) @ image_descriptors((native >= 128).to(torch.uint8) * 255).T
    neural = torch.stack(native_scores).mean(0)
    witnesses = {"neural": neural.topk(WITNESSES, dim=1).indices,
                 "contour": visual.topk(WITNESSES, dim=1).indices}
    report["mean_candidates_per_query"] = float(graph.valid.sum(1).float().mean())
    report["native_witness_gallery_indices"] = {name: inner[indices].tolist() for name, indices in witnesses.items()}
    report["shared_witness_fraction"] = float((witnesses["neural"][:, :, None] == witnesses["contour"][:, None, :]).any(2).float().mean())
    publish(report, OUTPUT, started, "witness_ocf_started")
    utilities, report["unique_witness_ocf_pairs"] = witness_utility(graph, rendered, native, witnesses, scorer)

    endpoint = PairedOCF(scorer, rendered, (images[held] >= 128).numpy(), device, 12)
    choices = {"v20": incumbent}
    for name, utility in utilities.items():
        combined = (1 - UTILITY_SHARE) * base + UTILITY_SHARE * utility
        choices[f"{name}_pure"] = graph.indices.gather(1, utility.argmax(1, keepdim=True))[:, 0]
        choices[f"{name}_blend"] = graph.indices.gather(1, combined.argmax(1, keepdim=True))[:, 0]
    outcomes = {name: endpoint.evaluate(choice[:, None])[:, 0] for name, choice in choices.items()}
    report["ocf"] = {name: summarize([output], [outcomes["v20"]]) for name, output in outcomes.items()}
    report["neural_vs_contour"] = {name: summarize([outcomes[f"neural_{name}"]], [outcomes[f"contour_{name}"]])
                                  for name in ("pure", "blend")}
    report["changed_choices_vs_v20"] = {name: int((choice != incumbent).sum()) for name, choice in choices.items()}
    report["paired_ocf_by_glyph"] = {name: output.tolist() for name, output in outcomes.items()}
    report["chosen_codepoints"] = {name: codepoints[choice].tolist() for name, choice in choices.items()}
    report["unique_endpoint_ocf_pairs"] = len(endpoint.values)
    publish(report, OUTPUT, started, "complete")


if __name__ == "__main__":
    main()
