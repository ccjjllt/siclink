"""Compare published PP-OCRv6 visual features and the current glyph readout."""

import argparse
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from glyph_atlas_prior import fit_atlas_prior
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import build_font_atlas, standardized
from kaggle_task1_common_visual_readout import SEED as COMMON_SEED, held_partition
from kaggle_task1_hsk_inventory import independent_split
from kaggle_task1_mice2_textverifier_recheck import fresh_cohort
from kaggle_task1_native_hanzi import feature_extractors, normalized_glyph, response_views
from kaggle_task1_native_hanzi_open_atlas import image_coordinates, visual_proxy
from kaggle_task1_textverifier import publish
from kaggle_v8_metric_validation import SESSIONS, summarize


SEED, HELD, SCORED = 224603, 320, 112
MODEL = "PaddlePaddle/PP-OCRv6_small_rec_onnx"
VERSION = "b8f84f0b80c529de40b4fbb3544b84fa7233a513"
SOURCE = f"https://huggingface.co/{MODEL}/resolve/{VERSION}/inference.onnx"
CHECKSUM = "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634"
OUTPUT = Path("/kaggle/working/task1_ppocrv6_readout.json")
MOUSE, STOCK = "mice2", "contour_and_hanzi"


def distinct_cohort(size, rng, previous_pp=None):
    _, direct, _ = fresh_cohort(size)
    _, common, _ = held_partition(size, np.random.default_rng(COMMON_SEED))
    (_, _, inventory), _ = independent_split(size)
    protected = {"v20_direct": direct, "shared_readout": common, "hsk_inventory": inventory}
    if previous_pp is not None:
        protected["ppocrv6_mice2"] = previous_pp
    excluded = np.unique(np.concatenate([values.numpy() for values in protected.values()]))
    held = np.sort(rng.choice(np.setdiff1d(np.arange(size), excluded), size=HELD, replace=False))
    fit = np.setdiff1d(np.arange(size), held)
    overlaps = {name: int(np.intersect1d(held, values.numpy()).size) for name, values in protected.items()}
    return torch.from_numpy(fit), torch.from_numpy(held), overlaps


def recognizer():
    missing = [package for package in ("onnx", "onnxruntime") if find_spec(package) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)
    import onnx
    import onnxruntime

    with urlopen(SOURCE, timeout=90) as response:
        contents = response.read()
    if sha256(contents).hexdigest() != CHECKSUM:
        raise ValueError("PP-OCRv6 weights differ from the published checksum.")
    graph = onnx.load_model_from_string(contents)
    weights = {entry.name: tuple(entry.dims) for entry in graph.graph.initializer}
    projection = []
    for node in graph.graph.node:
        if node.op_type not in {"MatMul", "Gemm"} or len(node.input) < 2:
            continue
        dims = weights.get(node.input[1], ())
        if len(dims) == 2 and max(dims) >= 5_000 and min(dims) <= 1_024:
            transposed = any(option.name == "transB" and option.i for option in node.attribute)
            projection.append((node.input[0], dims[1] if transposed else dims[0], node.name, dims))
    if len(projection) != 1:
        raise ValueError(f"Expected one visible CTC projection, found {projection}.")
    name, hidden, operation, dims = projection[0]
    if hidden > 1_024:
        raise ValueError(f"CTC projection orientation needs review: {dims}.")
    inferred = onnx.shape_inference.infer_shapes(graph)
    target = next((value for value in inferred.graph.value_info if value.name == name), None)
    if target is None:
        target = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [None, None, hidden])
    del graph.graph.output[:]
    graph.graph.output.append(target)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(graph.SerializeToString(), options, providers=["CPUExecutionProvider"])
    diagnostics = {"source": SOURCE, "sha256": CHECKSUM, "revision": VERSION, "license": "Apache-2.0",
                   "pre_ctc_tensor": name, "classifier_node": operation, "ctc_weight_shape": dims,
                   "hidden_dimension": hidden, "input_shape": session.get_inputs()[0].shape,
                   "onnxruntime_version": onnxruntime.__version__, "onnx_version": onnx.__version__}
    return session, diagnostics


def read_glyphs(recognizer_session, images, metadata):
    input_info = recognizer_session.get_inputs()[0]
    height = input_info.shape[-2] if isinstance(input_info.shape[-2], int) else 48
    width = input_info.shape[-1] if isinstance(input_info.shape[-1], int) else 64
    if not input_info.shape[1] == 3 or height != 48 or width < height:
        raise ValueError(f"Expected RGB 48px line images, received {input_info.shape}.")
    metadata.update(glyph_input_height=height, glyph_input_width=width, glyph_policy="left-aligned normalized 48px glyph")
    spatial, global_mean = [], []
    for batch in images.split(96):
        tiles = np.stack([np.asarray(Image.fromarray(normalized_glyph(tile)).resize((height, height),
                                                                    Image.Resampling.BILINEAR)) for tile in batch.numpy()])
        foreground = 1.0 - tiles.astype(np.float32) / 127.5
        formatted = np.zeros((len(batch), 3, height, width), dtype=np.float32)
        formatted[:, :, :, :height] = foreground[:, None]
        sequence = torch.from_numpy(recognizer_session.run(None, {input_info.name: formatted})[0])
        if sequence.shape[-1] != metadata["hidden_dimension"]:
            raise RuntimeError(f"Unexpected frozen text representation dimensions: {tuple(sequence.shape)}.")
        sequence = sequence.reshape(len(batch), -1, sequence.shape[-1])
        active = round(sequence.shape[1] * height / width)
        if active < 4 or not bool(torch.isfinite(sequence).all()):
            raise RuntimeError(f"Expected finite sequence features with four spatial cells, got {tuple(sequence.shape)}.")
        sequence = sequence[:, :active].transpose(1, 2)
        spatial.append(F.adaptive_avg_pool1d(sequence, 4).flatten(1))
        global_mean.append(sequence.mean(2))
    metadata.update(active_sequence_tokens=active, encoded_images=len(images), spatial_bins=4)
    return {"ppocrv6_spatial": torch.cat(spatial), "ppocrv6_global": torch.cat(global_mean)}


def fit_coordinates(features, fit_count):
    fitted = features[:fit_count]
    center = fitted.mean(0)
    scale = (fitted - center).square().mean().sqrt().clamp_min(1e-5)
    normalized = (features - center) / scale
    torch.manual_seed(SEED + 1)
    _, _, basis = torch.pca_lowrank(normalized[:fit_count], q=min(144, normalized.shape[1]), center=False, niter=3)
    coordinates = normalized @ basis[:, :128]
    coordinates /= coordinates[:fit_count].std(0).clamp_min(1e-4)
    fitted, atlas = coordinates.split((fit_count, len(features) - fit_count))
    return fitted, F.normalize(atlas).T.contiguous()


def score_readouts(coordinates, observations, prior):
    ridge, _, query = observations
    scores = {name: standardized(F.normalize(query @ (ridge @ labels)) @ bank)
              for name, (labels, bank) in coordinates.items()}
    stock = 0.5 * (scores["contour"] + scores["frozen_hanzi"])
    arms = {STOCK: stock, "contour_and_ppocrv6": 0.5 * (scores["contour"] + scores["ppocrv6_spatial"]),
            "stock_and_ppocrv6_spatial": 0.5 * (stock + scores["ppocrv6_spatial"]),
            "stock_and_ppocrv6_global": 0.5 * (stock + scores["ppocrv6_global"])}
    return {name: prior.variants(values)["usage_unseen"] for name, values in arms.items()}


def measure(scores, rows, exact, proxy, trusted, *, baseline=STOCK):
    results, recall = {}, {}
    for name, values in scores.items():
        top = values.topk(32, dim=1).indices
        match = top[trusted] == proxy[trusted, None]
        recall[name] = {str(k): float(match[:, :k].any(1).float().mean()) for k in (1, 16, 32)}
        results[name] = exact.evaluate(top[rows, :1])[:, 0]
    return {"ocf": {name: summarize([values], [results[baseline]]) for name, values in results.items()},
            "trusted_proxy_recall": recall}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", choices=("mice1", "mice2"), default=MOUSE)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    cpu = torch.device("cpu")
    seed = SEED if args.mouse == MOUSE else SEED + 1
    rng = np.random.default_rng(seed)
    root = locate_data()
    images = load_images(root, cpu)
    previous_pp = None if args.mouse == MOUSE else distinct_cohort(len(images), np.random.default_rng(SEED))[1]
    fit, held, overlaps = distinct_cohort(len(images), rng, previous_pp)
    rows = torch.from_numpy(np.sort(rng.choice(len(held), SCORED, replace=False)))
    groups = grouped_blocks(root, load_catalog(root), args.mouse, cpu, len(images))
    report = {"mouse": args.mouse, "seed": seed, "formal_queries_loaded": False, "fit_glyph_count": len(fit),
              "held_gallery_indices": held.tolist(), "scored_gallery_indices": held[rows].tolist(),
              "overlap": overlaps, "protocol": "Full public Noto bank and the same fitted ridge and Unihan prior in all rows. "
              "Compare the existing contour and frozen single-glyph OCR to PP-OCRv6 recognition features before CTC, with "
              "four spatial cells or a global average. One balanced same-day condition and known-correspondence sixteen-trial "
              "condition use the same whole-glyph holdout and fixed rendering."}
    session, metadata = recognizer()
    report["published_ocr"] = metadata
    publish(report, OUTPUT, started, "recognizer_ready")
    picked = rng.choice(fit.numpy(), size=128, replace=False)
    atlas, font, codes = build_font_atlas(images[picked], cpu, return_codepoints=True)
    inputs = torch.cat((images[fit], atlas.images))
    encoded = read_glyphs(session, inputs, metadata)
    coordinates = {name: fit_coordinates(values, len(fit)) for name, values in encoded.items()}
    del encoded, inputs, session
    extractors = feature_extractors()
    extractors.pop("random_mobilenet")
    coordinates.update(image_coordinates(images[fit], atlas.images, extractors))
    prior = fit_atlas_prior(atlas, codes, images[fit])
    report.update(atlas=font, candidates=len(codes), unihan=prior.diagnostics,
                  fitted_dimensions={name: labels.shape[-1] for name, (labels, _) in coordinates.items()})
    publish(report, OUTPUT, started, "frozen_representations_ready")

    views = response_views(groups, fit, held, rng)
    proxy, trusted, _ = visual_proxy(atlas, images[held])
    report["trusted_glyph_proxy_count"] = int(trusted.sum())
    day_order = torch.from_numpy(rng.permutation(len(held)) % len(SESSIONS))
    glyph_rows = torch.arange(len(held))
    all_scores = [score_readouts(coordinates, views[day], prior) for day in SESSIONS]
    stacked = {name: torch.stack([day[name] for day in all_scores]) for name in all_scores[0]}
    exact = PairedOCF(official_scorer(root), atlas.images, (images[held[rows]] >= 128).numpy(), cpu, 12)
    report["same_day_four"] = measure({name: values[day_order, glyph_rows] for name, values in stacked.items()},
                                      rows, exact, proxy, trusted)
    publish(report, OUTPUT, started, "same_day_scored")
    report["known_correspondence_sixteen"] = measure({name: values.mean(0) for name, values in stacked.items()},
                                                     rows, exact, proxy, trusted)
    report["unique_exact_pairs"] = len(exact.values)
    publish(report, OUTPUT, started, "complete")


if __name__ == "__main__":
    main()
