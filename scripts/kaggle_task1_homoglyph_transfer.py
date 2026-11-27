"""Test a published cross-font similarity space against matched pixel geometry."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import pickle
import time
from urllib.request import urlopen

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import build_font_atlas, image_descriptors, standardized
from kaggle_task1_mice2_textverifier_recheck import fresh_cohort
from kaggle_task1_microns_hanzi_recheck import fit_ocr_logits
from kaggle_task1_native_hanzi import fitted_pca, response_views
from kaggle_task1_native_hanzi_open_atlas import visual_proxy
from kaggle_v8_metric_validation import SESSIONS, summarize


SEED = 223541
MOUSE = "mice1"
LANDMARKS = 384
OCF_GLYPHS = 64
SHARE = 0.25
ASSET = "https://drive.usercontent.google.com/download?id=1BTfqCGW7w0UognnvTNrMiq18rGPdIDug&export=download&confirm=t"
OUTPUT = Path("/kaggle/working/task1_homoglyph_transfer.json")
BASE, PRIOR, CONTOUR, EXTERNAL = "contour_ocr", "coverage_prior", "pixel_geometry", "homoglyph"


def numeric_scalar(dtype, data):
    if dtype.kind not in "fiu" or len(data) != dtype.itemsize:
        raise pickle.UnpicklingError("Expected one numeric similarity value.")
    return np.frombuffer(data, dtype=dtype)[0].item()


class SimilarityReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "scalar":
            return numeric_scalar
        raise pickle.UnpicklingError(f"Unexpected object in similarity data: {module}.{name}")


def read_release():
    with urlopen(ASSET, timeout=90) as response:
        if response.headers.get_content_type() == "text/html":
            raise ValueError("The author's public similarity download returned a web page.")
        blob = response.read()
    release = SimilarityReader(BytesIO(blob)).load()
    if not isinstance(release, dict) or not release:
        raise ValueError("Expected the author's sparse character similarity dictionary.")
    return release, {"url": ASSET, "sha256": sha256(blob).hexdigest(), "bytes": len(blob), "keys": len(release)}


def coordinates(release, atlas, codepoints, mapped_fit, support, rng):
    letters = [chr(codepoints[index]) for index in support.tolist()]
    fit_positions = torch.searchsorted(support, mapped_fit)
    available = torch.unique(fit_positions).numpy()
    anchors = torch.from_numpy(np.sort(rng.choice(available, LANDMARKS, replace=False)))
    landmarks = [letters[index] for index in anchors.tolist()]

    # Both representations use the same columns and roughly the same neighbor budget.
    similarities = []
    for glyph in letters:
        row = release[glyph]
        similarities.append([1.0 if glyph == neighbor else float(row.get(neighbor, 0)) for neighbor in landmarks])
    external = torch.tensor(similarities, dtype=torch.float32)
    covered_contours = atlas.contours[support]
    controls = []
    for batch in covered_contours.split(256):
        pixel = batch @ covered_contours.T
        edge = pixel.topk(min(800, len(support)), dim=1).values[:, -1:]
        controls.append((pixel[:, anchors] >= edge) * pixel[:, anchors])
    control = torch.cat(controls)
    features = {}
    for name, feature in ((EXTERNAL, external), (CONTOUR, control)):
        if not bool(feature.isfinite().all()):
            raise ValueError("A geometry representation contains nonfinite similarities.")
        fitted, bank = fitted_pca(torch.cat((feature[fit_positions], feature)), len(fit_positions)).split(
            (len(fit_positions), len(feature)))
        features[name] = fitted, F.normalize(bank).T.contiguous()
    return features, landmarks


def alternatives(baseline, support, features, ridge, query):
    prior = baseline.clone()
    prior[:, support] += SHARE
    result = {BASE: baseline, PRIOR: prior}
    for name, (fit, bank) in features.items():
        correction = standardized(F.normalize(query @ (ridge @ fit)) @ bank)
        mixed = baseline.clone()
        mixed[:, support] += SHARE * correction
        result[name] = mixed
    return result


def measurements(scores, sample, metric, proxy, confident, dictionary_support):
    qualities, exact = {}, {}
    supported = confident & torch.isin(proxy, dictionary_support)
    for name, values in scores.items():
        candidates = values.topk(32, dim=1).indices
        matches = candidates == proxy[:, None]
        qualities[name] = {str(k): {"all": float(matches[:, :k].any(1).float().mean()),
                                   "trusted_covered": float(matches[supported, :k].any(1).float().mean())
                                   if bool(supported.any()) else None}
                           for k in (1, 16, 32)}
        exact[name] = metric.evaluate(candidates[sample, :1])[:, 0]
    return qualities, exact


def compare(blocks):
    paired = {name: summarize(scores, blocks[BASE]) for name, scores in blocks.items()}
    paired["published_vs_matched_pixels"] = summarize(blocks[EXTERNAL], blocks[CONTOUR])
    paired["published_vs_coverage"] = summarize(blocks[EXTERNAL], blocks[PRIOR])
    return paired


def publish(report, start, stage):
    report.update(stage=stage, seconds=time.monotonic() - start)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{stage} after {report['seconds']:.1f}s", flush=True)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    started = time.monotonic()
    rng = np.random.default_rng(SEED)
    device = torch.device("cpu")
    root = locate_data()
    release, provenance = read_release()
    report = {"mouse": MOUSE, "seed": SEED, "release": provenance, "recordings": {}}
    publish(report, started, "asset_loaded")
    images = load_images(root, device)
    fit, held, independence = fresh_cohort(len(images))
    style = rng.choice(fit.numpy(), size=128, replace=False)
    atlas, font, codes = build_font_atlas(images[style], device, return_codepoints=True)
    support = torch.tensor([i for i, cp in enumerate(codes.tolist()) if chr(cp) in release], dtype=torch.long)
    mapped, trusted, _ = visual_proxy(atlas, images[fit])
    use = trusted & torch.isin(mapped, support)
    features, landmarks = coordinates(release, atlas, codes, mapped[use], support, rng)
    del release
    fit = fit[use]
    references = images[held]
    span = torch.from_numpy(np.sort(rng.choice(len(held), OCF_GLYPHS, replace=False)))
    report.update(fit_glyphs=len(fit), held_glyphs=len(held), held_independence=independence,
                  atlas_size=len(codes), published_covered_candidates=len(support), font=font,
                  fit_anchor_codepoints=landmarks, graded_indices=held[span].tolist(), correction_share=SHARE,
                  protocol="Full Noto gallery and identical geometry-independent OCR/contour readout for all arms. "
                           "Fit on labeled glyphs whose geometric proxy belongs to the published dictionary. "
                           "Both auxiliary heads use the same trusted anchors, PCA and ridge. Score changes "
                           "are additive and supported only on dictionary codepoints. Coverage and pixel "
                           "neighbor controls preserve the same supported subset. True held images score "
                           "readouts after ranking. Use only official training spikes.")
    publish(report, started, "geometry_ready")

    groups = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    if tuple(groups) != SESSIONS:
        raise ValueError("Expected the four labeled recordings.")
    views = response_views(groups, fit, held, rng)
    del groups
    ocr = fit_ocr_logits(images[fit], atlas.images, views)
    contours = image_descriptors((images[fit] >= 128).to(torch.uint8) * 255)
    proxy, confident, _ = visual_proxy(atlas, references)
    report["trusted_proxy_in_dictionary"] = int((confident & torch.isin(proxy, support)).sum())
    report["trusted_proxy_glyphs"] = int(confident.sum())
    metric = PairedOCF(official_scorer(root), atlas.images, (references[span] >= 128).numpy(), device, 8)
    total, day_values = {}, {}
    for day, (ridge, _, query) in views.items():
        pixels = standardized(F.normalize(query @ (ridge @ contours)) @ atlas.contours.T)
        baseline = 0.5 * (pixels + ocr[day])
        scores = alternatives(baseline, support, features, ridge, query)
        recall, exact = measurements(scores, span, metric, proxy, confident, support)
        for name, values in scores.items():
            if name in total:
                total[name].add_(values)
            else:
                total[name] = values
            day_values.setdefault(name, []).append(exact[name])
        report["recordings"][day] = {"recall": recall, "ocf": compare({name: [v] for name, v in exact.items()})}
        publish(report, started, f"{day}_scored")
    recall, values = measurements(total, span, metric, proxy, confident, support)
    report["same_day_four_repeat"] = compare(day_values)
    report["pooled_16_known_correspondence"] = {"ocf": compare({name: [v] for name, v in values.items()}),
                                               "recall": recall}
    report["unique_exact_pairs"] = len(metric.values)
    publish(report, started, "complete")


if __name__ == "__main__":
    main()
