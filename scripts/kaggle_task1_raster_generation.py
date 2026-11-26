"""Test whether a shared native glyph rasterizer closes the public-font OCF gap."""

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.request import urlopen

from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features
import torch
from torch.nn import functional as F

from glyph_data import load_images, locate_data
from glyph_font_style import fit_aligned_font_style
from kaggle_diagnostics import official_scorer
from kaggle_font_family_coverage import fit_bank
from kaggle_gallery_validation import align_training_glyphs, build_font_atlas, image_descriptors, standard_error


RELEASE = "Sans2.004"
VARIABLE_FONT = (f"https://raw.githubusercontent.com/notofonts/noto-cjk/{RELEASE}"
                 "/Sans/Variable/OTF/Subset/NotoSansSC-VF.otf")
LICENSE = f"https://github.com/notofonts/noto-cjk/blob/{RELEASE}/LICENSE"
SIZE = 64
LOOKUP_DEPTH = 6
CONTEXT_GLYPHS = 1024
FIT_GLYPHS = 384
FOCUS_MARGIN = 0.015
AUDIT_MARGIN = 0.008
MIN_SIMILARITY = 0.65


@dataclass(frozen=True)
class Raster:
    oversample: int
    kernel: str
    weight: int
    em_pixels: float
    stretch_x: float
    shift_x: float
    shift_y: float


def load_variable_font():
    with urlopen(VARIABLE_FONT, timeout=90) as response:
        content = response.read()
    with TTFont(BytesIO(content)) as face:
        glyphs = {code for code, name in face.getBestCmap().items() if name != ".notdef"}
        axes = face["fvar"].axes
        if len(axes) != 1 or axes[0].axisTag != "wght" or not axes[0].minValue <= 300 <= 500 <= axes[0].maxValue:
            raise ValueError("Expected the pinned Noto Sans SC face with its native weight axis.")
        axis = {"tag": axes[0].axisTag, "min": axes[0].minValue, "default": axes[0].defaultValue,
                "max": axes[0].maxValue}
    provenance = {"source": VARIABLE_FONT, "version": RELEASE, "sha256": sha256(content).hexdigest(),
                  "license": "SIL OFL 1.1", "license_url": LICENSE, "variation_axis": axis,
                  "freetype_version": features.version_module("freetype2")}
    return content, glyphs, provenance


def rasterize(content, codes, recipe):
    if not len(codes):
        return torch.empty((0, SIZE, SIZE), dtype=torch.uint8)
    over = recipe.oversample
    width = SIZE * over
    mid = width / 2
    face = ImageFont.truetype(BytesIO(content), size=round(recipe.em_pixels * over), layout_engine=ImageFont.Layout.BASIC)
    face.set_variation_by_axes([recipe.weight])
    start = (mid + over * recipe.shift_x / recipe.stretch_x, mid + over * recipe.shift_y)
    matrices = np.empty((len(codes), SIZE, SIZE), dtype=np.uint8)
    stretch = (1 / recipe.stretch_x, 0, mid * (1 - 1 / recipe.stretch_x), 0, 1, 0)
    kernel = None if over == 1 else getattr(Image.Resampling, recipe.kernel)
    for index, code in enumerate(codes):
        layer = Image.new("L", (width, width))
        ImageDraw.Draw(layer).text(start, chr(int(code)), fill=255, font=face, anchor="mm")
        if abs(recipe.stretch_x - 1) >= 0.001:
            layer = layer.transform(layer.size, Image.Transform.AFFINE, stretch, resample=Image.Resampling.BILINEAR)
        if kernel is not None:
            layer = layer.resize((SIZE, SIZE), kernel)
        matrices[index] = np.asarray(layer)
    return torch.from_numpy(matrices)


def binary(images):
    return (images >= 128).to(torch.uint8) * 255


def descriptors(images):
    return torch.cat([image_descriptors(batch) for batch in binary(images).split(1024)])


def top_shapes(query, references, depth):
    values, columns = [], []
    for batch in query.split(32):
        best, index = (batch @ references.T).topk(depth, dim=1)
        values.append(best)
        columns.append(index)
    return torch.cat(values), torch.cat(columns)


def prepare_lookup(core, references, variable_codes, seed):
    cpu = torch.device("cpu")
    geometry = np.random.default_rng(seed + 2).choice(len(core), 128, replace=False)
    atlas, original, codes = build_font_atlas(core[geometry], cpu, return_codepoints=True)
    paired, _ = align_training_glyphs(atlas, core)
    style, trained = fit_aligned_font_style(paired, core, seed=seed)
    common = torch.tensor([code in variable_codes for code in codes.tolist()])
    base = torch.cat([(style.apply(group[:, None].float() / 255)[:, 0] * 255).round().to(torch.uint8)
                      for group in atlas.images[common].split(1024)])
    public_codes = codes[common]
    query = descriptors(references)
    base_values, base_columns = top_shapes(query, descriptors(base), LOOKUP_DEPTH)
    del atlas, paired
    check, check_codes, independent = fit_bank("simxihei", core, seed + 1)
    other_values, other_columns = top_shapes(query, descriptors(check.to(torch.uint8) * 255), 2)
    both = ((public_codes[base_columns[:, 0]] == check_codes[other_columns[:, 0]])
            & (base_values[:, 0] >= MIN_SIMILARITY) & (other_values[:, 0] >= MIN_SIMILARITY)
            & (base_values[:, 0] - base_values[:, 1] >= FOCUS_MARGIN)
            & (other_values[:, 0] - other_values[:, 1] >= AUDIT_MARGIN))
    provenance = {"noto_static": original, "style": asdict(style), "style_fit": trained,
                  "independent_codepoint_check": independent, "common_noto_columns": len(base),
                  "dropped_static_columns": int((~common).sum())}
    return base[base_columns], public_codes[base_columns], both, provenance


def boundary_metrics(source, reference, edge):
    delta = (source.float() - reference).abs() / 255
    edge_mae = (delta * edge).sum(dim=(-2, -1)) / edge.sum(dim=(-2, -1)).clamp_min(1)
    whole_mae = delta.mean(dim=(-2, -1))
    return {"loss": 0.70 * edge_mae + 0.30 * whole_mae, "edge_mae": edge_mae, "whole_image_mae": whole_mae}


def edge_pixels(images):
    target = (images >= 128).float()[:, None]
    dilated = F.max_pool2d(target, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-target, 3, stride=1, padding=1)
    return (dilated - eroded)[:, 0] > 0


def fit_raster(content, codes, reference, base_gray, provenance):
    geometry = provenance["noto_static"]["global_transform"]
    style = provenance["style"]
    if geometry["quarter_turns"] != 0 or geometry["mirror"]:
        raise ValueError("The common baseline did not use ordinary upright glyph orientation.")
    sx = style["scale_x"] * geometry["scale_x"]
    sy = style["scale_y"] * geometry["scale_y"]
    initial = Raster(2, "LANCZOS", 400, round(56 * sy), sx / sy,
                     32 * geometry["shift_x"] * style["scale_x"] + style["shift_x_pixels"],
                     32 * geometry["shift_y"] * style["scale_y"] + style["shift_y_pixels"])
    edge = edge_pixels(reference)
    log, cache = [], {}

    def stage(name, choices):
        for recipe in dict.fromkeys(choices):
            if recipe not in cache:
                losses = boundary_metrics(rasterize(content, codes, recipe), reference, edge)
                cache[recipe] = {key: float(value.mean()) for key, value in losses.items()}
                log.append({"stage": name, "recipe": asdict(recipe), **cache[recipe]})
        return min(choices, key=lambda recipe: cache[recipe]["loss"])

    sampling = ((1, "NATIVE"), (2, "BOX"), (2, "BILINEAR"), (2, "LANCZOS"), (4, "BOX"), (4, "LANCZOS"))
    recipe = stage("sampling", [replace(initial, oversample=over, kernel=kernel) for over, kernel in sampling])
    recipe = stage("weight", [recipe, *(replace(recipe, weight=w) for w in (300, 350, 450, 500))])
    recipe = stage("size", [recipe, *(replace(recipe, em_pixels=recipe.em_pixels + delta) for delta in (-2, -1, 1, 2))])
    recipe = stage("stretch", [recipe, *(replace(recipe, stretch_x=recipe.stretch_x + delta) for delta in (-0.04, 0.04))])
    for coordinate in ("shift_x", "shift_y"):
        recipe = stage(coordinate, [recipe, *(replace(recipe, **{coordinate: getattr(recipe, coordinate) + delta})
                                              for delta in (-0.75, -0.375, 0.375, 0.75))])
    baseline = {name: float(values.mean()) for name, values in boundary_metrics(base_gray, reference, edge).items()}
    return recipe, {"stock_gray": baseline, "selected_gray": cache[recipe], "tested_programs": log}


def compare(base, alternate):
    if not len(base):
        return {"n": 0}
    difference = alternate - base
    return {"n": len(base), "stock": float(base.mean()), "selected": float(alternate.mean()),
            "paired_delta": float(difference.mean()), "paired_se": standard_error(difference),
            "selected_higher": int((difference > 0).sum())}


def exact_score(scorer, candidates, reference):
    measured = scorer((candidates.flatten(0, 1) >= 128).numpy(),
                      (reference >= 128).repeat_interleave(candidates.shape[1], dim=0).numpy(),
                      device="cpu", batch_size=8)
    return np.asarray(measured).reshape(len(candidates), -1)


def evaluate(content, recipe, codes, old, locked, images, scorer):
    unique, inverse = torch.unique(codes.flatten(), sorted=True, return_inverse=True)
    rendered = rasterize(content, unique, recipe)[inverse].reshape(old.shape)
    src, target = exact_score(scorer, old, images), exact_score(scorer, rendered, images)
    edge = edge_pixels(images)
    base_gray = boundary_metrics(old[:, 0], images, edge)
    new_gray = boundary_metrics(rendered[:, 0], images, edge)
    scope = locked.numpy()
    return {"glyphs": len(images), "same_codepoint_agreement": int(locked.sum()),
            "candidate_slots_per_reference": codes.shape[1], "unique_public_codepoints": len(unique),
            "locked_codepoint_top1_ocf": compare(src[scope, 0], target[scope, 0]),
            "frozen_first_column_ocf": compare(src[:, 0], target[:, 0]),
            "fixed_six_oracle_ocf": compare(src.max(axis=1), target.max(axis=1)),
            "fixed_six_oracle_on_locked_subset_ocf": compare(src[scope].max(axis=1), target[scope].max(axis=1)),
            "locked_exact_binary_matches": {"stock": int(((binary(old[:, 0]) == binary(images)).flatten(1).all(1) & locked).sum()),
                                            "selected": int(((binary(rendered[:, 0]) == binary(images)).flatten(1).all(1) & locked).sum())},
            "locked_top1_gray": {key: compare(base_gray[key][locked].numpy(), new_gray[key][locked].numpy()) for key in base_gray},
            "all_top1_gray": {key: compare(base_gray[key].numpy(), new_gray[key].numpy()) for key in base_gray}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=32507)
    parser.add_argument("--outer-glyphs", type=int, default=80, choices=range(64, 97), metavar="64..96")
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_raster_generation.json"))
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the image-only raster diagnostic on Kaggle.")
    started = time.monotonic()
    torch.set_num_threads(4)
    root = locate_data()
    images = load_images(root, torch.device("cpu"))
    order = np.random.default_rng(args.seed).permutation(len(images))
    references = images[order[CONTEXT_GLYPHS:CONTEXT_GLYPHS + FIT_GLYPHS + 2 * args.outer_glyphs]]
    context = images[order[:CONTEXT_GLYPHS]]
    assert len(references) == FIT_GLYPHS + 2 * args.outer_glyphs
    content, supported, font = load_variable_font()
    baseline, codes, agreement, provenance = prepare_lookup(context, references, supported, args.seed)
    fit = agreement[:FIT_GLYPHS]
    if int(fit.sum()) < 48:
        raise RuntimeError("Fewer than 48 independently agreed calibration codepoints.")
    fitting_codes = codes[:FIT_GLYPHS, 0][fit]
    print(f"Fitting raw gray edges on {int(fit.sum())} public-outline codepoint agreements", flush=True)
    recipe, calibration = fit_raster(content, fitting_codes, references[:FIT_GLYPHS][fit],
                                     baseline[:FIT_GLYPHS, 0][fit], provenance)
    outer_locked = agreement & ~torch.isin(codes[:, 0], fitting_codes)
    report = {"seed": args.seed, "public_source": font, "baseline_and_lookup": provenance,
              "style_fit_glyphs": len(context), "raster_calibration_draw": FIT_GLYPHS,
              "raster_calibration_agreed_glyphs": int(fit.sum()), "outer_glyphs_each": args.outer_glyphs,
              "outer_agreed_codepoints_already_in_raster_calibration": int((agreement[FIT_GLYPHS:] & ~outer_locked[FIT_GLYPHS:]).sum()),
              "selected_raster_program": asdict(recipe), "fit_gray": calibration,
              "scope": "Image-only coverage on four disjoint training glyph groups. Two outer groups were unused for all fits. "
                       "Their reference images serve only to freeze public diagnostic lookup slots and score outcomes. "
                       "Both renderers use identical fixed Noto lookup codepoints. Agreement between static Noto and an "
                       "independent public SimXiHei face is a codepoint proxy. No spikes or formal test stimuli are read.",
              "cohorts": {}}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    scorer = official_scorer(root)
    for cohort, start in enumerate(range(FIT_GLYPHS, len(references), args.outer_glyphs), 1):
        selected = slice(start, start + args.outer_glyphs)
        outcome = evaluate(content, recipe, codes[selected], baseline[selected], outer_locked[selected], references[selected], scorer)
        report["cohorts"][str(cohort)] = outcome
        report["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"cohort": cohort, **outcome, "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
