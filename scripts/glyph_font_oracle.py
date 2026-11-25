"""Measure public font shape coverage on held-out SiClink training glyphs."""

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import blake2b, sha256
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import urlopen

from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F

from glyph_data import load_images, locate_data, split_glyphs
from kaggle_diagnostics import official_scorer


REPOSITORY = "https://github.com/notofonts/noto-cjk"
RAW = "https://raw.githubusercontent.com/notofonts/noto-cjk"
FONT_SOURCES = (
    ("Noto Sans CJK SC Regular", "Sans2.004", "Sans", "LICENSE"),
    ("Noto Serif CJK SC Regular", "Serif2.003", "Serif", "Serif/LICENSE"),
)
UNICODE_BLOCKS = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF))
SUPERSAMPLED_SIZE = 128


def render_font(family, version, directory, license_path, *, return_codepoints=False):
    stem = f"Noto{directory}CJKsc-Regular.otf"
    path = f"{directory}/OTF/SimplifiedChinese/{stem}"
    url = f"{RAW}/{version}/{path}"
    try:
        with urlopen(url, timeout=90) as response:
            font_bytes = response.read()
    except URLError as error:
        raise RuntimeError(f"Cannot fetch {family}. Check that Kaggle Notebook Internet is enabled.") from error
    with TTFont(BytesIO(font_bytes)) as table:
        cmap = table.getBestCmap()
    glyphs = sorted(codepoint for codepoint, name in cmap.items() if name != ".notdef" and any(
        low <= codepoint <= high for low, high in UNICODE_BLOCKS
    ))

    font = ImageFont.truetype(BytesIO(font_bytes), size=112, layout_engine=ImageFont.Layout.BASIC)
    masks = np.empty((len(glyphs), 64, 64), dtype=np.uint8)
    codepoints = np.empty(len(glyphs), dtype=np.int64) if return_codepoints else None
    signatures = set()
    count = 0
    for codepoint in glyphs:
        canvas = Image.new("L", (SUPERSAMPLED_SIZE, SUPERSAMPLED_SIZE))
        center = SUPERSAMPLED_SIZE // 2
        ImageDraw.Draw(canvas).text((center, center), chr(codepoint), fill=255, font=font, anchor="mm")
        bitmap = np.asarray(canvas.resize((64, 64), Image.Resampling.LANCZOS))
        if not (bitmap >= 128).any():
            continue
        signature = blake2b(bitmap.tobytes(), digest_size=16).digest()
        if signature not in signatures:
            signatures.add(signature)
            masks[count] = bitmap
            if codepoints is not None:
                codepoints[count] = codepoint
            count += 1
    provenance = {"family": family, "version": version, "source": url,
                  "sha256": sha256(font_bytes).hexdigest(), "license": "SIL OFL 1.1",
                  "license_url": f"{REPOSITORY}/blob/{version}/{license_path}",
                  "covered_codepoints": len(glyphs), "rendered_masks": count}
    rendered = torch.from_numpy(masks[:count])
    if codepoints is not None:
        return rendered, provenance, torch.from_numpy(codepoints[:count])
    return rendered, provenance


@dataclass(frozen=True)
class FontTransform:
    quarter_turns: int
    mirror: bool
    scale_x: float
    scale_y: float
    shift_x: float
    shift_y: float


def orient(images, turns, mirrored):
    selected = images.rot90(turns, dims=(-2, -1))
    return selected.flip((-1,)) if mirrored else selected


def warp(images, transform):
    oriented = orient(images, transform.quarter_turns, transform.mirror)
    # grid_sample uses the inverse map in normalized coordinates.
    sx, sy = transform.scale_x, transform.scale_y
    inverse = images.new_tensor([[1 / sx, 0, -transform.shift_x / sx],
                                 [0, 1 / sy, -transform.shift_y / sy]])[None]
    grid = F.affine_grid(inverse, (1, *images.shape[1:]), align_corners=False)
    return F.grid_sample(oriented, grid.expand(len(images), -1, -1, -1), align_corners=False)


def thumbnails(images):
    return torch.cat([F.avg_pool2d(batch[:, None].float() / 255.0, 4) for batch in images.split(512)])


def bounds(images):
    foreground = images[:, 0] > 0.16
    valid = foreground.flatten(1).any(dim=1)
    columns = foreground[valid].any(dim=1)
    rows = foreground[valid].any(dim=2)
    coordinates = torch.arange(images.shape[-1])
    left = torch.where(columns, coordinates, images.shape[-1]).amin(dim=1)
    right = torch.where(columns, coordinates, -1).amax(dim=1)
    top = torch.where(rows, coordinates, images.shape[-1]).amin(dim=1)
    bottom = torch.where(rows, coordinates, -1).amax(dim=1)
    width = (right - left + 1).float().median().item()
    height = (bottom - top + 1).float().median().item()
    center_x = (right + left + 1).float().median().item() / images.shape[-1] - 1
    center_y = (bottom + top + 1).float().median().item() / images.shape[-2] - 1
    return width, height, center_x, center_y


def descriptor(images):
    """Use spatial ink plus signed directional changes at two scales."""
    fine = F.normalize(images.flatten(1), dim=1)
    features = [fine * 0.77]
    for value, weight in ((images, 0.45), (F.avg_pool2d(images, 2), 0.45)):
        dx = F.pad(value[..., 1:] - value[..., :-1], (0, 1, 0, 0))
        dy = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
        changes = torch.cat((dx, dy, ((dx + dy) * 0.707), ((dx - dy) * 0.707)), dim=1)
        features.append(weight * F.normalize(changes.flatten(1), dim=1))
    return torch.cat(features, dim=1)


def shortlist(bank, transform, query, top_k):
    best_scores = query.new_empty((len(query), 0))
    best_indices = torch.empty((len(query), 0), dtype=torch.long)
    for start in range(0, len(bank), 2048):
        matches = query @ descriptor(warp(bank[start:start + 2048], transform)).T
        scores, indices = matches.topk(min(top_k, matches.shape[1]), dim=1)
        scores = torch.cat((best_scores, scores), dim=1)
        indices = torch.cat((best_indices, indices + start), dim=1)
        best_scores, positions = scores.topk(min(top_k, scores.shape[1]), dim=1)
        best_indices = indices.gather(1, positions)
    return best_scores, best_indices


def fit_font_transform(bank, train):
    query = descriptor(train)
    width, height, center_x, center_y = bounds(train)
    candidates = []
    for turns in range(4):
        for mirror in (False, True):
            w, h, x, y = bounds(orient(bank, turns, mirror))
            sx = min(max(width / w, 0.6), 1.5)
            sy = min(max(height / h, 0.6), 1.5)
            candidates.append(FontTransform(turns, mirror, sx, sy, center_x - sx * x, center_y - sy * y))

    def fit_score(transform):
        scores, _ = shortlist(bank, transform, query, top_k=1)
        return scores.mean().item()

    best = max(candidates, key=fit_score)
    refinement = [best]
    for delta in (-0.045, 0.045):
        refinement.append(replace(best, shift_x=best.shift_x + delta))
        refinement.append(replace(best, shift_y=best.shift_y + delta))
    for scale in (0.94, 1.06):
        refinement.append(replace(best, scale_x=best.scale_x * scale, scale_y=best.scale_y * scale))
    score, selected = max(((fit_score(candidate), candidate) for candidate in refinement), key=lambda item: item[0])
    return selected, score


def aggregate(scores):
    scores = np.asarray(scores)
    return {"n": len(scores), "mean": float(np.mean(scores)), "median": float(np.median(scores)),
            "p10": float(np.quantile(scores, 0.10)), "p90": float(np.quantile(scores, 0.90)),
            "fraction_ge_50": float(np.mean(scores >= 50))}


def candidate_scores(scorer, font, transform, candidates, targets):
    k = candidates.shape[1]
    pixels = []
    for indexes in candidates.flatten().split(128):
        masks = warp(font[indexes, None].float() / 255.0, transform)[:, 0]
        pixels.append((masks >= 0.5).float().numpy())
    predictions = np.concatenate(pixels, axis=0)
    references = np.repeat(targets, k, axis=0)
    values = scorer(predictions, references, device="cpu", batch_size=4)
    return np.asarray(values).reshape(len(targets), k)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--held-glyphs", type=int, default=320)
    parser.add_argument("--evaluation-glyphs", type=int, default=64)
    parser.add_argument("--fit-glyphs", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/font_oracle.json"))
    args = parser.parse_args()
    started = time.monotonic()
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run this font-download and OCF experiment in a Kaggle Notebook.")
    torch.set_num_threads(4)
    root = locate_data()
    scorer = official_scorer(root)
    images = load_images(root, torch.device("cpu"))
    if not 0 < args.evaluation_glyphs <= args.held_glyphs < len(images):
        parser.error("Require 0 < evaluation glyphs <= held glyphs < total training glyphs.")
    if not 0 < args.fit_glyphs <= len(images) - args.held_glyphs or args.top_k < 1:
        parser.error("Require a positive top-k and enough fit glyphs outside the held split.")

    fit, held = split_glyphs(len(images), args.held_glyphs, args.seed, torch.device("cpu"))
    rng = np.random.default_rng(args.seed + 1)
    fit = fit[torch.as_tensor(rng.choice(len(fit), args.fit_glyphs, replace=False))]
    held = held[torch.as_tensor(rng.choice(len(held), args.evaluation_glyphs, replace=False))]
    fit_images = thumbnails((images[fit] >= 128).to(torch.uint8) * 255)
    exact_references = (images[held] >= 128).numpy()
    held_query = descriptor(thumbnails((images[held] >= 128).to(torch.uint8) * 255))

    results = {"device": "cpu", "seed": args.seed, "held_glyphs": args.held_glyphs,
               "fit_glyphs": len(fit), "evaluation_glyphs": len(held), "top_k_per_font": args.top_k,
               "unicode_blocks": [[f"U+{low:04X}", f"U+{high:04X}"] for low, high in UNICODE_BLOCKS],
               "fonts": []}
    oracle = []
    for source in FONT_SOURCES:
        stage = time.monotonic()
        font, provenance = render_font(*source)
        if len(font) < args.top_k:
            raise ValueError(f"{provenance['family']} contains fewer than {args.top_k} rendered candidates.")
        print(f"Rendered {len(font):,} public glyphs for {provenance['family']}", flush=True)
        bank = thumbnails(font)
        transform, proxy = fit_font_transform(bank, fit_images)
        print(f"Fitted global transform using {len(fit)} fit glyphs. Scoring held image matches.", flush=True)
        _, candidates = shortlist(bank, transform, held_query, args.top_k)
        random = torch.as_tensor(rng.integers(len(font), size=(len(held), 1)))
        values = candidate_scores(scorer, font, transform, torch.cat((candidates, random), dim=1), exact_references)
        oracle.append(values[:, :-1].max(axis=1))
        record = {**provenance, "global_transform": asdict(transform), "training_similarity": proxy,
                  "held_top_approximate": aggregate(values[:, 0]),
                  "held_top_k_oracle": aggregate(oracle[-1]), "held_random": aggregate(values[:, -1]),
                  "seconds": time.monotonic() - stage}
        results["fonts"].append(record)
        results["combined_font_oracle"] = aggregate(np.max(oracle, axis=0))
        results["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record), flush=True)
    print(json.dumps({"combined_font_oracle": results["combined_font_oracle"]}), flush=True)


if __name__ == "__main__":
    main()
