"""Measure paired image-only OCF coverage of three public Chinese font families."""

import argparse
from dataclasses import asdict
from hashlib import blake2b, sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
from urllib.request import urlopen

from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from glyph_data import load_images, locate_data, split_glyphs
from glyph_font_oracle import (FONT_SOURCES, SUPERSAMPLED_SIZE, UNICODE_BLOCKS, fit_font_transform, render_font,
                               thumbnails, warp)
from glyph_font_style import fit_aligned_font_style
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import FontAtlas, align_training_glyphs, image_descriptors, standard_error


BASE = "noto_sans_sc"
FONTS = (BASE, "simxihei", "wenquanyi_zenhei")
DEPTH = 6
SIM_RELEASE = "https://github.com/lxgw/SimXiZhi/releases/tag/v1.103"
SIM_BINARY = "https://github.com/lxgw/SimXiZhi/releases/download/v1.103/SimXiHei.ttf"
SIM_SHA256 = "d3b7a4cb8552ddddfe23e5379c19f9b74f2ca63c238b7ef6882208bfde3cc1e3"
ZEN_SOURCE = "https://deb.debian.org/debian/pool/main/f/fonts-wqy-zenhei/fonts-wqy-zenhei_0.9.45.orig.tar.gz"
ZEN_SHA256 = "e4b7e306475bf9427d1757578f0e4528930c84c44eaa3f167d4c42f110ee75d6"
ZEN_UPSTREAM = "https://sourceforge.net/projects/wqy/files/wqy-zenhei/0.9.45%20%28Fighting-state%20RC1%29/"


def download(url):
    with urlopen(url, timeout=90) as response:
        return response.read()


def public_outline(name):
    if name == "simxihei":
        data = download(SIM_BINARY)
        if sha256(data).hexdigest() != SIM_SHA256:
            raise ValueError("SimXiHei's binary differs from the author's release SHA256.")
        return data, {"family": "SimXiHei", "version": "1.103", "source": SIM_BINARY,
                      "upstream": SIM_RELEASE, "license": "IPA Font License 1.0",
                      "license_url": "https://github.com/lxgw/SimXiZhi/blob/main/License.txt"}

    payload = download(ZEN_SOURCE)
    archive_digest = sha256(payload).hexdigest()
    if archive_digest != ZEN_SHA256:
        raise ValueError("The WenQuanYi upstream archive differs from the Debian package checksum.")
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        members = [member for member in archive.getmembers()
                   if member.isfile() and PurePosixPath(member.name).name == "wqy-zenhei.ttc"]
        if len(members) != 1:
            raise ValueError("Expected one WenQuanYi Zen Hei font collection in 0.9.45.")
        with archive.extractfile(members[0]) as stream:
            data = stream.read()
    return data, {"family": "WenQuanYi Zen Hei", "version": "0.9.45", "source": ZEN_SOURCE,
                  "upstream": ZEN_UPSTREAM, "source_archive_sha256": archive_digest,
                  "license": "GPL-2.0 with font embedding exception", "license_url": ZEN_UPSTREAM}


def rasterize(name):
    if name == BASE:
        return render_font(*FONT_SOURCES[0], return_codepoints=True)
    payload, provenance = public_outline(name)
    index = 0 if payload[:4] == b"ttcf" else -1
    with TTFont(BytesIO(payload), fontNumber=index) as outline:
        table = outline.getBestCmap()
    eligible = [codepoint for codepoint, glyph in sorted(table.items())
                if glyph != ".notdef" and any(lo <= codepoint <= hi for lo, hi in UNICODE_BLOCKS)]
    font = ImageFont.truetype(BytesIO(payload), size=112, index=0, layout_engine=ImageFont.Layout.BASIC)
    middle = SUPERSAMPLED_SIZE // 2
    images, codes, seen = [], [], set()
    for codepoint in eligible:
        canvas = Image.new("L", (SUPERSAMPLED_SIZE, SUPERSAMPLED_SIZE))
        ImageDraw.Draw(canvas).text((middle, middle), chr(codepoint), fill=255, font=font, anchor="mm")
        alpha = np.asarray(canvas.resize((64, 64), Image.Resampling.LANCZOS))
        identifier = blake2b(alpha.tobytes(), digest_size=16).digest()
        if (alpha >= 128).any() and identifier not in seen:
            images.append(alpha)
            codes.append(codepoint)
            seen.add(identifier)
    provenance.update(sha256=sha256(payload).hexdigest(), face_name=font.getname(),
                      covered_codepoints=len(eligible), rendered_masks=len(images))
    return torch.from_numpy(np.stack(images)), provenance, torch.as_tensor(codes)


@torch.no_grad()
def style_geometry(public, fit_images, seed):
    order = np.random.default_rng(seed + 2).permutation(len(fit_images))
    geometry_images = (fit_images[order[:128]] >= 128).to(torch.uint8) * 255
    transform, agreement = fit_font_transform(thumbnails(public), thumbnails(geometry_images))
    aligned = torch.cat([(warp(group[:, None].float() / 255, transform)[:, 0].clamp(0, 1) * 255).round().to(torch.uint8)
                         for group in public.split(512)])
    features = torch.cat([image_descriptors((group >= 128).to(torch.uint8) * 255) for group in aligned.split(1024)])
    style_references = fit_images[order[128:640]]
    matched, confidence = align_training_glyphs(FontAtlas(aligned, features), style_references)
    return aligned, matched, style_references, {"transform": asdict(transform), "geometry_similarity": agreement,
                                                  "style_match_similarity": confidence}


def fit_bank(name, fit_images, seed):
    public, provenance, codepoints = rasterize(name)
    aligned, matched, references, geometry = style_geometry(public, fit_images, seed)
    style, diagnostics = fit_aligned_font_style(matched, references, seed=seed)
    with torch.no_grad():
        rendered = torch.cat([style.apply(group[:, None].float() / 255)[:, 0] >= 0.5
                              for group in aligned.split(1024)])
    return rendered, codepoints, {**provenance, **geometry, "style": asdict(style), "style_fit": diagnostics}


def top_shapes(bank, reference):
    query = image_descriptors(reference.to(torch.uint8) * 255)
    descriptors = torch.cat([image_descriptors(part.to(torch.uint8) * 255) for part in bank.split(1024)])
    candidates = (query @ descriptors.T).topk(32, dim=1).indices
    choices = []
    for row in candidates:
        seen, taken = set(), []
        for column in row.tolist():
            key = bank[column].numpy().tobytes()
            if key not in seen:
                taken.append(column)
                seen.add(key)
            if len(taken) == DEPTH:
                break
        if len(taken) != DEPTH:
            raise ValueError("The image-only lookup returned too few unique glyphs.")
        choices.append(taken)
    return torch.as_tensor(choices)


def equal_depth_union(images, measurements):
    """Round-robin font nominations with a shared six-unique-image budget."""
    mixture = []
    for row in range(len(images[BASE])):
        seen, trial = set(), []
        for rank in range(DEPTH):
            for name in FONTS:
                image = images[name][row, rank].tobytes()
                if image not in seen:
                    seen.add(image)
                    trial.append(measurements[name][row, rank])
                if len(trial) == DEPTH:
                    break
            if len(trial) == DEPTH:
                break
        mixture.append(trial)
    return np.asarray(mixture)


def summarize(scores, baseline):
    result = {}
    for name, values in (("nearest", scores[:, 0]), ("oracle_best6", scores.max(axis=1))):
        comparison = baseline[:, 0] if name == "nearest" else baseline.max(axis=1)
        diff = values - comparison
        result[name] = {"ocf": float(values.mean()), "paired_delta_vs_noto": float(diff.mean()),
                        "paired_glyph_se": standard_error(diff), "n": len(values),
                        "n_higher": int((diff > 0).sum()), "fraction_ocf_ge_80": float((values >= 80).mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3547)
    parser.add_argument("--ocf-images", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/font_family_coverage.json"))
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle CPU with public font access and competition training images.")
    if not 2 <= args.ocf_images <= 64:
        parser.error("Choose between 2 and 64 paired exact-OCF glyphs.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, "cpu")
    fit, held = split_glyphs(len(images), 320, args.seed, images.device)
    sample = np.random.default_rng(args.seed + 1).choice(len(held), args.ocf_images, replace=False)
    reference = images[held[sample]] >= 128
    scorer = official_scorer(root)
    output = {"device": "cpu", "seed": args.seed, "held_glyphs": len(held), "paired_glyphs": len(reference),
              "candidate_budget": DEPTH, "fonts": {}, "metrics": {},
              "protocol": "Fit independently matched geometry on 128 labeled fit glyphs and bounded style on "
                          "512 other fit glyphs per font. A fixed descriptor selects six unique bitmaps per family "
                          "from entirely public outlines. The union uses the same six-image budget in round-robin "
                          "font order. Held reference images provide only retrospective candidate lookup and exact "
                          "OCF. This measures visual coverage and uses no spikes or competition test records."}

    def save():
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    rendered, measurements, coverage, picked_codes = {}, {}, {}, {}
    for name in FONTS:
        print(f"Render, align, and fit style for {name}", flush=True)
        bank, codes, provenance = fit_bank(name, images[fit], args.seed)
        indices = top_shapes(bank, reference)
        rendered[name] = bank[indices].numpy()
        picked_codes[name] = codes[indices].numpy()
        coverage[name] = set(codes.tolist())
        del bank
        exact = scorer(rendered[name].reshape(-1, 64, 64), np.repeat(reference.numpy(), DEPTH, axis=0),
                       device="cpu", batch_size=8)
        measurements[name] = exact.reshape(len(reference), DEPTH)
        output["fonts"][name] = provenance
        output["metrics"][name] = summarize(measurements[name], measurements[BASE])
        print(json.dumps({"font": name, "metrics": output["metrics"][name]}), flush=True)
        save()

    common = np.asarray(sorted(set.intersection(*coverage.values())))
    output["common_codepoints_in_deduplicated_atlases"] = len(common)
    for name, codes in picked_codes.items():
        output["fonts"][name]["nearest_candidate_in_common_codepoints"] = float(np.isin(codes[:, 0], common).mean())
    union = equal_depth_union(rendered, measurements)
    output["metrics"]["equal_budget_union"] = {"oracle_best6": summarize(union, measurements[BASE])["oracle_best6"]}
    reference_best = measurements[BASE].max(axis=1)
    all_scores = {**measurements, "equal_budget_union": union}
    output["paired_oracle_vs_noto_per_reference"] = [
        {"reference_ordinal": index, "noto_best6": float(reference_best[index]),
         "differences": {name: float(scores[index].max() - reference_best[index])
                         for name, scores in all_scores.items() if name != BASE}}
        for index in range(len(reference))
    ]
    save()
    print(json.dumps({"equal_budget_union": output["metrics"]["equal_budget_union"], "seconds": output["seconds"]}),
          flush=True)


if __name__ == "__main__":
    main()
