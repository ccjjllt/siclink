"""Image-only held-glyph validation of a frozen public lexicon support prior."""

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
from torch.nn import functional as F

from glyph_atlas_prior import _aligned_overlap, _usage_log_enrichment, seen_glyph_evidence, unihan_usage
from glyph_data import load_images, locate_data
from glyph_font_oracle import FONT_SOURCES, fit_font_transform, render_font, thumbnails, warp
from kaggle_gallery_validation import FontAtlas, image_descriptors, standard_error


SUBTLEX_ARTICLE = "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0010729"
SUBTLEX_URLS = (
    "https://journals.plos.org/plosone/article/file?id=10.1371%2Fjournal.pone.0010729.s002&type=supplementary",
    "https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexch/subtlexchchr.zip",
)


def _text_rows(payload):
    prefix = payload[:300]
    encodings = ("utf-16", "utf-16-le", "utf-16-be") if prefix.count(b"\x00") > 20 else ()
    for encoding in (*encodings, "utf-8-sig", "gb18030", "big5"):
        try:
            text = payload.decode(encoding)
        except UnicodeError:
            continue
        yield [line.split("\t") if "\t" in line else line.split() for line in text.splitlines()]
        yield csv.reader(text.splitlines())


def _parse_records(rows):
    records = {}
    for row in rows:
        if len(row) < 5:
            continue
        character = str(row[0]).strip().strip('"')
        if len(character) != 1 or not 0x3400 <= ord(character) <= 0x3134F:
            continue
        # The published CHR schema puts count and context count in columns 2 and 5.
        try:
            count = float(str(row[1]).replace(",", ""))
            contexts = float(str(row[4]).replace(",", ""))
        except ValueError:
            continue
        if np.isfinite(count) and np.isfinite(contexts) and count > 0 and contexts >= 0:
            records[ord(character)] = (count, contexts)
    return records


def _archive_records(payload):
    with ZipFile(BytesIO(payload)) as archive:
        members = [name for name in archive.namelist()
                   if "chr" in Path(name).name.lower() and not name.startswith("__MACOSX/")]
        for name in members:
            extension = Path(name).suffix.lower()
            if extension not in (".txt", ".tsv", ".csv", ".xls", ".xlsx", ""):
                continue
            contents = archive.read(name)
            if extension in (".xls", ".xlsx"):
                import pandas as pd

                try:
                    candidates = [pd.read_excel(BytesIO(contents), header=None).itertuples(index=False, name=None)]
                except (ImportError, ValueError):
                    continue
            else:
                candidates = _text_rows(contents)
            for rows in candidates:
                records = _parse_records(rows)
                if len(records) >= 4000:
                    return records, name
    raise ValueError("The public archive did not contain the documented character-frequency table")


def load_subtlex(codepoints):
    failures = []
    for source in SUBTLEX_URLS:
        try:
            with urlopen(Request(source, headers={"User-Agent": "SiClink research reproducibility"}), timeout=75) as response:
                payload = response.read()
            records, member = _archive_records(payload)
        except (BadZipFile, OSError, ValueError, ImportError) as error:
            failures.append(f"{source}: {error}")
            continue
        counts = torch.as_tensor([records.get(codepoint, (0, 0)) for codepoint in codepoints.tolist()], dtype=torch.float32)
        features = torch.cat(((counts[:, :1] > 0).float(), counts.log1p()), dim=1)
        return features, {"article": SUBTLEX_ARTICLE, "doi": "10.1371/journal.pone.0010729",
                          "attribution": "Qing Cai and Marc Brysbaert, SUBTLEX-CH, PLOS ONE (2010)",
                          "publication_license": "Creative Commons Attribution", "source": source,
                          "member": member, "sha256": sha256(payload).hexdigest(), "record_count": len(records),
                          "atlas_masks_with_frequency": int((counts[:, 0] > 0).sum())}
    raise RuntimeError("Could not load the authors' public SUBTLEX-CH table: " + " | ".join(failures))


@torch.no_grad()
def fit_atlas(raw, fit_images, seed):
    selected = np.random.default_rng(seed + 2).choice(len(fit_images), 128, replace=False)
    calibration = thumbnails((fit_images[selected] >= 128).to(torch.uint8) * 255)
    transform, fit_quality = fit_font_transform(thumbnails(raw), calibration)
    images = torch.cat([(warp(batch[:, None].float() / 255, transform)[:, 0] * 255).round().to(torch.uint8)
                        for batch in raw.split(512)])
    contours = torch.cat([image_descriptors((batch >= 128).to(torch.uint8) * 255) for batch in images.split(1024)])
    return FontAtlas(images, contours), {"transform": asdict(transform), "fit_quality": fit_quality}


@torch.no_grad()
def morphology(images):
    rows = []
    for group in images.split(512):
        mask = group >= 128
        ink = mask.float().mean(dim=(1, 2))
        width = mask.any(dim=1).float().mean(dim=1)
        height = mask.any(dim=2).float().mean(dim=1)
        horizontal = (mask[:, :, 1:] != mask[:, :, :-1]).float().mean(dim=(1, 2))
        vertical = (mask[:, 1:] != mask[:, :-1]).float().mean(dim=(1, 2))
        pooled = F.avg_pool2d(mask[:, None].float(), 16).flatten(1)
        rows.append(torch.cat((torch.stack((ink, width, height, horizontal, vertical), dim=1), pooled), dim=1))
    return torch.cat(rows)


def fit_density(features, seen):
    """Estimate fit-glyph to atlas density using soft positives and a uniform baseline."""
    features = ((features - features.mean(dim=0)) /
                features.std(dim=0, unbiased=False).clamp_min(0.04)).clamp(-6, 6)
    positive = seen / seen.sum().clamp_min(1e-8)
    weight = torch.nn.Parameter(features.new_zeros(features.shape[1]))
    bias = torch.nn.Parameter(features.new_zeros(()))
    optimizer = torch.optim.LBFGS([weight, bias], lr=1.0, max_iter=35, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        score = features @ weight + bias
        loss = (positive * F.softplus(-score)).sum() + F.softplus(score).mean() + 0.01 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        score = features @ weight + bias
        return (score - torch.logsumexp(score, dim=0) + np.log(len(score))).clamp(-2.5, 2.5)


def permute_corpus(corpus, tiers, shapes, seed):
    ink = shapes[:, 0]
    cuts = torch.quantile(ink, torch.tensor([0.25, 0.5, 0.75]))
    strata = tiers * 4 + torch.bucketize(ink, cuts)
    rng = np.random.default_rng(seed)
    shuffled = corpus.clone()
    for stratum in strata.unique():
        members = torch.where(strata == stratum)[0]
        shuffled[members] = corpus[members[rng.permutation(len(members))]]
    return shuffled


@torch.no_grad()
def visual_matches(atlas, held):
    scores, indices = [], []
    for group in held.split(32):
        truth = group >= 128
        descriptor = image_descriptors(truth.to(torch.uint8) * 255)
        cosine, columns = (descriptor @ atlas.contours.T).topk(32, dim=1)
        overlap = _aligned_overlap(truth[:, None], atlas.images[columns] >= 128)
        scores.append(0.65 * cosine + 0.35 * overlap)
        indices.append(columns)
    score, index = torch.cat(scores), torch.cat(indices)
    best = score.argmax(dim=1, keepdim=True)
    best_indices = index.gather(1, best).squeeze(1)
    best_score = score.gather(1, best)
    return index, best_indices, score >= (best_score - 0.035), (score / 0.025).softmax(dim=1), best_score[:, 0]


def coverage(prior, candidates, best, close, mass, budgets):
    # Expected selection at an exact score tie avoids any Unicode-order effect.
    values, inverse, counts = torch.unique(prior, sorted=True, return_inverse=True, return_counts=True)
    higher = len(prior) - counts.cumsum(dim=0)
    best_rank = higher[inverse[best]] + (counts[inverse[best]] + 1) / 2
    local_groups = inverse[candidates]
    result = {}
    for budget in budgets:
        size = min(budget, len(prior))
        threshold = values[torch.searchsorted(-higher, -(size - 1), right=False).clamp(max=len(values) - 1)]
        threshold_group = torch.searchsorted(values, threshold)
        strict = higher[threshold_group]
        # Include the entire prefix above the boundary and a uniform sample of its ties.
        boundary_count = counts[threshold_group]
        taken = size - strict
        fraction = (taken / boundary_count).clamp(0, 1)
        probability = torch.where(prior[candidates] > threshold, 1.0,
                                  torch.where(prior[candidates] == threshold, fraction, 0.0))
        definitely = ((local_groups > threshold_group) & close).any(dim=1)
        tied = ((local_groups == threshold_group) & close).sum(dim=1)
        offsets = torch.arange(candidates.shape[1])
        miss = ((boundary_count - taken - offsets).clamp_min(0) /
                (boundary_count - offsets).clamp_min(1))
        miss = torch.where(offsets[None] < tied[:, None], miss, 1).prod(dim=1)
        covered = torch.where(definitely, 1.0, 1 - miss)
        exact = torch.where(prior[best] > threshold, 1.0,
                            torch.where(prior[best] == threshold, fraction, 0.0))
        result[budget] = (exact.numpy(), covered.numpy(), (probability * mass).sum(dim=1).numpy())
    return best_rank.numpy(), result


def summarize(measured, baseline, cohort):
    rank, budgets = measured
    summary = {"n": int(cohort.sum()), "median_best_shape_prior_rank": float(np.median(rank[cohort])), "budgets": {}}
    for budget, (best, close, mass) in budgets.items():
        delta = (close - baseline[1][budget][1])[cohort]
        summary["budgets"][str(budget)] = {"best_shape_coverage": float(best[cohort].mean()),
                                          "near_best_coverage": float(close[cohort].mean()),
                                          "soft_alignment_mass": float(mass[cohort].mean()),
                                          "near_best_delta_vs_current": float(delta.mean()),
                                          "near_best_paired_se": standard_error(delta)}
    return summary


def evaluate_split(raw, images, fit, held, corpus, tiers, pinlu, seed, args):
    atlas, alignment = fit_atlas(raw, images[fit], seed)
    seen, seen_report = seen_glyph_evidence(atlas, images[fit])
    current, buckets = _usage_log_enrichment(tiers, pinlu, seen)
    shape = morphology(atlas.images)
    usage = torch.cat((F.one_hot(tiers, 4), F.one_hot(pinlu, 4)), dim=1).float()
    metadata = torch.cat((shape, usage), dim=1)
    shuffled = permute_corpus(corpus, tiers, shape, seed + 3000)
    lexical = {"current": current, "morphology": fit_density(shape, seen),
               "morphology_unihan": fit_density(metadata, seen), "subtlex": fit_density(corpus, seen),
               "combined": fit_density(torch.cat((metadata, corpus), dim=1), seen),
               "permuted_subtlex": fit_density(torch.cat((metadata, shuffled), dim=1), seen)}
    unseen = torch.log1p(-0.94 * seen)

    ids, best, close, mass, visual_quality = visual_matches(atlas, images[held])
    positive = corpus[:, 0] > 0
    low = corpus[positive, 1].quantile(0.25)
    rare = ((corpus[best, 0] == 0) | (corpus[best, 1] <= low)).numpy()
    controls = {"all": np.ones(len(held), dtype=bool), "low_frequency_or_unlisted": rare}
    report = {}
    base = coverage(0.55 * current + unseen, ids, best, close, mass, args.budgets)
    for name, log_prior in lexical.items():
        values = base if name == "current" else coverage(0.55 * log_prior + unseen, ids, best, close, mass, args.budgets)
        report[name] = {group: summarize(values, base, subset) for group, subset in controls.items() if subset.any()}
    return {"seed": seed, "fit_glyphs": len(fit), "held_glyphs": len(held), "alignment": alignment,
            "fit_match": seen_report, "current_buckets": buckets, "held_visual_match_mean": float(visual_quality.mean()),
            "held_visual_match_ge_0_65": float((visual_quality >= 0.65).float().mean()), "metrics": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs=2, type=int, default=(613, 1283))
    parser.add_argument("--held-glyphs", type=int, default=320)
    parser.add_argument("--budgets", nargs="+", type=int, default=(1000, 3000, 6000, 12000))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/lexical_support_validation.json"))
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the image-only diagnostic on Kaggle CPU.")
    torch.set_num_threads(4)
    started = time.monotonic()
    images = load_images(locate_data(), torch.device("cpu"))
    if args.held_glyphs < 2 or 2 * args.held_glyphs > len(images) or len(images) - args.held_glyphs < 128:
        parser.error("Choose two disjoint holdouts with enough remaining fit glyphs.")
    if min(args.budgets) < 1:
        parser.error("Ranking budgets must be positive.")
    public, provenance, codepoints = render_font(*FONT_SOURCES[0], return_codepoints=True)
    frequency, corpus = load_subtlex(codepoints)
    tiers, pinlu, unicode_source = unihan_usage(codepoints)
    print(json.dumps({"font": provenance, "subtlex": corpus}), flush=True)
    output = {"device": "cpu", "protocol": "Image-only oracle reachability; held images never fit candidate priors.",
              "font": provenance, "subtlex": corpus, "unihan": unicode_source, "budgets": args.budgets, "splits": []}
    previous = np.zeros(len(images), dtype=bool)
    for seed in args.seeds:
        order = np.random.default_rng(seed).permutation(len(images))
        held = np.sort(order[~previous[order]][:args.held_glyphs])
        previous[held] = True
        fit = np.setdiff1d(np.arange(len(images)), held)
        split = evaluate_split(public, images, fit, held, frequency, tiers, pinlu, seed, args)
        output["splits"].append(split)
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        core = {method: stats["all"]["budgets"] for method, stats in split["metrics"].items()}
        print(json.dumps({"seed": seed, "coverage": core}), flush=True)


if __name__ == "__main__":
    main()
