"""Estimate an open-atlas usage prior from public Unicode data and fit glyphs."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import re
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

import torch
from torch.nn import functional as F

from kaggle_gallery_validation import image_descriptors, standardized


UNIHAN_VERSION = "17.0.0"
UNIHAN_SOURCE = f"https://www.unicode.org/Public/{UNIHAN_VERSION}/ucd/Unihan.zip"
UNIHAN_LICENSE = "https://www.unicode.org/license.txt"
PINLU_COUNT = re.compile(r"\((\d+)\)")


def unihan_usage(codepoints):
    """Return modern usage tiers and frequency buckets in atlas order.

    kFrequency was removed in Unicode 16.0. kTGH supplies the three levels
    of the 2013 general-use character table, and kHanyuPinlu supplies corpus
    counts for the characters that have them. Zero represents missing data.
    """
    try:
        with urlopen(UNIHAN_SOURCE, timeout=90) as response:
            payload = response.read()
    except URLError as error:
        raise RuntimeError("Cannot fetch the pinned Unihan archive. Check Kaggle Notebook Internet.") from error

    wanted = set(codepoints.tolist())
    tier, counts = {}, {}
    with ZipFile(BytesIO(payload)) as archive:
        for entry in archive.namelist():
            if not entry.endswith(".txt"):
                continue
            with archive.open(entry) as lines:
                for line in lines:
                    if not line.startswith(b"U+"):
                        continue
                    raw_codepoint, field, value = line.decode("utf-8").rstrip("\r\n").split("\t", 2)
                    if field not in ("kTGH", "kHanyuPinlu"):
                        continue
                    codepoint = int(raw_codepoint[2:], 16)
                    if codepoint not in wanted:
                        continue
                    if field == "kTGH":
                        year, _, ordinal = value.partition(":")
                        if year == "2013":
                            index = int(ordinal)
                            tier[codepoint] = 1 if index <= 3500 else 2 if index <= 6500 else 3
                    else:
                        counts[codepoint] = sum(int(match) for match in PINLU_COUNT.findall(value))
    if not tier or not counts:
        raise ValueError(f"Expected both kTGH and kHanyuPinlu in Unihan {UNIHAN_VERSION}.")

    ordered = codepoints.tolist()
    usage_tier = torch.tensor([tier.get(codepoint, 0) for codepoint in ordered], dtype=torch.long)
    frequency = torch.tensor([counts.get(codepoint, 0) for codepoint in ordered], dtype=torch.float32)
    present = frequency > 0
    edges = torch.quantile(frequency[present], torch.tensor([1.0 / 3, 2.0 / 3]))
    frequency_bin = torch.zeros_like(usage_tier)
    frequency_bin[present] = torch.bucketize(frequency[present], edges) + 1
    metadata = {"version": UNIHAN_VERSION, "source": UNIHAN_SOURCE, "sha256": sha256(payload).hexdigest(),
                "license": "Unicode License V3", "license_url": UNIHAN_LICENSE,
                "documentation": "https://www.unicode.org/reports/tr38/",
                "fields": ["kTGH (2013 levels)", "kHanyuPinlu (atlas terciles)"]}
    return usage_tier, frequency_bin, metadata


def _aligned_overlap(reference, candidates):
    padded = F.pad(candidates, (2, 2, 2, 2))
    reference_mass = reference.sum(dim=(-1, -2))
    best = candidates.new_zeros(candidates.shape[:2], dtype=torch.float32)
    for x in (0, 2, 4):
        for y in (0, 2, 4):
            shifted = padded[..., y:y + 64, x:x + 64]
            intersection = (reference & shifted).sum(dim=(-1, -2))
            union = reference_mass + shifted.sum(dim=(-1, -2)) - intersection
            best = torch.maximum(best, intersection / union.clamp_min(1))
    return best


@torch.no_grad()
def seen_glyph_evidence(atlas, fit_images, *, neighbors=8, batch_size=96):
    """Collect soft fit-image matches and aggregate candidate membership evidence."""
    if not 1 <= neighbors <= len(atlas.images) or batch_size < 1 or not len(fit_images):
        raise ValueError("Provide fit glyphs and valid public-image shortlist settings.")
    fit_images = fit_images.to(atlas.images.device)
    absent_log = atlas.contours.new_zeros(len(atlas.images))
    qualities, probabilities = [], []
    for batch in fit_images.split(batch_size):
        masks = batch >= 128
        descriptor = image_descriptors(masks.to(torch.uint8) * 255)
        neighborhood, indices = (descriptor @ atlas.contours.T).topk(neighbors, dim=1)
        overlap = _aligned_overlap(masks[:, None], atlas.images[indices] >= 128)
        similarity = 0.65 * neighborhood + 0.35 * overlap
        best = similarity.max(dim=1).values
        confidence = ((best - 0.65) / 0.055).sigmoid()
        match = confidence[:, None] * (similarity / 0.025).softmax(dim=1)
        absent_log.scatter_add_(0, indices.flatten(), torch.log1p(-match.flatten().clamp(max=0.995)))
        qualities.append(best)
        probabilities.append(confidence)
    evidence = -torch.expm1(absent_log)
    return evidence, {"mean_fit_match_similarity": float(torch.cat(qualities).mean()),
                      "mean_fit_match_confidence": float(torch.cat(probabilities).mean()),
                      "expected_seen_masks": float(evidence.sum()),
                      "masks_with_seen_evidence_above_0_8": int((evidence > 0.8).sum())}


def _usage_log_enrichment(usage_tier, frequency_bin, seen_evidence):
    tier_size = torch.bincount(usage_tier, minlength=4)
    tier_mass = torch.bincount(usage_tier, weights=seen_evidence, minlength=4)
    global_rate = seen_evidence.mean().clamp_min(1e-5)
    tier_rate = (tier_mass + 128 * global_rate) / (tier_size + 128)
    bucket = usage_tier * 4 + frequency_bin
    bucket_size = torch.bincount(bucket, minlength=16)
    bucket_mass = torch.bincount(bucket, weights=seen_evidence, minlength=16)
    bucket_rate = (bucket_mass + 64 * tier_rate.repeat_interleave(4)) / (bucket_size + 64)
    enrichment = (bucket_rate[bucket] / global_rate).log().clamp(-2.5, 2.5)
    diagnostics = {"tier_atlas_counts": tier_size.tolist(), "tier_fit_evidence_mass": tier_mass.tolist(),
                   "bucket_atlas_counts": bucket_size.reshape(4, 4).tolist(),
                   "bucket_fit_evidence_mass": bucket_mass.reshape(4, 4).tolist()}
    return enrichment, diagnostics


@dataclass
class AtlasPrior:
    lexical_log: torch.Tensor
    unseen_log: torch.Tensor
    diagnostics: dict

    def variants(self, base_logits):
        """Apply fixed lexical and unseen weights to neural logits in this atlas's order."""
        if base_logits.ndim != 2 or base_logits.shape[1] != len(self.lexical_log):
            raise ValueError("Neural logits must have one column per fixed atlas template.")
        scores = standardized(base_logits)
        lexical = self.lexical_log.to(scores)
        unseen = self.unseen_log.to(scores)
        return {"usage_only": scores + 0.55 * lexical,
                "unseen_only": scores + unseen,
                "usage_unseen": scores + 0.55 * lexical + unseen,
                "usage_unseen_strong": scores + 0.9 * lexical + 1.45 * unseen}


@torch.no_grad()
def fit_atlas_prior(atlas, codepoints, fit_images):
    """Learn broad usage enrichment and soft visible-fit-glyph exclusion."""
    if codepoints.ndim != 1 or len(codepoints) != len(atlas.images):
        raise ValueError("Codepoints must be aligned one-to-one with public atlas masks.")
    usage, frequency, source = unihan_usage(codepoints.cpu())
    seen, fit_summary = seen_glyph_evidence(atlas, fit_images)
    lexical, buckets = _usage_log_enrichment(usage.to(seen.device), frequency.to(seen.device), seen)
    unseen = torch.log1p(-0.94 * seen)
    return AtlasPrior(lexical, unseen, {"unicode": source, "fit_alignment": fit_summary, "usage_buckets": buckets})
