"""Test whether public glyph components can inherit raster detail from labeled glyphs."""

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_data import load_images, locate_data
from kaggle_diagnostics import official_scorer
from kaggle_task1_ids_retrieval import component_dictionary
from kaggle_task1_raster_generation import (CONTEXT_GLYPHS, FIT_GLYPHS, compare, load_variable_font,
                                            prepare_lookup)


SEED, EARLIER_SEED = 97073, 32507
OUTER, PART_SIZE = 320, 24
OUTPUT = Path("/kaggle/working/task1_component_raster.json")
OPERATOR_AXIS = {"⿰": 1, "⿱": 0}


@dataclass(frozen=True)
class Part:
    image: torch.Tensor
    bounds: tuple[int, int, int, int]


@dataclass(frozen=True)
class Donor:
    component: str
    public: Part
    reference: Part


def picked_scope(size):
    older = np.random.default_rng(EARLIER_SEED).permutation(size)
    excluded = older[CONTEXT_GLYPHS + FIT_GLYPHS:CONTEXT_GLYPHS + FIT_GLYPHS + 160]
    selector = np.random.default_rng(SEED)
    held = np.sort(selector.choice(np.setdiff1d(np.arange(size), excluded), OUTER, replace=False))
    fit = selector.permutation(np.setdiff1d(np.arange(size), held))
    return torch.from_numpy(fit), torch.from_numpy(held), len(excluded)


def token_slots(tokens):
    layout = next((value.split("=")[1] for value in tokens if value.startswith("layout=")), None)
    if layout not in OPERATOR_AXIS:
        return None
    paired = tuple(next((token[3:] for token in tokens if token.startswith(f"{layout}{side}=")), None)
                   for side in range(2))
    return (layout, paired) if all(paired) and all(len(token) == 1 and token not in "⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻" for token in paired) else None


def reshape_image(image, size):
    return F.interpolate(image[None, None], size, mode="bilinear", align_corners=False, antialias=True)[0, 0]


def crop(image, mask, offset_y=0, offset_x=0):
    points = mask.nonzero()
    if len(points) < 28:
        return None
    top, left = map(int, points.min(0).values)
    bottom, right = map(int, points.max(0).values + 1)
    if min(bottom - top, right - left) < 4:
        return None
    return Part(reshape_image(image[top:bottom, left:right], (PART_SIZE, PART_SIZE)),
                (top + offset_y, bottom + offset_y, left + offset_x, right + offset_x))


def cut_parts(image, layout, aim=0.5):
    gray = image.float() / 255
    mask = gray >= 0.5
    occupied = mask.nonzero()
    if not len(occupied):
        return None
    axis = OPERATOR_AXIS[layout]
    first, last = occupied[:, axis].amin().item(), occupied[:, axis].amax().item() + 1
    length = last - first
    track = mask.float().sum(dim=1 - axis)
    cuts = torch.arange(first + max(4, int(0.20 * length)), first + min(length - 3, int(0.80 * length)))
    if not len(cuts):
        return None
    penalty = 0.5 * (track[cuts - 1] + track[cuts]) / track.max().clamp_min(1)
    cut = int(cuts[(penalty + 0.16 * ((cuts - first) / length - aim).abs()).argmin()])
    before, after = gray.split((cut, gray.shape[axis] - cut), dim=axis)
    leading = crop(before, before >= 0.5)
    trailing = crop(after, after >= 0.5, offset_y=cut if axis == 0 else 0, offset_x=cut if axis == 1 else 0)
    if leading is None or trailing is None:
        return None
    return (leading, trailing), (cut - first) / length


def aspect(part):
    top, bottom, left, right = part.bounds
    return np.log((right - left) / (bottom - top))


def make_donors(images, baseline, codes, agrees, parts):
    families, truth, emitted = defaultdict(list), set(), set()
    for actual, public, label, confident in zip(images, baseline, codes.tolist(), agrees.tolist()):
        parsed = parts.get(label)
        if not confident or parsed is None or label in truth:
            continue
        truth.add(label)
        layout, names = parsed
        fixed = cut_parts(public, layout)
        if fixed is None:
            continue
        observed = cut_parts(actual, layout, aim=fixed[1])
        if observed is None or abs(fixed[1] - observed[1]) > 0.14:
            continue
        for slot, (name, src, ref) in enumerate(zip(names, fixed[0], observed[0])):
            delta = (src.image - ref.image).abs().mean()
            if float(delta) <= 0.31 and abs(aspect(src) - aspect(ref)) <= 0.42:
                families[layout, slot].append(Donor(name, src, ref))
                emitted.add(label)
    counts = {f"{layout}{slot}": len(rows) for (layout, slot), rows in families.items()}
    return families, {"trusted_fitting_codepoints": len(truth), "fitting_codepoints_yielding_donors": len(emitted),
                      "donors_by_layout_and_slot": counts}


def prepare_banks(families):
    return {key: (rows, F.normalize(torch.stack([donor.public.image.flatten() for donor in rows]), dim=1),
                  torch.tensor([aspect(donor.public) for donor in rows])) for key, rows in families.items()}


def alternate_rasters(public, label, parts, banks):
    parsed = parts.get(label)
    if parsed is None:
        return public, public, public, []
    layout, names = parsed
    fixed = cut_parts(public, layout)
    if fixed is None:
        return public, public, public, []
    residual, matched, replaced = (public.float().clone() / 255 for _ in range(3))
    donors = []
    for slot, (name, target) in enumerate(zip(names, fixed[0])):
        if (layout, slot) not in banks:
            continue
        bank, descriptor, ratios = banks[layout, slot]
        eligible = torch.tensor([donor.component == name for donor in bank])
        if not bool(eligible.any()) or not bool((~eligible).any()):
            continue
        comparison = descriptor @ F.normalize(target.image.flatten(), dim=0) - 0.12 * (ratios - aspect(target)).abs()
        good = int(comparison.masked_fill(~eligible, -torch.inf).argmax())
        wrong = int(comparison.masked_fill(eligible, -torch.inf).argmax())
        if float(comparison[good]) < 0.70:
            continue
        top, bottom, left, right = target.bounds
        scope = (slice(top, bottom), slice(left, right))
        size = (bottom - top, right - left)
        true_donor, counterfactual = bank[good], bank[wrong]
        residual[scope] += 0.65 * reshape_image(true_donor.reference.image - true_donor.public.image, size)
        matched[scope] += 0.65 * reshape_image(counterfactual.reference.image - counterfactual.public.image, size)
        replaced[scope] = reshape_image(true_donor.reference.image, size)
        donors.append({"matching_public_similarity": float(comparison[good]), "control_public_similarity": float(comparison[wrong])})
    return (*(value.clamp(0, 1).mul(255).round().to(torch.uint8) for value in (residual, matched, replaced)), donors)


def exact(scorer, predicted, reference):
    return np.asarray(scorer((predicted >= 128).numpy(), (reference >= 128).numpy(), device="cpu", batch_size=8))


def main():
    if not Path("/kaggle/input").is_dir() or not OUTPUT.parent.is_dir():
        raise RuntimeError("Run this glyph-image diagnostic on a Kaggle CPU Notebook.")
    started = time.monotonic()
    torch.set_num_threads(4)
    root = locate_data()
    images = load_images(root, torch.device("cpu"))
    fit, held, old_count = picked_scope(len(images))
    _, supported, font = load_variable_font()
    scope = torch.cat((fit, held))
    bank, codes, agrees, shape = prepare_lookup(images[fit[:CONTEXT_GLYPHS]], images[scope], supported, SEED)
    baseline, identity = bank[:, 0].contiguous(), codes[:, 0]
    del bank, codes
    vocabulary = identity.unique()
    token_rows, dictionary = component_dictionary(vocabulary)
    labels = {int(code): token_slots(tokens) for code, tokens in zip(vocabulary, token_rows)}
    donor_bank, training = make_donors(images[fit], baseline[:len(fit)], identity[:len(fit)], agrees[:len(fit)], labels)
    prepared = prepare_banks(donor_bank)
    held_matches = agrees[len(fit):] & ~torch.isin(identity[len(fit):], identity[:len(fit)][agrees[:len(fit)]])
    predicted = {name: [] for name in ("stock", "same_component_residual", "nearest_other_component_residual", "same_component_replacement")}
    indices, candidates = [], []
    for index, (candidate, label) in enumerate(zip(baseline[len(fit):], identity[len(fit):].tolist())):
        if not bool(held_matches[index]):
            continue
        corrected, control, replacement, details = alternate_rasters(candidate, label, labels, prepared)
        if not details:
            continue
        indices.append(index)
        candidates.extend(details)
        for name, value in zip(predicted, (candidate, corrected, control, replacement)):
            predicted[name].append(value)
    report = {"seed": SEED, "fit_whole_glyphs": len(fit), "held_whole_glyphs": len(held),
              "previous_eval_glyphs_excluded": old_count, "high_agreement_held_codes": int(held_matches.sum()),
              "editable_held_codes": len(indices), "matched_component_slots": len(candidates), "donor_fit": training,
              "sources": {"font": font, "lookup": shape, "dictionary": dictionary},
              "protocol": "Freeze the same public codepoint per held image using two independently matched public faces. "
                          "A known reference is used only as a diagnostic of this codepoint agreement and in official OCF. "
                          "Masks for rendered components are derived from each candidate alone. Transfer fixed 65% of the "
                          "signed raster residual of the closest same-component fitting exemplar, versus the nearest exemplar "
                          "from another component within the same layout and side. This is image-only. No neural selection "
                          "or formal queries are used.", "status": "coverage_ready"}
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if len(indices) >= 2:
        scorer = official_scorer(root)
        reference = images[held[indices]]
        scores = {name: exact(scorer, torch.stack(rows), reference) for name, rows in predicted.items()}
        report["exact_ocf"] = {name: compare(scores["stock"], values) for name, values in scores.items()}
        report["same_vs_other_residual"] = compare(scores["nearest_other_component_residual"], scores["same_component_residual"])
        report["donor_similarity"] = {key: float(np.mean([row[key] for row in candidates])) for key in candidates[0]}
    report.update(seconds=time.monotonic() - started, status="complete")
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("sources", "protocol")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
