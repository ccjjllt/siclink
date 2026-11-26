"""Test binary image-space expected contour risk on fixed synthetic glyph mixtures.

The same known posterior and gallery neighborhood determine both gallery
actions and synthesized actions. Only the differentiable contour proxy
chooses actions. Exact OCF scores every posterior state after selection,
including a stronger retrospective best real-glyph control in the same pool.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage
import torch

from glyph_data import load_images, locate_data
from glyph_stroke_consensus import _align_to_anchor, _compose, _proposals
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import image_descriptors, standard_error
from ocf_training import directional_fields, local_pool


N_REAL = 8
N_PROPOSALS = 10
MAX_EDITS = 3
EDIT_FRACTION = 0.08
MIN_COMPONENT = 6
ALIGNMENT_RADIUS = 2
MIN_GAIN = 0.0001
SIGMAS = (1, 2, 4)
CONNECTIVITY = np.ones((3, 3), dtype=bool)
SCENARIOS = (("near_pair", 16, (0.55, 0.45)), ("near_triple", 12, (0.45, 0.35, 0.20)),
             ("single", 8, (1.0,)), ("unrelated_pair", 8, (0.55, 0.45)))


@dataclass(frozen=True)
class Mixture:
    scenario: str
    states: torch.Tensor
    probabilities: torch.Tensor
    neighborhood: torch.Tensor


@torch.no_grad()
def mixtures(descriptors, seed):
    rng = np.random.default_rng(seed)
    source = iter(rng.permutation(len(descriptors)).tolist())
    used = set()
    for scenario, count, probabilities in SCENARIOS:
        mass = torch.tensor(probabilities, dtype=torch.float32)
        for _ in range(count):
            anchor = next(index for index in source if index not in used)
            states = [anchor]
            if len(mass) > 1:
                likeness = descriptors[anchor] @ descriptors.T
                order = likeness.argsort(descending=True).tolist()
                if scenario == "unrelated_pair":
                    order = rng.permutation(order[len(order) // 2:]).tolist()
                states.extend([index for index in order if index != anchor and index not in used][:len(mass) - 1])
            used.update(states)
            ids = torch.tensor(states, dtype=torch.long)
            neighborhood_center = mass @ descriptors[ids]
            relevance = neighborhood_center @ descriptors.T
            relevance[ids] = -torch.inf
            nearby = torch.cat((ids, relevance.topk(N_REAL - len(ids)).indices))
            yield Mixture(scenario, ids, mass, nearby)


class ContourRisk:
    """The pairwise proxy in ocf_training with expectation after each ratio."""

    def __init__(self, targets, weights):
        self.weights = weights
        fields = [directional_fields(targets[:, None].float(), sigma) for sigma in SIGMAS]
        self.reference = [(value, local_pool(value.square().sum(dim=1, keepdim=True))) for value in fields]

    @torch.no_grad()
    def __call__(self, bitmaps):
        matched, overlap = [], []
        for sigma, (truth, energy) in zip(SIGMAS, self.reference):
            action = directional_fields(bitmaps[:, None].float(), sigma)
            action_energy = local_pool(action.square().sum(dim=1, keepdim=True))[:, None]
            cross = (action[:, None] * truth[None]).sum(dim=2, keepdim=True)
            cross = local_pool(cross.flatten(0, 1)).unflatten(0, cross.shape[:2]).clamp_min(0)
            correlation = (cross.square() / (action_energy * energy[None]).clamp_min(1e-10)).clamp_max(1)
            shared = torch.minimum(action_energy, energy[None]) * correlation
            total = action_energy + energy[None] - shared
            matched.append(shared.sum(dim=(2, 3, 4)))
            overlap.append(total.sum(dim=(2, 3, 4)))
        utilities = torch.stack(matched).sum(0) / torch.stack(overlap).sum(0).clamp_min(1e-8)
        return utilities @ self.weights


def edit_candidates(anchor, supporting, weights, budget):
    prior_support = np.tensordot(weights, supporting, axes=1)
    nearby = ndimage.distance_transform_edt(~anchor) <= ALIGNMENT_RADIUS
    boundary = anchor & ~ndimage.binary_erosion(anchor, structure=CONNECTIVITY)
    proposals = {}
    for candidate in supporting:
        for added, changed in ((True, candidate & ~anchor), (False, anchor & ~candidate)):
            connected, _ = ndimage.label(changed, structure=CONNECTIVITY)
            counts = np.bincount(connected.ravel())
            for item in np.flatnonzero((counts >= MIN_COMPONENT) & (counts <= budget)):
                if not item:
                    continue
                pixels = connected == item
                if (added and not nearby[pixels].any()) or (not added and not boundary[pixels].any()):
                    continue
                backing = prior_support[pixels] if added else 1 - prior_support[pixels]
                proposals[pixels.tobytes()] = (float(backing.mean() * np.sqrt(counts[item])), added, pixels)
    return sorted(proposals.values(), key=lambda proposal: -proposal[0])[:N_PROPOSALS]


def count_components(mask):
    _, count = ndimage.label(mask, structure=CONNECTIVITY)
    return count


def optimize_binary(anchor, support, weights, proxy, budget):
    proposals = edit_candidates(anchor, support, weights, budget)
    image = anchor.copy()
    changed = np.zeros_like(anchor)
    initial = current = float(proxy(torch.as_tensor(anchor[None]))[0])
    max_components = count_components(anchor) + 1
    n_edits = 0
    for _ in range(MAX_EDITS):
        next_masks, next_edits = [], []
        for _, added, pixels in proposals:
            if np.any(pixels & changed) or int(pixels.sum() + changed.sum()) > budget:
                continue
            trial = image.copy()
            trial[pixels] = added
            if count_components(trial) <= max_components:
                next_masks.append(trial)
                next_edits.append(pixels)
        if not next_masks:
            break
        utility = proxy(torch.as_tensor(np.stack(next_masks)))
        winner = int(utility.argmax())
        if float(utility[winner]) < current + MIN_GAIN:
            break
        image = next_masks[winner]
        changed |= next_edits[winner]
        current = float(utility[winner])
        n_edits += 1
    return image, {"candidate_parts": len(proposals), "accepted_parts": n_edits, "pixels_changed": int(changed.sum()),
                   "edit_budget": budget, "proxy_gain": current - initial}


def exact_utilities(actions, references, weights, scorer, cache):
    keys = [[(np.packbits(action).tobytes(), np.packbits(reference).tobytes()) for reference in references]
            for action in actions]
    missing = {}
    for row, group in enumerate(keys):
        for index, key in enumerate(group):
            if key not in cache:
                missing[key] = (actions[row], references[index])
    if missing:
        left, right = (np.stack(items) for items in zip(*missing.values()))
        measured = scorer(left, right, device="cpu", batch_size=16)
        cache.update(zip(missing, measured.tolist()))
    per_state = np.asarray([[cache[key] for key in group] for group in keys])
    intersection = (actions[:, None] & references[None]).sum(axis=(2, 3))
    union = (actions[:, None] | references[None]).sum(axis=(2, 3)).clip(min=1)
    return per_state @ weights, (intersection / union) @ weights


@torch.no_grad()
def evaluate(case, images, scorer, cache):
    references = images[case.states]
    hard = images[case.neighborhood]
    raw_proxy = ContourRisk(references, case.probabilities)
    anchor_index = int(raw_proxy(hard).argmax())
    anchor = hard[anchor_index]
    ordered = torch.cat((anchor[None], references))[None]
    aligned, _ = _align_to_anchor(ordered, ALIGNMENT_RADIUS)
    aligned = aligned[0, 1:]
    weight = case.probabilities.numpy()
    support_np, anchor_np = aligned.numpy(), anchor.numpy()
    budget = int(anchor.sum() * EDIT_FRACTION)
    vote_support = np.tensordot(weight, support_np, axes=1)
    original = _proposals(anchor_np, support_np, vote_support, weight, budget)
    sparse = _compose(anchor_np, original, budget, additions_only=False)
    vote = vote_support >= 0.5
    if not vote.any():
        vote = anchor_np
    registered_proxy = ContourRisk(aligned, case.probabilities)
    registered_hard_index = int(registered_proxy(hard).argmax())
    synthesis, info = optimize_binary(anchor_np, support_np, weight, registered_proxy, budget)
    images_to_score = np.concatenate((hard.numpy(), np.stack((vote, sparse, synthesis))))
    ocf, iou = exact_utilities(images_to_score, references.numpy(), weight, scorer, cache)
    names = {"map": 0, "hard_proxy_medoid": anchor_index, "hard_registered_proxy_medoid": registered_hard_index,
             "hard_exact_best_of8": int(ocf[:N_REAL].argmax()),
             "aligned_hard_vote": N_REAL, "budgeted_sparse_consensus": N_REAL + 1, "binary_preimage": N_REAL + 2}
    return {"scenario": case.scenario, "state_count": len(weight), "absolute_posterior_mass": float(weight.sum()),
            "expected_ocf": {name: float(ocf[index]) for name, index in names.items()},
            "expected_iou": {name: float(iou[index]) for name, index in names.items()}, **info}


def summarize(cases):
    metrics = {}
    for name in cases[0]["expected_ocf"]:
        exact = np.asarray([case["expected_ocf"][name] for case in cases])
        against_hard = exact - np.asarray([case["expected_ocf"]["hard_exact_best_of8"] for case in cases])
        against_medoid = exact - np.asarray([case["expected_ocf"]["hard_proxy_medoid"] for case in cases])
        binary_iou = np.asarray([case["expected_iou"][name] for case in cases])
        metrics[name] = {"mean_exact_posterior_ocf": float(exact.mean()), "mean_raw_binary_iou": float(binary_iou.mean()),
                         "delta_vs_hard_exact_best_of8": float(against_hard.mean()),
                         "paired_se_vs_hard_exact": standard_error(against_hard),
                         "delta_vs_hard_proxy_medoid": float(against_medoid.mean()),
                         "paired_se_vs_hard_proxy_medoid": standard_error(against_medoid),
                         "beats_hard_exact_cases": int((against_hard > 1e-6).sum())}
    return {"cases": len(cases), "preimages_changed": sum(case["pixels_changed"] > 0 for case in cases),
            "preimage_mean_pixels_changed": float(np.mean([case["pixels_changed"] for case in cases])),
            "preimage_mean_proxy_gain": float(np.mean([case["proxy_gain"] for case in cases])), "methods": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=13127)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_preimage_geometry.json"))
    args = parser.parse_args()
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run on Kaggle CPU with the official training input.")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    images = load_images(root, "cpu") >= 128
    descriptors = torch.cat([image_descriptors(batch.to(torch.uint8) * 255) for batch in images.split(256)])
    scorer = official_scorer(root)
    output = {"seed": args.seed, "device": "cpu", "settings": {"hard_neighborhood": N_REAL,
              "edit_fraction_of_anchor_ink": EDIT_FRACTION, "maximum_added_or_removed_components": MAX_EDITS,
              "stored_edit_proposals": N_PROPOSALS, "max_canvas_shift_pixels": ALIGNMENT_RADIUS, "posterior_groups": SCENARIOS},
              "protocol": "Fixed known mixtures of real training images. Each case has separate reference glyphs. "
                          "Every action sees the same posterior support and eight eligible real glyphs. The hard "
                          "baseline uses the per-state unregistered contour proxy; after fixed candidate-only "
                          "registration the synthesized action maximizes the same proxy using capped binary "
                          "edits of visually supported connected strokes. Exact OCF is computed only after all "
                          "actions are chosen, against unmodified real reference masks and the full probability "
                          "mass. The retrospective exact best of eight is a strong hard control, not a bound "
                          "over all 2,748 gallery glyphs. No spikes, actual test data or leaderboards are used.",
              "cases": []}
    cache = {}
    for index, case in enumerate(mixtures(descriptors, args.seed), start=1):
        result = evaluate(case, images, scorer, cache)
        output["cases"].append(result)
        if index % 4 == 0:
            print(json.dumps({"completed": index, "last_scenario": case.scenario, "exact_pairs": len(cache),
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
    all_cases = output["cases"]
    output["groups"] = {scenario: summarize([case for case in all_cases if case["scenario"] == scenario])
                        for scenario, _, _ in SCENARIOS}
    output["primary_near_mixture_groups"] = summarize([case for case in all_cases if case["scenario"].startswith("near_")])
    output["unique_exact_pairs"] = len(cache)
    output["seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"primary_near_mixture_groups": output["primary_near_mixture_groups"],
                      "unique_exact_pairs": len(cache), "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
