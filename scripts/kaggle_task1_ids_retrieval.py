"""Decode position-specific public IDS components for unseen glyph retrieval."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time
from urllib.request import urlopen

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from glyph_atlas_metric_ranker import describe_atlas_candidates, pack_atlas_candidates
from glyph_contour_retrieval import _response_features
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_retrieval import _standardize
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_gallery_validation import FontAtlas, image_descriptors, standardized
from kaggle_task1_crossfit_candidate import fit_crossfit_head
from kaggle_task1_metric_scale import ANCHOR, MODEL_SEED, NOISE_SIZE, ORIGINAL_SIZE
from kaggle_task1_peer_transport import choose_split as previous_split
from kaggle_task1_poisson_bridge import fresh_split
from kaggle_task1_repertoire_coverage import shape_matches
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, nested_ranker, rendered_bank, summarize


SEED = 33643
HELD_COUNT, ORACLE_COUNT = 320, 20
QUALITY, MARGIN = 0.65, 0.02
ADD = 4
IDS_PROPOSAL_WEIGHT, IDS_OCF_WEIGHT = 0.55, 3.0
MOUSE = "mice1"
ARMS = ("v14", "v8_depth", "ids_nomination", "ids_nomination_and_score", "stratified_shuffle")
OUTPUT = Path("/kaggle/working/task1_ids_retrieval.json")
REVISION = "bddc96d41bef78427ed0e034e9f7e31d71fd1b92"
REPO = "https://github.com/skishore/makemeahanzi"
URL = f"https://raw.githubusercontent.com/skishore/makemeahanzi/{REVISION}/dictionary.txt"
OPERATORS = {chr(code): (3 if code in (0x2FF2, 0x2FF3) else 2) for code in range(0x2FF0, 0x2FFC)}


def parse_ids(expression):
    def parse_at(index):
        token = expression[index]
        children = []
        index += 1
        for _ in range(OPERATORS.get(token, 0)):
            child, index = parse_at(index)
            children.append(child)
        return (token, children), index

    try:
        (root, branches), end = parse_at(0)
    except IndexError:
        return set()
    if not branches or end != len(expression):
        return set()
    result = {f"layout={root}"}
    for slot, (token, children) in enumerate(branches):
        position = f"{root}{slot}"
        if token != "？":
            result.add(f"{position}={token}")
        for subslot, (component, _) in enumerate(children):
            if component != "？":
                result.add(f"{position}/{token}{subslot}={component}")
        # Allow a familiar component to transfer between different layouts.
        for component in ((token,) if not children else tuple(value for value, _ in children)):
            if component not in OPERATORS and component != "？":
                result.add(f"component={component}")
    return result


def component_dictionary(codepoints):
    with urlopen(URL, timeout=90) as response:
        payload = response.read()
    entries = {}
    for line in payload.splitlines():
        item = json.loads(line)
        if len(item["character"]) == 1:
            entries[ord(item["character"])] = parse_ids(item["decomposition"])
    rows = [entries.get(codepoint, set()) for codepoint in codepoints.tolist()]
    return rows, {"source": URL, "sha256": sha256(payload).hexdigest(), "license": "LGPL-3.0-or-later",
                  "license_url": f"{REPO}/blob/{REVISION}/COPYING", "entries": len(entries)}


def split_glyphs(count, device):
    cpu = torch.device("cpu")
    _, peer_previous, _, _ = previous_split(count, cpu)
    _, bridge_previous, _ = fresh_split(count, cpu)
    eligible = np.setdiff1d(np.arange(count), np.union1d(peer_previous.numpy(), bridge_previous.numpy()))
    held = np.sort(np.random.default_rng(SEED).choice(eligible, size=HELD_COUNT, replace=False))
    return (torch.as_tensor(np.setdiff1d(np.arange(count), held), device=device),
            torch.as_tensor(held, device=device))


def visual_atlas(bank):
    cpu = bank.cpu()
    return FontAtlas(cpu, torch.cat([image_descriptors(part) for part in cpu.split(1024)]))


def matched_training(rows, code_map, references, device):
    confidence = rows.confident(QUALITY, MARGIN)
    accepted = [index for index in torch.where(confidence)[0].tolist() if code_map[int(rows.closest[index])]]
    support = Counter(token for index in accepted for token in code_map[int(rows.closest[index])])
    vocabulary = {token: index for index, token in enumerate(sorted(token for token, freq in support.items()
                                                                   if 10 <= freq < 0.95 * len(accepted)))}
    if len(accepted) < 500 or len(vocabulary) < 30:
        raise ValueError("Require enough visually identified training images and observed public components.")
    atlas_codes = torch.zeros((len(code_map), len(vocabulary)), dtype=torch.float32)
    for glyph, tokens in enumerate(code_map):
        atlas_codes[glyph, [vocabulary[token] for token in tokens if token in vocabulary]] = 1.0
    indices = torch.as_tensor(accepted, device=device)
    fit_targets = atlas_codes[rows.closest[accepted]].to(device)
    masks = references[indices] >= 128
    ink = masks.float().mean(dim=(1, 2))
    outline = ((masks[:, :, 1:] != masks[:, :, :-1]).sum(dim=(1, 2))
               + (masks[:, 1:, :] != masks[:, :-1, :]).sum(dim=(1, 2)))
    donors = torch.empty_like(indices)
    for bin_ink in ink.argsort().tensor_split(6):
        for bin_outline in bin_ink[outline[bin_ink].argsort()].tensor_split(4):
            donors[bin_outline] = bin_outline.roll(1)
    return indices, fit_targets, fit_targets[donors], atlas_codes.to(device), int(confidence.sum())


@torch.no_grad()
def projected_counts(model, counts, day):
    values = _response_features(counts, model.view, model.n_bins)
    return _standardize(values, *model.moments[day]).mean(dim=1) @ model.projection


def learn_component_scores(response, targets):
    features = torch.cat(tuple(response.values()))
    labels = targets.repeat(len(response), 1)
    center = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(0.05)
    features = (features - center) / scale
    prior = targets.mean(0).clamp(0.005, 0.98).logit()
    head = nn.Linear(features.shape[1], targets.shape[1], device=features.device)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(prior)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.012, weight_decay=0.08)
    rng = torch.Generator(device=features.device).manual_seed(SEED + 1)
    for _ in range(20):
        for rows in torch.randperm(len(features), generator=rng, device=features.device).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(head(features[rows]), labels[rows])
            loss += 0.6 * head.weight.square().mean()
            loss.backward()
            optimizer.step()
    return head, center, scale, prior


@torch.no_grad()
def ids_scores(learned, response, components):
    head, center, scale, prior = learned
    increment = head((response - center) / scale) - prior
    count = components.sum(1)
    available = count > 0
    value = (increment @ components.T) / count.clamp_min(1).sqrt()
    two_sided = standardized(value[:, available]).clamp(-4, 4)
    value[:, available] = two_sided
    return value


def replaced_pool(base, ranking, original, proposal):
    proposals = proposal.clone()
    rows, columns = base.valid.nonzero(as_tuple=True)
    proposals[rows, base.indices[rows, columns]] = -torch.inf
    new = proposals.topk(ADD, dim=1).indices
    original_count = base.valid.sum(1, keepdim=True)
    if bool((original_count <= ADD).any()):
        raise ValueError("Keep the incumbent plus the same four proposal slots in every arm.")
    order = ranking.argsort(dim=1, descending=True, stable=True)
    retained = base.indices.gather(1, order)
    slots = original_count - ADD + torch.arange(ADD, device=retained.device)
    retained.scatter_(1, slots, new)
    valid = torch.arange(retained.shape[1], device=retained.device) < original_count
    return describe_atlas_candidates(original, NOMINATIONS, base.baseline, base.rank_depth,
                                     retained.masked_fill(~valid, 0), valid)


@torch.no_grad()
def decide(raw, real, shuffled, ranker):
    base = pack_atlas_candidates(raw, NOMINATIONS, baseline="v8")
    ranking = ranker.rank(base, anchor_ocf=ANCHOR)
    candidates = {"v14": base, "v8_depth": replaced_pool(base, ranking, raw, raw["v8"])}
    for name, bonus in (("ids_nomination", real), ("stratified_shuffle", shuffled)):
        proposal = standardized(raw["v8"]) + IDS_PROPOSAL_WEIGHT * bonus
        candidates[name] = replaced_pool(base, ranking, raw, proposal)
    candidates["ids_nomination_and_score"] = candidates["ids_nomination"]
    selected = {}
    for name, graph in candidates.items():
        score = ranker.rank(graph, anchor_ocf=ANCHOR)
        if name in ("ids_nomination_and_score", "stratified_shuffle"):
            bonus = real if name == "ids_nomination_and_score" else shuffled
            score = score + IDS_OCF_WEIGHT * bonus.gather(1, graph.indices).clamp(-1, 1)
        selected[name] = graph.indices.gather(1, score.argmax(1, keepdim=True))[:, 0]
    return selected, candidates


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Use Kaggle T4 and the official training data with public font and IDS access.")
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    root = locate_data()
    images = load_images(root, device)
    fit, held = split_glyphs(len(images), device)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images))
    training = {day: rows[fit] for day, rows in recordings.items()}
    scorer = official_scorer(root)
    split = split_calibration(len(fit), ORIGINAL_SIZE, NOISE_SIZE, device, MODEL_SEED)
    _, noise, _, codes = nested_ranker(training, images[fit], split, scorer, MODEL_SEED)
    ranker, folds = fit_crossfit_head(training, images[fit], scorer, codes)
    model = fit_v8(training, images[fit], MODEL_SEED)
    if not torch.equal(model.codepoints.cpu(), codes):
        raise ValueError("The ranking heads and IDS table need one common atlas order.")
    bank = rendered_bank(model)
    atlas = visual_atlas(bank)
    code_map, provenance = component_dictionary(codes)
    matched = shape_matches(atlas, images[fit].cpu())
    rows, targets, mislabeled, components, confident = matched_training(matched, code_map, images[fit], device)
    response = {day: projected_counts(model.contour, values[rows], day) for day, values in training.items()}
    learned = learn_component_scores(response, targets)
    shuffled_head = learn_component_scores(response, mislabeled)
    del training, response

    ordering = np.random.default_rng(SEED + 1).permutation(len(held))
    oracle_count = np.sort(ordering[:ORACLE_COUNT])
    truth_matches = shape_matches(atlas, images[held].cpu())
    del atlas
    proxy = truth_matches.closest.to(device)
    proxy_valid = truth_matches.confident(QUALITY, MARGIN).to(device)
    paired = PairedOCF(scorer, bank, (images[held] >= 128).cpu().numpy(), device, 12)
    oracle = PairedOCF(scorer, bank, (images[held[oracle_count]] >= 128).cpu().numpy(), device, 12)
    measured = {name: [] for name in ARMS}
    report = {"mouse": MOUSE, "held_glyphs": len(held), "glyph_seed": SEED, "proposal_slots": ADD,
              "visual_mapping_quality_and_margin": [QUALITY, MARGIN], "fit_confident_public_code_proxy": confident,
              "fit_ids_rows": len(rows), "features_with_fit_support": components.shape[1],
              "public_columns_with_known_components": int((components.sum(1) > 0).sum()),
              "ids_source": provenance, "training_folds": folds, "sessions": {},
              "held_proxy_confident_rows": int(proxy_valid.sum()), "proposal_weight": IDS_PROPOSAL_WEIGHT,
              "ocf_bonus_weight": IDS_OCF_WEIGHT, "oracle_rows_per_session": ORACLE_COUNT,
              "protocol": "Use official training images matched to public glyph codepoints, public IDS as additional "
                          "targets, and fit a spike model for common layout and position-specific components. Replace four "
                          "slots at the same total candidate budget. Deep-lookup and shape-stratified mispairing use the "
                          "same slots. The independent held images are used only to score predictions and diagnostic "
                          "nearest-rendering proxy recall. No formal stimuli are loaded."}
    for day in SESSIONS:
        counts = recordings[day][held]
        query = projected_counts(model.contour, counts, day)
        original = atlas_scores(model, noise[day], counts, day)
        true = ids_scores(learned, query, components)
        fake = ids_scores(shuffled_head, query, components)
        picks, pools = decide(original, true, fake, ranker)
        report["sessions"][day] = {}
        for name in ARMS:
            graph = pools[name]
            values = paired.evaluate(picks[name][:, None])[:, 0]
            measured[name].append(values)
            recall = ((graph.indices == proxy[:, None]) & graph.valid).any(1)
            selected_graph = graph.indices[oracle_count]
            exact = oracle.evaluate(selected_graph)
            eligible = graph.valid[oracle_count].cpu().numpy()
            report["sessions"][day][name] = {
                "ocf": summarize([values], [measured["v14"][-1]]),
                "chosen_different": int((picks[name] != picks["v14"]).sum()),
                "nearest_rendering_proxy_recall": float(recall[proxy_valid].float().mean()) if bool(proxy_valid.any()) else None,
                "oracle_ocf": float(np.where(eligible, exact, -np.inf).max(axis=1).mean()),
            }
        report["pooled"] = {name: summarize(values, measured["v14"]) for name, values in measured.items()}
        report["ids_vs_deep_v8"] = summarize(measured["ids_nomination"], measured["v8_depth"])
        report["ids_vs_shuffle"] = summarize(measured["ids_nomination_and_score"], measured["stratified_shuffle"])
        report["seconds"] = time.monotonic() - started
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": day, "results": report["sessions"][day]}), flush=True)


if __name__ == "__main__":
    main()
