"""Test split-repeat proposal coverage with an unchanged full-count Task 1 ranker."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_atlas_metric_ranker import _deduplicate, describe_atlas_candidates, pack_atlas_candidates
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from kaggle_atlas_depth_diagnostics import PairedOCF
from kaggle_diagnostics import official_scorer
from kaggle_exact_rerank_validation import split_calibration
from kaggle_mice2_v9_metric_validation import outer_split
from kaggle_v8_metric_validation import NOMINATIONS, SESSIONS, atlas_scores, fit_v8, nested_ranker, rendered_bank, summarize


MODEL_SEED = 211
ANCHOR_OCF = 4.0
PREPOOL = 16
ADD_COUNT = 4
DEEPER_PER_SOURCE = 8
EARLIER_OUTER_SEEDS = (961, 2209)
DISAGREEMENT_WEIGHT = 0.75
DISAGREEMENT_CLAMP = 4.0


def twos(counts, seed):
    # Different random trial positions are chosen independently for each held glyph.
    order = torch.as_tensor(np.random.default_rng(seed).random(counts.shape[:2]).argsort(axis=1),
                            device=counts.device)
    arranged = counts.gather(1, order[:, :, None].expand_as(counts))
    return arranged[:, :2], arranged[:, 2:]


def sorted_proposals(scores, candidates):
    order = scores.argsort(dim=1, descending=True, stable=True)
    return candidates.indices.gather(1, order).masked_fill(~candidates.valid.gather(1, order), candidates.atlas_size)


def novel_prepool(source, base):
    already_present = ((source[:, :, None] == base.indices[:, None, :]) & base.valid[:, None, :]).any(dim=-1)
    novel = source.masked_fill(already_present, base.atlas_size)
    indices, valid = _deduplicate(novel, base.atlas_size)
    return indices[:, :PREPOOL], valid[:, :PREPOOL]


def deeper_prepool(scores, base):
    deeper_order = []
    for name, quota in NOMINATIONS.items():
        values, indices = scores[name].topk(base.rank_depth, dim=1)
        indices = indices.masked_fill(~torch.isfinite(values), base.atlas_size)
        deeper_order.append(indices[:, quota:quota + DEEPER_PER_SOURCE])
    return novel_prepool(torch.stack(deeper_order, dim=-1).flatten(1), base)


def extra_scores(indices, valid, scores, base, ranker):
    described = describe_atlas_candidates(scores, NOMINATIONS, base.baseline, base.rank_depth, indices, valid)
    ranked = ranker.rank(described, anchor_ocf=ANCHOR_OCF)
    chosen = ranked.argsort(dim=1, descending=True, stable=True)[:, :ADD_COUNT]
    return indices.gather(1, chosen), valid.gather(1, chosen), ranked.gather(1, chosen)


def expanded_choice(base, frozen_scores, indices, values):
    full = torch.cat((frozen_scores, values), dim=1)
    expanded = torch.cat((base.indices, indices), dim=1)
    return expanded.gather(1, full.argmax(dim=1, keepdim=True))[:, 0]


@torch.no_grad()
def choose_variants(scores, halves, ranker):
    base = pack_atlas_candidates(scores, NOMINATIONS, baseline="v8")
    base_values = ranker.rank(base, anchor_ocf=ANCHOR_OCF)
    baseline = base.indices.gather(1, base_values.argmax(dim=1, keepdim=True))[:, 0]

    ranked_halves = []
    for maps in halves:
        candidates = pack_atlas_candidates(maps, NOMINATIONS, baseline="v8")
        values = ranker.rank(candidates, anchor_ocf=ANCHOR_OCF)
        ranked_halves.append(sorted_proposals(values, candidates))
    half_order = torch.stack(ranked_halves, dim=-1).flatten(1)

    half_indices, half_valid = novel_prepool(half_order, base)
    deeper_indices, deeper_valid = deeper_prepool(scores, base)
    comparable = torch.minimum(half_valid.sum(dim=1), deeper_valid.sum(dim=1))
    prepool_valid = torch.arange(PREPOOL, device=comparable.device) < comparable[:, None]
    half = extra_scores(half_indices, prepool_valid, scores, base, ranker)
    deeper = extra_scores(deeper_indices, prepool_valid, scores, base, ranker)
    if not torch.equal(half[1].sum(dim=1), deeper[1].sum(dim=1)):
        raise ValueError("The two expanded policies must use the same additional candidate budget per glyph.")

    # Apply this fixed contrast only to the separate original-pool sensitivity arm.
    a, b = (maps["v8"].gather(1, base.indices) for maps in halves)
    disagreement = (a - b).abs().clamp_max(DISAGREEMENT_CLAMP)
    stable = (base_values - DISAGREEMENT_WEIGHT * disagreement).argmax(dim=1, keepdim=True)
    picks = {"v11": baseline,
             "half_additions": expanded_choice(base, base_values, half[0], half[2]),
             "deeper_full_additions": expanded_choice(base, base_values, deeper[0], deeper[2]),
             "half_disagreement_penalty": base.indices.gather(1, stable)[:, 0]}
    counts = {"half_novel_available_in_16": half_valid.sum(dim=1),
              "deeper_novel_available_in_16": deeper_valid.sum(dim=1), "matched_prepool": comparable,
              "added_to_each_arm": half[1].sum(dim=1)}
    return base, picks, {"half_additions": half[:2], "deeper_full_additions": deeper[:2]}, counts


def oracle(exact, base, additions, size):
    masks = {"v11": (base.indices, base.valid), **additions}
    results = {}
    for name, (ids, valid) in masks.items():
        values = exact.evaluate(ids[:size])
        results[name] = np.where(valid[:size].cpu().numpy(), values, -np.inf).max(axis=1)
    original = results["v11"]
    return {name: np.maximum(original, values) for name, values in results.items()}


def summary(values):
    result = {name: summarize(series, values["v11"]) for name, series in values.items()}
    delta = summarize(values["half_additions"], values["deeper_full_additions"])
    result["half_additions"].update(delta_vs_matched_deeper_full=delta["paired_delta"],
                                    paired_se_vs_matched_deeper_full=delta["paired_se"])
    return result


def count_metrics(counts):
    return {name: {"total": int(values.sum()), "mean": float(values.float().mean()),
                   "histogram": values.bincount(minlength=(ADD_COUNT if name == "added_to_each_arm" else PREPOOL) + 1).tolist()}
            for name, values in counts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-seed", type=int, default=3931)
    parser.add_argument("--half-seed", type=int, default=7517)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--oracle-images", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task1_half_proposal_validation.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the Task 1 labeled half-proposal validation on Kaggle T4.")
    if not 2 <= args.oracle_images <= args.ocf_images <= 320 or args.outer_seed in EARLIER_OUTER_SEEDS:
        parser.error("Use a fresh split seed and valid paired OCF sample sizes.")

    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    device = torch.device("cuda:0")
    images = load_images(root, device)
    fit, held = outer_split(len(images), args.outer_seed, EARLIER_OUTER_SEEDS, device)
    selected = np.random.default_rng(args.outer_seed + 1).choice(len(held), args.ocf_images, replace=False)
    scored = held[torch.as_tensor(selected, device=device)]
    fit_images = images[fit]
    blocks = grouped_blocks(root, load_catalog(root), "mice1", device, len(images))
    training = {name: counts[fit] for name, counts in blocks.items()}
    query = {name: counts[scored] for name, counts in blocks.items()}
    del blocks
    if any(counts.shape[1] != 4 for counts in query.values()):
        raise ValueError("The proxy requires four measured query repeats per unseen glyph.")
    split = split_calibration(len(fit), 96, 384, device, MODEL_SEED)
    scorer = official_scorer(root)
    print("Fit the exact four-session v11 ranker using disjoint nested labeled data", flush=True)
    ranker, likelihood, nested, codepoints = nested_ranker(training, fit_images, split, scorer, MODEL_SEED)
    model = fit_v8(training, fit_images, MODEL_SEED)
    del training
    if not torch.equal(codepoints, model.codepoints.cpu()):
        raise ValueError("Use the same public atlas order in nested and outer models.")
    exact = PairedOCF(scorer, rendered_bank(model), (images[scored] >= 128).cpu().numpy(), device, 12)
    output = {"mouse": "mice1", "model_seed": MODEL_SEED, "outer_seed": args.outer_seed, "half_seed": args.half_seed,
              "excluded_prior_holdout_seeds": EARLIER_OUTER_SEEDS, "fit_glyphs": len(fit), "held_glyphs": len(held),
              "paired_glyphs_per_session": len(scored), "oracle_glyphs_per_session": args.oracle_images,
              "query_repeats": 4, "anchor_ocf": ANCHOR_OCF, "original_nominations": NOMINATIONS,
              "matched_prepool_cap": PREPOOL, "added_candidate_cap": ADD_COUNT, "deeper_per_full_source": DEEPER_PER_SOURCE,
              "ranker": ranker.diagnostics, "outer_fit": model.report, "nested_fit": nested,
              "disagreement_penalty": {"weight": DISAGREEMENT_WEIGHT, "clamp": DISAGREEMENT_CLAMP}, "sessions": {},
              "protocol": "Split all outer evaluation glyphs away before atlas, encoder, noise and ranker fitting. "
                          "Use all four repeats for the v11 scores and original candidates. Add exactly the same "
                          "per-row number of extra proposals from matched prepools, scoring each by the original "
                          "full-count feature schema and frozen ranker. A fixed prefix of the randomly sampled "
                          "evaluation glyphs measures retrospective best-of coverage. No competition test records enter."}
    measured, best_of = {}, {}
    for session in SESSIONS:
        counts = query[session]
        a, b = twos(counts, args.half_seed + SESSIONS.index(session))
        with torch.autocast(device_type="cuda"):
            rates = model.encoder.expected_counts(model.atlas.images, session, batch_size=192).float()
        full = atlas_scores(model, likelihood[session], counts, session, rates=rates)
        halves = tuple(atlas_scores(model, likelihood[session], spikes, session, rates=rates) for spikes in (a, b))
        base, chosen, extras, sizes = choose_variants(full, halves, ranker)
        values = {name: exact.evaluate(indices[:, None])[:, 0] for name, indices in chosen.items()}
        diagnostic = oracle(exact, base, extras, args.oracle_images)
        for name, item in values.items():
            measured.setdefault(name, []).append(item)
        for name, item in diagnostic.items():
            best_of.setdefault(name, []).append(item)
        observation = summary({name: [item] for name, item in values.items()})
        for name in observation:
            observation[name]["changed_vs_v11"] = int((chosen[name] != chosen["v11"]).sum())
        output["sessions"][session] = {"exact_ocf": observation,
                                       "oracle_best_of": summary({name: [item] for name, item in diagnostic.items()}),
                                       "extra_counts": count_metrics(sizes)}
        output["summary"] = summary(measured)
        output["oracle_summary"] = summary(best_of)
        output["unique_exact_pairs"] = len(exact.values)
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"session": session, **output["sessions"][session]}), flush=True)
        del full, halves
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
