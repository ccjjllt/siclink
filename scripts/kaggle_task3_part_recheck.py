"""Measure novel local-part nominations against a refit of the v11 Task 3 policy.

Fit on three labeled repeats per session and reserve one real repeat per glyph.
Every compositional pool retains the first twelve v11 candidates and adds four
glyphs beyond the first sixteen. Compare with the first sixteen v11 candidates
and with a stratified shuffle. All choices precede use of held correspondence.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_contour_retrieval import blend_contour_poisson, contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_gain_likelihood import fit_gain_model, gain_template_logits
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_part_discovery import (SESSIONS, fit_codewords, fit_part_readouts, part_logits, patch_descriptors,
                                        permute_part_codes, standardize_rows)


WEIGHTS = (0.15, 0.25)
POOL_SIZE = 16
RETAIN = 12
CODEBOOK_SIZE = 32


@torch.no_grad()
def score_v11(mouse, contour, encoder, gain, query, images, session):
    direction = contour_retrieval_scores(contour, query, session)["contour_ridge"]
    rates = encoder.expected_counts(images, session)
    likelihood = (count_template_logits(query, rates, "timed") if mouse == "mice1"
                  else gain_template_logits(query, rates, gain))
    return blend_contour_poisson(direction, likelihood)


def nominate(v11, true_parts, shuffled_parts):
    base = v11.topk(POOL_SIZE, dim=1).indices
    score = standardize_rows(v11)
    part = standardize_rows(true_parts)
    shuffled = standardize_rows(shuffled_parts)
    pools = {"v11": base}
    for name, component in (("part", part), ("shuffled", shuffled)):
        nomination = 0.35 * score + 0.65 * component
        nomination.scatter_(1, base, -torch.inf)
        novel = nomination.topk(POOL_SIZE - RETAIN, dim=1).indices
        pools[name] = torch.cat((base[:, :RETAIN], novel), dim=1)
    decisions = {"v11": base[:, 0]}
    for name, pool, component in (("part", pools["part"], part), ("shuffled", pools["shuffled"], shuffled),
                                  ("within_v11", base, part)):
        for weight in WEIGHTS:
            ranking = ((1 - weight) * score + weight * component).gather(1, pool)
            decisions[f"{name}_{weight:.2f}"] = pool.gather(1, ranking.argmax(dim=1, keepdim=True))[:, 0]
    return pools, decisions


def score_references(pools, decisions, permutation, glyphs, masks, scorer, device, oracle_count):
    rows = permutation.argsort()[glyphs]
    picks = torch.stack([choice[rows] for choice in decisions.values()])
    shortlist = torch.stack([pool[rows[:oracle_count]] for pool in pools.values()])
    chosen_references = glyphs.expand_as(picks)
    oracle_references = glyphs[None, :oracle_count, None].expand_as(shortlist)
    exact, unique_pairs = unique_pair_ocf(torch.cat((picks.flatten(), shortlist.flatten())),
                                         torch.cat((chosen_references.flatten(), oracle_references.flatten())),
                                         masks, scorer, device, 384)
    selected, oracle = exact.split((picks.numel(), shortlist.numel()))
    selected = selected.reshape(picks.shape).cpu().numpy()
    oracle = oracle.reshape(shortlist.shape).max(dim=2).values.cpu().numpy()
    return dict(zip(decisions, selected)), dict(zip(pools, oracle)), unique_pairs


def paired_summary(values, references):
    delta = (values - references).mean(axis=0)
    return {"ocf": float(values.mean()), "delta": float(delta.mean()),
            "paired_glyph_se": float(delta.std(ddof=1) / np.sqrt(len(delta))), "scored_rows": int(values.size)}


def summarize(results):
    values = {name: np.stack(rows) for name, rows in results.items()}
    summary = {name: {"vs_v11": paired_summary(scores, values["v11"])} for name, scores in values.items()}
    for name, record in summary.items():
        if name == "part":
            record["vs_shuffled"] = paired_summary(values[name], values["shuffled"])
        elif name.startswith("part_"):
            weight = name.removeprefix("part_")
            for control in ("shuffled", "within_v11"):
                record[f"vs_{control}"] = paired_summary(values[name], values[f"{control}_{weight}"])
    return summary


def gallery_statistics(pools, decisions, labels):
    base = pools["v11"]
    recall = {}
    for name, pool in pools.items():
        matched = pool == labels[:, None]
        recall[name] = {"recall_top16": float(matched.any(dim=1).float().mean()),
                        "tail_recall": float(matched[:, RETAIN:].any(dim=1).float().mean())}
    choices = {name: {"top1": float((choice == labels).float().mean()),
                      "changed": float((choice != decisions["v11"]).float().mean()),
                      "outside_v11_top16": float((choice[:, None] != base).all(dim=1).float().mean())}
               for name, choice in decisions.items()}
    return {"shortlists": recall, "decisions": choices}


def append_cases(target, observed):
    for name, data in observed.items():
        target.setdefault(name, []).append(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--sessions", nargs="+", choices=SESSIONS, default=SESSIONS)
    parser.add_argument("--ocf-images", type=int, default=128)
    parser.add_argument("--oracle-images", type=int, default=24)
    parser.add_argument("--repeat-seed", type=int, default=887)
    parser.add_argument("--glyph-seed", type=int, default=5551)
    parser.add_argument("--codebook-seed", type=int, default=6143)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_part_discovery_recheck.json"))
    args = parser.parse_args()
    if not 2 <= args.oracle_images <= args.ocf_images or args.encoder_epochs < 1:
        parser.error("Choose a valid exact OCF sample and a positive number of encoder epochs.")
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the repeated exact OCF comparison on Kaggle GPU.")
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, device)
    masks = images >= 128
    if args.ocf_images > len(masks):
        parser.error("The exact OCF sample cannot exceed the known gallery.")
    signature, energy = contour_signatures(images)
    ink = masks.float().mean(dim=(1, 2))
    generator = torch.Generator(device="cpu").manual_seed(args.codebook_seed)
    codes = torch.stack([fit_codewords(region, CODEBOOK_SIZE, generator)
                         for region in patch_descriptors(masks)], dim=1)
    shuffled = permute_part_codes(codes, ink, energy, generator)
    rng = np.random.default_rng(args.glyph_seed)
    glyphs = torch.as_tensor(rng.choice(len(masks), args.ocf_images, replace=False), device=device)
    scorer = official_scorer(root)
    print(f"Prepared {len(masks)} glyphs and four patch dictionaries", flush=True)

    output = {"model_seed": args.seed, "repeat_seed": args.repeat_seed, "glyph_seed": args.glyph_seed,
              "codebook_seed": args.codebook_seed, "encoder_epochs": args.encoder_epochs,
              "codes_per_patch": CODEBOOK_SIZE, "candidate_budget": POOL_SIZE, "retain": RETAIN,
              "part_weights": WEIGHTS, "gallery_size": len(masks), "ocf_images_per_block": len(glyphs),
              "oracle_images_per_block": args.oracle_images, "device": str(device),
              "v11_refit": {"mice1": "contour + timed Poisson", "mice2": "contour + temporal gain"},
              "protocol": "All spike models fit on the three remaining training trials after reserving one per glyph. "
                          "Shuffle held rows; determine all rankings before consulting held glyph correspondence. "
                          "Part pools share the first twelve v11 entries; four nominations are outside v11 top16. "
                          "Within-v11 controls rerank its original top16 using the same real part signal.",
              "folds": [], "per_mouse": {}}
    total_selected, total_oracle = {}, {}
    for mouse in dict.fromkeys(args.mice):
        recordings = grouped_blocks(root, catalog, mouse, device, len(images))
        training, held = {}, {}
        for index, (session, counts) in enumerate(recordings.items()):
            if counts.shape[1] != 4:
                raise ValueError("The Task 3 proxy needs four training repeats per glyph.")
            training[session], held[session] = split_repeats(counts, args.repeat_seed + index)
        del recordings
        print(f"{mouse}: fitting exact v11 and part readouts on three repeats", flush=True)
        torch.manual_seed(args.seed)
        contour = fit_contour_retrieval(training, signature, energy, task="task3", seed=args.seed)
        part = fit_part_readouts(contour, training, codes)
        encoder = fit_glyph_encoder(training, images, epochs=args.encoder_epochs, seed=args.seed)
        gain = fit_gain_model(training, "timed", temporal=True) if mouse == "mice2" else None
        del training
        per_mouse_selected, per_mouse_oracle = {}, {}
        for session in dict.fromkeys(args.sessions):
            fold_started = time.monotonic()
            permutation = torch.as_tensor(rng.permutation(len(masks)), device=device)
            query = held[session][permutation]
            v11 = score_v11(mouse, contour, encoder, gain, query, images, session)
            true, control = part_logits(contour, part, query, session, codes, shuffled)
            pools, decisions = nominate(v11, true, control)
            selected, oracle, pairs = score_references(pools, decisions, permutation, glyphs, masks, scorer, device,
                                                       args.oracle_images)
            fold = {"mouse": mouse, "session": session, "unique_exact_pairs": pairs,
                    "gallery": gallery_statistics(pools, decisions, permutation),
                    "top1_ocf": summarize({name: [scores] for name, scores in selected.items()}),
                    "best_of16_ocf": summarize({name: [scores] for name, scores in oracle.items()}),
                    "seconds": time.monotonic() - fold_started}
            output["folds"].append(fold)
            append_cases(per_mouse_selected, selected)
            append_cases(per_mouse_oracle, oracle)
            append_cases(total_selected, selected)
            append_cases(total_oracle, oracle)
            output["per_mouse"][mouse] = {"top1_ocf": summarize(per_mouse_selected),
                                        "best_of16_ocf": summarize(per_mouse_oracle)}
            output["summary"] = {"top1_ocf": summarize(total_selected), "best_of16_ocf": summarize(total_oracle)}
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(fold), flush=True)
            del query, v11, true, control
        del contour, part, encoder, gain, held
    print(json.dumps({"summary": output["summary"], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
