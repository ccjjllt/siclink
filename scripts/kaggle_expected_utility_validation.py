"""Compare dense contour-utility regression with exact v7 Task 3 choices."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from glyph_contour_retrieval import blend_contour_poisson, contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_expected_utility import UtilityBasis, expected_utility_scores, fit_utility_basis, fit_utility_readout
from glyph_gain_likelihood import fit_gain_model, gain_template_logits
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_encoding_validation import seen_folds
from kaggle_task3_metric_validation import unique_pair_ocf


def _cluster_se(delta, glyphs):
    _, groups = np.unique(glyphs, return_inverse=True)
    totals = np.bincount(groups, weights=delta - delta.mean())
    if len(totals) < 2:
        return None
    return float(np.sqrt(len(totals) / (len(totals) - 1) * (totals**2).sum()) / len(delta))


def evaluate(utility, contour, encoder, gain, queries, images, score, device, args):
    choices, selected_glyphs, hits = {}, [], {}
    random = np.random.default_rng(args.seed)
    n_per_session = min(len(images), args.ocf_images // len(queries))
    labels = torch.arange(len(images), device=device)
    for session, observed in queries.items():
        selection = torch.as_tensor(random.choice(len(images), n_per_session, replace=False), device=device)
        selected_glyphs.append(selection)
        rates = encoder.expected_counts(images, session)
        shape = contour_retrieval_scores(contour, observed, session)["contour_ridge"]
        likelihood = (count_template_logits(observed, rates, "timed") if gain is None
                      else gain_template_logits(observed, rates, gain))
        v7 = blend_contour_poisson(shape, likelihood)
        expected = expected_utility_scores(utility, observed, session)
        gallery_size = (len(images),)
        anchor = (1 - args.blend_weight) * F.layer_norm(v7, gallery_size, eps=1e-12)
        shape_blend = anchor + args.blend_weight * F.layer_norm(shape, gallery_size, eps=1e-12)
        utility_blend = anchor + args.blend_weight * F.layer_norm(expected, gallery_size, eps=1e-12)
        rankings = {"v7": v7, "regular_contour": shape, "dense_utility": expected,
                    "v7_regular_blend": shape_blend, "v7_utility_blend": utility_blend,
                    "static_utility": utility.basis.unconditional[None].expand_as(v7)}
        for name, logits in rankings.items():
            ranked = logits.topk(5, dim=1).indices
            winner = logits.argmax(dim=1)
            top1, top5 = hits.get(name, (0, 0))
            hits[name] = (top1 + int((winner == labels).sum()),
                          top5 + int((ranked == labels[:, None]).any(dim=1).sum()))
            choices.setdefault(name, []).append(winner[selection])

    picked = torch.stack([torch.cat(blocks) for blocks in choices.values()])
    reference = torch.cat(selected_glyphs)
    measurements, unique = unique_pair_ocf(picked, reference.expand_as(picked), images >= 128, score, device, 384)
    scored = dict(zip(choices, measurements.cpu().numpy()))
    glyphs = reference.cpu().numpy()
    n = len(queries) * len(images)
    rows = []
    for index, (name, values) in enumerate(scored.items()):
        delta = values - scored["v7"]
        by_session = delta.reshape(len(queries), n_per_session).mean(axis=1)
        row = {"method": name, "ocf": float(values.mean()), "ocf_n": len(values),
               "paired_delta_vs_v7": float(delta.mean()), "paired_glyph_se_vs_v7": _cluster_se(delta, glyphs),
               "paired_delta_by_session": dict(zip(queries, by_session.tolist())),
               "changed_fraction": float((picked[index] != picked[0]).float().mean()),
               "top1": hits[name][0] / n, "top5": hits[name][1] / n, "top_n": n}
        if name == "v7_utility_blend":
            matched = values - scored["v7_regular_blend"]
            row.update(paired_delta_vs_matched_blend=float(matched.mean()),
                       paired_glyph_se_vs_matched_blend=_cluster_se(matched, glyphs))
        rows.append(row)
    return rows, unique


def run_mouse(mouse, device, root, catalog, score, gallery, args):
    started = time.monotonic()
    images, signature, energy, basis = gallery
    raw = grouped_blocks(root, catalog, mouse, device, len(images))
    training, queries = seen_folds(raw, args.repeat_seed, ("task3",))["task3"]
    del raw
    print(f"{mouse} on {device}: fitting outer-repeat controls", flush=True)
    utility = fit_utility_readout(training, basis)
    contour = fit_contour_retrieval(training, signature, energy, task="task3", seed=args.seed)
    encoder = fit_glyph_encoder(training, images, epochs=args.encoder_epochs, seed=args.seed)
    gain = fit_gain_model(training, "timed", temporal=True) if mouse == "mice2" else None
    metrics, unique = evaluate(utility, contour, encoder, gain, queries, images, score, device, args)
    for row in metrics:
        print(json.dumps({"mouse": mouse, **row}), flush=True)
    return {"mouse": mouse, "device": str(device), "sessions": list(training),
            "v7_policy": "contour_temporal_gain" if gain is not None else "contour_poisson",
            "basis": basis.diagnostics, "unique_exact_ocf_pairs": unique, "metrics": metrics,
            "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--ocf-images", type=int, default=512)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int, default=419)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--blend-weight", type=float, default=0.35)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/expected_utility_validation.json"))
    args = parser.parse_args()
    if args.ocf_images < 4 or min(args.rank, args.encoder_epochs) < 1 or not 0 <= args.blend_weight <= 1:
        parser.error("Choose valid image count, rank, epoch count, and score blend weight")

    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [torch.device("cpu")]
    torch.set_num_threads(4 if devices[0].type == "cpu" else 2)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    score = official_scorer(root)
    # The labeled image bank is shared by the mice. Compute its eigenspace
    # once and initialize CUDA linalg before starting concurrent fitting.
    images = load_images(root, devices[0])
    signature, energy = contour_signatures(images)
    basis = fit_utility_basis(signature, energy, rank=args.rank)
    galleries = {device: (images.to(device), signature.to(device), energy.to(device),
                          UtilityBasis(basis.targets.to(device), basis.actions.to(device),
                                       basis.unconditional.to(device), basis.diagnostics))
                 for device in devices[:len(args.mice)]}
    output = {"seed": args.seed, "repeat_seed": args.repeat_seed, "ocf_images": args.ocf_images,
              "rank": args.rank, "encoder_epochs": args.encoder_epochs, "blend_weight": args.blend_weight,
              "results": []}
    with ThreadPoolExecutor(max_workers=min(len(args.mice), len(devices))) as pool:
        jobs = [pool.submit(run_mouse, mouse, devices[index % len(devices)], root, catalog, score,
                            galleries[devices[index % len(devices)]], args)
                for index, mouse in enumerate(args.mice)]
        for job in as_completed(jobs):
            output["results"].append(job.result())
            output["results"].sort(key=lambda result: result["mouse"])
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
