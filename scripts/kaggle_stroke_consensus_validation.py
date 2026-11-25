"""Score sparse image construction under the unchanged Task 3 v7 rankings."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import torch

from glyph_contour_retrieval import blend_contour_poisson, contour_retrieval_scores, fit_contour_retrieval
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_encoding import count_template_logits, fit_glyph_encoder, make_count_templates
from glyph_gain_likelihood import fit_gain_model, gain_template_logits
from glyph_mbr import contour_signatures
from glyph_stroke_consensus import stroke_consensus_images
from kaggle_diagnostics import official_scorer
from kaggle_encoding_validation import seen_folds


def _cluster_se(delta, groups):
    _, ids = np.unique(groups, return_inverse=True)
    totals = np.bincount(ids, weights=delta - delta.mean())
    count = len(totals)
    if count < 2:
        return None
    return float(np.sqrt(count / (count - 1) * np.sum(totals**2)) / len(delta))


def evaluate(contour, bank, gain, queries, images, score, device, args):
    random = np.random.default_rng(args.seed)
    masks = (images >= 128).cpu().numpy()
    labels = torch.arange(len(images), device=device)
    n_per_session = min(len(images), args.ocf_images // len(queries))
    gold, predictions, held_glyphs = [], {}, []
    hits, total = 0, 0
    for session, observed in queries.items():
        selection = random.choice(len(images), n_per_session, replace=False)
        selected = torch.as_tensor(selection, device=device)
        held_glyphs.append(selection)
        gold.append(masks[selection])
        rates = bank.for_recording(session, observed)["forward"]
        contour_scores = contour_retrieval_scores(contour, observed, session)["contour_ridge"]
        if gain is None:
            likelihood = count_template_logits(observed, rates, "timed")
        else:
            likelihood = gain_template_logits(observed, rates, gain)
        logits = blend_contour_poisson(contour_scores, likelihood)
        hits += int((logits.argmax(dim=1) == labels).sum())
        total += len(labels)
        variants = stroke_consensus_images(logits[selected], images, top_k=args.top_k,
                                           temperature=args.temperature, radius=args.radius,
                                           edit_fraction=args.edit_fraction)
        for name, glyphs in variants.items():
            predictions.setdefault(name, []).append(glyphs.cpu().numpy() >= 128)

    reference = np.concatenate(gold)
    held_glyphs = np.concatenate(held_glyphs)
    rendered = {name: np.concatenate(blocks) for name, blocks in predictions.items()}
    scored = {name: np.asarray(score(masks, reference, device=str(device), batch_size=4))
              for name, masks in rendered.items()}
    original = rendered["v7_hard"]
    rows = []
    for name, values in scored.items():
        delta = values - scored["v7_hard"]
        changed_pixels = (rendered[name] != original).sum(axis=(1, 2))
        per_session = delta.reshape(len(queries), n_per_session).mean(axis=1)
        rows.append({"method": name, "ocf": float(values.mean()), "ocf_n": len(values),
                     "paired_delta": float(delta.mean()), "paired_glyph_cluster_se": _cluster_se(delta, held_glyphs),
                     "paired_delta_by_session": dict(zip(queries, per_session.tolist())),
                     "changed_samples": int((changed_pixels > 0).sum()),
                     "mean_changed_pixels": float(changed_pixels.mean())})
    return {"v7_top1": hits / total, "top_n": total, "metrics": rows}


def run_mouse(mouse, device, root, catalog, score, args):
    started = time.monotonic()
    images = load_images(root, device)
    raw = grouped_blocks(root, catalog, mouse, device, len(images))
    repeat_seed = args.seed if args.repeat_seed is None else args.repeat_seed
    training, queries = seen_folds(raw, repeat_seed, ("task3",))["task3"]
    del raw
    print(f"{mouse} on {device}: fitting v7 ranking", flush=True)
    contour = fit_contour_retrieval(training, *contour_signatures(images), task="task3", seed=args.seed)
    encoder = fit_glyph_encoder(training, images, epochs=args.encoder_epochs, seed=args.seed)
    bank = make_count_templates(encoder, images, training)
    gain = fit_gain_model(training, "timed", temporal=True) if mouse == "mice2" else None
    result = evaluate(contour, bank, gain, queries, images, score, device, args)
    for row in result["metrics"]:
        print(json.dumps({"mouse": mouse, **row}), flush=True)
    return {"mouse": mouse, "device": str(device), "sessions": list(training),
            "v7_policy": "contour_temporal_gain" if gain is not None else "contour_poisson",
            "seconds": time.monotonic() - started, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--ocf-images", type=int, default=384)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--edit-fraction", type=float, default=0.16)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/stroke_consensus_validation.json"))
    args = parser.parse_args()
    if args.ocf_images < 4 or args.encoder_epochs < 1:
        parser.error("Use at least four paired OCF images and positive encoder epochs")

    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [torch.device("cpu")]
    torch.set_num_threads(4 if devices[0].type == "cpu" else 2)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    score = official_scorer(root)
    output = {"seed": args.seed, "repeat_seed": args.seed if args.repeat_seed is None else args.repeat_seed,
              "ocf_images": args.ocf_images, "encoder_epochs": args.encoder_epochs, "top_k": args.top_k,
              "temperature": args.temperature, "radius": args.radius, "edit_fraction": args.edit_fraction,
              "results": []}
    with ThreadPoolExecutor(max_workers=min(len(args.mice), len(devices))) as pool:
        jobs = [pool.submit(run_mouse, mouse, devices[index % len(devices)], root, catalog, score, args)
                for index, mouse in enumerate(args.mice)]
        for job in as_completed(jobs):
            output["results"].append(job.result())
            output["results"].sort(key=lambda result: result["mouse"])
            output["seconds"] = time.monotonic() - started
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
