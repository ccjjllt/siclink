"""Evaluate paired-repeat spike denoising against the v6 Task 3 policy."""

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
from glyph_mbr import contour_signatures
from glyph_paired_denoise import fit_paired_denoiser, paired_glyph_scores
from kaggle_diagnostics import official_scorer
from kaggle_encoding_validation import seen_folds


def _row_z(scores):
    centered = scores - scores.mean(dim=1, keepdim=True)
    return centered / centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)


def evaluate(model, contour, poisson, queries, images, score, device, args):
    labels = torch.arange(len(images), device=device)
    masks = (images >= 128).cpu().numpy()
    random = np.random.default_rng(args.seed)
    n_per_session = min(len(images), args.ocf_images // len(queries))
    references, predictions, top1, top5 = [], {}, {}, {}

    for session, observed in queries.items():
        selection = random.choice(len(images), n_per_session, replace=False)
        selected = torch.as_tensor(selection, device=device)
        references.append(masks[selection])
        v6 = contour_retrieval_scores(contour, observed, session)["contour_ridge"]
        if poisson is not None:
            rates = poisson.for_recording(session, observed)["forward"]
            v6 = blend_contour_poisson(v6, count_template_logits(observed, rates, "timed"))

        latent = paired_glyph_scores(model, observed, session)
        blend = 0.65 * _row_z(v6) + 0.35 * _row_z(latent["paired_mlp"])
        for name, logits in {"v6": v6, **latent, "v6_paired_blend": blend}.items():
            top = logits.topk(5, dim=1).indices
            hits = top == labels[:, None]
            top1[name] = top1.get(name, 0) + int(hits[:, 0].sum())
            top5[name] = top5.get(name, 0) + int(hits.any(dim=1).sum())
            predictions.setdefault(name, []).append(masks[top[selected, 0].cpu().numpy()])

    reference = np.concatenate(references)
    scored = {name: np.asarray(score(np.concatenate(blocks), reference, device=str(device), batch_size=4))
              for name, blocks in predictions.items()}
    n = len(queries) * len(images)
    metrics = []
    for name, values in scored.items():
        delta = values - scored["v6"]
        per_session = delta.reshape(len(queries), n_per_session).mean(axis=1)
        metrics.append({"method": name, "ocf": float(values.mean()), "ocf_n": len(values),
                        "paired_delta_vs_v6": float(delta.mean()),
                        "paired_se_vs_v6": float(delta.std(ddof=1) / len(delta)**0.5),
                        "paired_delta_by_session": dict(zip(queries, per_session.tolist())),
                        "top1": top1[name] / n, "top5": top5[name] / n, "top_n": n})
    return metrics


def run_mouse(mouse, device, root, catalog, score, args):
    started = time.monotonic()
    images = load_images(root, device)
    raw = grouped_blocks(root, catalog, mouse, device, len(images))
    repeat_seed = args.seed if args.repeat_seed is None else args.repeat_seed
    training, queries = seen_folds(raw, repeat_seed, ("task3",))["task3"]
    del raw
    print(f"{mouse} on {device}: fitting independent repeat pairs", flush=True)
    model = fit_paired_denoiser(training, epochs=args.epochs, seed=args.seed)
    contour = fit_contour_retrieval(training, *contour_signatures(images), task="task3", seed=args.seed)
    poisson = None
    if mouse == "mice1":
        encoder = fit_glyph_encoder(training, images, epochs=args.encoder_epochs, seed=args.seed)
        poisson = make_count_templates(encoder, images, training)

    metrics = evaluate(model, contour, poisson, queries, images, score, device, args)
    for row in metrics:
        print(json.dumps({"mouse": mouse, **row}), flush=True)
    return {"mouse": mouse, "device": str(device), "sessions": list(training),
            "v6_policy": "contour_poisson" if poisson is not None else "contour_ridge",
            "denoising_fit": model.diagnostics, "metrics": metrics, "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ocf-images", type=int, default=384)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--repeat-seed", type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--encoder-epochs", type=int, default=18)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/paired_denoise_validation.json"))
    args = parser.parse_args()
    if args.ocf_images < 4 or min(args.epochs, args.encoder_epochs) < 1:
        parser.error("Use at least four paired OCF images and positive epoch counts")

    devices = [] if args.device == "cpu" else [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if args.device == "cuda" and not devices:
        parser.error("The requested CUDA device is unavailable")
    devices = devices or [torch.device("cpu")]
    torch.set_num_threads(4 if devices[0].type == "cpu" else 2)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    score = official_scorer(root)
    output = {"seed": args.seed, "repeat_seed": args.seed if args.repeat_seed is None else args.repeat_seed,
              "epochs": args.epochs, "encoder_epochs": args.encoder_epochs, "ocf_images": args.ocf_images,
              "fit_target_weights": {"same_session": 0.60, "other_sessions": 0.40},
              "score_weights": {"v6": 0.65, "paired_mlp": 0.35}, "results": []}
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
