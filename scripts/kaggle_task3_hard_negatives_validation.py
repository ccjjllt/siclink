"""Compare contour-matched and broad contrastive negatives on held Task 3 repeats."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from glyph_contour_retrieval import (_response_features, blend_contour_poisson, contour_retrieval_scores,
                                     fit_contour_retrieval)
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data, split_repeats
from glyph_encoding import count_template_logits, fit_glyph_encoder
from glyph_gain_likelihood import fit_gain_model, gain_template_logits
from glyph_mbr import contour_signatures
from glyph_retrieval import _standardize
from kaggle_diagnostics import official_scorer
from kaggle_task3_metric_validation import unique_pair_ocf


class ContrastiveScorer(nn.Module):
    def __init__(self, neural_size, image_size):
        super().__init__()
        self.response = nn.Sequential(nn.Linear(neural_size, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 64))
        self.image = nn.Sequential(nn.Linear(image_size, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 64))

    def query_vectors(self, spikes):
        return F.normalize(self.response(spikes), dim=-1)

    def gallery_vectors(self, shape):
        return F.normalize(self.image(shape), dim=-1)


@torch.no_grad()
def prepare_gallery(signatures, energy, masks, seed):
    device = masks.device
    torch.manual_seed(seed)
    centered = signatures - signatures.mean(dim=0)
    _, _, basis = torch.pca_lowrank(centered, q=160, center=False, niter=3)
    shape = F.normalize(centered @ basis, dim=1)
    log_ink = masks.float().mean(dim=(1, 2)).clamp_min(1 / 4096).log()
    metadata = torch.stack((log_ink, energy.log()), dim=1)
    scaled = metadata / metadata.std(dim=0).clamp_min(1e-6)
    difference = torch.cdist(scaled, scaled).fill_diagonal_(torch.inf)
    within_density = difference.topk(192, largest=False).indices
    cosine = shape @ shape.T
    close = cosine.gather(1, within_density).topk(48).indices
    matched = within_density.gather(1, close)
    sampler = torch.Generator(device=device).manual_seed(seed + 1)
    draws = torch.rand((len(shape), len(shape)), generator=sampler, device=device).fill_diagonal_(-1)
    draws.scatter_(1, matched, -1)
    broad = draws.topk(72).indices

    def describe(pool):
        mass_delta = (metadata[pool] - metadata[:, None]).abs()
        typical = mass_delta.flatten(0, 1).median(dim=0).values
        return {"pool_size": pool.shape[1], "median_absolute_log_ink_gap": float(typical[0]),
                "median_absolute_log_energy_gap": float(typical[1]),
                "average_projected_contour_cosine": float(cosine.gather(1, pool).mean())}

    return shape, matched, broad, {"hard": describe(matched), "broad": describe(broad)}


def response_features(contour, counts, session):
    response = _response_features(counts, contour.view, contour.n_bins)
    return _standardize(response, *contour.moments[session]) @ contour.projection


def fit_contrastive(features, labels, shapes, matched, broad, args, use_hard):
    device = features.device
    torch.manual_seed(args.learner_seed)
    rows_rng = torch.Generator(device=device).manual_seed(args.learner_seed + 1)
    sample_rng = torch.Generator(device=device).manual_seed(args.learner_seed + 2)
    model = ContrastiveScorer(features.shape[1], shapes.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)
    positive_column = torch.zeros(args.batch_size, dtype=torch.long, device=device)
    for epoch in range(args.epochs):
        order = torch.randperm(len(labels), generator=rows_rng, device=device)
        cumulative = features.new_zeros(())
        for rows in order.split(args.batch_size):
            target = labels[rows]
            broad_order = torch.rand((len(rows), broad.shape[1]), generator=sample_rng, device=device).argsort(dim=1)
            hard_order = torch.rand((len(rows), matched.shape[1]), generator=sample_rng, device=device).argsort(dim=1)
            general = broad[target].gather(1, broad_order[:, :36])
            hard = matched[target].gather(1, hard_order[:, :24])
            negatives = torch.cat((general[:, :12], hard), dim=1) if use_hard else general
            candidates = torch.cat((target[:, None], negatives), dim=1)
            optimizer.zero_grad(set_to_none=True)
            query = model.query_vectors(features[rows])
            gallery = model.gallery_vectors(shapes)
            scores = (query[:, None] * gallery[candidates]).sum(-1) / 0.12
            loss = F.cross_entropy(scores, positive_column[:len(rows)])
            loss.backward()
            optimizer.step()
            cumulative += loss.detach() * len(rows)
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 4 == 0 or epoch + 1 == args.epochs:
            print(json.dumps({"learner": "matched" if use_hard else "broad", "epoch": epoch + 1,
                              "train_loss": float(cumulative / len(labels))}), flush=True)
    return model.eval()


def ranks(scores, matched):
    correct = scores.diagonal()[:, None]
    overall = 1 + (scores > correct).sum(dim=1)
    local = 1 + (scores.gather(1, matched) > correct).sum(dim=1)
    return {"rows": len(scores), "top1": float((overall == 1).float().mean()),
            "top5": float((overall <= 5).float().mean()), "mrr": float(overall.float().reciprocal().mean()),
            "matched_48_top1": float((local == 1).float().mean()),
            "matched_48_mrr": float(local.float().reciprocal().mean())}


def paired_score(values, baseline):
    delta = values - baseline
    by_glyph = delta.mean(axis=0) if delta.ndim == 2 else delta
    return {"ocf": float(values.mean()), "rows": values.size, "paired_delta": float(delta.mean()),
            "paired_glyph_se": float(by_glyph.std(ddof=1) / np.sqrt(len(by_glyph)))}


def row_zscore(scores):
    return F.layer_norm(scores, (scores.shape[1],), eps=1e-10)


@torch.no_grad()
def evaluate(session, query, selected, mouse, contour, encoder, gain, models, gallery, bank, images, masks, scorer, weight):
    shape = contour_retrieval_scores(contour, query, session)["contour_ridge"]
    rates = encoder.expected_counts(images, session)
    likelihood = count_template_logits(query, rates, "timed") if mouse == "mice1" else gain_template_logits(query, rates, gain)
    v11 = blend_contour_poisson(shape, likelihood)
    features = response_features(contour, query, session).squeeze(1)
    broad = models[0].query_vectors(features) @ gallery[0].T
    hard = models[1].query_vectors(features) @ gallery[1].T
    anchored = (1 - weight) * row_zscore(v11)
    methods = {"v11": v11, "broad": broad, "hard": hard,
               "v11_broad": anchored + weight * row_zscore(broad), "v11_hard": anchored + weight * row_zscore(hard)}
    picked = torch.stack([score[selected].argmax(dim=1) for score in methods.values()])
    values, n_pairs = unique_pair_ocf(picked, selected.expand_as(picked), masks, scorer, masks.device, 384)
    exact = dict(zip(methods, values.cpu().numpy()))
    report = {"ranks": {name: ranks(scores, bank) for name, scores in methods.items()},
              "unique_ocf_pairs": n_pairs, "metrics": {name: paired_score(vals, exact["v11"]) for name, vals in exact.items()},
              "hard_fusion_vs_broad_fusion": paired_score(exact["v11_hard"], exact["v11_broad"])}
    return report, exact


def run_mouse(mouse, root, catalog, images, masks, signatures, energy, shape, matched, broad, scorer, selected, args):
    started = time.monotonic()
    raw = grouped_blocks(root, catalog, mouse, images.device, len(images))
    training, held = {}, {}
    for index, (session, responses) in enumerate(raw.items()):
        training[session], held[session] = split_repeats(responses, args.repeat_seed + index)
    del raw
    print(f"{mouse}: fitting v11 Task 3 baseline on three labeled repeats", flush=True)
    contour = fit_contour_retrieval(training, signatures, energy, task="task3", seed=211)
    encoder = fit_glyph_encoder(training, images, epochs=18, seed=211)
    gain = fit_gain_model(training, "timed", temporal=True) if mouse == "mice2" else None
    with torch.no_grad():
        by_session = [response_features(contour, trials, session) for session, trials in training.items()]
        repeats = by_session[0].shape[1]
        labels = torch.arange(len(images), device=images.device).repeat_interleave(repeats).repeat(len(by_session))
        rows = torch.cat([features.flatten(0, 1) for features in by_session])
    del training, by_session
    models = tuple(fit_contrastive(rows, labels, shape, matched, broad, args, use_hard=sample_hard)
                   for sample_hard in (False, True))
    with torch.no_grad():
        gallery = tuple(model.gallery_vectors(shape) for model in models)
    per_session, measurements = {}, {}
    for session, query in held.items():
        result, values = evaluate(session, query, selected, mouse, contour, encoder, gain, models, gallery,
                                  matched, images, masks, scorer, args.fusion)
        per_session[session] = result
        for method, value in values.items():
            measurements.setdefault(method, []).append(value)
        print(json.dumps({"mouse": mouse, "session": session, **result}), flush=True)
    measurements = {key: np.stack(value) for key, value in measurements.items()}
    summary = {name: paired_score(vals, measurements["v11"]) for name, vals in measurements.items()}
    average_ranks = {method: {metric: float(np.mean([report["ranks"][method][metric] for report in per_session.values()]))
                             for metric in ("top1", "top5", "mrr", "matched_48_top1", "matched_48_mrr")}
                     for method in summary}
    return {"sessions": per_session, "metrics": summary, "ranks": average_ranks,
            "hard_fusion_vs_broad_fusion": paired_score(measurements["v11_hard"], measurements["v11_broad"]),
            "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", choices=("mice1", "mice2"), default=("mice1", "mice2"))
    parser.add_argument("--ocf-per-session", type=int, default=128)
    parser.add_argument("--repeat-seed", type=int, default=419)
    parser.add_argument("--learner-seed", type=int, default=1613)
    parser.add_argument("--gallery-seed", type=int, default=5723)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fusion", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_hard_negatives_validation.json"))
    args = parser.parse_args()
    if args.ocf_per_session < 96 or min(args.epochs, args.batch_size) < 1 or not 0 < args.fusion < 1:
        parser.error("Need at least 96 OCF samples per recording and positive learner settings")
    if not torch.cuda.is_available():
        raise RuntimeError("Run the full-gallery experiment on Kaggle GPU")
    torch.set_num_threads(4)
    started = time.monotonic()
    root = locate_data()
    catalog = load_catalog(root)
    images = load_images(root, torch.device("cuda:0"))
    masks = images >= 128
    signatures, energy = contour_signatures(images)
    shape, matched, broad, pools = prepare_gallery(signatures, energy, masks, args.gallery_seed)
    rng = np.random.default_rng(args.gallery_seed)
    selected = torch.as_tensor(rng.choice(len(images), args.ocf_per_session, replace=False), device=images.device)
    scorer = official_scorer(root)
    output = {"baseline": {"mice1": "v11 equal contour + timed Poisson", "mice2": "v11 equal contour + temporal gain"},
              "pool_diagnostics": pools, "negatives_per_example": {"broad_control": 36, "hard": 24, "shared_broad": 12},
              "repeat_seed": args.repeat_seed, "learner_seed": args.learner_seed, "gallery_seed": args.gallery_seed,
              "ocf_per_session": args.ocf_per_session, "epochs": args.epochs, "batch_size": args.batch_size,
              "fusion": args.fusion, "mice": {}}
    print(json.dumps({"pool_diagnostics": pools}), flush=True)
    for mouse in args.mice:
        output["mice"][mouse] = run_mouse(mouse, root, catalog, images, masks, signatures, energy, shape,
                                         matched, broad, scorer, selected, args)
        output["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} in {output['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
