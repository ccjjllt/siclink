"""Select among the seven v12 Task 3 glyphs using calibrated OCF by posterior rank band."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_exact_utility import fit_exact_utility, gallery_geometry, gallery_utility, pair_features
from glyph_mbr import contour_signatures, fit_temperature, similarity_from_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task3_measured_metric_validation import paired
from kaggle_task3_metric_validation import nested_repeats, unique_pair_ocf
from kaggle_task3_repeat_posterior import fit_models, score_sources


WIDTH = 7
NEAR_EDGES = (0, 7, 32, 128)
ARMS = ("posterior", "contour", "exact")
SECTIONS = ((0, 24), (24, 128), (128, 512), (512, None))


def stage(name, started, **fields):
    print(json.dumps({"stage": name, "seconds": round(time.monotonic() - started, 2), **fields}), flush=True)


@torch.no_grad()
def stratified_pairs(pool, proxy, count, generator):
    nearby = proxy[pool[:, None], pool].clone().fill_diagonal_(-torch.inf)
    ordering = nearby.topk(512, dim=1).indices
    allowed = (count * 40 // 100, count * 30 // 100, count * 15 // 100)
    counts = (*allowed, count - sum(allowed))
    actions, references, strata = [], [], []
    for size, (lower, upper) in zip(counts, SECTIONS):
        q = torch.randint(len(pool), (size,), device=pool.device, generator=generator)
        if upper is None:
            tail = nearby[q].clone().scatter_(1, ordering[q], -torch.inf)
            weights = torch.isfinite(tail).float()
            a = torch.multinomial(weights, 1, generator=generator)[:, 0]
        else:
            index = torch.randint(lower, upper, (size,), device=pool.device, generator=generator)
            a = ordering[q, index]
        actions.append(pool[a])
        references.append(pool[q])
        strata.append(size)
    return torch.cat(actions), torch.cat(references), strata


def fit_visual_metric(masks, geometry, proxy, visual_fit, scorer, draw, args):
    action, reference, counts = stratified_pairs(visual_fit, proxy, args.visual_pairs, draw)
    observed, unique = unique_pair_ocf(action, reference, masks, scorer, masks.device, 384)
    metric = fit_exact_utility(geometry, action, reference, observed, seed=args.seed + 643)
    return metric, counts, unique


def fit_visual_tables(masks, signatures, energy, visual_fit, visual_hold, scorer, args):
    device = masks.device
    geometry = gallery_geometry(masks, signatures, energy)
    proxy = similarity_from_signatures(signatures, energy)
    draw = torch.Generator(device=device).manual_seed(args.seed + 641)
    metric, counts, unique = fit_visual_metric(masks, geometry, proxy, visual_fit, scorer, draw, args)
    a, b, strata = stratified_pairs(visual_hold, proxy, args.visual_check_pairs, draw)
    exact, held_unique = unique_pair_ocf(a, b, masks, scorer, device, 384)

    with torch.no_grad():
        fitted = metric.predict(pair_features(geometry, a, b))
        comparator = 100 * proxy[a, b]
        diagnostics = {"fit_glyphs": len(visual_fit), "held_glyphs": len(visual_hold),
                       "unique_fit_pairs": unique, "unique_held_pairs": held_unique, "strata": []}
        for span, count, true, raw, full in zip(SECTIONS, counts, exact.split(strata),
                                                comparator.split(strata), fitted.split(strata)):
            diagnostics["strata"].append({"visual_rank_zero_based": span, "fit_sampled_pairs": count,
                                           "held_sampled_pairs": len(true), "held_true_mean": float(true.mean()),
                                           "proxy_mae": float((raw - true).abs().mean()),
                                           "exact_fit_mae": float((full - true).abs().mean())})
        utility = gallery_utility(metric, geometry)
    return proxy * 100, utility, diagnostics


@dataclass(frozen=True)
class NeuralRows:
    base: torch.Tensor
    mixed: torch.Tensor
    action: torch.Tensor


@torch.no_grad()
def neural_rows(model, queries, gallery, session, args):
    base, mixed, action = [], [], []
    for counts in queries.split(args.query_batch):
        v11, b, fusion = score_sources(model, counts, gallery, session, "mice1", args.dimensions)
        leader = v11.argmax(dim=1, keepdim=True)
        following = v11.clone().scatter_(1, leader, -torch.inf).topk(15, dim=1).indices[:, :WIDTH - 1]
        action.append(torch.cat((leader, following), dim=1))
        base.append(b)
        mixed.append(fusion)
    return NeuralRows(torch.cat(base), torch.cat(mixed), torch.cat(action))


@dataclass(frozen=True)
class RankedPosterior:
    ordered: torch.Tensor
    inverse: torch.Tensor
    probability: torch.Tensor
    log_mass: torch.Tensor
    context: torch.Tensor


@torch.no_grad()
def rank_posterior(rows, temperature):
    logits, order = rows.mixed.sort(dim=1, descending=True, stable=True)
    inverse = torch.empty_like(order)
    inverse.scatter_(1, order, torch.arange(order.shape[1], device=order.device).expand_as(order))
    probability = (logits / temperature).softmax(dim=1)
    ends = (*NEAR_EDGES, order.shape[1])
    band_mass = torch.stack([probability[:, low:high].sum(dim=1) for low, high in zip(ends, ends[1:])], dim=1)
    base_best = rows.base.topk(2, dim=1).values
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=1)
    agree = (order[:, 0] == rows.action[:, 0]).float()
    context = torch.stack((logits[:, 0] - logits[:, 1], logits[:, 0] - logits[:, WIDTH - 1],
                           logits[:, WIDTH - 1] - logits[:, 31], base_best[:, 0] - base_best[:, 1], entropy, agree), dim=1)
    return RankedPosterior(order, inverse, probability, band_mass.clamp_min(1e-9).log(), context)


def true_bands(ranked, labels):
    rank = ranked.inverse.gather(1, labels[:, None])[:, 0]
    return torch.bucketize(rank, torch.as_tensor(NEAR_EDGES[1:], device=rank.device), right=True)


@dataclass(frozen=True)
class BandCalibration:
    adjustment: nn.Linear
    center: torch.Tensor
    scale: torch.Tensor

    def logits(self, log_mass, context):
        scaled = ((context - self.center) / self.scale).clamp(-5, 5)
        return log_mass + 1.75 * self.adjustment(scaled).tanh()


def fit_bands(panels, temp_labels, temperature):
    ranked = [rank_posterior(panel, temperature) for panel in panels]
    log_mass = torch.cat([panel.log_mass for panel in ranked])
    context = torch.cat([panel.context for panel in ranked])
    labels = torch.cat([true_bands(panel, temp_labels) for panel in ranked])
    center = context.mean(dim=0)
    scale = context.std(dim=0, unbiased=False).clamp_min(0.03)
    adjustment = nn.Linear(context.shape[1], log_mass.shape[1]).to(context.device)
    nn.init.zeros_(adjustment.weight)
    nn.init.zeros_(adjustment.bias)
    model = BandCalibration(adjustment, center, scale)
    optimizer = torch.optim.AdamW(adjustment.parameters(), lr=0.035, weight_decay=0.025)
    for _ in range(160):
        loss = F.cross_entropy(model.logits(log_mass, context), labels) + 0.06 * adjustment.weight.square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def center_actions(values):
    return values - values.mean(dim=1, keepdim=True)


def scale_actions(values, floor):
    values = center_actions(values)
    return values / values.square().mean(dim=1, keepdim=True).sqrt().clamp_min(floor)


def fixed_utility_score(anchor, expected):
    return 0.80 * anchor + 0.20 * scale_actions(expected, 0.50)


@torch.no_grad()
def bands_of_utility(action, posterior, utility, batch_size):
    ends = (*NEAR_EDGES, utility.shape[1])
    blocks = []
    for begin in range(0, len(action), batch_size):
        selection = slice(begin, begin + batch_size)
        order = posterior.ordered[selection]
        probability = posterior.probability[selection]
        table = utility[action[selection]].gather(2, order[:, None].expand(-1, WIDTH, -1))
        bands = []
        for low, high in zip(ends, ends[1:]):
            p = probability[:, None, low:high]
            bands.append((table[..., low:high] * p).sum(dim=-1) / p.sum(dim=-1).clamp_min(1e-9))
        blocks.append(torch.stack(bands, dim=-1))
    return torch.cat(blocks)


@dataclass(frozen=True)
class Decisions:
    anchor: torch.Tensor
    feature: dict[str, torch.Tensor]
    unsplit_exact: torch.Tensor
    calibrated_exact: torch.Tensor
    band_report: dict


@torch.no_grad()
def decision_features(rows, labels, temperature, band_model, proxy, exact, batch_size):
    posterior = rank_posterior(rows, temperature)
    mass = band_model.logits(posterior.log_mass, posterior.context).softmax(dim=1)
    observed = true_bands(posterior, labels)
    original_mass = posterior.log_mass.exp()
    anchor = scale_actions(rows.mixed.gather(1, rows.action), 0.30)
    base = scale_actions(rows.base.gather(1, rows.action), 0.30)
    ranks = posterior.inverse.gather(1, rows.action).to(anchor.dtype).log1p()
    column = torch.arange(WIDTH, device=anchor.device, dtype=anchor.dtype).log1p().expand_as(anchor)
    context = ((posterior.context - band_model.center) / band_model.scale).clamp(-5, 5)
    base_feature = torch.cat((torch.stack((anchor, base, center_actions(ranks), center_actions(column)), dim=-1),
                              context[:, None].expand(-1, WIDTH, -1), mass[:, None].expand(-1, WIDTH, -1)), dim=-1)

    proxy_band = center_actions(bands_of_utility(rows.action, posterior, proxy, batch_size))
    exact_band = center_actions(bands_of_utility(rows.action, posterior, exact, batch_size))
    features = {"posterior": torch.cat((base_feature, torch.zeros_like(exact_band)), dim=-1),
                "contour": torch.cat((base_feature, proxy_band * mass[:, None]), dim=-1),
                "exact": torch.cat((base_feature, exact_band * mass[:, None]), dim=-1)}
    unsplit = (exact_band * original_mass[:, None]).sum(dim=-1)
    calibrated = (exact_band * mass[:, None]).sum(dim=-1)
    report = {"observed": (torch.bincount(observed, minlength=mass.shape[1]) / len(observed)).tolist(),
              "single_temperature_mass": original_mass.mean(dim=0).tolist(), "calibrated_mass": mass.mean(dim=0).tolist(),
              "single_temperature_band_nll": float(F.nll_loss(posterior.log_mass, observed)),
              "calibrated_band_nll": float(F.nll_loss(mass.clamp_min(1e-9).log(), observed))}
    return Decisions(anchor, features, unsplit, calibrated, report)


@dataclass(frozen=True)
class ListwiseCorrection:
    adjustment: nn.Sequential
    center: torch.Tensor
    scale: torch.Tensor

    def logits(self, anchor, features):
        x = ((features - self.center) / self.scale).clamp(-6, 6)
        return anchor + 0.85 * self.adjustment(x).squeeze(-1).tanh()


def fit_listwise(anchor, features, utility, args):
    torch.manual_seed(args.seed + 661)
    draw = torch.Generator(device=anchor.device).manual_seed(args.seed + 663)
    center = features.mean(dim=(0, 1))
    scale = features.flatten(0, 1).std(dim=0, unbiased=False).clamp_min(0.045)
    dimensions = features.shape[-1]
    network = nn.Sequential(nn.Linear(dimensions, 24), nn.Tanh(), nn.Linear(24, 1)).to(anchor.device)
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    model = ListwiseCorrection(network, center, scale)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.0025, weight_decay=0.055)
    target = (utility / 6.0).softmax(dim=1)
    for _ in range(args.listwise_epochs):
        for idx in torch.randperm(len(anchor), generator=draw, device=anchor.device).split(128):
            logits = model.logits(anchor[idx], features[idx])
            cross_entropy = -(target[idx] * logits.log_softmax(dim=1)).sum(dim=1).mean()
            penalty = 0.035 * (logits - anchor[idx]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            (cross_entropy + penalty).backward()
            optimizer.step()
    return model


@torch.no_grad()
def exact_actions(rows, labels, masks, score):
    return unique_pair_ocf(rows.action, labels[:, None].expand_as(rows.action), masks, score, masks.device, 384)


@torch.no_grad()
def evaluate(rows, features, heads, exact, reorder):
    anchor = features.anchor
    variants = {"v12_top7": rows.mixed.gather(1, rows.action),
                "exact_single_temperature": fixed_utility_score(anchor, features.unsplit_exact),
                "exact_band_calibration": fixed_utility_score(anchor, features.calibrated_exact)}
    for name, head in heads.items():
        variants[f"listwise_{name}"] = head.logits(anchor, features.feature[name])
    positions = {name: value.argmax(dim=1) for name, value in variants.items()}
    positions["oracle_top7"] = exact.argmax(dim=1)
    observations = {name: exact.gather(1, choice[:, None])[:, 0][reorder].cpu().numpy()
                    for name, choice in positions.items()}
    changed = {name: int((choice != positions["v12_top7"]).sum()) for name, choice in positions.items()}
    return observations, changed


def summarize(values):
    base = values["v12_top7"]
    return {name: paired(value, base) for name, value in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-seed", type=int, default=9127)
    parser.add_argument("--repeat-seed", type=int, default=9121)
    parser.add_argument("--vision-check-glyphs", type=int, default=640)
    parser.add_argument("--temperature-glyphs", type=int, default=192)
    parser.add_argument("--calibration-glyphs", type=int, default=256)
    parser.add_argument("--ocf-images", type=int, default=192)
    parser.add_argument("--visual-pairs", type=int, default=14336)
    parser.add_argument("--visual-check-pairs", type=int, default=896)
    parser.add_argument("--query-batch", type=int, default=64)
    parser.add_argument("--listwise-epochs", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_rankband_ocf.json"))
    parser.set_defaults(seed=211, encoder_epochs=18, epochs=8, dimensions=128)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run the OCF comparison on Kaggle GPU with competition data mounted.")
    if min(args.temperature_glyphs, args.calibration_glyphs, args.ocf_images, args.query_batch,
           args.listwise_epochs) < 2 or min(args.visual_pairs, args.visual_check_pairs) < 16:
        parser.error("Use positive fit sizes and at least two scoring rows.")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    started = time.monotonic()
    root = locate_data()
    gallery = load_images(root, device)
    masks = gallery >= 128
    signatures, energy = contour_signatures(gallery)
    score = official_scorer(root)
    rng = np.random.default_rng(args.sample_seed)
    sizes = (args.vision_check_glyphs, args.temperature_glyphs, args.calibration_glyphs, args.ocf_images)
    fit_count = len(gallery) - sum(sizes)
    if min(fit_count, args.vision_check_glyphs) < 514:
        parser.error("Visual rank strata require at least 514 fit glyphs and 514 separate visual check glyphs.")
    cohort = torch.as_tensor(rng.permutation(len(gallery)), device=device)
    visual_hold, temp_labels, cal_labels, outer_labels, visual_fit = cohort.split((*sizes, fit_count))
    proxy, utility, vision_report = fit_visual_tables(masks, signatures, energy, visual_fit, visual_hold, score, args)
    stage("visual_tables", started, **vision_report)

    all_recordings = grouped_blocks(root, load_catalog(root), "mice1", device, len(gallery))
    inner_fit, inner_held, outer_fit, outer_held = nested_repeats(all_recordings, args.repeat_seed)
    del all_recordings
    inner, inner_fit_report = fit_models(inner_fit, gallery, signatures, energy, "mice1", args)
    del inner_fit
    temp_panels, cal_panels = {}, {}
    for session, query in inner_held.items():
        temp_panels[session] = neural_rows(inner, query[temp_labels], gallery, session, args)
        cal_panel = neural_rows(inner, query[cal_labels], gallery, session, args)
        cal_panels[session] = cal_panel
    joined = torch.cat([p.action for p in cal_panels.values()])
    references = cal_labels.repeat(len(cal_panels))[:, None].expand_as(joined)
    cal_values, count = unique_pair_ocf(joined, references, masks, score, device, 384)
    cal_exact = dict(zip(cal_panels, cal_values.split(len(cal_labels))))
    del inner, inner_held, joined, references
    stage("independent_ocf_calibration", started, unique_exact_pairs=count, **inner_fit_report)

    outer, outer_fit_report = fit_models(outer_fit, gallery, signatures, energy, "mice1", args)
    del outer_fit, signatures, energy
    report = {"protocol": {"mouse": "mice1", "gallery_size": len(gallery), "sample_seed": args.sample_seed,
                           "repeat_seed": args.repeat_seed, "outer_scored_per_day": len(outer_labels),
                           "listwise_calibration_per_source_day": len(cal_labels), "temperature_per_source_day": len(temp_labels),
                           "candidate_count": WIDTH, "query_batch": args.query_batch, "one_based_rank_bands": ["1–7", "8–32", "33–128", "129+"],
                           "actual_test_records_accessed": False, "listwise_target_temperature_ocf": 6.0,
                           "listwise_epochs": args.listwise_epochs,
                           "primary": "Listwise exact-utility versus exact v12 on the same seven candidates and versus matched posterior-only. "
                                      "Look for about +1 OCF in both comparisons and agreement in at least three target recordings.",
                           "separation": "The visual pair fit, visual holdout, band calibration glyphs, listwise calibration glyphs, "
                                         "and outer glyphs are disjoint. Each target day uses supervised calibrators from the other three days. "
                                         "The v12 anchor refits on legal known-glyph repeats in each recording. Outer measured repeats are reserved before all fits."},
              "visual": vision_report, "calibration_unique_exact_pairs": count,
              "v12_fit": {"inner": inner_fit_report, "outer": outer_fit_report}, "sessions": {}}
    measurements = {}

    for session, queries in outer_held.items():
        source_days = [other for other in temp_panels if other != session]
        temperature = fit_temperature([(temp_panels[day].mixed, temp_labels) for day in source_days])
        t = temperature["temperature"]
        band_model = fit_bands([temp_panels[day] for day in source_days], temp_labels, t)
        source = [decision_features(cal_panels[day], cal_labels, t, band_model, proxy, utility, args.query_batch)
                  for day in source_days]
        anchor = torch.cat([feature.anchor for feature in source])
        exact = torch.cat([cal_exact[day] for day in source_days])
        heads = {name: fit_listwise(anchor, torch.cat([feature.feature[name] for feature in source]), exact, args) for name in ARMS}

        shuffled = torch.as_tensor(rng.permutation(len(outer_labels)), device=device)
        labels = outer_labels[shuffled]
        observed = neural_rows(outer, queries[labels], gallery, session, args)
        features = decision_features(observed, labels, t, band_model, proxy, utility, args.query_batch)
        scores, pairs = exact_actions(observed, labels, masks, score)
        values, switches = evaluate(observed, features, heads, scores, shuffled.argsort())
        for name, observed_ocf in values.items():
            measurements.setdefault(name, []).append(observed_ocf)
        report["sessions"][session] = {"source_recordings": source_days, "temperature": temperature,
                                       "calibration_bands": {day: outcome.band_report for day, outcome in zip(source_days, source)},
                                       "outer_bands": features.band_report, "outer_exact_pairs": pairs,
                                       "switches_vs_v12": switches, "outcomes": summarize(values),
                                       "exact_vs_posterior": paired(values["listwise_exact"], values["listwise_posterior"]),
                                       "exact_vs_contour": paired(values["listwise_exact"], values["listwise_contour"])}
        report["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        stage("outer_ocf", started, session=session, **summarize(values))

    scores = {name: np.stack(values) for name, values in measurements.items()}
    report.update({"outcomes": summarize(scores),
                   "exact_vs_posterior": paired(scores["listwise_exact"], scores["listwise_posterior"]),
                   "exact_vs_contour": paired(scores["listwise_exact"], scores["listwise_contour"]),
                   "exact_vs_unsplit_exact": paired(scores["listwise_exact"], scores["exact_single_temperature"]),
                   "seconds": time.monotonic() - started})
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    stage("complete", started, outcomes=report["outcomes"], exact_vs_posterior=report["exact_vs_posterior"])


if __name__ == "__main__":
    main()
