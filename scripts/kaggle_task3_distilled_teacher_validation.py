"""Test whether repeat-resolved neural targets improve Task 3 single-trial decoding."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_episodic_assignment import _projection
from glyph_exact_utility import fit_exact_utility, gallery_geometry, pair_features, select_pairs
from glyph_mbr import contour_signatures, similarity_from_signatures
from kaggle_diagnostics import feature_view, official_scorer, standardize
from kaggle_task3_density_top7_recheck import top7_predictions
from kaggle_task3_measured_metric_validation import append_rows, paired, ranks, summarize
from kaggle_task3_metric_validation import nested_repeats, unique_pair_ocf
from kaggle_task3_part_discovery import standardize_rows
from kaggle_task3_repeat_posterior import fit_models, score_sources


VIEW = "evoked-multiscale"
ARMS = ("hard", "morphology", "teacher")
WEIGHTS = (0.0, 0.15, 0.25, 0.40)
PRIMARY_WEIGHT = 0.25
SOFT_MASS = 0.30
NEIGHBORS = 24


def stage(name, started, **details):
    print(json.dumps({"stage": name, "seconds": round(time.monotonic() - started, 2), **details}), flush=True)


def visual_representation(images, masks, signatures, energy, fit, scorer, args):
    device = images.device
    draw = torch.Generator(device=device).manual_seed(args.seed + 317)
    # Mine broadly in the inexpensive contour space, then learn OCF geometry.
    proxy = similarity_from_signatures(signatures[fit], energy[fit])
    local = torch.arange(len(fit), device=device)
    a, b = select_pairs(local, proxy, args.visual_pairs, draw)
    geometry = gallery_geometry(masks, signatures, energy)
    exact, count = unique_pair_ocf(fit[a], fit[b], masks, scorer, device, 384)
    metric = fit_exact_utility(geometry, fit[a], fit[b], exact, seed=args.seed + 313)

    with torch.no_grad():
        near = proxy.fill_diagonal_(-torch.inf).topk(96, dim=1).indices
        reference = fit[:, None].expand_as(near).flatten()
        nearby_ocf = torch.cat([metric.predict(geometry.features(action, target))
                               for action, target in zip(fit[near].flatten().split(2048), reference.split(2048))])
        utility, candidate_order = nearby_ocf.reshape_as(near).topk(NEIGHBORS, dim=1)
        neighbors = near.gather(1, candidate_order)
        landmarks = fit[torch.randperm(len(fit), generator=draw, device=device)[:160]]
        coords = torch.cat([metric.predict(geometry.features(rows, landmarks, crossed=True))
                            for rows in torch.arange(len(images), device=device).split(128)])
        center = coords[fit].mean(dim=0)
        scale = coords[fit].std(dim=0, unbiased=False).clamp_min(1.0)
        standardized = (coords - center) / scale
        torch.manual_seed(args.seed + 319)
        _, _, basis = torch.pca_lowrank(standardized[fit], q=64, center=False, niter=2)
        codes = F.normalize(standardized @ basis, dim=-1)
        estimate = metric.predict(pair_features(geometry, fit[a], fit[b]))
        fit_report = {"unique_fit_pairs": count, "landmarks": len(landmarks), "dimensions": codes.shape[1],
                      "fit_pair_mae": float((estimate - exact).abs().mean()),
                      "neighbors": NEIGHBORS, "visual_temperature_ocf": 8.0}
    return codes, neighbors, utility / 8.0, fit_report


@torch.no_grad()
def source_features(sources, args):
    projection, projected = _projection(sources, VIEW, args.neural_dimensions)
    moments = [standardize(feature_view(counts, VIEW))[1] for counts in sources.values()]
    target_moments = tuple(torch.stack(group).mean(dim=0) for group in zip(*moments))

    # Each fit glyph receives its own randomized trial permutation on each day.
    draw = torch.Generator(device=projected.device).manual_seed(args.seed + 331)
    order = torch.rand(projected.shape[:3], generator=draw, device=projected.device).argsort(dim=-1)
    projected = projected.gather(2, order[..., None].expand_as(projected))
    return projection, projected, target_moments


@torch.no_grad()
def teacher_targets(source, neighbors, shape_logits):
    days, classes, repeats, _ = source.shape
    if (days, repeats) != (3, 3):
        raise ValueError("The student/teacher/template partition needs three training days and three repeats per day.")
    summed = source.sum(dim=2)
    weights = source.new_empty(days, classes, repeats, neighbors.shape[1])
    single_top1, repeated_top1, total_shift = [], [], []
    visual_only = shape_logits.softmax(dim=1)

    for day in range(days):
        first, second = (day + 1) % days, (day + 2) % days
        for repeat in range(repeats):
            first_repeat, second_repeat = (repeat + 1) % repeats, (repeat + 2) % repeats
            first_trial = source[first, :, first_repeat]
            second_trial = source[second, :, second_repeat]
            # 1 real student trial, 4 disjoint teacher trials, 4 other template trials.
            teacher = (summed[day] - source[day, :, repeat]) / 6 + (first_trial + second_trial) / 3
            template = F.normalize(summed[first] + summed[second] - first_trial - second_trial, dim=-1)
            single = F.normalize(source[day, :, (repeat + 1) % repeats], dim=-1)
            teacher = F.normalize(teacher, dim=-1)
            near_template = template[neighbors]
            evidence = torch.einsum("gc,gkc->gk", teacher, near_template)
            control = torch.einsum("gc,gkc->gk", single, near_template)
            repeated_top1.append(((teacher * template).sum(dim=1) > evidence.amax(dim=1)).float().mean())
            single_top1.append(((single * template).sum(dim=1) > control.amax(dim=1)).float().mean())
            posterior = (shape_logits + 0.75 * standardize_rows(evidence).clamp(-3, 3)).softmax(dim=1)
            total_shift.append((0.5 * (posterior - visual_only).abs().sum(dim=1)).mean())
            weights[day, :, repeat] = posterior

    return weights.flatten(0, 2), {"source_teacher_top1_25": float(torch.stack(repeated_top1).mean()),
                                   "source_single_top1_25": float(torch.stack(single_top1).mean()),
                                   "teacher_neighbor_total_variation": float(torch.stack(total_shift).mean())}


def fit_students(projected, codes, neighbors, shape_logits, args):
    days, classes, repeats, dimensions = projected.shape
    source = projected.flatten(0, 2)
    labels = torch.arange(classes, device=source.device).repeat_interleave(repeats).repeat(days)
    teacher_mass, diagnostics = teacher_targets(projected, neighbors, shape_logits)
    shape_mass = shape_logits.softmax(dim=1)
    students, losses = {}, {}

    for arm in ARMS:
        torch.manual_seed(args.seed + 337)
        sampler = torch.Generator(device=source.device).manual_seed(args.seed + 347)
        student = nn.Sequential(nn.LayerNorm(dimensions), nn.Linear(dimensions, 128), nn.GELU(),
                                nn.Linear(128, codes.shape[1])).to(source.device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=0.0015, weight_decay=0.035)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_student, eta_min=0.0002)
        for _ in range(args.epochs_student):
            total = source.new_zeros(())
            for batch in torch.randperm(len(source), generator=sampler, device=source.device).split(256):
                target = labels[batch]
                log_prob = (12.0 * F.normalize(student(source[batch]), dim=-1) @ codes.T).log_softmax(dim=1)
                loss = -log_prob.gather(1, target[:, None]).mean()
                if arm != "hard":
                    mass = teacher_mass[batch] if arm == "teacher" else shape_mass[target]
                    soft_loss = -(log_prob.gather(1, neighbors[target]) * mass).sum(dim=1).mean()
                    loss = (1 - SOFT_MASS) * loss + SOFT_MASS * soft_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.detach() * len(batch)
            schedule.step()
        students[arm] = student.eval()
        losses[arm] = float(total / len(source))
    return students, {**diagnostics, "source_single_trials_per_arm": len(source), "source_loss_last_epoch": losses}


def student_key(arm, weight):
    return f"{arm}_fixed_pool_{weight:.2f}" if weight else "v12_top7"


def select(scores, candidates):
    winner = scores.gather(1, candidates).argmax(dim=1, keepdim=True)
    return candidates.gather(1, winner)[:, 0]


@torch.no_grad()
def score_query_batch(model, session, query, states, gallery, labels, dimensions, *, outer):
    projection, moments, students, codes = states
    if query.ndim != 3 or query.shape[1] != 1:
        raise ValueError("Each Task 3 query must contain exactly one measured repeat.")
    v11, _, v12 = score_sources(model, query, gallery, session, "mice1", dimensions)
    anchor, chosen = top7_predictions(v11, v12)
    next_six = v11.clone().scatter_(1, anchor[:, None], -torch.inf).topk(15, dim=1).indices[:, :6]
    original = torch.cat((anchor[:, None], next_six), dim=1)
    responses, _ = standardize(feature_view(query, VIEW), moments)
    encoded = responses[:, 0] @ projection
    choices = {"v12_top7": chosen, "v12_full": v12.argmax(dim=1)}
    pools = {"v12_top7": original}
    full = {"v11": ranks(v11, labels), "v12": ranks(v12, labels)}

    for arm, student in students.items():
        component = standardize_rows(F.normalize(student(encoded), dim=-1) @ codes.T)
        if component.shape != v12.shape:
            raise ValueError(f"{arm} and v12 scores must share [query, glyph] axes: {component.shape}, {v12.shape}.")
        for mix in WEIGHTS[1:]:
            adjusted = v12.lerp(component, mix)
            choices[student_key(arm, mix)] = select(adjusted, original)
        if outer:
            adjusted = v12.lerp(component, PRIMARY_WEIGHT)
            retain = original.gather(1, (original != chosen[:, None]).long().argsort(dim=1, descending=True, stable=True)[:, :4])
            nominated = component.clone().scatter_(1, original, -torch.inf).topk(2, dim=1).indices
            repl = torch.cat((chosen[:, None], retain, nominated), dim=1)
            key = f"{arm}_nominated_top7"
            pools[key] = repl
            choices[key] = select(adjusted, repl)
            choices[f"{arm}_full_0.25"] = adjusted.argmax(dim=1)
            full[arm] = ranks(component, labels)
            full[f"{arm}_blend_0.25"] = ranks(adjusted, labels)

    return choices, {"full_gallery_rank": full,
                     "pool_recall": {name: float((pool == labels[:, None]).any(dim=1).float().mean())
                                     for name, pool in pools.items()},
                     "changes_vs_v12_top7": {name: int((guess != chosen).sum()) for name, guess in choices.items()}}


@torch.no_grad()
def evaluate(model, session, query, states, gallery, masks, scorer, sample, ordering, args, *, outer):
    # Rows are independent. Score the shuffled paired cohort in bounded batches against every gallery candidate.
    presented = ordering[torch.isin(ordering, sample)]
    supplied_position = presented.argsort()[sample.argsort().argsort()]
    all_choices, rank_metrics, coverage, changes = {}, {}, {}, {}
    for labels in presented.split(args.query_batch):
        choices, metrics = score_query_batch(model, session, query[labels], states, gallery, labels, args.dimensions, outer=outer)
        fraction = len(labels) / len(sample)
        for name, values in metrics["full_gallery_rank"].items():
            totals = rank_metrics.setdefault(name, {})
            for limit, truth_rate in values.items():
                totals[limit] = totals.get(limit, 0.0) + truth_rate * fraction
        for name, value in metrics["pool_recall"].items():
            coverage[name] = coverage.get(name, 0.0) + value * fraction
        for name, value in metrics["changes_vs_v12_top7"].items():
            changes[name] = changes.get(name, 0) + value
        for name, choice in choices.items():
            all_choices.setdefault(name, []).append(choice)

    predictions = torch.stack([torch.cat(guess)[supplied_position] for guess in all_choices.values()])
    values, count = unique_pair_ocf(predictions, sample.expand_as(predictions), masks, scorer, gallery.device, 384)
    return dict(zip(all_choices, values.cpu().numpy())), {"full_gallery_rank": rank_metrics, "pool_recall": coverage,
                                                          "changes_vs_v12_top7": changes, "unique_paired_ocf": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sample-seed", type=int, default=8329)
    parser.add_argument("--repeat-seed", type=int, default=8327)
    parser.add_argument("--ocf-images", type=int, default=192)
    parser.add_argument("--calibration-images", type=int, default=128)
    parser.add_argument("--visual-pairs", type=int, default=3200)
    parser.add_argument("--neural-dimensions", type=int, default=96)
    parser.add_argument("--epochs-student", type=int, default=8)
    parser.add_argument("--query-batch", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/task3_distilled_teacher.json"))
    parser.set_defaults(seed=211, encoder_epochs=18, epochs=8, dimensions=128)
    args = parser.parse_args()
    if min(args.ocf_images, args.calibration_images, args.visual_pairs, args.neural_dimensions,
           args.epochs_student, args.query_batch) < 2:
        parser.error("Provide at least two observations and positive fit budgets.")
    if not Path("/kaggle/input").is_dir() or (args.device == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("Run on Kaggle with the competition data and requested accelerator.")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    started = time.monotonic()
    root = locate_data()
    gallery = load_images(root, device)
    masks = gallery >= 128
    scorer = official_scorer(root)
    signatures, energy = contour_signatures(gallery)
    rng = np.random.default_rng(args.sample_seed)
    order = torch.as_tensor(rng.permutation(len(gallery)), device=device)
    outer_rows, inner_rows, fit = order.split((args.ocf_images, args.calibration_images,
                                             len(gallery) - args.ocf_images - args.calibration_images))
    if len(fit) < 512:
        parser.error("Keep at least 512 glyphs for teacher and image representation training.")
    codes, neighbors, shape_logits, visual_report = visual_representation(gallery, masks, signatures, energy, fit, scorer, args)
    stage("image_pretraining", started, **visual_report)

    recordings = grouped_blocks(root, load_catalog(root), "mice1", device, len(gallery))
    inner_fit, inner_held, outer_fit, outer_held = nested_repeats(recordings, args.repeat_seed)
    del recordings
    inner_model, inner_base = fit_models(inner_fit, gallery, signatures, energy, "mice1", args)
    del inner_fit
    stage("inner_v12_fit", started, **inner_base)
    outer_model, outer_base = fit_models(outer_fit, gallery, signatures, energy, "mice1", args)
    source_pool = {day: values[fit] for day, values in outer_fit.items()}
    del outer_fit, signatures, energy
    stage("outer_v12_fit", started, **outer_base)

    report = {"protocol": {"mouse": "mice1", "gallery": len(gallery), "sample_seed": args.sample_seed,
                           "repeat_seed": args.repeat_seed, "training_seed": args.seed, "device": str(device),
                           "image_and_student_fit_glyphs": len(fit), "inner_calibration_glyphs_per_day": len(inner_rows),
                           "outer_glyphs_per_day": len(outer_rows), "frozen_mix": PRIMARY_WEIGHT,
                           "calibration_weight_grid": WEIGHTS, "same_candidate_budget": 7,
                           "soft_target_mass": SOFT_MASS, "student_epochs": args.epochs_student,
                           "full_gallery_query_batch": args.query_batch,
                           "teacher_partition_per_example": {"student": 1, "teacher": 4, "teacher_templates": 4},
                           "teacher_uses_target_recording": False, "teacher_uses_held_glyphs": False,
                           "actual_test_spikes_read": False,
                           "primary_comparison": "Teacher minus morphology-only OCF at fixed 0.25 Rerank weight "
                                                 "on the same seven v12 candidates. Look for about +1 OCF and agreement in at least three days.",
                           "comparison": "The v12 baseline is re-fit with all known glyphs and permitted same-day repeats. "
                                         "Novel model normalization and weights use source recordings only. "
                                         "Image embeddings for the full known gallery are evaluated from frozen visual parameters."},
              "visual_pretraining": visual_report, "v12": {"inner": inner_base, "outer": outer_base}, "sessions": {}}
    inner_scores, outer_scores = {}, {}

    for session in source_pool:
        sources = {day: counts for day, counts in source_pool.items() if day != session}
        projection, projected, moments = source_features(sources, args)
        students, teacher_report = fit_students(projected, codes[fit], neighbors, shape_logits, args)
        del projected
        stage("student_fit", started, session=session, **teacher_report)
        states = (projection, moments, students, codes)
        inner_order = torch.as_tensor(rng.permutation(len(gallery)), device=device)
        outer_order = torch.as_tensor(rng.permutation(len(gallery)), device=device)
        first, inner_diag = evaluate(inner_model, session, inner_held[session], states, gallery,
                                     masks, scorer, inner_rows, inner_order, args, outer=False)
        last, outer_diag = evaluate(outer_model, session, outer_held[session], states, gallery,
                                    masks, scorer, outer_rows, outer_order, args, outer=True)
        append_rows(inner_scores, first)
        append_rows(outer_scores, last)
        contrast = paired(last[student_key("teacher", PRIMARY_WEIGHT)], last[student_key("morphology", PRIMARY_WEIGHT)])
        report["sessions"][session] = {"source_days": list(sources), "teacher_fit_diagnostics": teacher_report,
                                       "inner": inner_diag, "outer": outer_diag,
                                       "teacher_vs_morphology_fixed": contrast, "paired_ocf": summarize(last)}
        report["seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        stage("paired_outer", started, session=session, teacher_vs_morphology=contrast)

    inner = {name: np.stack(samples) for name, samples in inner_scores.items()}
    outer = {name: np.stack(samples) for name, samples in outer_scores.items()}
    # Select a common optional mix using only the two controls' inner OCF.
    weight = max(WEIGHTS, key=lambda mix: ((inner[student_key("hard", mix)].mean()
                                          + inner[student_key("morphology", mix)].mean()) / 2, -mix))
    teacher, morphology, hard = (outer[student_key(arm, PRIMARY_WEIGHT)] for arm in ("teacher", "morphology", "hard"))
    report.update({"outer_paired_ocf": summarize(outer), "inner_paired_ocf": summarize(inner),
                   "primary_teacher_vs_morphology": paired(teacher, morphology),
                   "primary_teacher_vs_hard": paired(teacher, hard),
                   "primary_morphology_vs_hard": paired(morphology, hard),
                   "teacher_vs_morphology_nominated_top7": paired(outer["teacher_nominated_top7"], outer["morphology_nominated_top7"]),
                   "inner_control_selected_weight": weight,
                   "outer_common_calibrated_ocf": {arm: paired(outer[student_key(arm, weight)], outer["v12_top7"]) for arm in ARMS},
                   "seconds": time.monotonic() - started})
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    stage("complete", started, primary_teacher_vs_morphology=report["primary_teacher_vs_morphology"])


if __name__ == "__main__":
    main()
