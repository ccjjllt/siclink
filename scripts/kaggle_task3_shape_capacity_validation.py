"""Test visual-group capacities and cross-recording posterior corrections for Task 3."""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_exact_utility import gallery_geometry, gallery_utility
from glyph_mbr import contour_signatures, fit_temperature, glyph_posterior, similarity_from_signatures
from kaggle_diagnostics import official_scorer
from kaggle_task3_measured_metric_validation import paired
from kaggle_task3_metric_validation import nested_repeats, unique_pair_ocf
from kaggle_task3_rankband_ocf_recheck import glyph_cohorts
from kaggle_task3_rankband_ocf_validation import fit_visual_metric, scale_actions, stage
from kaggle_task3_repeat_posterior import fit_models, score_sources
from kaggle_task3_utility_nomination_recheck import independent_cohorts


SEED = 27937
REPEAT_SEED = 27943
VISUAL_GROUPS = 96
EVAL = 256
TEMP = 192
OUTPUT = Path("/kaggle/working/task3_shape_capacity.json")
ACTUAL = "v14"
INDEPENDENT = "independent_ocf"
JOINT = "shape_joint"


@dataclass(frozen=True)
class GlyphGroups:
    membership: torch.Tensor
    incidence: torch.Tensor
    counts: torch.Tensor


@dataclass(frozen=True)
class NeuralPanel:
    logits: torch.Tensor
    shortlist: torch.Tensor | None


def cohorts(size, device):
    visual, _, _ = glyph_cohorts(size, device)
    recent = independent_cohorts(size, device).outer
    earlier = torch.as_tensor(np.random.default_rng(1531).permutation(size)[2432:2560], device=device)
    eligible = torch.arange(size, device=device)
    eligible = eligible[~torch.isin(eligible, torch.cat((visual, recent, earlier)))]
    if len(eligible) < TEMP + EVAL:
        raise RuntimeError("Need independent image-fit, temperature and new OCF glyphs.")
    order = torch.as_tensor(np.random.default_rng(SEED).permutation(len(eligible))[:TEMP + EVAL], device=device)
    temperature, scored = eligible[order].split((TEMP, EVAL))
    return visual, temperature, scored


def membership_groups(ids):
    membership = F.one_hot(ids, num_classes=VISUAL_GROUPS).float()
    counts = membership.sum(dim=0)
    if not (counts > 0).all():
        raise RuntimeError("Every learned visual group needs at least one known glyph.")
    return GlyphGroups(ids, membership, counts)


@torch.no_grad()
def fit_visual_groups(signatures, energy, masks, visual_fit, utility):
    device = signatures.device
    torch.manual_seed(SEED)
    _, _, axes = torch.pca_lowrank(signatures[visual_fit], q=40, center=True, niter=2)
    flattened = F.normalize((signatures - signatures[visual_fit].mean(dim=0)) @ axes, dim=1)
    ink = masks.float().mean(dim=(1, 2))
    scalars = torch.stack((ink, energy.log()), dim=1)
    scalars = (scalars - scalars.mean(dim=0)) / scalars.std(dim=0).clamp_min(1e-6)
    descriptors = torch.cat((flattened, 0.12 * scalars), dim=1)
    ids = KMeans(n_clusters=VISUAL_GROUPS, n_init=4, random_state=SEED).fit_predict(descriptors.cpu().numpy())
    groups = membership_groups(torch.as_tensor(ids, dtype=torch.long, device=device))

    quantiles = ink.new_tensor((0.25, 0.5, 0.75))
    area = torch.bucketize(ink.contiguous(), torch.quantile(ink, quantiles))
    contour = torch.bucketize(energy.contiguous(), torch.quantile(energy, quantiles))
    strata = 4 * area + contour
    rng = torch.Generator(device=device).manual_seed(SEED + 31)
    shuffled = groups.membership.clone()
    for stratum in strata.unique():
        rows = (strata == stratum).nonzero()[:, 0]
        shuffled[rows] = groups.membership[rows[torch.randperm(len(rows), device=device, generator=rng)]]
    control = membership_groups(shuffled)
    if not torch.equal(groups.counts, control.counts):
        raise RuntimeError("The matched shuffled geometry must preserve all visual-group capacities.")

    def coherence(partition):
        relation = (partition.membership[:, None] == partition.membership).fill_diagonal_(False)
        return float(utility[relation].mean())

    report = {"count": VISUAL_GROUPS, "min_groupsize": int(groups.counts.min()), "max_groupsize": int(groups.counts.max()),
              "mean_fit_utility_inside_group": coherence(groups), "mean_shuffled_utility_inside_group": coherence(control)}
    return groups, control, report


@torch.no_grad()
def predict_panel(model, counts, gallery, session, mouse, args):
    scores, picks = [], []
    for batch in counts.split(args.query_batch):
        v11, _, mixed = score_sources(model, batch, gallery, session, mouse, args.dimensions)
        if mouse == "mice1":
            anchor = v11.argmax(dim=1, keepdim=True)
            tail = v11.clone().scatter_(1, anchor, -torch.inf).topk(15, dim=1).indices[:, :6]
            picks.append(torch.cat((anchor, tail), dim=1))
        scores.append(mixed if mouse == "mice1" else v11)
    return NeuralPanel(torch.cat(scores), torch.cat(picks) if picks else None)


@torch.no_grad()
def capacity_potential(probability, groups):
    """Compute the KL projection's groupwise log multipliers with fixed row mass."""
    area = (probability @ groups.incidence).clamp_min(1e-12).log()
    target = groups.counts.log()
    potential = torch.zeros_like(target)
    for _ in range(48):
        current = (area + potential).softmax(dim=1).sum(dim=0)
        potential += target - current.clamp_min(1e-12).log()
    return potential - (potential * groups.counts).sum() / groups.counts.sum()


@torch.no_grad()
def condition(probability, potential, groups):
    return (probability.clamp_min(1e-30).log() + potential[groups.membership]).softmax(dim=1)


@torch.no_grad()
def check_posterior(probability, truth, groups):
    true_group = groups.membership[truth]
    group_mass = probability @ groups.incidence
    glyph_true = probability.gather(1, truth[:, None])[:, 0].clamp_min(1e-30)
    group_true = group_mass.gather(1, true_group[:, None])[:, 0].clamp_min(1e-30)
    return {"glyph_nll": float(-glyph_true.log().mean()), "group_nll": float(-group_true.log().mean()),
            "glyph_top1": float((probability.argmax(dim=1) == truth).float().mean()),
            "glyph_top32": float((probability.topk(32, dim=1).indices == truth[:, None]).any(dim=1).float().mean()),
            "group_top1": float((group_mass.argmax(dim=1) == true_group).float().mean()),
            "group_mass_total_variation": float((group_mass.sum(dim=0) - groups.counts).abs().sum() / (2 * len(truth)))}


def nominate(scores, shortlist):
    if shortlist is None:
        return scores.argmax(dim=1)
    return shortlist.gather(1, scores.argmax(dim=1, keepdim=True))[:, 0]


@torch.no_grad()
def decide(panel, rows, posteriors, utility):
    anchor = panel.logits[rows]
    available = None if panel.shortlist is None else panel.shortlist[rows]
    if available is not None:
        anchor = anchor.gather(1, available)
    choices = {ACTUAL: nominate(anchor, available)}
    target = 0.80 * scale_actions(anchor, 0.30)
    population_value = utility.mean(dim=1)
    for name, posterior in posteriors.items():
        expected = posterior[rows] @ utility.T - population_value
        if available is not None:
            expected = expected.gather(1, available)
        choices[name] = nominate(target + 0.20 * scale_actions(expected, 0.50), available)
    return choices


def ocf_statistics(scores):
    result = {name: paired(values, scores[ACTUAL]) for name, values in scores.items()}
    result["joint_vs_ordinary_independent_ocf"] = paired(scores[JOINT], scores[INDEPENDENT])
    result["joint_vs_local_equal_strength"] = paired(scores[JOINT], scores["shape_local"])
    result["joint_vs_local_shrunken"] = paired(scores[JOINT], scores["shape_local_half"])
    result["joint_vs_matched_shuffled_groups"] = paired(scores[JOINT], scores["shape_joint_shuffled"])
    result["joint_vs_glyph_sinkhorn"] = paired(scores[JOINT], scores["glyph_sinkhorn"])
    return result


@torch.no_grad()
def score_recordings(panels, raw, truth, temp, visual, shuffled, utility, sampled, masks, scorer, save_block):
    fields = {name: capacity_potential(post, visual) for name, post in raw.items()}
    null_fields = {name: capacity_potential(post, shuffled) for name, post in raw.items()}
    measured = {}
    for name, post in raw.items():
        peers = [other for other in raw if other != name]
        shared = torch.stack([fields[peer] for peer in peers]).median(dim=0).values
        null_shared = torch.stack([null_fields[peer] for peer in peers]).median(dim=0).values
        posterior = {INDEPENDENT: post, "glyph_sinkhorn": glyph_posterior(panels[name].logits, temp[name]["temperature"], balance=True),
                     "shape_local": condition(post, fields[name], visual),
                     "shape_local_half": condition(post, 0.5 * fields[name], visual),
                     JOINT: condition(post, 0.5 * (fields[name] + shared), visual),
                     "shape_joint_shuffled": condition(post, 0.5 * (null_fields[name] + null_shared), shuffled)}
        rows = truth[name].argsort()[sampled]
        choices = decide(panels[name], rows, posterior, utility)
        predictions = torch.stack(list(choices.values()))
        values, count = unique_pair_ocf(predictions, sampled.expand_as(predictions), masks, scorer, masks.device, 384)
        block = dict(zip(choices, values.cpu().numpy()))
        for arm, observed in block.items():
            measured.setdefault(arm, []).append(observed)
        # True row identity is used here and for OCF, after all corrections and choices are fixed.
        diagnostics = {arm: check_posterior(p, truth[name], visual) for arm, p in posterior.items()}
        save_block(name, {"ocf": ocf_statistics(block), "posterior": diagnostics, "temperature": temp[name],
                         "peer_recordings": peers, "unique_exact_pairs": count,
                         "changes_vs_v14": {arm: int((value != choices[ACTUAL]).sum()) for arm, value in choices.items()},
                         "local_peer_potential_correlation": float(torch.corrcoef(torch.stack((fields[name], shared)))[0, 1])})
        del posterior
    return measured


def main():
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("Run Task 3 whole-recording calibration on Kaggle T4 with official training input.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cuda:0")
    args = SimpleNamespace(seed=211, encoder_epochs=18, epochs=8, dimensions=128, visual_pairs=14336, query_batch=64)
    root = locate_data()
    catalog = load_catalog(root)
    gallery = load_images(root, device)
    masks = gallery >= 128
    signatures, energy = contour_signatures(gallery)
    scorer = official_scorer(root)
    fit_glyphs, temp_glyphs, held_glyphs = cohorts(len(gallery), device)

    geometry = gallery_geometry(masks, signatures, energy)
    approximate = similarity_from_signatures(signatures, energy)
    rng = torch.Generator(device=device).manual_seed(args.seed + 641)
    table_model, _, exact_pairs = fit_visual_metric(masks, geometry, approximate, fit_glyphs, scorer, rng, args)
    utility = gallery_utility(table_model, geometry)
    del geometry, approximate, table_model
    visual, shuffled, geometry_report = fit_visual_groups(signatures, energy, masks, fit_glyphs, utility)
    stage("visual_capacity_fit", started, exact_image_pairs=exact_pairs, **geometry_report)

    report = {"protocol": {"seed": SEED, "repeat_seed": REPEAT_SEED, "gallery": len(gallery), "ocf_glyphs_per_block": EVAL,
                           "temperature_subset_per_source_day": TEMP, "utility_fit_glyphs": len(fit_glyphs),
                           "actual_test_accessed": False, "joint_local_and_peer_fraction": 0.50, "fixed_utility_weight": 0.20,
                           "row_assumption": "Each recording has one trial from every official known gallery glyph. Shape capacities "
                                             "count known gallery glyphs in each shape group. Neither the observed anonymous rows nor "
                                             "output images are required to be one-to-one.",
                           "separation": "Each held recording is independently row shuffled. The same recording's row logits and other "
                                         "three days' unlabeled aggregate potentials are used with no row alignment or identity metadata. "
                                         "Temperatures use other days' independently measured inner repeats. Current OCF glyphs exclude "
                                         "the earlier exact exclusion and recent nomination-recheck scoring cohorts.",
                           "baseline": "mice1 uses the literal v14 predecessor v12 first seven selection. mice2 uses the actual equal "
                                       "directional contour and timed population gain blend over all known glyphs. Candidate lists and "
                                       "rendered images are fixed across posterior methods in each mouse."},
              "exact_visual_fit_pairs": exact_pairs, "visual_groups": geometry_report, "mice": {}}
    observations = {}
    order_rng = np.random.default_rng(SEED + 7)

    def save():
        report["seconds"] = time.monotonic() - started
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for mouse in ("mice1", "mice2"):
        training = grouped_blocks(root, catalog, mouse, device, len(gallery))
        inner_fit, inner_held, outer_fit, outer_held = nested_repeats(training, REPEAT_SEED)
        del training
        inner, inner_details = fit_models(inner_fit, gallery, signatures, energy, mouse, args)
        del inner_fit
        calibration = {name: predict_panel(inner, trials[temp_glyphs], gallery, name, mouse, args).logits
                       for name, trials in inner_held.items()}
        del inner, inner_held
        outer, outer_details = fit_models(outer_fit, gallery, signatures, energy, mouse, args)
        del outer_fit
        panels, truth, temp, probabilities = {}, {}, {}, {}
        for name, trials in outer_held.items():
            temp[name] = fit_temperature([(logits, temp_glyphs) for peer, logits in calibration.items() if peer != name])
            truth[name] = torch.as_tensor(order_rng.permutation(len(gallery)), device=device)
            panels[name] = predict_panel(outer, trials[truth[name]], gallery, name, mouse, args)
            probabilities[name] = glyph_posterior(panels[name].logits, temp[name]["temperature"])
        del outer, outer_held, calibration
        snapshot = {"inner_fit": inner_details, "outer_fit": outer_details, "sessions": {}}
        report["mice"][mouse] = snapshot

        def save_block(session, outcome):
            snapshot["sessions"][session] = outcome
            save()
            stage("scored_recording", started, mouse=mouse, session=session, ocf=outcome["ocf"])

        measured = score_recordings(panels, probabilities, truth, temp, visual, shuffled, utility, held_glyphs, masks, scorer, save_block)
        del panels, probabilities, truth
        snapshot["summary"] = ocf_statistics({name: np.stack(values) for name, values in measured.items()})
        for name, values in measured.items():
            observations.setdefault(name, []).extend(values)
        save()

    report["both_mice"] = ocf_statistics({name: np.stack(values) for name, values in observations.items()})
    save()
    stage("complete", started, both_mice=report["both_mice"])


if __name__ == "__main__":
    main()
