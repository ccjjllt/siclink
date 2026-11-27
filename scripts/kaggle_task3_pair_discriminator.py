"""Learn ordered OCF comparisons from real spike contrasts within v20's seven actions."""

from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_mbr import contour_signatures
from kaggle_diagnostics import official_scorer
from kaggle_posterior_utility_validation import fit_neural, v11_logits
from kaggle_task3_count_occupancy import paired_delta
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_mice2_full_balance_replication import cohorts
from kaggle_task3_mice2_utility_candidate import MOUSE, SHORTLIST, TABLE_SHA, validated_utility
from kaggle_task3_rankband_ocf_validation import scale_actions
from kaggle_task3_relational_consensus import UTILITY_WEIGHT
from kaggle_task3_repeat_posterior import expected_utility
from kaggle_task3_v20_candidate_headroom import DAYS, GLYPHS_PER_DAY, held_repeats
from kaggle_task3_v20_exact_utility import held_panels, original_panel, paired_summary


SEED = 364381
FIT_GLYPHS, TRAIN_EXACT_PER_DAY, SCORE_PER_DAY = 1792, 16, 64
TOP_UNITS, WIDTH = 64, 40
WINDOWS = ((50, 100), (100, 175), (175, 250))
EPOCHS, EXACT_EPOCHS, BATCH, OCF_SCALE = 7, 3, 128, 10.0
V20, DIRECT, SHUFFLED, SUMMARY = "v20", "spike_contrast", "activity_matched_shuffle", "summary_only"
OUTPUT = Path("/kaggle/working/task3_pair_discriminator.json")
LEFT, RIGHT = torch.triu_indices(SHORTLIST, SHORTLIST, 1)
INCIDENCE = F.one_hot(LEFT, SHORTLIST).float() - F.one_hot(RIGHT, SHORTLIST).float()


@dataclass
class Panel:
    candidates: torch.Tensor
    reference: torch.Tensor
    observed: torch.Tensor
    template: torch.Tensor
    signed: torch.Tensor
    context: torch.Tensor
    stock: torch.Tensor

    def inputs(self, arm, rows, permuted=None):
        signed = self.signed[rows]
        if arm == SUMMARY:
            width = self.observed.shape[-1]
            raw = signed.new_zeros((*signed.shape[:2], width))
        else:
            observed = self.observed[rows if permuted is None else permuted[rows], None]
            rates = self.template[rows]
            left, right = rates[:, LEFT], rates[:, RIGHT]
            raw = ((left - right) * (observed - 0.5 * (left + right))).clamp(-8, 8)
        return torch.cat((signed, raw), dim=-1), self.context[rows]


class PairJudge(nn.Module):
    def __init__(self, signed_width, context_width):
        super().__init__()
        self.signed = nn.Linear(signed_width, WIDTH, bias=False)
        self.context = nn.Sequential(nn.Linear(context_width, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH))
        self.out = nn.Linear(WIDTH, 1, bias=False)

    def forward(self, signed, context):
        interaction = torch.tanh(self.signed(signed)) * (0.5 + self.context(context).sigmoid())
        return 0.65 * signed[..., 0] + self.out(interaction).squeeze(-1)


def timing(counts):
    bins = counts.unflatten(-1, (-1, 10))
    return torch.stack([(bins[..., start // 25:end // 25].sum(-1) + 0.375).sqrt()
                        for start, end in WINDOWS], dim=-1)


@torch.no_grad()
def supported_templates(support, query, fit):
    """Estimate consistent unit reliability and means from independent support spikes."""
    centers, observations, unit_strength = {}, {}, []
    for day, counts in support.items():
        transformed = timing(counts)
        train = transformed[fit]
        mean = train.mean((0, 1))
        noise = train.var(1, unbiased=True).mean(0).add(0.12).sqrt()
        centers[day] = (transformed.mean(1) - mean) / noise
        observations[day] = (timing(query[day][:, 0]) - mean) / noise
        stable = centers[day][fit].var(0, unbiased=False) - 1 / counts.shape[1]
        unit_strength.append(stable.mean(dim=1))
    good = torch.stack(unit_strength).mean(0).topk(TOP_UNITS).indices.sort().values
    shared = torch.stack([value[:, good] for value in centers.values()]).mean(0)
    return {day: (observations[day][:, good].flatten(1), (0.75 * centers[day][:, good] + 0.25 * shared).flatten(1))
            for day in observations}, good.tolist()


@torch.no_grad()
def make_panel(logits, truth, observed, templates, ink, utility, temperature):
    candidates, posterior, winner = original_panel(logits, temperature, utility)
    anchor = scale_actions(logits.gather(1, candidates), 0.30)
    excess = expected_utility(posterior, candidates, utility) - utility.mean(dim=1)[candidates]
    effect = scale_actions(excess, 0.50)
    score = (1 - UTILITY_WEIGHT) * anchor + UTILITY_WEIGHT * effect
    stock = score.argmax(1)
    if not torch.equal(winner, candidates.gather(1, stock[:, None])[:, 0]):
        raise ValueError("Unchanged anchor and utility must reproduce v20 exactly.")
    candidate_mass = posterior.gather(1, candidates).sqrt() * 5
    values = torch.stack((score, anchor, effect, candidate_mass, ink[candidates] * 4), dim=-1)
    left, right = values[:, LEFT], values[:, RIGHT]
    signed = left - right
    a, b = candidates[:, LEFT], candidates[:, RIGHT]
    pair_utility = (utility[a, b] + utility[b, a])[..., None] / 50
    distance = (templates[a] - templates[b]).square().mean(-1, keepdim=True).sqrt()
    context = torch.cat((signed.abs(), 0.5 * (left + right), pair_utility, distance), dim=-1)
    return Panel(candidates, truth, observed, templates[candidates], signed, context, stock)


def stack_panels(panels):
    return Panel(*(torch.cat([getattr(panel, name) for panel in panels]) for name in Panel.__dataclass_fields__))


def matched_shuffle(panel, rows_per_day, strata, seed):
    """Preserve day and approximate activity while breaking trial linkage."""
    rng = torch.Generator().manual_seed(seed)
    permutation = torch.arange(len(panel.reference))
    for rows in torch.arange(len(panel.reference)).split(rows_per_day):
        activity = panel.observed[rows].sum(1)
        for stratum in rows[activity.argsort()].tensor_split(strata):
            order = stratum[torch.randperm(len(stratum), generator=rng)]
            permutation[order] = order.roll(1)
    if (permutation == torch.arange(len(permutation))).any() or len(permutation.unique()) != len(permutation):
        raise ValueError("Training shuffle must be a derangement within day/activity groups.")
    return permutation


def train_phase(model, panel, target, arm, permutation, epochs, lr, seed):
    rng = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.035)
    margin = (target[:, LEFT] - target[:, RIGHT]) / OCF_SCALE
    history = []
    for _ in range(epochs):
        total = 0.0
        for rows in torch.randperm(len(target), generator=rng).split(BATCH):
            signed, context = panel.inputs(arm, rows, permutation)
            prediction = model(signed, context)
            loss = F.smooth_l1_loss(prediction, margin[rows], beta=0.5)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        history.append(total / len(target))
    return history


@torch.no_grad()
def choices(model, panel, arm):
    comparison = torch.cat([model(*panel.inputs(arm, rows)) for rows in torch.arange(len(panel.reference)).split(BATCH)])
    return (comparison @ INCIDENCE).argmax(1), comparison


def fit_pair_judges(training, proxy, exact, measured):
    permute_train = matched_shuffle(training, FIT_GLYPHS, 16, SEED + 17)
    permute_exact = matched_shuffle(exact, TRAIN_EXACT_PER_DAY, TRAIN_EXACT_PER_DAY // 2, SEED + 18)
    models, fit_report = {}, {}
    for arm in (DIRECT, SHUFFLED, SUMMARY):
        torch.manual_seed(SEED)
        model = PairJudge(training.signed.shape[-1] + training.observed.shape[-1], training.context.shape[-1])
        permutation = permute_train if arm == SHUFFLED else None
        warm = train_phase(model, training, proxy, arm, permutation, EPOCHS, 0.0015, SEED + 20)
        permutation = permute_exact if arm == SHUFFLED else None
        fine = train_phase(model, exact, measured, arm, permutation, EXACT_EPOCHS, 0.00035, SEED + 21)
        models[arm] = model.eval()
        fit_report[arm] = {"proxy_train_loss": warm, "exact_train_loss": fine,
                           "parameters": sum(parameter.numel() for parameter in model.parameters())}
        print(json.dumps({"fitted": arm, **fit_report[arm]}), flush=True)
    return models, fit_report


def publish(report, started, stage):
    report.update(stage=stage, seconds=time.monotonic() - started)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{stage} after {report['seconds']:.1f}s", flush=True)


def main():
    torch.set_num_threads(4)
    started = time.monotonic()
    device = torch.device("cpu")
    root = locate_data()
    images = load_images(root, device)
    masks = images >= 128
    ink = masks.float().mean((1, 2))
    utility, temperatures, _ = validated_utility(images, device)
    visual, held, calibration, _ = cohorts(len(images), device)
    prior_exact, _ = held_panels(held, torch.cat((visual, calibration)), GLYPHS_PER_DAY)
    available = held[~torch.isin(held, torch.cat(tuple(prior_exact.values())))]
    rng = torch.Generator().manual_seed(SEED)
    selected = available[torch.randperm(len(available), generator=rng)[:len(DAYS) * SCORE_PER_DAY]]
    references = dict(zip(DAYS, selected.split(SCORE_PER_DAY), strict=True))
    gallery = torch.arange(len(images))
    independent = gallery[~torch.isin(gallery, held)]
    independent = independent[torch.randperm(len(independent), generator=rng)]
    proxy_fit = independent[:FIT_GLYPHS]
    exact_fit = independent[FIT_GLYPHS:FIT_GLYPHS + 4 * TRAIN_EXACT_PER_DAY]
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, device, len(images), bin_ms=25)
    support, query, repeat_overlap = held_repeats(recordings, held)
    del recordings
    report = {"mouse": MOUSE, "seed": SEED, "candidate_budget": SHORTLIST, "pair_count": len(LEFT),
              "query_days": list(DAYS), "paired_glyphs_per_day": SCORE_PER_DAY, "fit_glyphs": len(proxy_fit),
              "exact_fit_glyphs": len(exact_fit), "held_glyph_group": len(held), "repeat_overlap": repeat_overlap,
              "utility_sha256": TABLE_SHA, "windows_ms": WINDOWS, "fit_epochs": [EPOCHS, EXACT_EPOCHS],
              "protocol": "Use the fixed v20 seven actions for every arm. Predict all signed OCF margins from "
              "observed spike contrasts around two independently fitted response templates. Match network size, data "
              "and training schedule for the real neural contrast, within-day activity-matched shuffled training "
              "linkage, and baseline summaries alone. Warm up with the inherited table, then fit on official OCF "
              "from a disjoint 64-glyph cohort. Score all seven actions on 128 glyphs with held query repeats and "
              "no prior headroom endpoints. The stock model and candidate response templates use the three other "
              "labeled repetitions per glyph. All spikes come from the official training files.", "sessions": {}}
    publish(report, started, "split_ready")

    neural = fit_neural(support, images, *contour_signatures(images), MOUSE, 211)
    templates, units = supported_templates(support, query, proxy_fit)
    report["retained_units"] = units
    fit_panels, exact_panels, test = [], [], {}
    for index, day in enumerate(support):
        order = torch.randperm(len(images), generator=rng)
        values = v11_logits(neural, query[day][order], day, images)[order.argsort()]
        observed, expected = templates[day]
        temperature = temperatures[day]["temperature"]
        exact_labels = exact_fit[index * TRAIN_EXACT_PER_DAY:(index + 1) * TRAIN_EXACT_PER_DAY]
        fit_panels.append(make_panel(values[proxy_fit], proxy_fit, observed[proxy_fit], expected, ink, utility, temperature))
        exact_panels.append(make_panel(values[exact_labels], exact_labels, observed[exact_labels], expected, ink, utility,
                                       temperature))
        if day in DAYS:
            labels = references[day]
            test[day] = make_panel(values[labels], labels, observed[labels], expected, ink, utility, temperature)
    del neural, support, query, templates
    training, exact = stack_panels(fit_panels), stack_panels(exact_panels)
    proxy = utility[training.candidates, training.reference[:, None]]
    report["proxy_fit_trials"] = len(proxy)
    publish(report, started, "candidate_evidence_ready")

    scorer = official_scorer(root)
    measured, seen_pairs = unique_pair_ocf(exact.candidates, exact.reference[:, None], masks, scorer, device, 96)
    report["exact_fit_pairs"] = seen_pairs
    publish(report, started, "exact_fit_labels_ready")
    models, report["fits"] = fit_pair_judges(training, proxy, exact, measured)
    publish(report, started, "comparators_ready")

    results = {}
    for day, panel in test.items():
        values, unique = unique_pair_ocf(panel.candidates, panel.reference[:, None], masks, scorer, device, 96)
        decisions, margins = {V20: panel.stock}, {V20: panel.signed[..., 0]}
        for arm, model in models.items():
            decisions[arm], margins[arm] = choices(model, panel, arm)
        base = values[torch.arange(len(values)), panel.stock]
        oracle = values.amax(1)
        quality = values[:, LEFT] - values[:, RIGHT]
        consequential = quality.abs() >= 3
        measured = {arm: values[torch.arange(len(values)), position] for arm, position in decisions.items()}
        results[day] = measured
        report["sessions"][day] = {
            "held_glyphs": panel.reference.tolist(), "distinct_ocf_pairs": unique,
            "oracle_ocf": float(oracle.mean()), "oracle_gap": paired_delta(oracle, base),
            "ocf": paired_summary(measured),
            "switches": {arm: int((position != panel.stock).sum()) for arm, position in decisions.items()},
            "exact_pair_accuracy_margin_ge_3": {arm: float(((predicted * quality)[consequential] > 0).float().mean())
                                                for arm, predicted in margins.items()},
        }
        publish(report, started, f"{day}_official_ocf")
        print(json.dumps(report["sessions"][day]["ocf"]), flush=True)
    pooled = {arm: torch.cat([values[arm] for values in results.values()]) for arm in next(iter(results.values()))}
    report["pooled_ocf"] = paired_summary(pooled)
    report["spike_contrast_vs_controls"] = {arm: paired_delta(pooled[DIRECT], pooled[arm]) for arm in (SUMMARY, SHUFFLED)}
    publish(report, started, "complete")
    print(json.dumps(report["pooled_ocf"]), flush=True)


if __name__ == "__main__":
    main()
