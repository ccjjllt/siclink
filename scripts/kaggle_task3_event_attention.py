"""Compare causal population representations on genuine and resampled fine spike timing."""

import argparse
from itertools import permutations
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_mbr import contour_signatures, glyph_posterior
from kaggle_diagnostics import official_scorer
from kaggle_gallery_validation import standardized
from kaggle_posterior_utility_validation import fit_neural, v11_logits
from kaggle_task3_count_occupancy import paired_delta, posterior_metrics
from kaggle_task3_glyph_covariance import split_glyphs as prior_covariance_split
from kaggle_task3_metric_validation import unique_pair_ocf
from kaggle_task3_mice2_full_balance_replication import cohorts, early_cohorts
from kaggle_task3_mice2_utility_candidate import MOUSE, SHORTLIST, TABLE_SHA, validated_utility
from kaggle_task3_native_hanzi_nomination import fresh_glyphs
from kaggle_task3_rankband_ocf_validation import NeuralRows
from kaggle_task3_relational_consensus import choose_images
from kaggle_task3_v20_candidate_headroom import held_repeats
from kaggle_task3_v20_exact_utility import original_panel, paired_summary


SEED, BIN_MS, PER_WINDOW = 368443, 5, 5
WIDTH, EMBEDDING, MASK_FRACTION = 24, 64, 0.15
N_HELD, N_EXACT, MIX = 320, 64, 0.40
DAYS = ("session2", "session4")
HISTORY_MS = (5, 15, 35)
COARSE, SHUFFLED, TRUE, V20 = "coarse_25ms", "shuffled_5ms", "real_5ms", "v20"
MODES = (COARSE, SHUFFLED, TRUE)
OUTPUT = Path("/kaggle/working/task3_event_attention.json")


class EventReadout(nn.Module):
    def __init__(self, units, sessions, width):
        super().__init__()
        self.width = width
        self.population = nn.Linear(units, width, bias=False)
        self.recording = nn.Embedding(sessions, width)
        self.query = nn.Linear(width, width, bias=False)
        self.history_key = nn.Linear(width, width, bias=False)
        self.history_out = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.pool = nn.Sequential(nn.Linear(width * 3, EMBEDDING), nn.LayerNorm(EMBEDDING), nn.GELU())
        self.embedding = nn.Linear(EMBEDDING, EMBEDDING, bias=False)
        self.independent_repeat = nn.Linear(EMBEDDING, units * 2)
        # Kernel entries run from oldest to most recent. The history stops before the current bin.
        distance = torch.arange(12, 0, -1).float() * BIN_MS
        kernels = (-distance / torch.tensor(HISTORY_MS)[:, None]).exp()
        kernels /= kernels.sum(1, keepdim=True)
        self.register_buffer("kernels", kernels[:, None].repeat(width, 1, 1))

    def forward(self, spikes, day):
        instant = self.population(spikes) + 0.20 * self.recording(day)[:, None]
        past = F.pad(instant.transpose(1, 2), (self.kernels.shape[-1], 0))
        history = F.conv1d(past, self.kernels, groups=self.width)[..., :instant.shape[1]]
        history = history.reshape(len(spikes), self.width, len(HISTORY_MS), -1).permute(0, 3, 2, 1)
        comparison = (self.query(instant)[:, :, None] * self.history_key(history)).sum(-1) / self.width ** 0.5
        context = (comparison.softmax(dim=2)[..., None] * history).sum(2)
        tokens = F.gelu(self.norm(instant + self.history_out(context)))
        state = self.pool(torch.cat([part.mean(1) for part in tokens.split((10, 15, 15), dim=1)], dim=1))
        return F.normalize(self.embedding(state), dim=-1), self.independent_repeat(state).reshape(len(spikes), -1, 2)


def args_for_run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or a CUDA device")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--fit-glyphs", type=int, default=1728)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def fresh_cohort(size, fit_size):
    cpu = torch.device("cpu")
    visual, earlier, calibration, _ = cohorts(size, cpu)
    native, _, _ = fresh_glyphs(size, cpu)
    _, early = early_cohorts(size, cpu)
    _, covariance, _, _, _ = prior_covariance_split(size, cpu)
    gallery = torch.arange(size)
    excluded = torch.unique(torch.cat((visual, earlier, calibration, native, early, covariance)))
    eligible = gallery[~torch.isin(gallery, excluded)]
    rng = torch.Generator().manual_seed(SEED)
    if len(eligible) < N_HELD or fit_size > size - N_HELD:
        raise ValueError("Keep the fitted glyphs independent of the complete held ranking cohort.")
    held = eligible[torch.randperm(len(eligible), generator=rng)[:N_HELD]]
    references = dict(zip(DAYS, held[:len(DAYS) * N_EXACT].split(N_EXACT), strict=True))
    rest = gallery[~torch.isin(gallery, held)]
    fitting = rest[torch.randperm(len(rest), generator=rng)[:fit_size]]
    return fitting, held, references, earlier, len(eligible)


def windows(raw):
    return raw.unflatten(-1, (-1, 10, PER_WINDOW))


def coarse_counts(raw):
    return windows(raw).sum(-1).flatten(-2)


def compact_events(raw):
    fine = windows(raw)[..., 2:, :]
    if float(fine.max()) > 255:
        raise ValueError("Fine spike counts must fit the compact integer representation.")
    return fine.to(torch.uint8).contiguous()


def shuffled_events(events, seed):
    """Permute the five fine bins independently for every unit, trial and 25 ms window."""
    rng = torch.Generator().manual_seed(seed)
    all_orders = torch.tensor(list(permutations(range(PER_WINDOW))))
    flat = events.reshape(-1, *events.shape[-3:])
    arranged = []
    for batch in flat.split(256):
        serial = torch.randint(len(all_orders), batch.shape[:-1], generator=rng)
        permuted = batch.gather(-1, all_orders[serial])
        if not torch.equal(permuted.sum(-1), batch.sum(-1)):
            raise RuntimeError("The conditional shuffle must preserve every individual 25 ms count.")
        arranged.append(permuted)
    return torch.cat(arranged).reshape_as(events)


def calibration_targets(events, fit, device):
    # The response targets are always taken from a different measured repeat.
    groups = events.sum(-1, dtype=torch.float32)
    rate = groups[:, fit].mean((1, 2, 4)) / PER_WINDOW
    center, scale = rate.to(device), (0.04 + rate).sqrt().to(device)
    response = torch.stack((groups[..., :2].sum(-1), groups[..., 2:].sum(-1)), dim=-1)
    response = (response + 0.375).sqrt()
    mean = response[:, fit].mean((1, 2), keepdim=True)
    spread = response[:, fit].std((1, 2), unbiased=False, keepdim=True).clamp_min(0.15)
    return center, scale, ((response - mean) / spread).to(device)


def input_view(events, mode, day, center, scale):
    if mode == COARSE:
        data = (events.float() / PER_WINDOW).expand(*events.shape[:-1], PER_WINDOW)
    else:
        data = events.float()
    steps = data.flatten(-2).transpose(-1, -2)
    return (steps - center[day, None]) / scale[day, None]


def train_model(events, counts, fit, mode, center, scale, args, device):
    torch.manual_seed(SEED)
    units = events.shape[-3]
    model = EventReadout(units, len(events), args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.035)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0003)
    rng = torch.Generator().manual_seed(SEED + 11)
    history = []
    for epoch in range(args.epochs):
        losses = []
        for source in range(len(events)):
            order = fit[torch.randperm(len(fit), generator=rng)]
            for glyph in order.split(args.batch_size):
                n = len(glyph)
                next_day = (source + torch.randint(len(events), (n,), generator=rng)).to(device) % len(events)
                current_day = torch.full_like(next_day, source)
                anchor = torch.randint(events.shape[2], (n,), generator=rng)
                positive = (anchor + torch.randint(1, events.shape[2], (n,), generator=rng)) % events.shape[2]
                glyph, anchor, positive = glyph.to(device), anchor.to(device), positive.to(device)
                a = input_view(events[current_day, glyph, anchor], mode, current_day, center, scale)
                b = input_view(events[next_day, glyph, positive], mode, next_day, center, scale)
                a_mask, b_mask = (torch.rand((n, units), generator=rng) < MASK_FRACTION).to(device), \
                    (torch.rand((n, units), generator=rng) < MASK_FRACTION).to(device)
                z_a, pred_a = model(a.masked_fill(a_mask[:, None, :], 0), current_day)
                z_b, pred_b = model(b.masked_fill(b_mask[:, None, :], 0), next_day)
                relation = z_a @ z_b.T / 0.12
                target = torch.arange(n, device=device)
                contrast = 0.5 * (F.cross_entropy(relation, target) + F.cross_entropy(relation.T, target))
                error_a = F.smooth_l1_loss(pred_a, counts[next_day, glyph, positive], reduction="none")
                error_b = F.smooth_l1_loss(pred_b, counts[current_day, glyph, anchor], reduction="none")
                forecast = ((error_a * a_mask[..., None]).sum() + (error_b * b_mask[..., None]).sum())
                forecast /= 2 * (a_mask.sum() + b_mask.sum()).clamp_min(1)
                optimizer.zero_grad(set_to_none=True)
                (contrast + 0.25 * forecast).backward()
                optimizer.step()
                losses.append(torch.stack((contrast.detach(), forecast.detach())))
        scheduler.step()
        contrast_mean, forecast_mean = torch.stack(losses).mean(0).tolist()
        history.append({"contrast": contrast_mean, "independent_repeat_masked_loss": forecast_mean})
        print(f"{mode} epoch {epoch + 1}/{args.epochs}: contrast={contrast_mean:.4f} "
              f"held-unit repeat={forecast_mean:.4f}", flush=True)
    return model.eval(), history


@torch.no_grad()
def embed_day(model, events, mode, index, center, scale, batch_size):
    output = []
    for batch in events.split(batch_size):
        label = torch.full((len(batch),), index, device=events.device, dtype=torch.long)
        representation, _ = model(input_view(batch, mode, label, center, scale), label)
        output.append(representation)
    return torch.cat(output)


@torch.no_grad()
def gallery_logits(model, support, query, days, mode, center, scale, args):
    prototypes = []
    for index, block in enumerate(support):
        measured = embed_day(model, block.flatten(0, 1), mode, index, center, scale, args.batch_size)
        prototypes.append(measured.reshape(*block.shape[:2], -1).mean(1))
    scores = {}
    for day in DAYS:
        index = days.index(day)
        other = torch.stack([part for peer, part in enumerate(prototypes) if peer != index]).mean(0)
        template = F.normalize(prototypes[index].lerp(other, 0.30), dim=1)
        trial = embed_day(model, query[index], mode, index, center, scale, args.batch_size)
        scores[day] = standardized(trial @ template.T)
    return scores


def publish(report, started, stage, output):
    report.update(stage=stage, seconds=time.monotonic() - started)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{stage} after {report['seconds']:.1f}s", flush=True)


def main():
    args = args_for_run()
    if min(args.epochs, args.width, args.batch_size, args.fit_glyphs) < 2:
        raise ValueError("Use a shared positive fit size, batch size, epoch count and model width.")
    torch.set_num_threads(4)
    started = time.monotonic()
    device, cpu = torch.device(args.device), torch.device("cpu")
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Choose CPU or an available remote CUDA device.")
    root = locate_data()
    images_cpu = load_images(root, cpu)
    images = images_cpu.to(device)
    utility, temperatures, _ = validated_utility(images_cpu, device)
    fit, held, references, previous, eligible = fresh_cohort(len(images), args.fit_glyphs)
    recordings = grouped_blocks(root, load_catalog(root), MOUSE, cpu, len(images), bin_ms=BIN_MS)
    raw_support, raw_query, overlap = held_repeats(recordings, previous)
    del recordings
    days = tuple(raw_support)
    truth = held.to(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"mouse": MOUSE, "device": str(device), "seed": SEED, "fine_bin_ms": BIN_MS,
              "conditional_coarse_bin_ms": BIN_MS * PER_WINDOW, "causal_timescales_ms": HISTORY_MS,
              "time_window_ms": [50, 250], "shared_width": args.width, "shared_epochs": args.epochs,
              "shared_batch": args.batch_size, "fit_glyphs": len(fit), "new_held_glyphs": len(held),
              "eligible_new_glyphs": eligible, "training_days": days, "scored_days": DAYS,
              "repeats_excluded": overlap, "gallery_size": len(images), "candidate_budget": SHORTLIST,
              "exact_glyphs_per_day": N_EXACT, "posterior_mix": MIX, "utility_sha256": TABLE_SHA,
              "protocol": "All controls use the same causal history network, five-ms grid, training glyphs, "
              "source/positive repeat draws, unit masks and number of updates. Match an independent same-glyph "
              "repeat, with a masked-unit auxiliary count target from that different presentation. The real "
              "sequence, a fixed independent per-unit permutation inside 25-ms blocks, and a uniform envelope "
              "retain the same 25-ms spike totals. Conditional norms use shared fitting counts. Three support "
              "repeats per glyph/day provide known-gallery prototypes, independent of all anonymous held queries. "
              "Score each full 2748-glyph gallery, then compare v20 with fixed-seven posterior changes, equal "
              "seven re-nomination under the same utility, and stand-alone decisions. All spikes are from the "
              "official labeled training recordings.", "fit_curves": {}, "sessions": {}}
    publish(report, started, "split_ready", args.output)

    stock_support = {day: coarse_counts(raw).to(device) for day, raw in raw_support.items()}
    stock_model = fit_neural(stock_support, images, *contour_signatures(images), MOUSE, 211)
    stock = {day: v11_logits(stock_model, coarse_counts(raw_query[day][held]).to(device), day, images) for day in DAYS}
    del stock_support, stock_model
    support = torch.stack([compact_events(values) for values in raw_support.values()])
    queries = torch.stack([compact_events(raw_query[day][held, 0]) for day in days])
    del raw_support, raw_query
    center, scale, independent = calibration_targets(support, fit, device)
    publish(report, started, "v20_and_paired_repeats_ready", args.output)

    scores = {}
    for mode in MODES:
        if mode == COARSE:
            mode_support, mode_queries = (part.sum(-1, keepdim=True, dtype=torch.float32).to(device)
                                          for part in (support, queries))
        elif mode == SHUFFLED:
            mode_support = shuffled_events(support, SEED + 20).to(device)
            mode_queries = shuffled_events(queries, SEED + 21).to(device)
        else:
            mode_support, mode_queries = support.to(device), queries.to(device)
        model, history = train_model(mode_support, independent, fit, mode, center, scale, args, device)
        scores[mode] = gallery_logits(model, mode_support, mode_queries, days, mode, center, scale, args)
        report["fit_curves"][mode] = {"history": history, "parameters": sum(p.numel() for p in model.parameters()),
                                      "standalone_top32": {day: float((value.topk(32).indices == truth[:, None])
                                                                       .any(1).float().mean())
                                                           for day, value in scores[mode].items()}}
        print(json.dumps({"model": mode, **report["fit_curves"][mode]}), flush=True)
        del model, mode_support, mode_queries
        publish(report, started, f"{mode}_ranks_ready", args.output)

    scorer = official_scorer(root)
    raw_results = {}
    for day in DAYS:
        logits = stock[day]
        temperature = temperatures[day]["temperature"]
        rows = torch.where(torch.isin(truth, references[day].to(device)))[0]
        baseline_rank, baseline_nll = posterior_metrics(glyph_posterior(logits, temperature), truth)
        actions, original_mass, incumbent = original_panel(logits[rows], temperature, utility)
        fixed = {V20: original_mass}
        choices, metrics = {V20: incumbent}, {V20: baseline_rank}
        for mode in MODES:
            standalone = scores[mode][day]
            fused = logits + MIX * standalone
            mass = glyph_posterior(fused, temperature)
            standalone_rank, _ = posterior_metrics(glyph_posterior(standalone, temperature), truth)
            fused_rank, nll = posterior_metrics(mass, truth)
            fixed[mode + "_fixed7"] = mass[rows]
            new_actions, _, new_choice = original_panel(fused[rows], temperature, utility)
            choices[mode + "_renominated7"] = new_choice
            choices[mode + "_standalone"] = standalone[rows].argmax(1)
            metrics[mode] = {"standalone": standalone_rank, "fused": fused_rank,
                             "fused_identity_nll_vs_v20": paired_delta(nll.cpu(), baseline_nll.cpu()),
                             "v20_seven_seen_truth": int((actions == truth[rows, None]).any(1).sum()),
                             "renominated_seven_seen_truth": int((new_actions == truth[rows, None]).any(1).sum())}
        panel = NeuralRows(logits[rows], logits[rows], actions)
        replacements = choose_images(panel, fixed, utility)
        if not torch.equal(replacements[V20], incumbent):
            raise RuntimeError("The original seven-action decision must reproduce v20.")
        choices.update({arm: replacements[arm] for arm in fixed if arm != V20})
        selected = torch.stack(tuple(choices.values()))
        exact, unique = unique_pair_ocf(selected, truth[rows].expand_as(selected), images >= 128, scorer, device, 96)
        raw_results[day] = dict(zip(choices, exact.cpu(), strict=True))
        report["sessions"][day] = {"ranks": metrics, "held_exact_glyphs": truth[rows].tolist(), "exact_pairs": unique,
                                   "changed_vs_v20": {key: int((value != incumbent).sum()) for key, value in choices.items()},
                                   "exact_ocf": paired_summary(raw_results[day])}
        publish(report, started, f"{day}_official_ocf", args.output)
    pooled = {arm: torch.cat([values[arm] for values in raw_results.values()])
              for arm in next(iter(raw_results.values()))}
    report["pooled_ocf"] = paired_summary(pooled)
    report["real_vs_control"] = {policy: {arm: paired_delta(pooled[f"{TRUE}_{policy}"], pooled[f"{arm}_{policy}"])
                                           for arm in (COARSE, SHUFFLED)}
                                 for policy in ("fixed7", "renominated7", "standalone")}
    publish(report, started, "complete", args.output)
    print(json.dumps(report["pooled_ocf"]), flush=True)


if __name__ == "__main__":
    main()
