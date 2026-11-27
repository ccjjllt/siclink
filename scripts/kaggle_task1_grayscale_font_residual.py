"""Compare paired grayscale font translation at locked public Task 1 codepoints."""

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from glyph_atlas_prior import fit_atlas_prior
from glyph_data import grouped_blocks, load_catalog, load_images, locate_data
from glyph_font_style import fit_aligned_font_style
from kaggle_diagnostics import official_scorer
from kaggle_font_family_coverage import fit_bank
from kaggle_gallery_validation import align_training_glyphs, build_font_atlas, image_descriptors, standardized
from kaggle_task1_component_raster import picked_scope
from kaggle_task1_local_font_style import fresh_outer
from kaggle_task1_mice2_textverifier_recheck import tensor_hash
from kaggle_task1_microns_hanzi_recheck import fit_ocr_logits
from kaggle_task1_native_hanzi import response_views
from kaggle_task1_peer_transport import RASTER_FONT_SHA, RASTER_RECIPE, RASTER_SEED, choose_split
from kaggle_task1_raster_generation import (
    AUDIT_MARGIN, CONTEXT_GLYPHS, FIT_GLYPHS, FOCUS_MARGIN, MIN_SIMILARITY, boundary_metrics, descriptors,
    edge_pixels, load_variable_font, rasterize, top_shapes,
)
from kaggle_task1_stroke_edit_validation import exact_score
from kaggle_task1_textverifier import publish
from kaggle_v8_metric_validation import SESSIONS, summarize


SEED = 224467
HELD, CALIBRATION, ORACLE, CAL_EXACT, NEURAL = 320, 112, 96, 48, 48
FIT_CAP, UPDATES, BATCH = 1536, 400, 16
MOUSE = "mice2"
OUTPUT = Path("/kaggle/working/task1_grayscale_font_residual.json")
STOCK, NATIVE, PAIR, SHUFFLED, SELECTED = "global_style", "native_recipe", "paired_grayscale", "shuffled_residual", "calibrated_grayscale"


class GrayscaleResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.full = nn.Conv2d(1, 12, 3, padding=1)
        self.half_conv = nn.Conv2d(12, 16, 3, padding=1)
        self.context = nn.Sequential(nn.Conv2d(16, 24, 3, padding=2, dilation=2), nn.GELU(),
                                     nn.Conv2d(24, 24, 3, padding=1), nn.GELU())
        self.merge = nn.Conv2d(40, 16, 3, padding=1)
        self.local = nn.Conv2d(28, 12, 3, padding=1)
        self.output = nn.Conv2d(12, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, source):
        full = F.gelu(self.full(source))
        half = F.gelu(self.half_conv(F.avg_pool2d(full, 2)))
        context = self.context(F.avg_pool2d(half, 2))
        context = F.interpolate(context, scale_factor=2, mode="bilinear", align_corners=False)
        half = F.gelu(self.merge(torch.cat((half, context), dim=1)))
        half = F.interpolate(half, scale_factor=2, mode="bilinear", align_corners=False)
        change = 0.75 * self.output(F.gelu(self.local(torch.cat((full, half), dim=1)))).tanh()
        support = F.max_pool2d((source > 0.04).float(), 7, stride=1, padding=3)
        return (source + support * change).clamp(0, 1)


def cohorts(size):
    rng = np.random.default_rng(SEED)
    old = {"native_recipe_fit_and_scoring": np.random.default_rng(RASTER_SEED).permutation(size)[:CONTEXT_GLYPHS + FIT_GLYPHS + 160],
           "native_recipe_neural_scoring": choose_split(size, torch.device("cpu"))[1],
           "component_transfer": picked_scope(size)[1],
           "local_edge": fresh_outer(size, 15919)[1]}
    primary = tuple(value for name, value in old.items() if name != "local_edge")
    eligible = np.setdiff1d(np.arange(size), np.concatenate(primary))
    clean = np.setdiff1d(eligible, old["local_edge"])
    if len(clean) >= HELD + CALIBRATION:
        eligible = clean
    chosen = rng.choice(eligible, size=HELD + CALIBRATION, replace=False)
    held, calibration = np.sort(chosen[:HELD]), np.sort(chosen[HELD:])
    fit = np.setdiff1d(np.arange(size), chosen)
    overlap = {name: int(np.intersect1d(held, rows).size) for name, rows in old.items()}
    return tuple(torch.from_numpy(rows) for rows in (fit, calibration, held)), overlap


@torch.no_grad()
def match_columns(images, public, codepoints, audit, audit_codes):
    query = descriptors(images)
    main_scores, columns = top_shapes(query, public, 2)
    audit_scores, audit_columns = top_shapes(query, audit, 2)
    same = codepoints[columns[:, 0]] == audit_codes[audit_columns[:, 0]]
    eligible = same & (main_scores[:, 0] >= MIN_SIMILARITY) & (audit_scores[:, 0] >= MIN_SIMILARITY)
    reliable = eligible & ((main_scores[:, 0] - main_scores[:, 1]) >= FOCUS_MARGIN)
    reliable &= (audit_scores[:, 0] - audit_scores[:, 1]) >= AUDIT_MARGIN
    return columns[:, 0], reliable, main_scores[:, 0]


def matched_indices(columns, reliable, quality, rng):
    highest = torch.where(reliable)[0]
    highest = highest[quality[highest].argsort(descending=True)]
    unique = {}
    for row in highest.tolist():
        unique.setdefault(int(columns[row]), row)
    rows = np.array(list(unique.values()), dtype=np.int64)
    if len(rows) < 384:
        raise ValueError("The independent fonts should agree on at least 384 fit codepoints.")
    return torch.from_numpy(rng.choice(rows, min(FIT_CAP, len(rows)), replace=False))


def permute_residual(source, reference, rng):
    density = (source >= 0.5).flatten(1).float().mean(1)
    sorted_rows = density.argsort()
    donors = torch.empty_like(sorted_rows)
    for group in sorted_rows.tensor_split(10):
        order = group[rng.permutation(len(group))]
        donors[order] = order.roll(1)
    return (source + (reference - source)[donors]).clamp(0, 1)


def fit_renderer(source, reference):
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED + 1)
    model = GrayscaleResidual()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES, eta_min=0.0002)
    boundary = 1 + 2 * edge_pixels((reference[:, 0] * 255).round().to(torch.uint8)).float()[:, None]
    for _ in range(UPDATES):
        rows = torch.randperm(len(source), generator=generator)[:BATCH]
        output = model(source[rows])
        truth = reference[rows]
        loss = (boundary[rows] * F.smooth_l1_loss(output, truth, beta=0.12, reduction="none")).mean()
        # Penalize shape errors at two scales while keeping a grayscale target.
        for scale in (2, 4):
            loss = loss + 0.25 * F.l1_loss(F.avg_pool2d(output, scale), F.avg_pool2d(truth, scale))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
    return model.eval(), {"parameters": sum(value.numel() for value in model.parameters()), "final_loss": float(loss.detach())}


@torch.inference_mode()
def renders(columns, codepoints, base, native_content, supported, true_model, control_model, strength):
    unique, inverse = torch.unique(columns, return_inverse=True)
    code = codepoints[unique]
    valid = torch.tensor([int(point) in supported for point in code])
    native = base[unique].clone()
    native[valid] = rasterize(native_content, code[valid], RASTER_RECIPE)
    alpha = native[:, None].float() / 255
    mask = valid[:, None, None, None]
    true = torch.cat([true_model(batch) for batch in alpha.split(64)])
    control = torch.cat([control_model(batch) for batch in alpha.split(64)])
    true = torch.where(mask, true, alpha)
    control = torch.where(mask, control, alpha)
    selected = alpha.lerp(true, strength)
    return {STOCK: base[columns], NATIVE: native[inverse],
            **{name: (value[:, 0] * 255).round().to(torch.uint8)[inverse]
               for name, value in ((PAIR, true), (SHUFFLED, control), (SELECTED, selected))}}


def compare_rendered(variants, references, scorer, cache):
    expected = (references >= 128).numpy()
    binary = {name: value >= 128 for name, value in variants.items()}
    measured = {name: exact_score(value, expected, scorer, cache) for name, value in binary.items()}
    stats = {name: {"vs_global": summarize([values], [measured[STOCK]]),
                    "vs_native": summarize([values], [measured[NATIVE]]),
                    "exact_masks": int((binary[name].numpy() == expected).reshape(len(values), -1).all(1).sum()),
                    "mean_pixels_changed_vs_native": float((binary[name] != binary[NATIVE]).flatten(1).sum(1).float().mean())}
             for name, values in measured.items()}
    stats["correct_pair_vs_shuffle"] = summarize([measured[PAIR]], [measured[SHUFFLED]])
    return measured, stats


def main():
    if not OUTPUT.parent.is_dir():
        raise RuntimeError("Run on Kaggle CPU with the official labeled training data.")
    torch.set_num_threads(4)
    started = time.monotonic()
    cpu = torch.device("cpu")
    rng = np.random.default_rng(SEED + 3)
    root = locate_data()
    images = load_images(root, cpu)
    if images.shape != (2748, 64, 64) or images.dtype != torch.uint8:
        raise ValueError("Use the official labeled 64×64 grayscale raster gallery.")
    (fit, calibration, held), overlap = cohorts(len(images))
    labeled = images[fit]
    source, supported, variable = load_variable_font()
    if variable["sha256"] != RASTER_FONT_SHA:
        raise ValueError("The pinned native recipe uses an exact public Noto release.")
    atlas, provenance, codes = build_font_atlas(labeled[rng.choice(len(labeled), 128, replace=False)], cpu, return_codepoints=True)
    aligned, _ = align_training_glyphs(atlas, labeled)
    style, style_fit = fit_aligned_font_style(aligned, labeled, seed=SEED)
    base = torch.cat([(style.apply(part[:, None].float() / 255)[:, 0] * 255).round().to(torch.uint8)
                      for part in atlas.images.split(1024)])
    public = descriptors(base)
    audit, audit_codes, audit_metadata = fit_bank("simxihei", labeled, SEED + 1)
    audit = descriptors(audit.to(torch.uint8) * 255)
    fit_cols, confidence, quality = match_columns(labeled, public, codes, audit, audit_codes)
    supported_cols = torch.tensor([int(point) in supported for point in codes])
    selected = matched_indices(fit_cols, confidence & supported_cols[fit_cols], quality, rng)
    fit_columns = fit_cols[selected]
    paired_source = rasterize(source, codes[fit_columns], RASTER_RECIPE)[:, None].float() / 255
    paired_reference = labeled[selected, None].float() / 255
    report = {"seed": SEED, "mouse": MOUSE, "fit_glyphs": len(fit), "calibration_glyphs": len(calibration),
              "held_glyphs": len(held), "old_render_overlap": overlap, "formal_queries_loaded": False,
              "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "raster_gallery_sha256": tensor_hash(images),
              "codepoint_order_sha256": tensor_hash(codes), "old_global_render_sha256": tensor_hash(base),
              "noto": provenance, "audit_font": audit_metadata, "native_font": variable, "native_recipe": asdict(RASTER_RECIPE),
              "old_style": asdict(style), "old_style_fit": style_fit, "fit_pairs": len(selected), "updates_per_model": UPDATES,
              "protocol": "Fit only on mutually agreed Noto/SimXiHei codepoint proxies in the fit glyphs. Train the same "
                          "small full-grayscale residual model on real pairs and on residual targets permuted within ink bins. "
                          "Freeze candidate identities for the independent visual and neural endpoints. References of the held "
                          "queries are used for evaluation, and the neural shortlist and rendering arm are independent."}
    publish(report, OUTPUT, started, "renderer_training_started")
    model, report["paired_fit"] = fit_renderer(paired_source, paired_reference)
    control, report["shuffled_fit"] = fit_renderer(paired_source, permute_residual(paired_source, paired_reference, rng))
    publish(report, OUTPUT, started, "renderer_training_complete")
    scorer = official_scorer(root)

    def locked_rows(queries, sample_size):
        columns, agreement, _ = match_columns(images[queries], public, codes, audit, audit_codes)
        allowed = agreement & supported_cols[columns] & ~torch.isin(columns, fit_columns)
        eligible = torch.where(allowed)[0]
        amount = min(sample_size, len(eligible))
        rows = eligible[np.sort(rng.choice(len(eligible), amount, replace=False))]
        return queries[rows], columns[rows]

    def apply(columns, strength):
        return renders(columns, codes, base, source, supported, model, control, strength)

    calibration_rows, columns = locked_rows(calibration, CAL_EXACT)
    if not len(calibration_rows):
        raise RuntimeError("Need independent calibration glyphs with agreed public codepoints.")
    calibration_refs = images[calibration_rows]
    ref_masks = (calibration_refs >= 128).numpy()
    variants = apply(columns, 0.5)
    cache = {}
    settings = {0.0: variants[NATIVE], 0.5: variants[SELECTED], 1.0: variants[PAIR]}
    measured = {gain: exact_score(render >= 128, ref_masks, scorer, cache) for gain, render in settings.items()}
    strength = max(measured, key=lambda value: float(measured[value].mean()))
    report["fit_only_strength"] = {"selected": strength, "glyphs": len(calibration_refs),
                                   "settings": {str(gain): summarize([vals], [measured[0.0]]) for gain, vals in measured.items()}}
    model_path = OUTPUT.with_name("grayscale_renderers.pt")
    torch.save({"paired": model.state_dict(), "shuffled": control.state_dict(), "locked_recipe": asdict(RASTER_RECIPE),
                "calibrated_strength": strength}, model_path)
    report["renderer_sha256"] = sha256(model_path.read_bytes()).hexdigest()

    visual_rows, visual_columns = locked_rows(held, ORACLE)
    if not len(visual_rows):
        raise RuntimeError("Need independent held glyphs with agreed public codepoints.")
    reference = images[visual_rows]
    variants = apply(visual_columns, strength)
    visual_cache = {}
    _, report["visual_locked_codepoints"] = compare_rendered(variants, reference, scorer, visual_cache)
    edge = edge_pixels(reference)
    report["visual_grayscale"] = {name: {key: float(vals.mean()) for key, vals in boundary_metrics(render, reference, edge).items()}
                                  for name, render in variants.items()}
    report["visual_gallery_indices"] = visual_rows.tolist()
    report["visual_codepoints"] = codes[visual_columns].tolist()
    report["unique_visual_pairs"] = len(visual_cache)
    publish(report, OUTPUT, started, "visual_complete")

    selected = torch.from_numpy(np.sort(rng.choice(len(held), NEURAL, replace=False)))
    neural_rows = held[selected]
    groups = grouped_blocks(root, load_catalog(root), MOUSE, cpu, len(images))
    if tuple(groups) != SESSIONS:
        raise ValueError("Expected all four labeled training recordings.")
    views = response_views(groups, fit, neural_rows, rng)
    del groups
    ocr = fit_ocr_logits(labeled, atlas.images, views)
    prior = fit_atlas_prior(atlas, codes, labeled)
    fitting = image_descriptors((labeled >= 128).to(torch.uint8) * 255)
    scores = {}
    for day, (ridge, _, query) in views.items():
        contour = standardized(F.normalize(query @ (ridge @ fitting)) @ atlas.contours.T)
        scores[day] = prior.variants(0.5 * (contour + ocr[day]))["usage_unseen"]
    choices = {day: scores[day].argmax(1) for day in (SESSIONS[0], SESSIONS[2])}
    choices["pooled16_known_correspondence"] = torch.stack(tuple(scores.values())).mean(0).argmax(1)
    neural_reference = images[neural_rows]
    cache = {}
    same_day = {}
    report["neural_gallery_indices"] = neural_rows.tolist()
    report["neural_frozen_codepoints"] = {}
    for condition, fixed in choices.items():
        measurements, values = compare_rendered(apply(fixed, strength), neural_reference, scorer, cache)
        if condition in SESSIONS:
            for name, value in measurements.items():
                same_day.setdefault(name, []).append(value)
        report["neural_frozen_codepoints"][condition] = {"paired_ocf": values,
            "fit_codepoint_reuse_count": int(torch.isin(fixed, fit_columns).sum()), "supported_native_candidates": int(supported_cols[fixed].sum()),
            "codepoints": codes[fixed].tolist()}
        publish(report, OUTPUT, started, f"{condition}_scored")
    report["neural_same_day_four"] = {name: {"vs_global": summarize(values, same_day[STOCK]),
                                               "vs_native": summarize(values, same_day[NATIVE])}
                                        for name, values in same_day.items()}
    report["unique_neural_pairs"] = len(cache)
    publish(report, OUTPUT, started, "complete")


if __name__ == "__main__":
    main()
