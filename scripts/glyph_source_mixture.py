"""Mix labeled recording styles using unlabeled split-repeat predictive evidence."""

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import torch

from glyph_hierarchical_gallery import (HierarchicalGallery, _repeat_covariance, _symmetric,
                                        _whiten_covariance, fit_hierarchical_gallery, response_features)
from glyph_retrieval import _template_logits, _whiten
from kaggle_diagnostics import standardize


@dataclass
class SourceStyleMixture:
    hierarchy: HierarchicalGallery
    names: tuple[str, ...]
    templates: torch.Tensor
    diagnostics: dict

    @torch.no_grad()
    def logits(self, query, *, query_repeat_mix=0.5, temperature=0.12, row_batch_size=128):
        """Return uniform and evidence-weighted style scores for a four-repeat block.

        Conditional evidence compares a uniform-glyph marginal likelihood
        from all four trials with those from the six possible two-trial
        subsets. The contrast between half means is independent of style
        under the shared Gaussian repeat noise model and drops out.
        """
        if query.ndim != 3 or query.shape[1] != 4 or not len(query):
            raise ValueError("Source-style evidence needs four repeats per query glyph.")
        if not 0 <= query_repeat_mix <= 1 or temperature <= 0 or row_batch_size < 1:
            raise ValueError("Use valid repeat noise, evidence temperature and batch settings.")
        core = self.hierarchy
        values, _ = standardize(response_features(query, core.view), core.normalization)
        repeat_noise = core.repeat_covariance
        if query_repeat_mix:
            measured = _whiten_covariance(_repeat_covariance(values), core.noise_whitener)
            repeat_noise = repeat_noise.lerp(measured, query_repeat_mix)

        nuisance = core.session_covariance + core.posterior_covariance
        nuisance.diagonal().add_(core.predictive_floor)
        full_factor = torch.linalg.cholesky(_symmetric(nuisance + repeat_noise / 4))
        half_factor = torch.linalg.cholesky(_symmetric(nuisance + repeat_noise / 2))
        shared_features = _whiten(values.flatten(0, 1), core.noise_whitener).reshape_as(values)
        full_query = _whiten(shared_features.mean(dim=1), full_factor)
        subset = torch.as_tensor(tuple(combinations(range(4), 2)), device=values.device)
        half = shared_features[:, subset].mean(dim=2)
        half_query = _whiten(half.flatten(0, 1), half_factor).reshape_as(half)

        complete, evidence = [], []
        for templates in self.templates:
            full_centers = _whiten(templates, full_factor)
            half_centers = _whiten(templates, half_factor)
            full_scores = _template_logits(full_query, full_centers)
            complete.append(full_scores)
            half_evidence = []
            for partial in half_query.split(row_batch_size):
                partial_scores = _template_logits(partial.flatten(0, 1), half_centers)
                half_evidence.append(partial_scores.logsumexp(dim=1).reshape(len(partial), -1).mean(dim=1))
            evidence.append(full_scores.logsumexp(dim=1) - torch.cat(half_evidence))

        # A fixed shared-style component and a uniform floor stop a single
        # favorable cohort estimate from explaining the entire new recording.
        difference = torch.stack([item - evidence[0] for item in evidence[1:]])
        relative = difference.clamp(-2, 2).mean(dim=1)
        source_probability = (relative / temperature).softmax(dim=0)
        uniform_weights = relative.new_full((len(complete),), 0.55 / len(relative))
        uniform_weights[0] = 0.45
        weighted = uniform_weights.clone()
        weighted[1:] = 0.5 * weighted[1:] + 0.275 * source_probability

        def marginal(weights):
            prior = weights.log()
            combined = complete[0] + prior[0]
            for scores, mass in zip(complete[1:], prior[1:]):
                combined = torch.logaddexp(combined, scores + mass)
            return combined

        diagnostics = {"half_subsets": len(subset), "relative_held_repeat_evidence": dict(zip(self.names[1:], relative.tolist())),
                       "evidence_weights": dict(zip(self.names, weighted.tolist())),
                       "uniform_weights": dict(zip(self.names, uniform_weights.tolist()))}
        return {"uniform_source_mixture": marginal(uniform_weights),
                "predictive_source_mixture": marginal(weighted)}, diagnostics


@torch.no_grad()
def fit_source_mixture(training: Mapping[str, torch.Tensor], *, view="total") -> SourceStyleMixture:
    """Fit shared stable responses and shrink labeled recording-specific deviations.

    Session residuals are damped by finite-repeat noise and shared-template
    uncertainty. All session styles use the same predictive covariance, so
    the relative split likelihood cannot simply favor a broader noise model.
    """
    core = fit_hierarchical_gallery(training, view=view)
    common = core.posterior_templates
    templates = [common]
    ratio = {}
    for name, counts in training.items():
        normalized, _ = standardize(response_features(counts, view))
        measured = _whiten(normalized.mean(dim=1), core.noise_whitener)
        covariance = _whiten_covariance(_repeat_covariance(normalized), core.noise_whitener)
        residual_noise = (core.session_covariance + covariance / counts.shape[1] + core.posterior_covariance)
        residual_noise.diagonal().add_(core.predictive_floor)
        factor = torch.linalg.cholesky(_symmetric(residual_noise))
        style = torch.cholesky_solve(_symmetric(core.session_covariance), factor)
        observed = measured - common
        adjustment = observed @ style
        templates.append(common + adjustment)
        ratio[name] = float(adjustment.norm() / observed.norm().clamp_min(1e-8))

    return SourceStyleMixture(core, ("shared", *training), torch.stack(templates),
                              {"source_residual_rms_ratio": ratio, "hierarchy": core.diagnostics})
