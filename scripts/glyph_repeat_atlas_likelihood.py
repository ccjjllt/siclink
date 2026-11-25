"""Repeat-aware predictive scoring of public glyph code templates."""

from dataclasses import dataclass

import torch

from glyph_predictive_likelihood import PredictiveLikelihood, stabilized_rate, stabilized_spikes


@dataclass
class RepeatAtlasLikelihood:
    calibration: PredictiveLikelihood
    noise_pseudocount: float = 12.0
    disagreement_weight: float = 0.20

    @torch.no_grad()
    def logits(self, observed, expected):
        """Keep code residual uncertainty common to both halves of the repeats."""
        repeats = observed.shape[1]
        if repeats < 2 or repeats % 2:
            raise ValueError("Provide two equally sized, disjoint repeat groups")
        calibration = self.calibration
        templates = stabilized_rate(expected) + calibration.offset
        center = templates.mean(dim=0)
        templates = (templates - center) / calibration.scale
        template_square = templates.square()
        trials = (stabilized_spikes(observed) - center) / calibration.scale
        left, right = trials.chunk(2, dim=1)
        left_mean, right_mean = left.mean(dim=1), right.mean(dim=1)
        shot = calibration.trial_covariance.diagonal().clamp_min(1e-4)
        code = calibration.model_covariance.diagonal().clamp_min(0) + 0.05
        precision = (shot / repeats + code).reciprocal()
        mean = 0.5 * (left_mean + right_mean)
        fixed = (mean * precision) @ templates.T - 0.5 * (template_square * precision).sum(dim=1)

        half_size = left.shape[1]
        if half_size > 1:
            shrinkage = (half_size - 1) / (half_size - 1 + self.noise_pseudocount)
            left_var = (left.var(dim=1, unbiased=True) / shot).clamp(0.25, 4.0)
            right_var = (right.var(dim=1, unbiased=True) / shot).clamp(0.25, 4.0)
            left_precision = half_size / (shot * (1 + shrinkage * (left_var - 1)))
            right_precision = half_size / (shot * (1 + shrinkage * (right_var - 1)))
            total = left_precision + right_precision
            inverse_mean = (left_precision * left_mean + right_precision * right_mean) / total
            inverse_precision = (total.reciprocal() + code).reciprocal()
            inverse = (inverse_mean * inverse_precision) @ templates.T - 0.5 * inverse_precision @ template_square.T
        else:
            # One observation per half gives no within-half variance estimate.
            inverse = fixed

        half_gap = (0.5 * (left_mean - right_mean) * precision) @ templates.T
        invariant = fixed - self.disagreement_weight * half_gap.abs()
        return {"fixed_diagonal": fixed, "split_inverse_variance": inverse, "repeat_invariant": invariant}
