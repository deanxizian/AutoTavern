"""Explicit local scheduling choice: linear interpolation of the paper's LR endpoints."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearLearningRate:
    initial: float = 6e-6
    final: float = 5e-6

    def __call__(self, progress_remaining):
        progress = min(1.0, max(0.0, float(progress_remaining)))
        return self.final + (self.initial - self.final) * progress
