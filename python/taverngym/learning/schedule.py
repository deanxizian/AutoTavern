"""Linear learning-rate interpolation; endpoints are local experiment choices."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearLearningRate:
    initial: float = 3e-4
    final: float = 3e-6

    def __call__(self, progress_remaining):
        progress = min(1.0, max(0.0, float(progress_remaining)))
        return self.final + (self.initial - self.final) * progress
