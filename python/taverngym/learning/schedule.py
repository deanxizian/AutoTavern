"""Stage-local learning-rate schedules; endpoints are experiment choices."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LinearLearningRate:
    initial: float = 3e-4
    final: float = 3e-6

    def __call__(self, progress_remaining):
        progress = min(1.0, max(0.0, float(progress_remaining)))
        return self.final + (self.initial - self.final) * progress


@dataclass(frozen=True)
class WarmupLinearLearningRate:
    initial: float = 3e-6
    peak: float = 1e-5
    final: float = 5e-6
    warmup_fraction: float = 0.05

    def __post_init__(self):
        values = (self.initial, self.peak, self.final, self.warmup_fraction)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Learning-rate settings must be finite")
        if min(self.initial, self.final) < 0 or self.peak <= 0:
            raise ValueError("Learning rates must be nonnegative with a positive peak")
        if self.peak < max(self.initial, self.final):
            raise ValueError("Peak learning rate must be at least both endpoints")
        if not 0 < self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be strictly between zero and one")

    def __call__(self, progress_remaining):
        elapsed = 1 - min(1.0, max(0.0, float(progress_remaining)))
        if elapsed <= self.warmup_fraction:
            return self.initial + (self.peak - self.initial) * elapsed / self.warmup_fraction
        decay = (elapsed - self.warmup_fraction) / (1 - self.warmup_fraction)
        return self.peak + (self.final - self.peak) * decay


def learning_rate_schedule(config):
    kind = config.get("learning_rate_schedule", "linear")
    if kind == "linear":
        return LinearLearningRate(config["learning_rate_start"], config["learning_rate_end"])
    if kind == "warmup_linear":
        return WarmupLinearLearningRate(
            config["learning_rate_start"], config["learning_rate_peak"],
            config["learning_rate_end"], config["learning_rate_warmup_fraction"],
        )
    raise ValueError("learning_rate_schedule must be linear or warmup_linear")
