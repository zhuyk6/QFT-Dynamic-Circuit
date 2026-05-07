"""Protocol definitions for pluggable Shor strict benchmark components."""

import random
from typing import Protocol


class ConditionalSampler(Protocol):
    """Sample integer y from a conditional distribution P(y | s)."""

    def sample_y(self, s: int, rng: random.Random) -> int:
        """Sample one measurement result for a given phase label s."""


class StrictPostprocessor(Protocol):
    """Predict order with strict Shor post-processing."""

    def predict_order(self, samples_y: list[int]) -> int | None:
        """Return predicted order, or None for null output."""


class ArithmeticStrictEstimator(Protocol):
    """Compute strict arithmetic-ideal success probability."""

    def estimate_p_ord_strict(self, k: int) -> float:
        """Return strict success probability for K samples."""
