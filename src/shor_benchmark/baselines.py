"""Baseline samplers and arithmetic-ideal estimator for strict benchmark."""

import logging
import random
from dataclasses import dataclass
from math import cos, pi, sin
from typing import Literal

from shor_benchmark.types import BenchmarkInstance

LOGGER: logging.Logger = logging.getLogger(__name__)


def finite_q_ideal_probability(y: int, s: int, instance: BenchmarkInstance) -> float:
    """Return finite-Q ideal P(y | s) for inverse QFT.

    Args:
        y: Measured integer in [0, Q - 1].
        s: Phase label in [0, r - 1].
        instance: Benchmark instance data.

    Returns:
        Ideal conditional probability.
    """

    q_value: int = instance.q
    delta: float = (s / instance.r) - (y / q_value)

    denominator: float = sin(pi * delta)
    if abs(denominator) < 1e-12:
        return 1.0

    numerator: float = sin(pi * q_value * delta)
    probability: float = (numerator * numerator) / (
        (q_value * q_value) * (denominator * denominator)
    )
    return probability


@dataclass(frozen=True)
class FiniteQIdealSampler:
    """Conditional sampler from finite-Q ideal distributions.

    This sampler supports two exact implementations of the same finite-Q ideal
    distribution P(y | s):

    - ``"bitwise"``: sample output bits using the exact semiclassical inverse
      QFT measurement rule. This costs O(m) per sample and is suitable for
      large m.
    - ``"enumerate"``: materialize the full discrete distribution over
      ``y in [0, Q - 1]`` and sample from it using those probabilities as
      weights. This costs O(Q) per sample and is useful for debugging and
      cross-checking small instances.

    Args:
        instance: Benchmark instance data.
        sample_method: Exact sampling strategy for the finite-Q ideal model.
    """

    instance: BenchmarkInstance
    sample_method: Literal["bitwise", "enumerate"] = "bitwise"

    def _enumerated_weights_for_s(self, s: int) -> list[float]:
        """Build unnormalized exact weights for all y at fixed s.

        Args:
            s: Phase label in [0, r - 1].

        Returns:
            One exact probability weight per y in [0, Q - 1].
        """

        q_value: int = self.instance.q
        weights: list[float] = [
            finite_q_ideal_probability(y=y, s=s, instance=self.instance)
            for y in range(q_value)
        ]
        return weights

    def _sample_y_by_enumeration(self, s: int, rng: random.Random) -> int:
        """Sample y by explicit distribution enumeration.

        Args:
            s: Phase label in [0, r - 1].
            rng: Random generator.

        Returns:
            Sampled integer y in [0, Q - 1].
        """

        population: range = range(self.instance.q)
        weights: list[float] = self._enumerated_weights_for_s(s=s)
        sampled_y: int = rng.choices(population=population, weights=weights, k=1)[0]
        return sampled_y

    def _sample_bit(
        self,
        phase_exponent: float,
        rng: random.Random,
    ) -> int:
        """Sample one output bit from a single-qubit phase state.

        Args:
            phase_exponent: Phase exponent theta in exp(2 pi i theta).
            rng: Random generator.

        Returns:
            Measured bit after a Hadamard-basis measurement.
        """

        phase_mod_1: float = phase_exponent % 1.0
        probability_zero: float = 0.5 + 0.5 * cos(2.0 * pi * phase_mod_1)
        probability_zero = min(1.0, max(0.0, probability_zero))
        sampled_bit: int = 0 if rng.random() < probability_zero else 1
        return sampled_bit

    def _sample_y_by_bitwise_iqft(self, s: int, rng: random.Random) -> int:
        """Sample y using the semiclassical inverse-QFT measurement rule.

        Args:
            s: Phase label in [0, r - 1].
            rng: Random generator.

        Returns:
            Sampled integer y in [0, Q - 1].
        """

        phase_fraction: float = s / self.instance.r
        sampled_y: int = 0
        place_value: int = 1

        # Classical feed-forward phase accumulated from already sampled lower
        # significance bits in the semiclassical inverse-QFT picture.
        phase_correction: float = 0.0

        qubit_index: int
        for qubit_index in range(self.instance.m - 1, -1, -1):
            phase_exponent: float = (
                phase_fraction * (2**qubit_index)
            ) - phase_correction
            sampled_bit: int = self._sample_bit(
                phase_exponent=phase_exponent,
                rng=rng,
            )
            sampled_y += sampled_bit * place_value
            place_value *= 2

            phase_correction = (phase_correction + (sampled_bit / 2.0)) / 2.0

        return sampled_y

    def sample_y(self, s: int, rng: random.Random) -> int:
        """Sample y from finite-Q ideal P(y | s).

        Args:
            s: Phase label in [0, r - 1].
            rng: Random generator.

        Returns:
            Sampled integer y in [0, Q - 1].

        Raises:
            ValueError: If `sample_method` is unsupported.
        """
        match self.sample_method:
            case "enumerate":
                if self.instance.q > 2**10:
                    q_value: int = self.instance.q
                    LOGGER.warning(
                        "Enumeration-based finite-Q sampling with Q=%d may be slow "
                        "and memory-intensive; consider sample_method='bitwise' "
                        "for larger instances.",
                        q_value,
                    )

                return self._sample_y_by_enumeration(s=s, rng=rng)
            case "bitwise":
                return self._sample_y_by_bitwise_iqft(s=s, rng=rng)
            case method:
                raise ValueError(f"unsupported finite-Q sample method: {method}")


@dataclass(frozen=True)
class UniformSampler:
    """Uniform random sampler over y in [0, Q - 1]."""

    instance: BenchmarkInstance

    def sample_y(self, s: int, rng: random.Random) -> int:
        """Sample y uniformly; s is ignored by design."""

        _ignored_s: int = s
        sampled_y: int = rng.randrange(self.instance.q)
        return sampled_y


def _distinct_prime_factors(value: int) -> list[int]:
    """Return distinct prime factors of a positive integer."""

    if value <= 0:
        raise ValueError("value must be positive")

    factors: list[int] = []
    n: int = value

    if n % 2 == 0:
        factors.append(2)
        while n % 2 == 0:
            n //= 2

    p: int = 3
    while p * p <= n:
        if n % p == 0:
            factors.append(p)
            while n % p == 0:
                n //= p
        p += 2

    if n > 1:
        factors.append(n)

    return factors


@dataclass(frozen=True)
class ArithmeticIdealEstimator:
    """Closed-form arithmetic-ideal estimator.

    P_arith^(K) = product_{p | r} (1 - p^{-K})
    where p runs over distinct prime factors of r.
    """

    instance: BenchmarkInstance

    def estimate_p_ord_strict(self, k: int) -> float:
        """Estimate arithmetic-ideal strict success probability for K samples."""

        if k <= 0:
            raise ValueError("k must be positive")

        factors: list[int] = _distinct_prime_factors(self.instance.r)
        probability: float = 1.0
        prime_factor: int
        for prime_factor in factors:
            probability *= 1.0 - (prime_factor ** (-k))
        return probability
