"""Strict Shor post-processing math primitives and implementation."""

import logging
from dataclasses import dataclass
from fractions import Fraction
from math import lcm

from .types import BenchmarkInstance

logger = logging.getLogger(__name__)


def continued_fraction_denominator(y: int, q: int, n: int) -> int:
    """Return denominator from bounded continued-fraction approximation.

    The input ratio is treated exactly as y / q. The approximation denominator
    is bounded by n - 1 to match the benchmark definition.

    Args:
        y: Measured integer in [0, q - 1].
        q: Power-of-two register size, q = 2^m.
        n: Modulus in the Shor instance.

    Returns:
        Candidate denominator q_i. When the approximated numerator is zero,
        this function returns 1 (uninformative sample).
    """

    max_denominator: int = n - 1
    if max_denominator < 1:
        raise ValueError("n must be >= 2")

    ratio: Fraction = Fraction(y, q)
    approx: Fraction = ratio.limit_denominator(max_denominator)
    if approx.numerator == 0:
        return 1
    return approx.denominator


def build_lcm_candidates(denominators: list[int], n: int) -> set[int]:
    """Build reachable LCM candidates with dynamic updates.

    Args:
        denominators (list[int]): Candidate denominators extracted from K samples.
        n (int): Modulus in the Shor instance.

    Returns:
        Set of candidate L values satisfying 1 < L < n.
    """

    reachable: set[int] = {1}
    for denominator in denominators:
        if denominator <= 1:
            continue

        next_values: set[int] = set(reachable)
        base_value: int
        for base_value in reachable:
            new_lcm: int = lcm(base_value, denominator)
            if 1 < new_lcm < n:
                next_values.add(new_lcm)
        reachable = next_values

    filtered: set[int] = {value for value in reachable if 1 < value < n}
    return filtered


def validated_orders(a: int, n: int, candidates: set[int]) -> list[int]:
    """Filter candidates by modular-exponentiation validation.

    Args:
        a: Base integer in the Shor instance.
        n: Modulus in the Shor instance.
        candidates: Candidate L values.

    Returns:
        Sorted validated candidate list V = {L | a^L == 1 (mod n)}.
    """

    validated: list[int] = []
    candidate: int
    for candidate in sorted(candidates):
        if pow(a, candidate, n) == 1:
            validated.append(candidate)
    return validated


def strict_predict_order(a: int, n: int, denominators: list[int]) -> int | None:
    """Run strict post-processing and return predicted order.

    Args:
        a: Base integer in the Shor instance.
        n: Modulus in the Shor instance.
        denominators: Continued-fraction denominators from K samples.

    Returns:
        Predicted order as min(V) when validation set is not empty,
        otherwise None.
    """

    l_candidates: set[int] = build_lcm_candidates(denominators=denominators, n=n)
    v_candidates: list[int] = validated_orders(a=a, n=n, candidates=l_candidates)
    if not v_candidates:
        return None
    return min(v_candidates)


@dataclass(frozen=True)
class DefaultStrictPostprocessor:
    """Default strict post-processor implementation."""

    instance: BenchmarkInstance

    def predict_order(self, samples_y: list[int]) -> int | None:
        """Predict order from K sampled y values.

        Args:
            samples_y: Integer measurement samples in [0, Q - 1].

        Returns:
            Predicted order or None.
        """

        q_values: list[int] = [
            continued_fraction_denominator(
                y=y,
                q=self.instance.q,
                n=self.instance.n,
            )
            for y in samples_y
        ]
        prediction: int | None = strict_predict_order(
            a=self.instance.a,
            n=self.instance.n,
            denominators=q_values,
        )
        logger.debug(
            f"Samples {samples_y} - q values {q_values} - prediction {prediction}"
        )

        return prediction
