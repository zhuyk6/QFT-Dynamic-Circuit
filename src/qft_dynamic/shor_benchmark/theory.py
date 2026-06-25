"""Pure theory formulas for Shor strict order recovery."""

from math import comb
from typing import Sequence


def arithmetic_success_probability(
    k_value: int,
    order_primes: Sequence[int],
) -> float:
    """Compute the arithmetic strict-success probability.

    This implements
    ``P_arith^(K) = prod_{p|r} (1 - p^{-K})``.

    Args:
        k_value: Number of independent samples.
        order_primes: Distinct prime factors of the order ``r``.

    Returns:
        Arithmetic-limit strict order-recovery success probability.

    Raises:
        ValueError:
            - If ``k_value`` is negative.
            - If ``order_primes`` contains non-distinct elements.
    """

    if k_value < 0:
        raise ValueError(f"k_value must be non-negative, got {k_value}.")
    if k_value == 0:
        return 0.0

    if len(set(order_primes)) != len(order_primes):
        raise ValueError("order_primes must be distinct.")

    probability: float = 1.0
    prime: int
    for prime in order_primes:
        probability *= 1.0 - float(prime ** (-k_value))
    return probability


def depolarized_approx_success_probability(
    k_value: int,
    lambda_value: float,
    order_primes: Sequence[int],
) -> float:
    """Compute the approximate depolarized strict-success probability.

    This implements the binomial-mixture form from
    ``docs/order_finding_success_theory.md`` Section 5.1:

    ``sum_j binom(K,j) (1-lambda)^j lambda^(K-j) P_arith^(j)``.

    Args:
        k_value: Number of independent samples.
        lambda_value: Depolarized noise mixture weight in ``[0, 1]``.
        order_primes: Distinct prime factors of the order ``r``.

    Returns:
        Approximate strict order-recovery success probability.

    Raises:
        ValueError:
            - If ``k_value`` is negative.
            - If ``lambda_value`` is outside the interval [0, 1].
            - If ``order_primes`` contains non-distinct elements.
    """

    if k_value < 0:
        raise ValueError(f"k_value must be non-negative, got {k_value}.")
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError(
            f"lambda_value must be in the interval [0, 1], got {lambda_value}."
        )
    if k_value == 0:
        return 0.0

    probability: float = 0.0
    ideal_count: int
    for ideal_count in range(1, k_value + 1):
        mixture_weight: float = (
            float(comb(k_value, ideal_count))
            * ((1.0 - lambda_value) ** ideal_count)
            * (lambda_value ** (k_value - ideal_count))
        )
        probability += mixture_weight * arithmetic_success_probability(
            k_value=ideal_count,
            order_primes=order_primes,
        )
    return probability
