"""Tests for Shor strict-success theory formulas."""

import pytest

from qft_dynamic.shor_benchmark.theory import (
    arithmetic_success_probability,
    depolarized_approx_success_probability,
)


def test_arithmetic_success_probability() -> None:
    """Check the arithmetic product formula for a composite order."""

    probability: float = arithmetic_success_probability(
        k_value=1,
        order_primes=[2, 3, 5],
    )

    assert probability == pytest.approx((1.0 / 2.0) * (2.0 / 3.0) * (4.0 / 5.0))


def test_depolarized_approx_success_probability_endpoints() -> None:
    """Check depolarized approximation endpoints for a small factorization."""

    order_primes: list[int] = [2, 3]
    arithmetic_probability: float = arithmetic_success_probability(
        k_value=2,
        order_primes=order_primes,
    )

    assert depolarized_approx_success_probability(
        k_value=2,
        lambda_value=0.0,
        order_primes=order_primes,
    ) == pytest.approx(arithmetic_probability)
    assert depolarized_approx_success_probability(
        k_value=2,
        lambda_value=1.0,
        order_primes=order_primes,
    ) == pytest.approx(0.0)


def test_depolarized_approx_success_probability_rejects_invalid_lambda() -> None:
    """Check that invalid depolarized mixture weights are rejected."""

    with pytest.raises(ValueError, match="lambda_value"):
        depolarized_approx_success_probability(
            k_value=1,
            lambda_value=1.1,
            order_primes=[2],
        )


def test_arithmetic_success_probability_rejects_invalid_order_primes() -> None:
    """Check that non-distinct order primes are rejected."""

    with pytest.raises(ValueError, match="order_primes"):
        arithmetic_success_probability(
            k_value=1,
            order_primes=[2, 2],
        )
