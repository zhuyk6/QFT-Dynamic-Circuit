"""Unit tests for Shor strict baseline components."""

import random

from shor_benchmark.baselines import (
    ArithmeticIdealEstimator,
    FiniteQIdealSampler,
    finite_q_ideal_probability,
)
from shor_benchmark.types import BenchmarkInstance


def _empirical_distribution_for_sampler(
    sampler: FiniteQIdealSampler,
    s: int,
    seed: int,
    num_samples: int,
) -> list[float]:
    """Estimate the output distribution of a sampler for one fixed s.

    Args:
        sampler: Finite-Q ideal sampler under test.
        s: Phase label in [0, r - 1].
        seed: Random seed for reproducibility.
        num_samples: Number of Monte Carlo samples.

    Returns:
        Empirical probability vector over y in [0, Q - 1].
    """

    rng: random.Random = random.Random(seed)
    counts: list[int] = [0 for _ in range(sampler.instance.q)]

    sample_index: int
    for sample_index in range(num_samples):
        _ignored_sample_index: int = sample_index
        sampled_y: int = sampler.sample_y(s=s, rng=rng)
        counts[sampled_y] += 1

    empirical: list[float] = [count / num_samples for count in counts]
    return empirical


def test_arithmetic_ideal_matches_closed_form_for_r_equals_four() -> None:
    """Arithmetic ideal should match product formula for r=4."""

    instance: BenchmarkInstance = BenchmarkInstance(n=15, a=2, r=4, m=4)
    estimator: ArithmeticIdealEstimator = ArithmeticIdealEstimator(instance=instance)

    assert estimator.estimate_p_ord_strict(k=1) == 0.5
    assert estimator.estimate_p_ord_strict(k=2) == 0.75
    assert estimator.estimate_p_ord_strict(k=4) == 0.9375


def test_finite_q_ideal_distribution_is_normalized_for_fixed_s() -> None:
    """Finite-Q ideal probabilities over y should sum to one for fixed s."""

    instance: BenchmarkInstance = BenchmarkInstance(n=15, a=2, r=4, m=4)

    total_probability: float = sum(
        finite_q_ideal_probability(y=y, s=1, instance=instance)
        for y in range(instance.q)
    )

    assert abs(total_probability - 1.0) < 1e-9


def test_finite_q_ideal_sampler_matches_closed_form_distribution() -> None:
    """Bitwise finite-Q sampler should match the closed-form distribution."""

    instance: BenchmarkInstance = BenchmarkInstance(n=15, a=2, r=4, m=2)
    sampler: FiniteQIdealSampler = FiniteQIdealSampler(
        instance=instance,
        sample_method="bitwise",
    )
    num_samples: int = 20000
    empirical: list[float] = _empirical_distribution_for_sampler(
        sampler=sampler,
        s=1,
        seed=123,
        num_samples=num_samples,
    )
    expected: list[float] = [
        finite_q_ideal_probability(y=y, s=1, instance=instance)
        for y in range(instance.q)
    ]

    y_value: int
    for y_value in range(instance.q):
        assert abs(empirical[y_value] - expected[y_value]) < 0.02


def test_finite_q_ideal_enumeration_sampler_matches_closed_form_distribution() -> None:
    """Enumerated finite-Q sampler should match the closed-form distribution."""

    instance: BenchmarkInstance = BenchmarkInstance(n=15, a=2, r=4, m=2)
    sampler: FiniteQIdealSampler = FiniteQIdealSampler(
        instance=instance,
        sample_method="enumerate",
    )
    num_samples: int = 20000
    empirical: list[float] = _empirical_distribution_for_sampler(
        sampler=sampler,
        s=1,
        seed=321,
        num_samples=num_samples,
    )
    expected: list[float] = [
        finite_q_ideal_probability(y=y, s=1, instance=instance)
        for y in range(instance.q)
    ]

    y_value: int
    for y_value in range(instance.q):
        assert abs(empirical[y_value] - expected[y_value]) < 0.02


def test_finite_q_sampling_methods_agree_for_medium_m() -> None:
    """Bitwise and enumerated finite-Q samplers should agree for medium m."""

    instance: BenchmarkInstance = BenchmarkInstance(n=21, a=2, r=6, m=6)
    bitwise_sampler: FiniteQIdealSampler = FiniteQIdealSampler(
        instance=instance,
        sample_method="bitwise",
    )
    enumerate_sampler: FiniteQIdealSampler = FiniteQIdealSampler(
        instance=instance,
        sample_method="enumerate",
    )
    num_samples: int = 30000

    bitwise_empirical: list[float] = _empirical_distribution_for_sampler(
        sampler=bitwise_sampler,
        s=1,
        seed=11,
        num_samples=num_samples,
    )
    enumerate_empirical: list[float] = _empirical_distribution_for_sampler(
        sampler=enumerate_sampler,
        s=1,
        seed=29,
        num_samples=num_samples,
    )

    y_value: int
    for y_value in range(instance.q):
        assert abs(bitwise_empirical[y_value] - enumerate_empirical[y_value]) < 0.02


def test_finite_q_ideal_sampler_handles_large_m_without_enumerating_q() -> None:
    """Finite-Q sampler should support large m by bitwise sampling."""

    instance: BenchmarkInstance = BenchmarkInstance(n=21, a=2, r=6, m=40)
    sampler: FiniteQIdealSampler = FiniteQIdealSampler(
        instance=instance,
        sample_method="bitwise",
    )
    rng: random.Random = random.Random(7)

    sampled_y: int = sampler.sample_y(s=1, rng=rng)

    assert 0 <= sampled_y < instance.q
