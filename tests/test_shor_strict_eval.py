"""Unit tests for strict Monte Carlo evaluators."""

import random

from shor_benchmark.baselines import ArithmeticIdealEstimator, UniformSampler
from shor_benchmark.strict_eval import (
    evaluate_arithmetic_curve,
    evaluate_strict_metrics_for_k,
)
from shor_benchmark.strict_postprocess import DefaultStrictPostprocessor
from shor_benchmark.types import BenchmarkInstance


def test_strict_metrics_probability_mass_is_conserved() -> None:
    """Strict metrics should satisfy p_ord_strict + p_wrong + p_null = 1."""

    instance: BenchmarkInstance = BenchmarkInstance(n=15, a=2, r=4, m=4)
    sampler: UniformSampler = UniformSampler(instance=instance)
    postprocessor: DefaultStrictPostprocessor = DefaultStrictPostprocessor(
        instance=instance
    )
    rng: random.Random = random.Random(123)

    metrics = evaluate_strict_metrics_for_k(
        instance=instance,
        sampler=sampler,
        postprocessor=postprocessor,
        k=3,
        m_mc=400,
        rng=rng,
    )

    total: float = metrics.p_ord_strict + metrics.p_wrong + metrics.p_null
    assert abs(total - 1.0) < 1e-12


def test_evaluate_arithmetic_curve_returns_all_k_values() -> None:
    """Arithmetic curve should return one value for each requested K."""

    instance: BenchmarkInstance = BenchmarkInstance(n=15, a=2, r=4, m=4)
    estimator: ArithmeticIdealEstimator = ArithmeticIdealEstimator(instance=instance)

    curve = evaluate_arithmetic_curve(estimator=estimator, k_list=[1, 2, 4, 8])

    assert sorted(curve.p_ord_strict_by_k.keys()) == [1, 2, 4, 8]
