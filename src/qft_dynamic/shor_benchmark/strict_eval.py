"""Monte Carlo evaluator for strict Shor benchmark."""

import logging
import random

from .protocols import ConditionalSampler, StrictPostprocessor
from .samplers import ArithmeticIdealEstimator
from .types import (
    ArithmeticCurveResult,
    BenchmarkInstance,
    StrictCurveResult,
    StrictMetrics,
)

logger = logging.getLogger(__name__)


def evaluate_strict_metrics_for_k(
    instance: BenchmarkInstance,
    sampler: ConditionalSampler,
    postprocessor: StrictPostprocessor,
    k: int,
    m_mc: int,
    rng: random.Random,
) -> StrictMetrics:
    """Evaluate strict metrics for one K with Monte Carlo resampling.

    Args:
        instance: Benchmark instance.
        sampler: Conditional sampler P(y | s).
        postprocessor: Strict post-processing implementation.
        k: Number of quantum samples used by one algorithm run.
        m_mc: Number of Monte Carlo trials.
        rng: Random generator.

    Returns:
        StrictMetrics for the given K.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if m_mc <= 0:
        raise ValueError("m_mc must be positive")

    success_count: int = 0
    wrong_count: int = 0
    null_count: int = 0

    trial_index: int
    for trial_index in range(m_mc):
        _ignored_trial_index: int = trial_index

        logger.debug(f"Trial {trial_index + 1}/{m_mc} for K={k}")

        samples_s: list[int] = [rng.randrange(instance.r) for _ in range(k)]
        samples_y: list[int] = [sampler.sample_y(s=s_i, rng=rng) for s_i in samples_s]

        logger.debug(f"Samples s: {samples_s}")
        logger.debug(f"Samples y: {samples_y}")

        predicted_order: int | None = postprocessor.predict_order(samples_y=samples_y)
        if predicted_order is None:
            null_count += 1
        elif predicted_order == instance.r:
            success_count += 1
        else:
            wrong_count += 1

    p_ord_strict: float = success_count / m_mc
    p_wrong: float = wrong_count / m_mc
    p_null: float = null_count / m_mc

    metrics: StrictMetrics = StrictMetrics(
        p_ord_strict=p_ord_strict,
        p_wrong=p_wrong,
        p_null=p_null,
    )
    return metrics


def evaluate_strict_curve(
    instance: BenchmarkInstance,
    sampler: ConditionalSampler,
    postprocessor: StrictPostprocessor,
    k_list: list[int],
    m_mc: int,
    seed: int,
) -> StrictCurveResult:
    """Evaluate strict metrics curves for all K values."""

    rng: random.Random = random.Random(seed)
    metrics_by_k: dict[int, StrictMetrics] = {}

    k_value: int
    for k_value in k_list:
        metrics_by_k[k_value] = evaluate_strict_metrics_for_k(
            instance=instance,
            sampler=sampler,
            postprocessor=postprocessor,
            k=k_value,
            m_mc=m_mc,
            rng=rng,
        )

    result: StrictCurveResult = StrictCurveResult(metrics_by_k=metrics_by_k)
    return result


def evaluate_arithmetic_curve(
    estimator: ArithmeticIdealEstimator,
    k_list: list[int],
) -> ArithmeticCurveResult:
    """Evaluate arithmetic-ideal strict curve for all K values."""

    p_ord_strict_by_k: dict[int, float] = {}
    k_value: int
    for k_value in k_list:
        p_ord_strict_by_k[k_value] = estimator.estimate_p_ord_strict(k=k_value)

    result: ArithmeticCurveResult = ArithmeticCurveResult(
        p_ord_strict_by_k=p_ord_strict_by_k
    )
    return result
