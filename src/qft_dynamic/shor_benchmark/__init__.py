"""Shor benchmark package."""

from .samplers import (
    ArithmeticIdealEstimator,
    FiniteQIdealSampler,
    HistogramSampler,
    UniformSampler,
)
from .schemas import StrictBenchmarkResultFileModel
from .strict_eval import (
    evaluate_arithmetic_curve,
    evaluate_strict_curve,
)
from .strict_postprocess import DefaultStrictPostprocessor
from .theory import (
    arithmetic_success_probability,
    depolarized_approx_success_probability,
)
from .types import (
    ArithmeticCurveResult,
    BenchmarkInstance,
    CombinedCurveResult,
    StrictCurveResult,
    StrictMetrics,
)

__all__: list[str] = [
    "ArithmeticCurveResult",
    "ArithmeticIdealEstimator",
    "BenchmarkInstance",
    "CombinedCurveResult",
    "DefaultStrictPostprocessor",
    "FiniteQIdealSampler",
    "HistogramSampler",
    "StrictCurveResult",
    "StrictBenchmarkResultFileModel",
    "StrictMetrics",
    "UniformSampler",
    "arithmetic_success_probability",
    "depolarized_approx_success_probability",
    "evaluate_arithmetic_curve",
    "evaluate_strict_curve",
]
