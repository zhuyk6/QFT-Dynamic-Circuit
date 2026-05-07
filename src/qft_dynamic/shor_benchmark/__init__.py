"""Shor benchmark package."""

from .config import ShorBenchmarkPaths, resolve_shor_benchmark_paths
from .samplers import (
    ArithmeticIdealEstimator,
    FiniteQIdealSampler,
    HistogramSampler,
    UniformSampler,
)
from .schemas import HistogramFileModel, StrictBenchmarkResultFileModel
from .strict_eval import (
    evaluate_arithmetic_curve,
    evaluate_strict_curve,
)
from .strict_postprocess import DefaultStrictPostprocessor
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
    "HistogramFileModel",
    "StrictCurveResult",
    "StrictBenchmarkResultFileModel",
    "StrictMetrics",
    "ShorBenchmarkPaths",
    "UniformSampler",
    "evaluate_arithmetic_curve",
    "evaluate_strict_curve",
    "resolve_shor_benchmark_paths",
]
