"""Shor benchmark package."""

from shor_benchmark.config import ShorBenchmarkPaths, resolve_shor_benchmark_paths
from shor_benchmark.samplers import (
    ArithmeticIdealEstimator,
    FiniteQIdealSampler,
    HistogramSampler,
    UniformSampler,
)
from shor_benchmark.schemas import HistogramFileModel, StrictBenchmarkResultFileModel
from shor_benchmark.strict_eval import (
    evaluate_arithmetic_curve,
    evaluate_strict_curve,
)
from shor_benchmark.strict_postprocess import DefaultStrictPostprocessor
from shor_benchmark.types import (
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
