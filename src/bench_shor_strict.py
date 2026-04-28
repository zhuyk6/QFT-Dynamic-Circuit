"""Strict Shor benchmark runner.

This script computes strict metrics for finite-Q ideal and uniform baselines via
Monte Carlo, and arithmetic-ideal strict success via closed-form expression.
"""

import argparse
import json
import logging
from math import gcd
from pathlib import Path

from shor_benchmark.baselines import (
    ArithmeticIdealEstimator,
    FiniteQIdealSampler,
    UniformSampler,
)
from shor_benchmark.strict_eval import (
    ArithmeticCurveResult,
    CombinedStrictBenchmarkResult,
    StrictCurveResult,
    evaluate_arithmetic_curve,
    evaluate_strict_curve,
)
from shor_benchmark.strict_postprocess import DefaultStrictPostprocessor
from shor_benchmark.types import BenchmarkInstance, StrictMetrics

logger = logging.getLogger(__name__)


def parse_k_list(value: str) -> list[int]:
    """Parse comma-separated K values such as '1,2,4,8,16'."""

    raw_tokens: list[str] = [
        token.strip() for token in value.split(",") if token.strip()
    ]
    if not raw_tokens:
        raise argparse.ArgumentTypeError("k-list cannot be empty")

    parsed: list[int] = []
    token: str
    for token in raw_tokens:
        try:
            k_value: int = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid K value: {token}") from exc
        if k_value <= 0:
            raise argparse.ArgumentTypeError("all K values must be positive")
        parsed.append(k_value)

    unique_sorted: list[int] = sorted(set(parsed))
    return unique_sorted


def _strict_curve_to_dict(curve: StrictCurveResult) -> dict[str, dict[str, float]]:
    """Convert strict curve result to JSON-serializable dict."""

    result: dict[str, dict[str, float]] = {}
    k_value: int
    metrics: StrictMetrics
    for k_value, metrics in sorted(curve.metrics_by_k.items()):
        result[str(k_value)] = {
            "p_ord_strict": metrics.p_ord_strict,
            "p_wrong": metrics.p_wrong,
            "p_null": metrics.p_null,
        }
    return result


def _arithmetic_curve_to_dict(curve: ArithmeticCurveResult) -> dict[str, float]:
    """Convert arithmetic curve result to JSON-serializable dict."""

    result: dict[str, float] = {}
    k_value: int
    probability: float
    for k_value, probability in sorted(curve.p_ord_strict_by_k.items()):
        result[str(k_value)] = probability
    return result


def run_strict_benchmark(
    instance: BenchmarkInstance,
    k_list: list[int],
    m_mc: int,
    seed: int,
) -> CombinedStrictBenchmarkResult:
    """Run strict benchmark for ideal, uniform, and arithmetic baselines."""

    if gcd(instance.a, instance.n) != 1:
        raise ValueError("a and n must be coprime")
    if instance.r <= 0:
        raise ValueError("r must be positive")

    postprocessor: DefaultStrictPostprocessor = DefaultStrictPostprocessor(
        instance=instance
    )

    ideal_sampler: FiniteQIdealSampler = FiniteQIdealSampler(instance=instance)
    uniform_sampler: UniformSampler = UniformSampler(instance=instance)
    arithmetic_estimator: ArithmeticIdealEstimator = ArithmeticIdealEstimator(
        instance=instance
    )

    ideal_curve: StrictCurveResult = evaluate_strict_curve(
        instance=instance,
        sampler=ideal_sampler,
        postprocessor=postprocessor,
        k_list=k_list,
        m_mc=m_mc,
        seed=seed,
    )
    uniform_curve: StrictCurveResult = evaluate_strict_curve(
        instance=instance,
        sampler=uniform_sampler,
        postprocessor=postprocessor,
        k_list=k_list,
        m_mc=m_mc,
        seed=seed + 1,
    )
    arithmetic_curve: ArithmeticCurveResult = evaluate_arithmetic_curve(
        estimator=arithmetic_estimator,
        k_list=k_list,
    )

    combined: CombinedStrictBenchmarkResult = CombinedStrictBenchmarkResult(
        ideal=ideal_curve,
        uniform=uniform_curve,
        arithmetic=arithmetic_curve,
    )
    return combined


def setup_logging(verbose: bool = False):
    """Setup logging config."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """CLI entry point."""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Shor strict benchmark (ideal, uniform, arithmetic ideal)."
    )
    parser.add_argument("--n", type=int, required=True, help="Modulus N")
    parser.add_argument("--a", type=int, required=True, help="Base a")
    parser.add_argument("--r", type=int, required=True, help="Order r")
    parser.add_argument(
        "--m", type=int, required=True, help="Control register qubit count"
    )
    parser.add_argument(
        "--k-list",
        type=parse_k_list,
        default=[1, 2, 4, 8, 16],
        help="Comma-separated K values (default: 1,2,4,8,16)",
    )
    parser.add_argument(
        "--m-mc",
        type=int,
        default=5000,
        help="Monte Carlo trials for each K (default: 5000)",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    args: argparse.Namespace = parser.parse_args()
    setup_logging(args.verbose)

    logger.debug(args)

    instance: BenchmarkInstance = BenchmarkInstance(
        n=args.n,
        a=args.a,
        r=args.r,
        m=args.m,
    )

    result: CombinedStrictBenchmarkResult = run_strict_benchmark(
        instance=instance,
        k_list=args.k_list,
        m_mc=args.m_mc,
        seed=args.seed,
    )

    output_payload: dict[str, object] = {
        "instance": {
            "n": instance.n,
            "a": instance.a,
            "r": instance.r,
            "m": instance.m,
            "q": instance.q,
        },
        "k_list": args.k_list,
        "m_mc": args.m_mc,
        "seed": args.seed,
        "baselines": {
            "ideal": {
                "metrics_by_k": _strict_curve_to_dict(result.ideal),
            },
            "uniform": {
                "metrics_by_k": _strict_curve_to_dict(result.uniform),
            },
            "arithmetic": {
                "p_ord_strict_by_k": _arithmetic_curve_to_dict(result.arithmetic),
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(output_payload, output_file, indent=2)


if __name__ == "__main__":
    main()
