"""Strict Shor benchmark runner.

This script computes strict metrics for finite-Q ideal and uniform baselines via
Monte Carlo, arithmetic-ideal strict success via closed-form expression, and
optionally a histogram-driven simulation baseline.
"""

import argparse
import logging
from pathlib import Path

from qft_dynamic.shor_benchmark.samplers import (
    ArithmeticIdealEstimator,
    FiniteQIdealSampler,
    HistogramSampler,
    UniformSampler,
)
from qft_dynamic.shor_benchmark.schemas import StrictBenchmarkResultFileModel
from qft_dynamic.shor_benchmark.strict_eval import (
    evaluate_arithmetic_curve,
    evaluate_strict_curve,
)
from qft_dynamic.shor_benchmark.strict_postprocess import DefaultStrictPostprocessor
from qft_dynamic.shor_benchmark.types import (
    ArithmeticCurveResult,
    BenchmarkInstance,
    CombinedCurveResult,
    StrictCurveResult,
)

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


def run_strict_benchmark(
    instance: BenchmarkInstance,
    k_list: list[int],
    m_mc: int,
    seed: int,
    histogram_path: Path | None = None,
) -> CombinedCurveResult:
    """Run strict benchmark for ideal, uniform, arithmetic, and simulation baselines.

    Args:
        instance: Benchmark instance.
        k_list: Sample-count values K.
        m_mc: Monte Carlo trial count per K.
        seed: Random seed.
        histogram_path: Optional JSON file containing per-s histograms.

    Returns:
        Combined strict benchmark results.
    """

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
    simulation_curve: StrictCurveResult | None = None
    if histogram_path is not None:
        simulation_sampler: HistogramSampler = HistogramSampler.from_file(
            histogram_path=histogram_path,
            instance=instance,
        )
        simulation_curve = evaluate_strict_curve(
            instance=instance,
            sampler=simulation_sampler,
            postprocessor=postprocessor,
            k_list=k_list,
            m_mc=m_mc,
            seed=seed + 2,
        )

    combined: CombinedCurveResult = CombinedCurveResult(
        ideal=ideal_curve,
        uniform=uniform_curve,
        arithmetic=arithmetic_curve,
        simulation=simulation_curve,
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
        "--simulation-histograms",
        type=Path,
        default=None,
        help="Optional JSON file with per-s simulation histograms",
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

    result: CombinedCurveResult = run_strict_benchmark(
        instance=instance,
        k_list=args.k_list,
        m_mc=args.m_mc,
        seed=args.seed,
        histogram_path=args.simulation_histograms,
    )

    output_payload: StrictBenchmarkResultFileModel = StrictBenchmarkResultFileModel(
        instance=instance,
        k_list=args.k_list,
        m_mc=args.m_mc,
        seed=args.seed,
        result=result,
        simulation_histogram_file=(
            str(args.simulation_histograms)
            if args.simulation_histograms is not None
            else None
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_payload.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
