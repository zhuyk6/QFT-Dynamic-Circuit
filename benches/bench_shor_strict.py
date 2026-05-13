"""Strict Shor benchmark runner.

This script computes strict metrics for finite-Q ideal and uniform baselines via
Monte Carlo, arithmetic-ideal strict success via closed-form expression, and
optionally a histogram-driven simulation baseline.
"""

import logging
from pathlib import Path
from typing import Annotated

import typer

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

app = typer.Typer()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging config."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


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


@app.command()
def main(
    n: Annotated[int, typer.Argument(help="Modulus N")],
    a: Annotated[int, typer.Argument(help="Base a")],
    r: Annotated[int, typer.Argument(help="Order r")],
    m: Annotated[int, typer.Argument(help="Control register qubit count")],
    output: Annotated[Path, typer.Argument(help="Output JSON path")],
    k_list: Annotated[
        list[int],
        typer.Option(help="K values"),
    ] = [1, 2, 4, 8, 16],
    m_mc: Annotated[int, typer.Option(help="Monte Carlo trials for each K")] = 5000,
    seed: Annotated[int, typer.Option(help="Random seed")] = 7,
    simulation_histograms: Annotated[
        Path | None,
        typer.Option(help="JSON file with per-s histograms"),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable debug logging")
    ] = False,
) -> None:
    """Shor strict benchmark (ideal, uniform, arithmetic ideal)."""
    setup_logging(verbose)

    logger.debug(
        "args: n=%s a=%s r=%s m=%s k_list=%s m_mc=%s seed=%s",
        n,
        a,
        r,
        m,
        k_list,
        m_mc,
        seed,
    )

    instance: BenchmarkInstance = BenchmarkInstance(n, a, r, m)

    result: CombinedCurveResult = run_strict_benchmark(
        instance=instance,
        k_list=k_list,
        m_mc=m_mc,
        seed=seed,
        histogram_path=simulation_histograms,
    )

    output_payload: StrictBenchmarkResultFileModel = StrictBenchmarkResultFileModel(
        instance=instance,
        k_list=k_list,
        m_mc=m_mc,
        seed=seed,
        result=result,
        simulation_histogram_file=(
            str(simulation_histograms) if simulation_histograms is not None else None
        ),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(output_payload.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    app()
