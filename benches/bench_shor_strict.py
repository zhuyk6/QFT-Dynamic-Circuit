"""Strict Shor benchmark runner.

This script computes strict metrics for finite-Q ideal and uniform baselines via
Monte Carlo, arithmetic-ideal strict success via closed-form expression, and
optionally experiments via histograms.
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
    histogram_paths: list[Path],
) -> CombinedCurveResult:
    """Run strict benchmark for ideal, uniform, arithmetic baselines,
    and optionally experiments via histograms.

    Args:
        instance: Benchmark instance.
        k_list: Sample-count values K.
        m_mc: Monte Carlo trial count per K.
        seed: Random seed.
        histogram_paths: Optional JSON file containing per-s histograms.

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

    experiments_curves: list[StrictCurveResult] = []
    for filepath in histogram_paths:
        histogram_sampler: HistogramSampler = HistogramSampler.from_file(
            histogram_path=filepath,
            instance=instance,
        )
        exp_curve = evaluate_strict_curve(
            instance=instance,
            sampler=histogram_sampler,
            postprocessor=postprocessor,
            k_list=k_list,
            m_mc=m_mc,
            seed=seed + 2,
        )
        experiments_curves.append(exp_curve)

    combined: CombinedCurveResult = CombinedCurveResult(
        ideal=ideal_curve,
        uniform=uniform_curve,
        arithmetic=arithmetic_curve,
        experiments=experiments_curves,
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
    experiments_histograms: Annotated[
        list[Path],
        typer.Option(help="JSON files with per-s histograms"),
    ] = [],
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable debug logging")
    ] = False,
) -> None:
    """Shor strict benchmark: ideal, uniform, arithmetic ideal baselines and experiments results."""
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
        histogram_paths=experiments_histograms,
    )

    output_payload: StrictBenchmarkResultFileModel = StrictBenchmarkResultFileModel(
        instance=instance,
        k_list=k_list,
        m_mc=m_mc,
        seed=seed,
        result=result,
        experiments_histogram_files=experiments_histograms,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(output_payload.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    app()
