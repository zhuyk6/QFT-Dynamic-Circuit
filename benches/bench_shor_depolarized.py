"""Run depolarized finite-Q Shor strict robustness analysis.

This script evaluates the application-level noise model

    P_lambda(y | s) = (1 - lambda) P_ideal(y | s) + lambda / Q

without running a quantum circuit simulation.  Each Monte Carlo sample first
chooses the finite-Q ideal sampler with probability ``1 - lambda`` and the
uniform sampler with probability ``lambda``.
"""

import logging
import random
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import typer
from pydantic import BaseModel

from qft_dynamic.shor_benchmark.samplers import FiniteQIdealSampler, UniformSampler
from qft_dynamic.shor_benchmark.strict_eval import evaluate_strict_curve
from qft_dynamic.shor_benchmark.strict_postprocess import DefaultStrictPostprocessor
from qft_dynamic.shor_benchmark.types import (
    BenchmarkInstance,
    StrictCurveResult,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Run depolarized finite-Q Shor strict robustness analysis.",
)
logger: logging.Logger = logging.getLogger(__name__)

type LambdaCurveResult = tuple[float, StrictCurveResult]
type IndexedLambdaCurveResult = tuple[int, LambdaCurveResult]


@dataclass(frozen=True)
class DepolarizedFiniteQSampler:
    """Sampler for the depolarized finite-Q ideal output distribution.

    Args:
        ideal_sampler: Sampler for the finite-Q ideal conditional distribution.
        uniform_sampler: Sampler for the fully mixed output distribution.
        noise_lambda: Mixture weight for the uniform component.
    """

    ideal_sampler: FiniteQIdealSampler
    uniform_sampler: UniformSampler
    noise_lambda: float

    def __post_init__(self) -> None:
        """Validate the mixture weight."""

        if not 0.0 <= self.noise_lambda <= 1.0:
            raise ValueError("noise_lambda must satisfy 0 <= lambda <= 1")

    def sample_y(self, s: int, rng: random.Random) -> int:
        """Sample y from the mixture distribution P_lambda(y | s).

        Args:
            s: Phase label in [0, r - 1].
            rng: Random generator.

        Returns:
            Sampled integer y in [0, Q - 1].
        """

        if rng.random() < self.noise_lambda:
            return self.uniform_sampler.sample_y(s=s, rng=rng)
        return self.ideal_sampler.sample_y(s=s, rng=rng)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration.

    Args:
        verbose: Whether to enable debug logging.
    """

    level: int = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _evaluate_lambda_curve_worker(
    instance: BenchmarkInstance,
    k_list: list[int],
    m_mc: int,
    sample_method: Literal["bitwise", "enumerate"],
    lambda_index: int,
    noise_lambda: float,
    seed: int,
) -> IndexedLambdaCurveResult:
    """Evaluate one lambda value in a worker process.

    Args:
        instance: Benchmark instance.
        k_list: Sample-count values K.
        m_mc: Monte Carlo trial count for each K.
        sample_method: Finite-Q ideal sampling strategy.
        lambda_index: Original position of this lambda value.
        noise_lambda: Mixture weight for the uniform component.
        seed: Random seed for this lambda value.

    Returns:
        Tuple containing the original lambda index and its curve result.
    """

    postprocessor: DefaultStrictPostprocessor = DefaultStrictPostprocessor(
        instance=instance
    )
    ideal_sampler: FiniteQIdealSampler = FiniteQIdealSampler(
        instance=instance,
        sample_method=sample_method,
    )
    uniform_sampler: UniformSampler = UniformSampler(instance=instance)
    sampler: DepolarizedFiniteQSampler = DepolarizedFiniteQSampler(
        ideal_sampler=ideal_sampler,
        uniform_sampler=uniform_sampler,
        noise_lambda=noise_lambda,
    )
    curve: StrictCurveResult = evaluate_strict_curve(
        instance=instance,
        sampler=sampler,
        postprocessor=postprocessor,
        k_list=k_list,
        m_mc=m_mc,
        seed=seed,
    )
    return lambda_index, (noise_lambda, curve)


def run_depolarized_benchmark(
    instance: BenchmarkInstance,
    k_list: list[int],
    lambdas: list[float],
    m_mc: int,
    seed: int,
    sample_method: Literal["bitwise", "enumerate"],
    max_workers: int | None = None,
) -> list[LambdaCurveResult]:
    """Run strict metrics for a depolarized finite-Q lambda sweep.

    Args:
        instance: Benchmark instance.
        k_list: Sample-count values K.
        lambdas: Mixture weights for the uniform component.
        m_mc: Monte Carlo trial count for each K and lambda.
        seed: Base random seed.
        sample_method: Finite-Q ideal sampling strategy.
        max_workers: Maximum number of worker processes. If ``None``, the
            process pool chooses a default based on available CPUs.

    Returns:
        List of ``(lambda, strict curve)`` results, preserving lambda order.
    """

    if not lambdas:
        return []
    if max_workers is not None and max_workers <= 0:
        raise ValueError("max_workers must be positive when provided")

    curves_by_index: list[LambdaCurveResult | None] = [None for _ in lambdas]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future[IndexedLambdaCurveResult], int] = {}
        lambda_index: int
        noise_lambda: float
        for lambda_index, noise_lambda in enumerate(lambdas):
            logger.info("Submitting lambda=%.6g", noise_lambda)
            future: Future[IndexedLambdaCurveResult] = executor.submit(
                _evaluate_lambda_curve_worker,
                instance,
                k_list,
                m_mc,
                sample_method,
                lambda_index,
                noise_lambda,
                seed + lambda_index,
            )
            futures[future] = lambda_index

        completed_future: Future[IndexedLambdaCurveResult]
        for completed_future in as_completed(futures):
            result_index: int
            result: LambdaCurveResult
            result_index, result = completed_future.result()
            result_lambda: float = result[0]
            logger.info("Finished lambda=%.6g", result_lambda)
            curves_by_index[result_index] = result

    curves_by_lambda: list[LambdaCurveResult] = []
    curve_result: LambdaCurveResult | None
    for curve_result in curves_by_index:
        if curve_result is None:
            raise RuntimeError("missing lambda curve result after process pool finish")
        curves_by_lambda.append(curve_result)
    return curves_by_lambda


def _normalize_max_workers(max_workers: int) -> int | None:
    """Convert a CLI worker count into a process-pool worker limit.

    Args:
        max_workers: CLI worker count. ``0`` means using the executor default.

    Returns:
        ``None`` for executor default, otherwise a positive worker count.

    Raises:
        ValueError: If ``max_workers`` is negative.
    """

    if max_workers < 0:
        raise ValueError("max_workers must be non-negative")
    if max_workers == 0:
        return None
    return max_workers


class OutputPayload(BaseModel):
    """Output payload."""

    model: str = "depolarized_finite_q"
    description: str = "P_lambda(y|s) = (1 - lambda) P_ideal(y|s) + lambda / Q"
    instance: BenchmarkInstance
    k_list: list[int]
    lambdas: list[float]
    m_mc: int
    seed: int
    sample_method: Literal["bitwise", "enumerate"]
    max_workers: int
    curves_by_lambda: list[LambdaCurveResult]


def main(
    instance: BenchmarkInstance,
    output: Path,
    k_list: list[int],
    num_lambdas: int,
    m_mc: int,
    seed: int,
    sample_method: Literal["bitwise", "enumerate"],
    max_workers: int,
    verbose: bool,
) -> None:
    """Evaluate Shor strict metrics under depolarized finite-Q ideal noise.

    Args:
        instance: Benchmark instance.
        output: Output JSON path.
        k_list: Sample-count values K.
        num_lambdas: Number of linearly spaced lambda values.
        m_mc: Monte Carlo trial count for each K and lambda.
        seed: Base random seed.
        sample_method: Finite-Q ideal sampling strategy.
        max_workers: CLI worker-process setting. ``0`` means executor default.
        verbose: Whether to enable debug logging.
    """

    setup_logging(verbose=verbose)
    logger.debug(
        f"args: {instance.n=},{instance.a=},{instance.r=},{instance.m=},{k_list=},{num_lambdas=},{m_mc=},{seed=},{max_workers=}",
    )

    selected_lambdas: list[float] = np.linspace(
        0, 1, num_lambdas, dtype=np.float64
    ).tolist()
    normalized_max_workers: int | None = _normalize_max_workers(max_workers=max_workers)

    curves_by_lambda = run_depolarized_benchmark(
        instance=instance,
        k_list=k_list,
        lambdas=selected_lambdas,
        m_mc=m_mc,
        seed=seed,
        sample_method=sample_method,
        max_workers=normalized_max_workers,
    )

    output_payload = OutputPayload(
        instance=instance,
        k_list=k_list,
        lambdas=selected_lambdas,
        m_mc=m_mc,
        seed=seed,
        sample_method=sample_method,
        max_workers=max_workers,
        curves_by_lambda=curves_by_lambda,
    )
    output.write_text(output_payload.model_dump_json(indent=2), encoding="utf-8")

    logger.info(f"Wrote output to {output}")


@app.command("input")
def cli_manual_input(
    n: Annotated[int, typer.Argument(help="Modulus N")],
    a: Annotated[int, typer.Argument(help="Base a")],
    r: Annotated[int, typer.Argument(help="Order r")],
    m: Annotated[int, typer.Argument(help="Control register qubit count")],
    output: Annotated[Path, typer.Argument(help="Output JSON path")],
    k_list: Annotated[list[int], typer.Option(help="K values")] = [1, 2, 4, 8, 16],
    num_lambdas: Annotated[
        int,
        typer.Option(help="Number of linspace(0, 1, num_lambdas) lambda values", min=2),
    ] = 11,
    m_mc: Annotated[int, typer.Option(help="Monte Carlo trials for each K")] = 5000,
    seed: Annotated[int, typer.Option(help="Random seed")] = 7,
    sample_method: Annotated[
        Literal["bitwise", "enumerate"],
        typer.Option(help="Finite-Q ideal sampling method"),
    ] = "bitwise",
    max_workers: Annotated[
        int,
        typer.Option(
            help="Maximum worker processes for lambda parallelism; 0 uses default",
            min=0,
        ),
    ] = 0,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable debug logging")
    ] = False,
) -> None:
    """Manually input instance parameters."""
    main(
        BenchmarkInstance(n=n, a=a, r=r, m=m),
        output,
        k_list,
        num_lambdas,
        m_mc,
        seed,
        sample_method,
        max_workers,
        verbose,
    )


class FileInstance(BaseModel):
    generator: str
    n: int
    a: int
    r: int
    m: int
    order_factors: list[int] = []


@app.command("file")
def cli_from_file(
    input: Annotated[Path, typer.Argument(help="Input instance JSON path")],
    output: Annotated[Path, typer.Argument(help="Output result JSON path")],
    k_list: Annotated[list[int], typer.Option(help="K values")] = [1, 2, 4, 8, 16],
    num_lambdas: Annotated[
        int,
        typer.Option(help="Number of linspace(0, 1, num_lambdas) lambda values", min=2),
    ] = 11,
    m_mc: Annotated[int, typer.Option(help="Monte Carlo trials for each K")] = 5000,
    seed: Annotated[int, typer.Option(help="Random seed")] = 7,
    sample_method: Annotated[
        Literal["bitwise", "enumerate"],
        typer.Option(help="Finite-Q ideal sampling method"),
    ] = "bitwise",
    max_workers: Annotated[
        int,
        typer.Option(
            help="Maximum worker processes for lambda parallelism; 0 uses default",
            min=0,
        ),
    ] = 0,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable debug logging")
    ] = False,
) -> None:
    """Load instance parameters from a JSON file."""
    json_str = input.read_text(encoding="utf-8")
    file_instance = FileInstance.model_validate_json(json_str)

    instance = BenchmarkInstance(
        n=file_instance.n,
        a=file_instance.a,
        r=file_instance.r,
        m=file_instance.m,
    )

    main(
        instance,
        output,
        k_list,
        num_lambdas,
        m_mc,
        seed,
        sample_method,
        max_workers,
        verbose,
    )


if __name__ == "__main__":
    app()
