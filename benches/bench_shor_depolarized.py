"""Run depolarized finite-Q Shor strict robustness analysis.

This script evaluates the application-level noise model

    P_lambda(y | s) = (1 - lambda) P_ideal(y | s) + lambda / Q

without running a quantum circuit simulation.  Each Monte Carlo sample first
chooses the finite-Q ideal sampler with probability ``1 - lambda`` and the
uniform sampler with probability ``lambda``.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import numpy as np
import typer

from qft_dynamic.shor_benchmark.samplers import FiniteQIdealSampler, UniformSampler
from qft_dynamic.shor_benchmark.strict_eval import evaluate_strict_curve
from qft_dynamic.shor_benchmark.strict_postprocess import DefaultStrictPostprocessor
from qft_dynamic.shor_benchmark.types import (
    BenchmarkInstance,
    StrictCurveResult,
    StrictMetrics,
)

app = typer.Typer()
logger: logging.Logger = logging.getLogger(__name__)
LambdaCurveResult: TypeAlias = tuple[float, StrictCurveResult]


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


def strict_metrics_to_dict(metrics: StrictMetrics) -> dict[str, float]:
    """Serialize strict metrics into a JSON-compatible dictionary.

    Args:
        metrics: Strict metrics for one K value.

    Returns:
        Dictionary containing strict success, wrong-output, and null rates.
    """

    payload: dict[str, float] = {
        "p_ord_strict": metrics.p_ord_strict,
        "p_wrong": metrics.p_wrong,
        "p_null": metrics.p_null,
    }
    return payload


def strict_curve_to_dict(curve: StrictCurveResult) -> dict[str, dict[str, float]]:
    """Serialize a strict curve into a JSON-compatible dictionary.

    Args:
        curve: Strict metrics indexed by K.

    Returns:
        Dictionary keyed by stringified K values.
    """

    payload: dict[str, dict[str, float]] = {}
    k_value: int
    metrics: StrictMetrics
    for k_value, metrics in sorted(curve.metrics_by_k.items()):
        payload[str(k_value)] = strict_metrics_to_dict(metrics=metrics)
    return payload


def run_depolarized_benchmark(
    instance: BenchmarkInstance,
    k_list: list[int],
    lambdas: list[float],
    m_mc: int,
    seed: int,
    sample_method: Literal["bitwise", "enumerate"],
) -> list[LambdaCurveResult]:
    """Run strict metrics for a depolarized finite-Q lambda sweep.

    Args:
        instance: Benchmark instance.
        k_list: Sample-count values K.
        lambdas: Mixture weights for the uniform component.
        m_mc: Monte Carlo trial count for each K and lambda.
        seed: Base random seed.
        sample_method: Finite-Q ideal sampling strategy.

    Returns:
        List of ``(lambda, strict curve)`` results, preserving lambda order.
    """

    postprocessor: DefaultStrictPostprocessor = DefaultStrictPostprocessor(
        instance=instance
    )
    ideal_sampler: FiniteQIdealSampler = FiniteQIdealSampler(
        instance=instance,
        sample_method=sample_method,
    )
    uniform_sampler: UniformSampler = UniformSampler(instance=instance)

    curves_by_lambda: list[LambdaCurveResult] = []
    lambda_index: int
    noise_lambda: float
    for lambda_index, noise_lambda in enumerate(lambdas):
        logger.info("Evaluating lambda=%.6g", noise_lambda)
        sampler = DepolarizedFiniteQSampler(
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
            seed=seed + lambda_index,
        )
        curves_by_lambda.append((noise_lambda, curve))

    return curves_by_lambda


def build_output_payload(
    instance: BenchmarkInstance,
    k_list: list[int],
    m_mc: int,
    seed: int,
    sample_method: Literal["bitwise", "enumerate"],
    curves_by_lambda: list[LambdaCurveResult],
) -> dict[str, object]:
    """Build the JSON payload for the lambda sweep.

    Args:
        instance: Benchmark instance.
        k_list: Sample-count values K.
        m_mc: Monte Carlo trial count for each K and lambda.
        seed: Base random seed.
        sample_method: Finite-Q ideal sampling strategy.
        curves_by_lambda: Strict curves paired with lambda values.

    Returns:
        JSON-compatible payload.
    """

    lambdas: list[float] = []
    curves_payload: list[dict[str, object]] = []
    noise_lambda: float
    curve: StrictCurveResult
    for noise_lambda, curve in curves_by_lambda:
        lambdas.append(noise_lambda)
        curves_payload.append(
            {
                "lambda": noise_lambda,
                "metrics_by_k": strict_curve_to_dict(curve=curve),
            }
        )

    payload: dict[str, object] = {
        "model": "depolarized_finite_q",
        "description": ("P_lambda(y|s) = (1 - lambda) P_ideal(y|s) + lambda / Q"),
        "instance": {
            "n": instance.n,
            "a": instance.a,
            "r": instance.r,
            "m": instance.m,
            "q": instance.q,
        },
        "k_list": k_list,
        "lambdas": lambdas,
        "m_mc": m_mc,
        "seed": seed,
        "sample_method": sample_method,
        "curves": curves_payload,
    }
    return payload


@app.command()
def main(
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
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable debug logging")
    ] = False,
) -> None:
    """Evaluate Shor strict metrics under depolarized finite-Q ideal noise."""
    setup_logging(verbose=verbose)
    logger.debug(
        f"args: {n=},{a=},{r=},{m=},{k_list=},{num_lambdas=},{m_mc=},{seed=}",
    )

    selected_lambdas: list[float] = np.linspace(
        0, 1, num_lambdas, dtype=np.float64
    ).tolist()

    instance: BenchmarkInstance = BenchmarkInstance(n=n, a=a, r=r, m=m)

    curves_by_lambda = run_depolarized_benchmark(
        instance=instance,
        k_list=k_list,
        lambdas=selected_lambdas,
        m_mc=m_mc,
        seed=seed,
        sample_method=sample_method,
    )

    output_payload: dict[str, object] = build_output_payload(
        instance=instance,
        k_list=k_list,
        m_mc=m_mc,
        seed=seed,
        sample_method=sample_method,
        curves_by_lambda=curves_by_lambda,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    app()
