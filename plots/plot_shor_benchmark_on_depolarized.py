"""Plot Shor benchmark points on depolarized probability curves."""

from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Annotated, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib_config import (
    PlotConfig,
    get_latex_figsize,
    plot_context,
)
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from qft_dynamic.shor_benchmark import BenchmarkInstance
from qft_dynamic.shor_benchmark.schemas import StrictBenchmarkResultFileModel
from qft_dynamic.shor_benchmark.theory import (
    depolarized_approx_success_probability,
)
from qft_dynamic.shor_benchmark.types import StrictCurveResult

app = typer.Typer(
    help="Plot benchmark strict success points on depolarized curves.",
    no_args_is_help=True,
)

PLOT_DIR: Path = Path(__file__).resolve().parent
PLOT_CONFIG_PATH: Path = PLOT_DIR / "plot_config.toml"

type CurveFunction = Callable[[float], float]


@dataclass(frozen=True)
class BenchmarkPoint:
    """Benchmark point projected onto a depolarized curve.

    Args:
        k_value: Sample-count value ``K``.
        p_ord_strict: Benchmark strict success probability.
        lambda_value: Effective depolarized ``lambda`` on the matching depolarized curve.
    """

    k_value: int
    p_ord_strict: float
    lambda_value: float


@dataclass(frozen=True)
class BenchmarkOnCurveData:
    """Data needed to plot benchmark points against depolarized curves.

    Args:
        instance: Benchmark instance metadata.
        k_list: Sample-count values ``K``.
        lambdas: Depolarized curve lambda grid.
        depolarized_prob_by_k: Depolarized probabilities indexed by ``K``.
        benchmark_points: Benchmark points projected onto depolarized curves.
        order_primes: Distinct prime factors of the order.
        curve_name: Benchmark curve name used for star points.
    """

    instance: BenchmarkInstance
    k_list: list[int]
    lambdas: list[float]
    depolarized_prob_by_k: dict[int, list[float]]
    benchmark_points: list[BenchmarkPoint]
    order_primes: list[int]
    curve_name: str


class InstanceFactorData(BaseModel):
    """Serializable schema for generated Shor instance JSON files."""

    generator: str | None = None
    n: int
    a: int
    r: int
    m: int
    order_factors: list[int] = Field(default_factory=list)


def _load_instance_factor_data(input_path: Path) -> InstanceFactorData:
    """Load generated instance metadata with order factorization.

    Args:
        input_path: Path to generated instance JSON.

    Returns:
        Parsed instance factor metadata.

    Raises:
        ValueError: If ``order_factors`` are missing.
    """

    data = InstanceFactorData.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )
    if not data.order_factors:
        raise ValueError("Instance file must contain order_factors.")
    return data


def _validate_matching_instance(
    benchmark_instance: BenchmarkInstance,
    factor_instance: InstanceFactorData,
) -> None:
    """Validate benchmark metadata against generated instance metadata.

    Args:
        benchmark_instance: Instance metadata from benchmark output.
        factor_instance: Instance metadata with order factors.

    Raises:
        ValueError: If the instance descriptions differ.
    """

    mismatch_fields: list[str] = []
    if benchmark_instance.n != factor_instance.n:
        mismatch_fields.append("n")
    if benchmark_instance.a != factor_instance.a:
        mismatch_fields.append("a")
    if benchmark_instance.r != factor_instance.r:
        mismatch_fields.append("r")
    if benchmark_instance.m != factor_instance.m:
        mismatch_fields.append("m")
    if mismatch_fields:
        fields: str = ", ".join(mismatch_fields)
        raise ValueError(f"Instance file does not match benchmark data: {fields}.")


def _distinct_order_primes(instance: InstanceFactorData) -> list[int]:
    """Return distinct order factors and validate their product equals ``r``.

    Args:
        instance: Instance metadata with ``order_factors``.

    Returns:
        Sorted distinct factors of ``r``.

    Raises:
        ValueError: If the factorization is invalid.
    """

    factor_product: int = prod(instance.order_factors)
    if factor_product != instance.r:
        raise ValueError(
            "order_factors must be a complete factorization: "
            f"product={factor_product}, r={instance.r}."
        )

    factor: int
    for factor in instance.order_factors:
        if factor <= 1:
            raise ValueError(f"order_factors must be greater than 1, got {factor}.")
        if instance.r % factor != 0:
            raise ValueError(f"order factor {factor} does not divide r={instance.r}.")

    primes: list[int] = sorted(set(instance.order_factors))
    return primes


def _select_strict_curve(
    payload: StrictBenchmarkResultFileModel,
    curve_name: str,
) -> StrictCurveResult:
    """Select one strict benchmark curve from a result file.

    Args:
        payload: Parsed strict benchmark result file.
        curve_name: One of ``ideal``, ``uniform``, or ``exp{i}``.

    Returns:
        Selected strict curve.

    Raises:
        ValueError: If ``curve_name`` is unknown.
    """

    if curve_name == "ideal":
        return payload.result.ideal
    if curve_name == "uniform":
        return payload.result.uniform
    if curve_name.startswith("exp"):
        index_text: str = curve_name.removeprefix("exp")
        if index_text.isdecimal():
            index: int = int(index_text)
            if 0 <= index < len(payload.result.experiments):
                return payload.result.experiments[index]
    raise ValueError(f"Unknown benchmark curve name: {curve_name}.")


def _build_theory_probability_function(
    k_value: int,
    order_primes: Sequence[int],
) -> CurveFunction:
    """Build a continuous theoretical depolarized probability function.

    Args:
        k_value: Sample-count value ``K``.
        order_primes: Distinct prime factors of the order.

    Returns:
        Function mapping ``lambda`` to approximate strict success probability.
    """

    def probability(lambda_value: float) -> float:
        result: float = depolarized_approx_success_probability(
            k_value=k_value,
            lambda_value=lambda_value,
            order_primes=order_primes,
        )
        return result

    return probability


def _build_polynomial_probability_function(
    k_value: int,
    lambdas: Sequence[float],
    probabilities: Sequence[float],
) -> CurveFunction:
    """Fit a degree-``K`` polynomial probability curve from sampled data.

    Args:
        k_value: Sample-count value ``K`` and polynomial degree.
        lambdas: Sampled depolarized noise mixture weights.
        probabilities: Sampled strict success probabilities.

    Returns:
        Function mapping ``lambda`` to the fitted strict success probability.

    Raises:
        ValueError: If the sampled data cannot determine a degree-``K`` fit.
    """

    if len(lambdas) != len(probabilities):
        raise ValueError("lambdas and probabilities must have the same length.")
    if len(lambdas) <= k_value:
        raise ValueError(
            f"Need at least {k_value + 1} samples to fit a degree-{k_value} curve."
        )

    lambda_array: NDArray[np.float64] = np.asarray(lambdas, dtype=np.float64)
    probability_array: NDArray[np.float64] = np.asarray(probabilities, dtype=np.float64)
    coefficients = np.polynomial.polynomial.polyfit(
        lambda_array,
        probability_array,
        deg=k_value,
    )

    def probability(lambda_value: float) -> float:
        result: float = float(
            np.polynomial.polynomial.polyval(lambda_value, coefficients)
        )
        return result

    return probability


def _infer_lambda_by_bisection(
    p_ord_strict: float,
    probability_function: CurveFunction,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    """Infer effective lambda for one benchmark success probability.

    The depolarized curve is assumed to be monotone decreasing in lambda.
    Values outside the curve range are clamped to the nearest endpoint.

    Args:
        p_ord_strict: Benchmark strict success probability.
        probability_function: Continuous probability function of lambda.
        tolerance: Tolerance for the bisection search.
        max_iterations: Maximum bisection iterations.

    Returns:
        Effective depolarized noise mixture weight.
    """

    lo_value: float = 0.0
    hi_value: float = 1.0
    p_at_zero: float = probability_function(lo_value)
    p_at_one: float = probability_function(hi_value)

    if p_ord_strict >= p_at_zero:
        return lo_value
    if p_ord_strict <= p_at_one:
        return hi_value

    _iteration: int
    for _iteration in range(max_iterations):
        mid_value: float = 0.5 * (lo_value + hi_value)
        p_mid: float = probability_function(mid_value)
        if abs(p_mid - p_ord_strict) <= tolerance:
            return mid_value
        if p_mid > p_ord_strict:
            lo_value = mid_value
        else:
            hi_value = mid_value

    lambda_value: float = 0.5 * (lo_value + hi_value)
    return lambda_value


class DepolarizedCurvePayload(BaseModel):
    """Serializable schema for depolarized curve JSON files."""

    instance: BenchmarkInstance
    k_list: list[int]
    lambdas: list[float]
    curves_by_lambda: list[tuple[float, StrictCurveResult]]


def _validate_depolarized_payload(
    payload: DepolarizedCurvePayload,
    benchmark_instance: BenchmarkInstance,
    k_list: Sequence[int],
) -> None:
    """Validate sampled depolarized curve data against benchmark metadata.

    Args:
        payload: Parsed depolarized curve payload.
        benchmark_instance: Benchmark instance metadata.
        k_list: Benchmark sample-count values.

    Raises:
        ValueError: If the sampled curve data are incompatible.
    """

    if payload.instance != benchmark_instance:
        raise ValueError(
            "Depolarized curve instance does not match benchmark instance."
        )
    if len(payload.lambdas) != len(payload.curves_by_lambda):
        raise ValueError("Depolarized lambdas and curves_by_lambda length mismatch.")

    missing_k_values: list[int] = [
        k_value for k_value in k_list if k_value not in payload.k_list
    ]
    if missing_k_values:
        raise ValueError(f"Depolarized data are missing K values: {missing_k_values}.")

    index: int
    payload_lambda: float
    curve_lambda: float
    _curve: StrictCurveResult
    for index, (payload_lambda, (curve_lambda, _curve)) in enumerate(
        zip(payload.lambdas, payload.curves_by_lambda, strict=True)
    ):
        if abs(payload_lambda - curve_lambda) > 1e-12:
            raise ValueError(
                "Depolarized lambda grid does not match curves_by_lambda at "
                f"index {index}: lambdas={payload_lambda}, curve={curve_lambda}."
            )


def _build_theory_curve_functions(
    k_list: Sequence[int],
    order_primes: Sequence[int],
) -> dict[int, CurveFunction]:
    """Build theoretical depolarized probability functions for each K.

    Args:
        k_list: Sample-count values ``K``.
        order_primes: Distinct prime factors of the order.

    Returns:
        Probability functions indexed by ``K``.
    """

    curve_functions: dict[int, CurveFunction] = {}
    k_value: int
    for k_value in k_list:
        curve_functions[k_value] = _build_theory_probability_function(
            k_value=k_value,
            order_primes=order_primes,
        )
    return curve_functions


def _build_sample_curve_functions(
    payload: DepolarizedCurvePayload,
    k_list: Sequence[int],
) -> dict[int, CurveFunction]:
    """Fit sampled depolarized data into degree-``K`` polynomial functions.

    Args:
        payload: Parsed sampled depolarized curve payload.
        k_list: Sample-count values ``K`` to fit.

    Returns:
        Fitted probability functions indexed by ``K``.
    """

    sampled_probabilities_by_k: dict[int, list[float]] = {
        k_value: [] for k_value in k_list
    }
    _lambda_value: float
    curve: StrictCurveResult
    for _lambda_value, curve in payload.curves_by_lambda:
        k_value: int
        for k_value in k_list:
            sampled_probabilities_by_k[k_value].append(
                curve.metrics_by_k[k_value].p_ord_strict
            )

    curve_functions: dict[int, CurveFunction] = {}
    for k_value in k_list:
        curve_functions[k_value] = _build_polynomial_probability_function(
            k_value=k_value,
            lambdas=payload.lambdas,
            probabilities=sampled_probabilities_by_k[k_value],
        )
    return curve_functions


def build_plot_data(
    benchmark_path: Path,
    instance_path: Path,
    curve_name: str,
    lambda_points: int,
    depolarized_curve_path: Path | None = None,
) -> BenchmarkOnCurveData:
    """Build depolarized curves and benchmark star points for plotting.

    Args:
        benchmark_path: Path to strict benchmark result JSON.
        instance_path: Path to generated instance JSON with ``order_factors``.
        curve_name: Benchmark curve to project onto depolarized curves.
        lambda_points: Number of lambda grid points for depolarized curves.
        depolarized_curve_path:
            - If provided, use this path to load pre-computed depolarized curves instead of computing them.
            - If None, compute depolarized curves using theoretical method on-the-fly.

    Returns:
        Data ready for plotting.
    """

    payload: StrictBenchmarkResultFileModel = (
        StrictBenchmarkResultFileModel.model_validate_json(
            benchmark_path.read_text(encoding="utf-8")
        )
    )
    factor_instance: InstanceFactorData = _load_instance_factor_data(instance_path)
    _validate_matching_instance(
        benchmark_instance=payload.instance,
        factor_instance=factor_instance,
    )
    order_primes: list[int] = _distinct_order_primes(factor_instance)
    bench_curve: StrictCurveResult = _select_strict_curve(
        payload=payload,
        curve_name=curve_name,
    )

    depolarized_curve_by_k: dict[int, list[float]] = {}
    benchmark_points: list[BenchmarkPoint] = []
    lambdas: list[float] = np.linspace(0.0, 1.0, num=lambda_points).tolist()
    curve_functions_by_k: dict[int, CurveFunction]

    if depolarized_curve_path is not None:
        depolarized_payload: DepolarizedCurvePayload = (
            DepolarizedCurvePayload.model_validate_json(
                depolarized_curve_path.read_text(encoding="utf-8")
            )
        )
        _validate_depolarized_payload(
            payload=depolarized_payload,
            benchmark_instance=payload.instance,
            k_list=payload.k_list,
        )
        curve_functions_by_k = _build_sample_curve_functions(
            payload=depolarized_payload,
            k_list=payload.k_list,
        )
    else:
        curve_functions_by_k = _build_theory_curve_functions(
            k_list=payload.k_list,
            order_primes=order_primes,
        )

    k_value: int
    for k_value in payload.k_list:
        probability_function: CurveFunction = curve_functions_by_k[k_value]
        depolarized_curve_by_k[k_value] = [
            probability_function(lambda_value) for lambda_value in lambdas
        ]

    for k_value in payload.k_list:
        p_ord_strict: float = bench_curve.metrics_by_k[k_value].p_ord_strict
        lambda_value: float = _infer_lambda_by_bisection(
            p_ord_strict=p_ord_strict,
            probability_function=curve_functions_by_k[k_value],
        )

        benchmark_points.append(
            BenchmarkPoint(
                k_value=k_value,
                p_ord_strict=p_ord_strict,
                lambda_value=lambda_value,
            )
        )

    data: BenchmarkOnCurveData = BenchmarkOnCurveData(
        instance=payload.instance,
        k_list=payload.k_list,
        lambdas=lambdas,
        depolarized_prob_by_k=depolarized_curve_by_k,
        benchmark_points=benchmark_points,
        order_primes=order_primes,
        curve_name=curve_name,
    )
    return data


def _color_norm(k_list: Sequence[int]) -> Normalize | LogNorm:
    """Build a color normalization for K values.

    Args:
        k_list: Sample-count values K.

    Returns:
        Log normalization for positive non-degenerate K values, otherwise
        linear normalization.
    """

    min_k: int = min(k_list)
    max_k: int = max(k_list)
    if min_k > 0 and min_k != max_k:
        return LogNorm(vmin=float(min_k), vmax=float(max_k))
    return Normalize(
        vmin=float(min_k), vmax=float(max_k if min_k != max_k else min_k + 1)
    )


def _plot_benchmark_on_depolarized(
    data: BenchmarkOnCurveData,
    output_path: Path,
    cmap_name: str,
    height_ratio: float,
    short_title: bool,
    config: PlotConfig,
) -> None:
    """Plot depolarized curves and benchmark star points.

    Args:
        data: Plot data containing depolarized curves and benchmark points.
        output_path: Path to save the figure.
        cmap_name: Matplotlib colormap name.
        height_ratio: Height ratio for the figure.
        short_title: Whether to use a compact bit-length title.
    """

    cmap: Colormap = plt.get_cmap(cmap_name)
    norm: Normalize | LogNorm = _color_norm(k_list=data.k_list)

    fig: Figure
    ax: Axes
    figsize: tuple[float, float] = get_latex_figsize(
        config,
        width="column",
        fraction=0.95,
        height_ratio=height_ratio,
    )
    fig, ax = plt.subplots(figsize=figsize)

    k_value: int
    for k_value in sorted(data.k_list):
        color: tuple[float, float, float, float] = cmap(norm(float(k_value)))
        ax.plot(
            data.lambdas,
            data.depolarized_prob_by_k[k_value],
            linestyle="--",
            marker=None,
            color=color,
            label=f"K={k_value}",
        )

    # plot benchmark point
    arr_lambdas: NDArray[np.float64] = np.array(
        [point.lambda_value for point in data.benchmark_points],
        dtype=np.float64,
    )
    arr_prob: NDArray[np.float64] = np.array(
        [point.p_ord_strict for point in data.benchmark_points],
        dtype=np.float64,
    )
    ax.scatter(
        arr_lambdas,
        arr_prob,
        zorder=3,
        marker="*",
        color="black",
        label=f"{data.curve_name}",
    )

    # analysis the point
    ave_lambdas: float = float(np.mean(arr_lambdas))
    std_lambdas: float = float(np.std(arr_lambdas))
    print(f"Average lambda: {ave_lambdas:.4f}, Std: {std_lambdas:.4f}")

    # plot a vertical line for the average lambda
    ax.axvline(
        x=ave_lambdas,
        color="red",
        linestyle=":",
    )

    # add text for the average lambda
    ax.text(
        ave_lambdas + 0.02,
        0.05,
        f"$\\bar\\lambda$ = {ave_lambdas:.4f}",
        color="red",
        fontsize="small",
    )

    ax.set_xlabel(r"Noise mixture $\lambda$")
    ax.set_ylabel(r"$P_{\rm ord,strict}^{(K)}$")
    ax.set_title(f"Benchmark {data.curve_name} on Depolarized Curve")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # put legent on the upper right corner
    ax.legend(loc="upper right", fontsize="small", framealpha=0.5)

    instance: BenchmarkInstance = data.instance
    if short_title:
        n_bits: int = instance.n.bit_length()
        r_bits: int = instance.r.bit_length()
        fig.suptitle(f"N={n_bits} bits, r={r_bits} bits, m={instance.m}")
    else:
        fig.suptitle(
            f"Instance (N={instance.n}, a={instance.a}, r={instance.r}, m={instance.m})"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_benchmark_on_depolarized(
    data: BenchmarkOnCurveData,
    output_path: Path,
    cmap_name: str,
    height_ratio: float,
    short_title: bool,
) -> None:
    """Plot benchmark points using scoped Matplotlib configuration."""
    with plot_context(PLOT_CONFIG_PATH, palette="nature") as config:
        _plot_benchmark_on_depolarized(
            data=data,
            output_path=output_path,
            cmap_name=cmap_name,
            height_ratio=height_ratio,
            short_title=short_title,
            config=config,
        )


@app.command("theory")
def cli_theory(
    benchmark: Annotated[
        Path,
        typer.Argument(help="Path to strict benchmark result JSON"),
    ],
    instance: Annotated[
        Path,
        typer.Argument(help="Path to instance JSON with order_factors"),
    ],
    output: Annotated[Path, typer.Argument(help="Path to save the output plot")],
    curve: Annotated[
        str,
        typer.Option(help="Benchmark curve name: ideal, uniform, exp0, exp1, ..."),
    ] = "exp0",
    lambda_points: Annotated[
        int,
        typer.Option(help="Number of lambda grid points for depolarized curves", min=2),
    ] = 201,
    cmap: Annotated[str, typer.Option(help="Matplotlib colormap name")] = "viridis",
    height_ratio: Annotated[float, typer.Option(help="Height ratio")] = 1.0,
    short_title: Annotated[bool, typer.Option(help="Use short title")] = False,
) -> None:
    """Use theoretical formula to calculate depolarized curve.

    ALERT: this formula is approximated and only valid for big instance.
    """

    data: BenchmarkOnCurveData = build_plot_data(
        benchmark_path=benchmark,
        instance_path=instance,
        curve_name=curve,
        lambda_points=lambda_points,
    )
    plot_benchmark_on_depolarized(
        data=data,
        output_path=output,
        cmap_name=cmap,
        height_ratio=height_ratio,
        short_title=short_title,
    )


@app.command("sample")
def cli_sample(
    benchmark: Annotated[
        Path,
        typer.Argument(help="Path to strict benchmark result JSON"),
    ],
    instance: Annotated[
        Path,
        typer.Argument(help="Path to instance JSON with order_factors"),
    ],
    depolarized_data: Annotated[
        Path,
        typer.Argument(help="Path to depolarized curve JSON data"),
    ],
    output: Annotated[Path, typer.Argument(help="Path to save the output plot")],
    curve: Annotated[
        str,
        typer.Option(help="Benchmark curve name: ideal, uniform, exp0, exp1, ..."),
    ] = "exp0",
    lambda_points: Annotated[
        int,
        typer.Option(help="Number of lambda grid points for sample curves", min=2),
    ] = 201,
    cmap: Annotated[str, typer.Option(help="Matplotlib colormap name")] = "viridis",
    height_ratio: Annotated[float, typer.Option(help="Height ratio")] = 1.0,
    short_title: Annotated[bool, typer.Option(help="Use short title")] = False,
) -> None:
    """Use sample method to calculate depolarized curve."""
    data: BenchmarkOnCurveData = build_plot_data(
        benchmark_path=benchmark,
        instance_path=instance,
        curve_name=curve,
        lambda_points=lambda_points,
        depolarized_curve_path=depolarized_data,
    )
    plot_benchmark_on_depolarized(
        data=data,
        output_path=output,
        cmap_name=cmap,
        height_ratio=height_ratio,
        short_title=short_title,
    )


if __name__ == "__main__":
    app()
