"""Plot approximate Shor depolarized strict-success theory curves."""

from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Annotated, Sequence

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
from pydantic import BaseModel, Field

from qft_dynamic.shor_benchmark.theory import (
    depolarized_approx_success_probability,
)

app = typer.Typer()
PLOT_DIR: Path = Path(__file__).resolve().parent
PLOT_CONFIG_PATH: Path = PLOT_DIR / "plot_config.toml"


@dataclass(frozen=True)
class InstanceData:
    """Shor order-finding instance metadata with known order factorization.

    Args:
        generator: Optional generator family name from the instance file.
        n: Modulus used in order finding.
        a: Base integer whose order is ``r`` modulo ``n``.
        r: Multiplicative order of ``a`` modulo ``n``.
        m: Number of control qubits, so ``Q = 2^m``.
        order_factors: Prime factorization of ``r``, with multiplicity.
    """

    generator: str | None
    n: int
    a: int
    r: int
    m: int
    order_factors: list[int]


@dataclass(frozen=True)
class TheoryCurveData:
    """Approximate depolarized strict-success theory data for plotting.

    Args:
        instance: Instance metadata loaded from JSON.
        lambdas: Noise mixture weights.
        k_list: Sample-count values ``K``.
        distinct_order_primes: Distinct prime factors of the order ``r``.
        p_ord_strict_by_k: Approximate strict-success values indexed by ``K``.
    """

    instance: InstanceData
    lambdas: list[float]
    k_list: list[int]
    distinct_order_primes: list[int]
    p_ord_strict_by_k: dict[int, list[float]]


class InstancePayload(BaseModel):
    """Serializable schema for generated Shor instance JSON files."""

    generator: str | None = None
    n: int
    a: int
    r: int
    m: int
    order_factors: list[int] = Field(default_factory=list)


def load_instance(input_path: Path) -> InstanceData:
    """Load a generated Shor instance JSON file.

    Args:
        input_path: Path to a JSON file produced by
            ``devtools/generate_shor_instance.py``.

    Returns:
        Parsed instance metadata.

    Raises:
        ValueError: If the file does not contain ``order_factors``.
    """

    payload: InstancePayload = InstancePayload.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )
    if not payload.order_factors:
        raise ValueError(
            "Instance file must contain order_factors for big-instance theory curves."
        )

    instance: InstanceData = InstanceData(
        generator=payload.generator,
        n=payload.n,
        a=payload.a,
        r=payload.r,
        m=payload.m,
        order_factors=payload.order_factors,
    )
    return instance


def distinct_order_primes(instance: InstanceData) -> list[int]:
    """Return distinct order factors and validate their product equals ``r``.

    Args:
        instance: Instance metadata with an ``order_factors`` factorization.

    Returns:
        Sorted distinct factors of ``r``.

    Raises:
        ValueError: If the factor list is empty, contains invalid factors, or
            does not multiply to ``r``.
    """

    if not instance.order_factors:
        raise ValueError("order_factors must be non-empty.")

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


def build_theory_curve_data(
    instance: InstanceData,
    k_list: Sequence[int],
    lambdas: Sequence[float],
) -> TheoryCurveData:
    """Compute approximate depolarized theory curves for one plot.

    Args:
        instance: Instance metadata with a known factorization of ``r``.
        k_list: Sample-count values ``K``.
        lambdas: Noise mixture weights.

    Returns:
        Theory data ready for plotting.
    """

    order_primes: list[int] = distinct_order_primes(instance)
    k_values: list[int] = list(k_list)
    lambda_values: list[float] = list(lambdas)
    p_ord_strict_by_k: dict[int, list[float]] = {}

    k_value: int
    for k_value in k_values:
        if k_value < 1:
            raise ValueError(f"All K values must be positive, got {k_value}.")
        probabilities: list[float] = [
            depolarized_approx_success_probability(
                k_value=k_value,
                lambda_value=lambda_value,
                order_primes=order_primes,
            )
            for lambda_value in lambda_values
        ]
        p_ord_strict_by_k[k_value] = probabilities

    data: TheoryCurveData = TheoryCurveData(
        instance=instance,
        lambdas=lambda_values,
        k_list=k_values,
        distinct_order_primes=order_primes,
        p_ord_strict_by_k=p_ord_strict_by_k,
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


def plot_theory_curve(
    ax: Axes,
    data: TheoryCurveData,
    cmap_name: str = "viridis",
    label_prefix: str = "Theory",
) -> None:
    """Plot approximate depolarized theory curves on an existing axis.

    Args:
        ax: Matplotlib axis to draw on.
        data: Precomputed theory curve data.
        cmap_name: Matplotlib colormap name.
        label_prefix: Prefix used in legend labels.
    """

    cmap: Colormap = plt.get_cmap(cmap_name)
    norm: Normalize | LogNorm = _color_norm(k_list=data.k_list)

    k_value: int
    for k_value in sorted(data.k_list):
        color: tuple[float, float, float, float] = cmap(norm(float(k_value)))
        ax.plot(
            data.lambdas,
            data.p_ord_strict_by_k[k_value],
            marker=None,
            linestyle="--",
            color=color,
            label=f"{label_prefix} K={k_value}",
        )

    ax.set_xlabel(r"Noise mixture $\lambda$")
    ax.set_ylabel(r"$P_{\rm ord,strict}^{(K)}$")
    ax.set_title("Approximate Shor Strict Theory")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)


def plot_instance_theory_curve(
    ax: Axes,
    instance_path: Path,
    k_list: Sequence[int],
    lambdas: Sequence[float],
    cmap_name: str = "viridis",
    label_prefix: str = "Theory",
) -> TheoryCurveData:
    """Load an instance file, compute theory curves, and plot them.

    Args:
        ax: Matplotlib axis to draw on.
        instance_path: Path to a generated Shor instance JSON file.
        k_list: Sample-count values ``K``.
        lambdas: Noise mixture weights.
        cmap_name: Matplotlib colormap name.
        label_prefix: Prefix used in legend labels.

    Returns:
        Computed theory curve data.
    """

    instance: InstanceData = load_instance(instance_path)
    data: TheoryCurveData = build_theory_curve_data(
        instance=instance,
        k_list=k_list,
        lambdas=lambdas,
    )
    plot_theory_curve(
        ax=ax,
        data=data,
        cmap_name=cmap_name,
        label_prefix=label_prefix,
    )
    return data


def _save_theory_plot(
    data: TheoryCurveData,
    output_path: Path,
    cmap_name: str,
    height_ratio: float,
    short_title: bool,
    config: PlotConfig,
) -> None:
    """Save a standalone approximate theory plot.

    Args:
        data: Precomputed theory curve data.
        output_path: Path to save the figure.
        cmap_name: Matplotlib colormap name.
        height_ratio: Height ratio for the figure.
        short_title: Whether to use a compact bit-length title.
    """

    fig: Figure
    ax: Axes
    figsize: tuple[float, float] = get_latex_figsize(
        config,
        width="column",
        fraction=0.95,
        height_ratio=height_ratio,
    )
    fig, ax = plt.subplots(figsize=figsize)
    plot_theory_curve(ax=ax, data=data, cmap_name=cmap_name)
    ax.legend()

    instance: InstanceData = data.instance
    if short_title:
        n_bits: int = instance.n.bit_length()
        r_bits: int = instance.r.bit_length()
        fig.suptitle(f"N={n_bits} bits, r={r_bits} bits, m={instance.m}")
    else:
        fig.suptitle(
            (
                "Approximate Theory "
                f"(N={instance.n}, a={instance.a}, r={instance.r}, m={instance.m})"
            ),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_theory_plot(
    data: TheoryCurveData,
    output_path: Path,
    cmap_name: str,
    height_ratio: float,
    short_title: bool,
) -> None:
    """Save a theory plot using scoped Matplotlib configuration."""
    with plot_context(PLOT_CONFIG_PATH, palette="nature") as config:
        _save_theory_plot(
            data=data,
            output_path=output_path,
            cmap_name=cmap_name,
            height_ratio=height_ratio,
            short_title=short_title,
            config=config,
        )


def parse_k_list(k_values: str) -> list[int]:
    """Parse a comma-separated list of positive K values.

    Args:
        k_values: Comma-separated integer list, for example ``"1,2,4,8"``.

    Returns:
        Parsed positive integer values.

    Raises:
        ValueError: If no positive K values are supplied.
    """

    parsed_values: list[int] = [
        int(part.strip()) for part in k_values.split(",") if part.strip()
    ]
    if not parsed_values:
        raise ValueError("At least one K value is required.")

    k_value: int
    for k_value in parsed_values:
        if k_value < 1:
            raise ValueError(f"K values must be positive, got {k_value}.")
    return parsed_values


@app.command()
def main(
    input: Annotated[Path, typer.Argument(help="Path to generated instance JSON")],
    output: Annotated[Path, typer.Argument(help="Path to save the output plot")],
    k_values: Annotated[
        str,
        typer.Option(help="Comma-separated K values"),
    ] = "1,2,4,8,16",
    lambda_points: Annotated[
        int,
        typer.Option(help="Number of lambda grid points"),
    ] = 101,
    cmap: Annotated[str, typer.Option(help="Matplotlib colormap name")] = "viridis",
    height_ratio: Annotated[float, typer.Option(help="Height ratio")] = 1.0,
    short_title: Annotated[bool, typer.Option(help="Use short title")] = False,
) -> None:
    """Plot approximate depolarized strict-success theory curves."""

    instance: InstanceData = load_instance(input)
    k_list: list[int] = parse_k_list(k_values)
    lambdas: list[float] = np.linspace(0.0, 1.0, lambda_points).tolist()
    data: TheoryCurveData = build_theory_curve_data(
        instance=instance,
        k_list=k_list,
        lambdas=lambdas,
    )
    save_theory_plot(
        data=data,
        output_path=output,
        cmap_name=cmap,
        height_ratio=height_ratio,
        short_title=short_title,
    )


if __name__ == "__main__":
    app()
