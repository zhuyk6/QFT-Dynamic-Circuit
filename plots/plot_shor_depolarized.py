"""Plot Shor depolarized finite-Q robustness curves."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib_config import PlotConfig, configure_matplotlib, get_latex_figsize
from plot_shor_depolarized_theory import (
    InstanceData,
    TheoryCurveData,
    build_theory_curve_data,
)
from plot_shor_depolarized_theory import (
    load_instance as load_theory_instance,
)
from pydantic import BaseModel

from qft_dynamic.shor_benchmark import BenchmarkInstance
from qft_dynamic.shor_benchmark.types import StrictCurveResult, StrictMetrics

app = typer.Typer()
PLOT_DIR: Path = Path(__file__).resolve().parent
PLOT_CONFIG: PlotConfig = configure_matplotlib(PLOT_DIR / "plot_config.toml")


@dataclass(frozen=True)
class RobustnessData:
    """Parsed depolarized robustness data.

    Args:
        instance: Benchmark instance metadata.
        lambdas: Noise mixture weights.
        k_list: Sample-count values K.
        p_ord_strict_by_k: Strict success values indexed by K.
    """

    instance: BenchmarkInstance
    lambdas: list[float]
    k_list: list[int]
    p_ord_strict_by_k: dict[int, list[float]]


class DataPayload(BaseModel):
    model: str = "depolarized_finite_q"
    description: str = "P_lambda(y|s) = (1 - lambda) P_ideal(y|s) + lambda / Q"
    instance: BenchmarkInstance
    k_list: list[int]
    lambdas: list[float]
    m_mc: int
    seed: int
    sample_method: Literal["bitwise", "enumerate"]
    max_workers: int
    curves_by_lambda: list[tuple[float, StrictCurveResult]]


def load_data(input_path: Path) -> RobustnessData:
    """Load depolarized benchmark JSON data.

    Args:
        input_path: Path to the depolarized benchmark JSON file.

    Returns:
        Parsed robustness data for plotting.

    Raises:
        ValueError: If the JSON payload does not match the expected schema.
    """

    payload: DataPayload = DataPayload.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )

    instance: BenchmarkInstance = payload.instance
    k_list: list[int] = payload.k_list
    lambdas: list[float] = payload.lambdas
    p_ord_strict_by_k: dict[int, list[float]] = {k_value: [] for k_value in k_list}

    _lambda_val: float
    curve: StrictCurveResult
    for _lambda_val, curve in payload.curves_by_lambda:
        metrics_by_k: dict[int, StrictMetrics] = curve.metrics_by_k
        for k_value in k_list:
            metrics: StrictMetrics = metrics_by_k[k_value]
            p_ord_strict: float = metrics.p_ord_strict
            p_ord_strict_by_k[k_value].append(p_ord_strict)

    data: RobustnessData = RobustnessData(
        instance=instance,
        lambdas=lambdas,
        k_list=k_list,
        p_ord_strict_by_k=p_ord_strict_by_k,
    )
    return data


def _color_norm(k_list: list[int]) -> Normalize:
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


def _validate_matching_instance(
    data_instance: BenchmarkInstance,
    theory_instance: InstanceData,
) -> None:
    """Validate that sample data and theory data describe the same instance.

    Args:
        data_instance: Instance metadata from depolarized sampled data.
        theory_instance: Instance metadata with ``order_factors``.

    Raises:
        ValueError: If the two instance descriptions differ.
    """

    mismatch_fields: list[str] = []
    if data_instance.n != theory_instance.n:
        mismatch_fields.append("n")
    if data_instance.a != theory_instance.a:
        mismatch_fields.append("a")
    if data_instance.r != theory_instance.r:
        mismatch_fields.append("r")
    if data_instance.m != theory_instance.m:
        mismatch_fields.append("m")
    if mismatch_fields:
        fields: str = ", ".join(mismatch_fields)
        raise ValueError(f"Theory instance does not match sampled data: {fields}.")


def _apply_common_axes_style(ax: Axes, title: str | None) -> None:
    """Apply common depolarized robustness axis labels and limits.

    Args:
        ax: Matplotlib axis to style.
        title: Axis title.
    """

    ax.set_xlabel(r"Noise mixture $\lambda$")
    ax.set_ylabel(r"$P_{\mathrm ord}^{(K)}$")

    if title is not None:
        ax.set_title(title)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)


def plot_robustness_curve(
    data: RobustnessData,
    output_path: Path,
    cmap_name: str,
    height_ratio: float,
    short_title: bool,
) -> None:
    """Plot P_ord_strict versus lambda with one curve per K.

    Args:
        data: Parsed depolarized robustness data.
        output_path: Path to save the figure.
        cmap_name: Matplotlib colormap name.
        height_ratio: Height ratio for the figure.
        short_title: Whether to use a short title.
    """

    cmap: Colormap = plt.get_cmap(cmap_name)
    norm: Normalize = _color_norm(k_list=data.k_list)

    fig: Figure
    ax: Axes
    figsize: tuple[float, float] = get_latex_figsize(
        PLOT_CONFIG,
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
            data.p_ord_strict_by_k[k_value],
            marker="o",
            color=color,
            label=f"K={k_value}",
        )

    _apply_common_axes_style(ax=ax, title="Depolarized Shor Strict Robustness")
    ax.legend()

    instance: BenchmarkInstance = data.instance
    if not short_title:
        fig.suptitle(
            (
                f"Instance (N={instance.n}, a={instance.a}, r={instance.r}, m={instance.m})"
            ),
        )
    else:
        n_bits: int = instance.n.bit_length()
        r_bits: int = instance.r.bit_length()
        fig.suptitle(f"N={n_bits} bits, r={r_bits} bits, m={instance.m}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_sample_points_with_theory(
    ax: Axes,
    data: RobustnessData,
    theory_data: TheoryCurveData,
    cmap: Colormap,
    norm: Normalize,
) -> None:
    """Plot sampled points and approximate theory curves on one axis.

    Args:
        ax: Matplotlib axis to draw on.
        data: Parsed depolarized sampled data.
        theory_data: Precomputed approximate theory curves.
        cmap_name: Matplotlib colormap name.

    Raises:
        ValueError: If the theory data do not contain every sampled ``K``.
    """

    missing_k_values: list[int] = [
        k_value
        for k_value in data.k_list
        if k_value not in theory_data.p_ord_strict_by_k
    ]
    if missing_k_values:
        raise ValueError(f"Theory data are missing K values: {missing_k_values}.")

    k_value: int
    for k_value in sorted(data.k_list):
        color: tuple[float, float, float, float] = cmap(norm(float(k_value)))
        ax.plot(
            theory_data.lambdas,
            theory_data.p_ord_strict_by_k[k_value],
            linestyle="--",
            marker=None,
            color=color,
            label="_nolegend_",
        )
        ax.plot(
            data.lambdas,
            data.p_ord_strict_by_k[k_value],
            linestyle="None",
            marker="o",
            color=color,
            label="_nolegend_",
        )


def draw_sample_theory_table_legend(
    legend_ax: Axes,
    k_list: list[int],
    cmap: Colormap,
    norm: Normalize,
) -> None:
    """Draw a matrix legend on a dedicated legend axis.

    Args:
        legend_ax: Axis used only for the manually drawn legend table.
        k_list: Sample-count values ``K``.
        cmap: Matplotlib colormap.
        norm: Color normalization for ``K`` values.
    """

    sorted_k_list: list[int] = sorted(k_list)
    row_count: int = len(sorted_k_list) + 1

    # set limit of x, y
    legend_ax.set_xlim(0.0, 1.0)
    legend_ax.set_ylim(float(row_count), 0.0)
    legend_ax.axis("off")

    fontsize = 6

    y_start = 0.5
    row_gap = 0.5
    column_gap = 0.20
    column_widths = [0.2, 0.4, 0.4]

    x_k: float = 0.0
    x_sample: float = column_gap * 1 + column_widths[0]
    x_theory: float = column_gap * 2 + column_widths[0] + column_widths[1]

    # draw header
    legend_ax.text(
        x_k,
        y_start,
        "K",
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    legend_ax.text(
        x_sample,
        y_start,
        "Sample",
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    legend_ax.text(
        x_theory,
        y_start,
        "Theory",
        ha="center",
        va="center",
        fontsize=fontsize,
    )

    row_index: int
    k_value: int
    for row_index, k_value in enumerate(sorted_k_list):
        y_value: float = float(row_index + 1) * row_gap + y_start
        color: tuple[float, float, float, float] = cmap(norm(float(k_value)))

        legend_ax.text(
            x_k,
            y_value,
            str(k_value),
            ha="center",
            va="center",
            fontsize=fontsize,
        )

        legend_ax.plot(
            [x_sample],
            [y_value],
            linestyle="None",
            marker="o",
            color=color,
            clip_on=False,
        )

        legend_ax.plot(
            [x_theory - 0.5 * column_widths[2], x_theory + 0.5 * column_widths[2]],
            [y_value, y_value],
            linestyle="--",
            marker=None,
            color=color,
            clip_on=False,
        )


def _build_sample_theory_data(
    data: RobustnessData,
    theory_instance_path: Path,
    theory_lambda_points: int,
) -> TheoryCurveData:
    """Load and compute approximate theory data for sampled robustness data.

    Args:
        data: Parsed depolarized robustness data.
        theory_instance_path: Path to generated instance JSON with
            ``order_factors``.
        theory_lambda_points: Number of points in the smooth theory grid.

    Returns:
        Approximate theory curves matching ``data.k_list``.
    """

    theory_instance: InstanceData = load_theory_instance(theory_instance_path)
    _validate_matching_instance(
        data_instance=data.instance,
        theory_instance=theory_instance,
    )
    theory_lambdas: list[float] = np.linspace(0, 1, theory_lambda_points).tolist()
    theory_data: TheoryCurveData = build_theory_curve_data(
        instance=theory_instance,
        k_list=data.k_list,
        lambdas=theory_lambdas,
    )
    return theory_data


def _apply_instance_title(
    fig: Figure,
    instance: BenchmarkInstance,
    short_title: bool,
) -> None:
    """Apply a figure-level instance title.

    Args:
        fig: Matplotlib figure receiving the title.
        instance: Shor benchmark instance metadata.
        short_title: Whether to use compact bit-length metadata.
    """

    if not short_title:
        fig.suptitle(
            (
                f"Instance (N={instance.n}, a={instance.a}, r={instance.r}, m={instance.m})"
            ),
        )
    else:
        n_bits: int = instance.n.bit_length()
        r_bits: int = instance.r.bit_length()
        fig.suptitle(f"N={n_bits} bits, r={r_bits} bits, m={instance.m}")


def _save_figure(fig: Figure, output_path: Path) -> None:
    """Save and close a Matplotlib figure.

    Args:
        fig: Figure to save.
        output_path: Path to save the figure.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_robustness_curve_with_theory(
    data: RobustnessData,
    theory_instance_path: Path,
    output_path: Path,
    cmap_name: str,
    height_ratio: float,
    short_title: bool,
    theory_lambda_points: int,
) -> None:
    """Plot sampled depolarized data against approximate theory curves.

    Sampled data are drawn as circles without connecting lines. Theory curves
    are drawn as dashed lines with matching colors for each ``K``.

    Args:
        data: Parsed depolarized robustness data.
        theory_instance_path: Path to generated instance JSON with
            ``order_factors``.
        output_path: Path to save the figure.
        cmap_name: Matplotlib colormap name.
        height_ratio: Height ratio for the figure.
        short_title: Whether to use a short title.
        theory_lambda_points: Number of points in the smooth theory grid.
    """

    theory_data: TheoryCurveData = _build_sample_theory_data(
        data=data,
        theory_instance_path=theory_instance_path,
        theory_lambda_points=theory_lambda_points,
    )

    cmap: Colormap = plt.get_cmap(cmap_name)
    norm: Normalize = _color_norm(k_list=data.k_list)

    fig: Figure
    axes: np.ndarray[tuple[int], np.dtype[np.object_]]
    figsize: tuple[float, float] = get_latex_figsize(
        PLOT_CONFIG,
        width="column",
        fraction=0.95,
        height_ratio=height_ratio,
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.0, 0.3]},
    )
    ax: Axes = axes[0]
    legend_ax: Axes = axes[1]

    plot_sample_points_with_theory(
        ax=ax,
        data=data,
        theory_data=theory_data,
        cmap=cmap,
        norm=norm,
    )

    _apply_common_axes_style(
        ax=ax,
        title="Sample vs Theory",
    )
    draw_sample_theory_table_legend(
        legend_ax=legend_ax,
        k_list=data.k_list,
        cmap=cmap,
        norm=norm,
    )
    _apply_instance_title(
        fig=fig,
        instance=data.instance,
        short_title=short_title,
    )
    _save_figure(fig=fig, output_path=output_path)


@app.command()
def main(
    input: Annotated[Path, typer.Argument(help="Path to depolarized JSON output")],
    output: Annotated[Path, typer.Argument(help="Path to save the output plot")],
    cmap: Annotated[str, typer.Option(help="Matplotlib colormap name")] = "viridis",
    height_ratio: Annotated[float, typer.Option(help="Height ratio")] = 1.0,
    short_title: Annotated[bool, typer.Option(help="Use short title")] = False,
    with_theory: Annotated[
        bool,
        typer.Option(help="Overlay approximate theory curves"),
    ] = False,
    theory_instance: Annotated[
        Path | None,
        typer.Option(help="Path to generated instance JSON with order_factors"),
    ] = None,
    theory_lambda_points: Annotated[
        int,
        typer.Option(help="Number of lambda grid points for theory curves"),
    ] = 201,
) -> None:
    """Plot Shor depolarized robustness curves."""

    data: RobustnessData = load_data(input_path=input)
    if with_theory:
        if theory_instance is None:
            raise typer.BadParameter(
                "--theory-instance is required when --with-theory is set."
            )
        plot_robustness_curve_with_theory(
            data=data,
            theory_instance_path=theory_instance,
            output_path=output,
            cmap_name=cmap,
            height_ratio=height_ratio,
            short_title=short_title,
            theory_lambda_points=theory_lambda_points,
        )
    else:
        plot_robustness_curve(
            data=data,
            output_path=output,
            cmap_name=cmap,
            height_ratio=height_ratio,
            short_title=short_title,
        )


if __name__ == "__main__":
    app()
