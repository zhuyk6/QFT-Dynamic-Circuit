"""Plot Shor depolarized finite-Q robustness curves."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import typer
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib_config import PlotConfig, configure_matplotlib, get_latex_figsize
from pydantic import BaseModel

from qft_dynamic.shor_benchmark import BenchmarkInstance
from qft_dynamic.shor_benchmark.types import StrictCurveResult

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

    payload = DataPayload.model_validate_json(input_path.read_text(encoding="utf-8"))

    instance: BenchmarkInstance = payload.instance
    k_list = payload.k_list
    lambdas = payload.lambdas
    p_ord_strict_by_k: dict[int, list[float]] = {k_value: [] for k_value in k_list}

    for lambda_val, curve in payload.curves_by_lambda:
        metrics_by_k = curve.metrics_by_k
        for k_value in k_list:
            metrics = metrics_by_k[k_value]
            p_ord_strict = metrics.p_ord_strict
            p_ord_strict_by_k[k_value].append(p_ord_strict)

    data: RobustnessData = RobustnessData(
        instance=instance,
        lambdas=lambdas,
        k_list=k_list,
        p_ord_strict_by_k=p_ord_strict_by_k,
    )
    return data


def _color_norm(k_list: list[int]) -> Normalize | LogNorm:
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
    norm: Normalize | LogNorm = _color_norm(k_list=data.k_list)

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

    ax.set_xlabel(r"Noise mixture $\lambda$")
    ax.set_ylabel(r"$P_{\rm ord,strict}^{(K)}$")
    ax.set_title("Depolarized Shor Strict Robustness")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    instance: BenchmarkInstance = data.instance
    if not short_title:
        fig.suptitle(
            (
                f"Instance (N={instance.n}, a={instance.a}, r={instance.r}, m={instance.m})"
            ),
        )
    else:
        n_bits = instance.n.bit_length()
        r_bits = instance.r.bit_length()
        fig.suptitle(f"N={n_bits} bits, r={r_bits} bits, m={instance.m}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


@app.command()
def main(
    input: Annotated[Path, typer.Argument(help="Path to depolarized JSON output")],
    output: Annotated[Path, typer.Argument(help="Path to save the output plot")],
    cmap: Annotated[str, typer.Option(help="Matplotlib colormap name")] = "viridis",
    height_ratio: Annotated[float, typer.Option(help="Height ratio")] = 1.0,
    short_title: Annotated[bool, typer.Option(help="Use short title")] = False,
) -> None:
    """Plot Shor depolarized robustness curves."""

    data: RobustnessData = load_data(input_path=input)
    plot_robustness_curve(
        data=data,
        output_path=output,
        cmap_name=cmap,
        height_ratio=height_ratio,
        short_title=short_title,
    )


if __name__ == "__main__":
    app()
