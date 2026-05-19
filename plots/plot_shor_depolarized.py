"""Plot Shor depolarized finite-Q robustness curves."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import matplotlib.pyplot as plt
import typer
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib_config import PlotConfig, configure_matplotlib, get_latex_figsize

from qft_dynamic.shor_benchmark import BenchmarkInstance

app = typer.Typer()
PLOT_DIR: Path = Path(__file__).resolve().parent
PLOT_CONFIG: PlotConfig = configure_matplotlib(PLOT_DIR / "plot_config.toml")


def _required_int(payload: Mapping[str, object], key: str) -> int:
    """Parse a required integer field from a JSON object.

    Args:
        payload: JSON object containing the field.
        key: Field name to parse.

    Returns:
        Integer field value.

    Raises:
        ValueError: If the field cannot be parsed as an integer.
    """

    return _parse_int(value=payload[key], field_name=key)


def _parse_int(value: object, field_name: str) -> int:
    """Parse an integer-compatible JSON value.

    Args:
        value: JSON value to parse.
        field_name: Field name for error messages.

    Returns:
        Parsed integer.

    Raises:
        ValueError: If the value cannot be parsed as an integer.
    """

    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"Expected integer-compatible value for {field_name!r}.")
    return int(value)


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


def _parse_instance(value: object) -> BenchmarkInstance:
    """Parse benchmark instance metadata.

    Args:
        value: Parsed JSON instance object.

    Returns:
        Benchmark instance metadata.
    """

    payload: dict[str, object] = cast(dict[str, object], value)
    instance = BenchmarkInstance(
        n=_required_int(payload=payload, key="n"),
        a=_required_int(payload=payload, key="a"),
        r=_required_int(payload=payload, key="r"),
        m=_required_int(payload=payload, key="m"),
    )
    return instance


def load_data(input_path: Path) -> RobustnessData:
    """Load depolarized benchmark JSON data.

    Args:
        input_path: Path to the depolarized benchmark JSON file.

    Returns:
        Parsed robustness data for plotting.

    Raises:
        ValueError: If the JSON payload does not match the expected schema.
    """

    payload: dict[str, object] = json.loads(input_path.read_text(encoding="utf-8"))

    instance: BenchmarkInstance = _parse_instance(value=payload["instance"])

    raw_k_list: object = payload["k_list"]
    assert isinstance(raw_k_list, list)
    k_values: list[object] = cast(list[object], raw_k_list)
    k_list: list[int] = [
        _parse_int(value=value, field_name="k_list") for value in k_values
    ]

    raw_curves: object = payload["curves"]
    assert isinstance(raw_curves, list)
    curves: list[object] = cast(list[object], raw_curves)

    lambdas: list[float] = []
    p_ord_strict_by_k: dict[int, list[float]] = {k_value: [] for k_value in k_list}

    for curve_value in curves:
        curve_payload: dict[str, object] = cast(dict[str, object], curve_value)
        noise_lambda: float = float(cast(float | int | str, curve_payload["lambda"]))
        raw_metrics_by_k: object = curve_payload["metrics_by_k"]
        assert isinstance(raw_metrics_by_k, dict)
        metrics_by_k: dict[str, object] = cast(dict[str, object], raw_metrics_by_k)

        lambdas.append(noise_lambda)

        k_value: int
        for k_value in k_list:
            metrics: dict[str, object] = cast(
                dict[str, object], metrics_by_k[str(k_value)]
            )
            p_ord_strict: float = float(
                cast(float | int | str, metrics["p_ord_strict"])
            )
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
) -> None:
    """Plot P_ord_strict versus lambda with one curve per K.

    Args:
        data: Parsed depolarized robustness data.
        output_path: Path to save the figure.
        cmap_name: Matplotlib colormap name.
        height_ratio: Height ratio for the figure.
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
    fig.suptitle(
        (
            "Instance "
            f"(N={instance.n}, a={instance.a}, r={instance.r}, "
            f"m={instance.m}, Q={instance.q})"
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


@app.command()
def main(
    input: Annotated[Path, typer.Argument(help="Path to depolarized JSON output")],
    output: Annotated[Path, typer.Argument(help="Path to save the output plot")],
    cmap: Annotated[str, typer.Option(help="Matplotlib colormap name")] = "viridis",
    height_ratio: Annotated[float, typer.Option(help="Height ratio")] = 1.0,
) -> None:
    """Plot Shor depolarized robustness curves."""

    data: RobustnessData = load_data(input_path=input)
    plot_robustness_curve(
        data=data,
        output_path=output,
        cmap_name=cmap,
        height_ratio=height_ratio,
    )


if __name__ == "__main__":
    app()
