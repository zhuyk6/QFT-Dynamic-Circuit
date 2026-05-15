"""Plot Shor strict benchmark results from JSON output."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import typer
from matplotlib.axes import Axes

from qft_dynamic.shor_benchmark import (
    ArithmeticCurveResult,
    BenchmarkInstance,
    StrictBenchmarkResultFileModel,
    StrictCurveResult,
)

app = typer.Typer()


def load_data(
    input_path: Path,
) -> tuple[
    list[int], dict[str, StrictCurveResult], ArithmeticCurveResult, BenchmarkInstance
]:
    """Load and parse the benchmark JSON file.

    Args:
        input_path (Path): Path to JSON file.

    Returns:
        tuple containing k_list, strict_curves, arithmetic_curve, instance_info
    """
    payload = StrictBenchmarkResultFileModel.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )

    instance = payload.instance
    k_list: list[int] = payload.k_list
    result = payload.result

    strict_curves: dict[str, StrictCurveResult] = {
        "ideal": result.ideal,
        "uniform": result.uniform,
    }
    for i, exp in enumerate(result.experiments):
        strict_curves[f"exp{i}"] = exp

    arithmetic_curve = result.arithmetic

    return k_list, strict_curves, arithmetic_curve, instance


def plot_benchmark(
    k_list: list[int],
    strict_curves: dict[str, StrictCurveResult],
    arithmetic_curve: ArithmeticCurveResult,
    instance: BenchmarkInstance,
    output_path: Path,
    dpi: int,
    experiments_labels: list[str] | None = None,
) -> None:
    """Plot benchmark results and save to output_path.

    Args:
        k_list: List of K values.
        strict_curves: Dict of baseline name to {K: {metric: value}}.
        arithmetic_curve: Dict of "arithmetic" to {K: p_ord_strict}.
        instance: Instance info dict with n, a, r, m, q.
        output_path: Path to save the figure.
        dpi: Output DPI.
        experiments_labels: Optional list of labels for experiments.
    """
    colors: dict[str, str] = {
        "ideal": "#1f77b4",
        "uniform": "#d62728",
        "arithmetic": "#2ca02c",
        "exp0": "#ff7f0e",
        "exp1": "#9467bd",
        "exp2": "#8c564b",
    }
    metric_labels: dict[str, str] = {
        "p_ord_strict": "Success",
        "p_wrong": "Wrong",
        "p_null": "Reject",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: P_ord_strict vs K ---
    def get_label(name: str) -> str:
        if "exp" in name and experiments_labels is not None:
            idx = int(name[3:])
            return experiments_labels[idx]
        else:
            return name.title()

    def get_mark(name: str) -> str:
        if "exp" in name:
            return "o"
        else:
            return "s"

    ax1: Axes = axes[0]
    for name in strict_curves:
        y_vals = [strict_curves[name].metrics_by_k[k].p_ord_strict for k in k_list]
        ax1.plot(
            k_list,
            y_vals,
            marker=get_mark(name),
            color=colors.get(name),
            label=get_label(name),
            linewidth=1.5,
        )

    name = "arithmetic"
    y_vals = [arithmetic_curve.p_ord_strict_by_k[k] for k in k_list]
    ax1.plot(
        k_list,
        y_vals,
        marker=None,
        color=colors.get(name, "#2ca02c"),
        linestyle="--",
        label=name.title(),
        linewidth=1.5,
    )

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(k_list)
    ax1.set_xticklabels([str(k) for k in k_list])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel("K")
    ax1.set_ylabel("$P_{\\rm ord,strict}^{(K)}$")
    ax1.set_title("Strict Order-Recovery Success")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # --- Right panel: breakdown for the selected curve ---
    ax2: Axes = axes[1]
    # name = "ideal"
    name = "exp0"
    ideal_curve = strict_curves[name].metrics_by_k
    success: list[float] = [ideal_curve[k].p_ord_strict for k in k_list]
    wrong: list[float] = [ideal_curve[k].p_wrong for k in k_list]
    null_: list[float] = [ideal_curve[k].p_null for k in k_list]

    k_positions: range = range(len(k_list))
    bar_width: float = 0.5
    ax2.bar(
        k_positions,
        success,
        bar_width,
        color="#2ca02c",
        label=metric_labels["p_ord_strict"],
    )
    ax2.bar(
        k_positions,
        wrong,
        bar_width,
        bottom=success,
        color="#d62728",
        label=metric_labels["p_wrong"],
    )
    bottom_wrong: list[float] = [s + w for s, w in zip(success, wrong)]
    ax2.bar(
        k_positions,
        null_,
        bar_width,
        bottom=bottom_wrong,
        color="#7f7f7f",
        label=metric_labels["p_null"],
    )

    ax2.set_xticks(k_positions)
    ax2.set_xticklabels([str(k) for k in k_list])
    ax2.set_xlabel("K")
    ax2.set_ylabel("Probability")
    ax2.set_title(f"{get_label(name)} — Breakdown")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.05)

    n_val: int = instance.n
    a_val: int = instance.a
    r_val: int = instance.r
    m_val: int = instance.m
    q_val: int = instance.q
    fig.suptitle(
        f"Shor Strict — Instance (N={n_val}, a={a_val}, r={r_val}, m={m_val}, Q={q_val})",
        fontsize=11,
        y=1.02,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


@app.command()
def main(
    input: Annotated[Path, typer.Argument(help="Path to the benchmark JSON file")],
    output: Annotated[Path, typer.Argument(help="Path to save the output plot")],
    dpi: Annotated[int, typer.Option(help="Output image DPI")] = 150,
    experiments_labels: Annotated[
        list[str] | None, typer.Option(help="Experiments labels")
    ] = None,
) -> None:
    """Plot Shor strict benchmark results from a JSON file."""
    k_list, strict_curves, arithmetic_curve, instance = load_data(input)

    plot_benchmark(
        k_list=k_list,
        strict_curves=strict_curves,
        arithmetic_curve=arithmetic_curve,
        instance=instance,
        output_path=output,
        dpi=dpi,
        experiments_labels=experiments_labels,
    )


if __name__ == "__main__":
    app()
