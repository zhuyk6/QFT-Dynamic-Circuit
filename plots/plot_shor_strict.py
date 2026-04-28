"""Plot Shor strict benchmark results from JSON output."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def build_cli() -> argparse.ArgumentParser:
    """Build argument parser for the plot script."""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Plot Shor strict benchmark results from a JSON file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the benchmark JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the output plot (PNG, PDF, etc.).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI (default: 150).",
    )
    return parser


def load_data(
    input_path: Path,
) -> tuple[
    list[int],
    dict[str, dict[int, dict[str, float]]],
    dict[str, dict[int, float]],
    dict[str, int],
]:
    """Load and parse the benchmark JSON file.

    Args:
        input_path (Path): Path to JSON file.

    Returns:
        tuple containing k_list, strict_curves, arithmetic_curve, instance_info
    """

    with input_path.open("r", encoding="utf-8") as f:
        data: dict = json.load(f)

    instance: dict[str, int] = data["instance"]
    k_list: list[int] = data["k_list"]

    strict_curves: dict[str, dict[int, dict[str, float]]] = {}
    baselines: dict = data["baselines"]
    for name in ("ideal", "uniform"):
        raw: dict[str, dict[str, float]] = baselines[name]["metrics_by_k"]
        strict_curves[name] = {
            int(k): {
                "p_ord_strict": v["p_ord_strict"],
                "p_wrong": v["p_wrong"],
                "p_null": v["p_null"],
            }
            for k, v in raw.items()
        }

    arithmetic_curve: dict[str, dict[int, float]] = {
        "arithmetic": {
            int(k): float(v)
            for k, v in baselines["arithmetic"]["p_ord_strict_by_k"].items()
        }
    }

    return k_list, strict_curves, arithmetic_curve, instance


def plot_benchmark(
    k_list: list[int],
    strict_curves: dict[str, dict[int, dict[str, float]]],
    arithmetic_curve: dict[str, dict[int, float]],
    instance: dict[str, int],
    output_path: Path,
    dpi: int,
) -> None:
    """Plot benchmark results and save to output_path.

    Args:
        k_list: List of K values.
        strict_curves: Dict of baseline name to {K: {metric: value}}.
        arithmetic_curve: Dict of "arithmetic" to {K: p_ord_strict}.
        instance: Instance info dict with n, a, r, m, q.
        output_path: Path to save the figure.
        dpi: Output DPI.
    """

    colors: dict[str, str] = {
        "ideal": "#1f77b4",
        "uniform": "#d62728",
        "arithmetic": "#2ca02c",
    }
    metric_labels: dict[str, str] = {
        "p_ord_strict": "Success",
        "p_wrong": "Wrong",
        "p_null": "Reject",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: P_ord_strict vs K ---
    ax1: Axes = axes[0]
    y_vals: list[float]
    name: str
    for name in ("ideal", "uniform"):
        y_vals = [strict_curves[name][k]["p_ord_strict"] for k in k_list]
        ax1.plot(
            k_list,
            y_vals,
            marker="o",
            color=colors[name],
            label=name.title(),
            linewidth=1.5,
        )

    for name, curve in arithmetic_curve.items():
        y_vals = [curve[k] for k in k_list]
        ax1.plot(
            k_list,
            y_vals,
            marker="s",
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

    # --- Right panel: breakdown for ideal baseline ---
    ax2: Axes = axes[1]
    name = "ideal"
    ideal_curve: dict[int, dict[str, float]] = strict_curves[name]
    success: list[float] = [ideal_curve[k]["p_ord_strict"] for k in k_list]
    wrong: list[float] = [ideal_curve[k]["p_wrong"] for k in k_list]
    null_: list[float] = [ideal_curve[k]["p_null"] for k in k_list]

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
    ax2.set_title(f"{name.title()} Baseline — Breakdown")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.05)

    n_val: int = instance["n"]
    a_val: int = instance["a"]
    r_val: int = instance["r"]
    m_val: int = instance["m"]
    q_val: int = instance["q"]
    fig.suptitle(
        f"Shor Strict — Instance (N={n_val}, a={a_val}, r={r_val}, m={m_val}, Q={q_val})",
        fontsize=11,
        y=1.02,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entry point."""

    parser: argparse.ArgumentParser = build_cli()
    args: argparse.Namespace = parser.parse_args()

    k_list, strict_curves, arithmetic_curve, instance = load_data(args.input)

    plot_benchmark(
        k_list=k_list,
        strict_curves=strict_curves,
        arithmetic_curve=arithmetic_curve,
        instance=instance,
        output_path=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
