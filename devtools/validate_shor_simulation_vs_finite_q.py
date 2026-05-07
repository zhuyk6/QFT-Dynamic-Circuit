"""Visualize noiseless Shor simulation against the finite-Q ideal baseline.

This script compares two discrete probability distributions:

- finite-Q ideal probabilities
- simulation histograms normalized into probabilities

For this task, violin plots are not a good primary choice because they are
better suited to continuous samples or large collections of raw draws. Here the
most informative view is:

- a top panel with a mirrored probability view
- a bottom panel with the signed residual ``simulation - ideal``

The script can either:

- load an existing histogram JSON file, or
- run a fresh noiseless simulation and then plot the comparison.

The figure includes one subplot per selected phase label ``s`` and an optional
equal-weight mixture over all phase labels.
"""

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from shor_benchmark.samplers import finite_q_ideal_probability
from shor_benchmark.schemas import HistogramFileModel
from shor_benchmark.simulation import simulate_histograms_for_instance
from shor_benchmark.types import BenchmarkInstance

NUMERICAL_ZERO_TOLERANCE: float = 1e-15


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the validation script.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Plot noiseless simulation histograms against the finite-Q ideal "
            "distribution for a Shor benchmark instance."
        )
    )
    parser.add_argument("--n", type=int, help="Modulus N")
    parser.add_argument("--a", type=int, help="Base a")
    parser.add_argument("--r", type=int, help="Order r")
    parser.add_argument("--m", type=int, help="Control qubit count")
    parser.add_argument(
        "--histogram",
        type=Path,
        help="Existing histogram JSON file produced by the Shor simulation flow",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Tile size of the optimized QFT block when running simulation",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=4096,
        help="Simulation shots per phase label when running a fresh simulation",
    )
    parser.add_argument(
        "--s",
        type=int,
        nargs="*",
        help="Phase labels to plot; default is all phase labels",
    )
    parser.add_argument(
        "--hide-mixture",
        action="store_true",
        help="Do not add the equal-weight mixture subplot over all s values",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output image path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving",
    )
    return parser.parse_args()


def load_or_simulate_histograms(
    args: argparse.Namespace,
) -> tuple[BenchmarkInstance, dict[int, Counter[int]], str]:
    """Load histogram data from disk or run a noiseless simulation.

    Args:
        args: Parsed CLI arguments.

    Returns:
        tuple[BenchmarkInstance, dict[int, Counter[int]], str]: Benchmark
            instance, per-s histograms, and a short source label for plotting.

    Raises:
        ValueError: If required arguments are missing or inconsistent.
    """

    if args.histogram is not None:
        histogram_file: HistogramFileModel = HistogramFileModel.load(args.histogram)
        instance: BenchmarkInstance = histogram_file.instance
        histograms: dict[int, Counter[int]] = {
            s_value: Counter(y_counts)
            for s_value, y_counts in histogram_file.histograms.items()
        }
        source_label: str = f"loaded histogram ({args.histogram})"
        return instance, histograms, source_label

    missing_fields: list[str] = []
    field_name: str
    for field_name in ("n", "a", "r", "m", "batch_size"):
        if getattr(args, field_name) is None:
            missing_fields.append(f"--{field_name.replace('_', '-')}")
    if missing_fields:
        missing_options: str = ", ".join(missing_fields)
        raise ValueError(
            f"fresh simulation requires the following options: {missing_options}"
        )
    if args.num_shots <= 0:
        raise ValueError("--num-shots must be positive")

    instance = BenchmarkInstance(
        n=args.n,
        a=args.a,
        r=args.r,
        m=args.m,
    )
    histograms = simulate_histograms_for_instance(
        instance=instance,
        batch_size=args.batch_size,
        num_shots=args.num_shots,
        gate_error=False,
        readout_error=False,
        thermal_relaxation=False,
    )
    source_label = (
        "fresh noiseless simulation "
        f"(batch_size={args.batch_size}, shots={args.num_shots})"
    )
    return instance, histograms, source_label


def resolve_selected_s(
    selected_s_values: list[int] | None,
    instance: BenchmarkInstance,
) -> list[int]:
    """Resolve and validate which phase labels should be plotted.

    Args:
        selected_s_values: Optional phase labels from the CLI.
        instance: Benchmark instance.

    Returns:
        list[int]: Sorted phase labels to plot.

    Raises:
        ValueError: If a phase label lies outside ``[0, r - 1]``.
    """

    if selected_s_values is None or not selected_s_values:
        return list(range(instance.r))

    unique_s_values: list[int] = sorted(set(selected_s_values))
    s_value: int
    for s_value in unique_s_values:
        if not (0 <= s_value < instance.r):
            raise ValueError(
                f"phase label s={s_value} is out of range for r={instance.r}"
            )
    return unique_s_values


def finite_q_distribution_for_s(
    instance: BenchmarkInstance,
    s: int,
) -> list[float]:
    """Compute the closed-form finite-Q ideal distribution for a fixed s.

    Args:
        instance: Benchmark instance.
        s: Phase label in ``[0, r - 1]``.

    Returns:
        list[float]: Probability vector over ``y in [0, Q - 1]``.
    """

    raw_distribution: list[float] = [
        finite_q_ideal_probability(y=y, s=s, instance=instance)
        for y in range(instance.q)
    ]
    distribution: list[float] = clamp_near_zero_values(raw_distribution)
    return distribution


def histogram_to_probability_vector(
    histogram: Counter[int],
    q_value: int,
) -> list[float]:
    """Convert histogram counts into a normalized probability vector.

    Args:
        histogram: Shot counts keyed by decoded integer ``y``.
        q_value: Total number of possible ``y`` values.

    Returns:
        list[float]: Empirical probability vector over ``y in [0, Q - 1]``.

    Raises:
        ValueError: If the histogram is empty.
    """

    total_shots: int = sum(histogram.values())
    if total_shots <= 0:
        raise ValueError("histogram must contain at least one shot")

    probabilities: list[float] = [0.0 for _ in range(q_value)]
    y_value: int
    count: int
    for y_value, count in histogram.items():
        probabilities[y_value] = count / total_shots
    return probabilities


def average_distributions(distributions: Iterable[list[float]]) -> list[float]:
    """Average several probability vectors elementwise.

    Args:
        distributions: Probability vectors with identical length.

    Returns:
        list[float]: Elementwise average of the inputs.

    Raises:
        ValueError: If the iterable is empty or vector lengths disagree.
    """

    distribution_list: list[list[float]] = list(distributions)
    if not distribution_list:
        raise ValueError("at least one distribution is required")

    q_value: int = len(distribution_list[0])
    averaged: list[float] = [0.0 for _ in range(q_value)]
    distribution: list[float]
    for distribution in distribution_list:
        if len(distribution) != q_value:
            raise ValueError("all distributions must share the same length")
        index: int
        probability: float
        for index, probability in enumerate(distribution):
            averaged[index] += probability

    num_distributions: int = len(distribution_list)
    index = 0
    while index < q_value:
        averaged[index] /= num_distributions
        index += 1
    return clamp_near_zero_values(averaged)


def clamp_near_zero_values(values: list[float]) -> list[float]:
    """Clamp tiny floating-point noise to exact zero.

    Args:
        values: Probability-like values that may contain numerical noise.

    Returns:
        list[float]: Values with tiny floating-point artifacts set to zero.
    """

    cleaned_values: list[float] = []
    value: float
    for value in values:
        if abs(value) < NUMERICAL_ZERO_TOLERANCE:
            cleaned_values.append(0.0)
        else:
            cleaned_values.append(value)
    return cleaned_values


def compute_tvd(
    reference_distribution: list[float],
    empirical_distribution: list[float],
) -> float:
    """Compute total variation distance between two probability vectors.

    Args:
        reference_distribution: Baseline probability vector.
        empirical_distribution: Comparison probability vector.

    Returns:
        float: Total variation distance.
    """

    total_difference: float = 0.0
    reference_probability: float
    empirical_probability: float
    for reference_probability, empirical_probability in zip(
        reference_distribution,
        empirical_distribution,
        strict=True,
    ):
        total_difference += abs(reference_probability - empirical_probability)
    return 0.5 * total_difference


def compute_max_absolute_difference(
    reference_distribution: list[float],
    empirical_distribution: list[float],
) -> float:
    """Compute the largest pointwise absolute difference between two vectors.

    Args:
        reference_distribution: Baseline probability vector.
        empirical_distribution: Comparison probability vector.

    Returns:
        float: Maximum absolute pointwise difference.
    """

    max_difference: float = 0.0
    reference_probability: float
    empirical_probability: float
    for reference_probability, empirical_probability in zip(
        reference_distribution,
        empirical_distribution,
        strict=True,
    ):
        absolute_difference: float = abs(reference_probability - empirical_probability)
        if absolute_difference > max_difference:
            max_difference = absolute_difference
    return max_difference


def build_subplot_layout(num_panels: int) -> tuple[int, int]:
    """Choose a compact subplot grid for the requested panel count.

    Args:
        num_panels: Number of panels to place.

    Returns:
        tuple[int, int]: Number of rows and columns.
    """

    if num_panels <= 3:
        return 1, num_panels

    num_columns: int = min(3, num_panels)
    num_rows: int = math.ceil(num_panels / num_columns)
    return num_rows, num_columns


def draw_distribution_panel(
    top_axis: Axes,
    bottom_axis: Axes,
    y_values: list[int],
    ideal_distribution: list[float],
    empirical_distribution: list[float],
    panel_title: str,
    simulation_label: str,
    ideal_label: str,
) -> None:
    """Draw one mirrored probability panel with residuals.

    Args:
        top_axis: Axis for the main distribution comparison.
        bottom_axis: Axis for the signed residuals.
        y_values: Discrete support values.
        ideal_distribution: Finite-Q ideal probability vector.
        empirical_distribution: Simulation-derived probability vector.
        panel_title: Title for the top panel.
        simulation_label: Legend label for the simulation bars.
        ideal_label: Legend label for the ideal bars.
    """

    residuals: list[float] = [
        empirical_probability - ideal_probability
        for ideal_probability, empirical_probability in zip(
            ideal_distribution,
            empirical_distribution,
            strict=True,
        )
    ]
    residuals = clamp_near_zero_values(residuals)
    residual_colors: list[str] = [
        "#2a9d8f" if residual >= 0.0 else "#e76f51" for residual in residuals
    ]
    probability_ceiling: float = max(
        max(ideal_distribution),
        max(empirical_distribution),
    )
    residual_ceiling: float = max(abs(residual) for residual in residuals)
    if residual_ceiling == 0.0:
        residual_ceiling = 1e-12

    top_axis.bar(
        y_values,
        empirical_distribution,
        width=0.9,
        color="#4c72b0",
        alpha=0.65,
        label=simulation_label,
    )
    top_axis.bar(
        y_values,
        [-probability for probability in ideal_distribution],
        width=0.9,
        color="#dd8452",
        alpha=0.55,
        label=ideal_label,
    )
    top_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    top_axis.set_title(panel_title)
    top_axis.set_ylabel("Probability")
    top_axis.set_xlim(-0.75, y_values[-1] + 0.75)
    top_axis.set_ylim(
        -probability_ceiling * 1.15 if probability_ceiling > 0 else -1.0,
        probability_ceiling * 1.15 if probability_ceiling > 0 else 1.0,
    )
    top_axis.grid(axis="y", alpha=0.25)
    top_axis.tick_params(labelbottom=False)

    bottom_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    bottom_axis.bar(
        y_values,
        residuals,
        width=0.85,
        color=residual_colors,
        alpha=0.75,
    )
    bottom_axis.set_xlim(-0.75, y_values[-1] + 0.75)
    bottom_axis.set_ylim(-residual_ceiling * 1.15, residual_ceiling * 1.15)
    bottom_axis.set_xlabel("y")
    bottom_axis.set_ylabel("Delta")
    bottom_axis.grid(axis="y", alpha=0.25)


def plot_comparison(
    instance: BenchmarkInstance,
    histograms: dict[int, Counter[int]],
    selected_s_values: list[int],
    include_mixture: bool,
    source_label: str,
    output_path: Path,
    show_plot: bool,
) -> None:
    """Render the finite-Q ideal vs simulation comparison figure.

    Args:
        instance: Benchmark instance.
        histograms: Per-s simulation histograms.
        selected_s_values: Phase labels to visualize.
        include_mixture: Whether to add an equal-weight mixture subplot.
        source_label: Short description of where the histogram data came from.
        output_path: Figure output path.
        show_plot: Whether to open the figure interactively.
    """

    ideal_by_s: dict[int, list[float]] = {}
    empirical_by_s: dict[int, list[float]] = {}
    s_value: int
    for s_value in selected_s_values:
        ideal_by_s[s_value] = finite_q_distribution_for_s(instance=instance, s=s_value)
        empirical_by_s[s_value] = histogram_to_probability_vector(
            histogram=histograms[s_value],
            q_value=instance.q,
        )

    panel_labels: list[str] = [f"s={s_value}" for s_value in selected_s_values]
    if include_mixture:
        panel_labels.append("mixture")

    num_rows: int
    num_columns: int
    num_rows, num_columns = build_subplot_layout(num_panels=len(panel_labels))
    figure_width: float = 5.4 * num_columns
    figure_height: float = 5.6 * num_rows
    fig: Figure
    axes: object
    fig, axes = plt.subplots(
        num_rows * 2,
        num_columns,
        figsize=(figure_width, figure_height),
        squeeze=False,
        sharex=False,
        gridspec_kw={
            "height_ratios": [ratio for _ in range(num_rows) for ratio in (3.0, 1.2)]
        },
    )
    axes_grid: list[list[Axes]] = axes.tolist()
    y_values: list[int] = list(range(instance.q))

    panel_index: int
    for panel_index, s_value in enumerate(selected_s_values):
        row_index: int = panel_index // num_columns
        column_index: int = panel_index % num_columns
        top_axis: Axes = axes_grid[row_index * 2][column_index]
        bottom_axis: Axes = axes_grid[(row_index * 2) + 1][column_index]
        ideal_distribution: list[float] = ideal_by_s[s_value]
        empirical_distribution: list[float] = empirical_by_s[s_value]
        tvd_value: float = compute_tvd(
            reference_distribution=ideal_distribution,
            empirical_distribution=empirical_distribution,
        )
        max_difference: float = compute_max_absolute_difference(
            reference_distribution=ideal_distribution,
            empirical_distribution=empirical_distribution,
        )
        draw_distribution_panel(
            top_axis=top_axis,
            bottom_axis=bottom_axis,
            y_values=y_values,
            ideal_distribution=ideal_distribution,
            empirical_distribution=empirical_distribution,
            panel_title=(
                f"s={s_value} | TVD={tvd_value:.4f} | max|Delta|={max_difference:.4f}"
            ),
            simulation_label="simulation",
            ideal_label="finite-Q ideal",
        )

    if include_mixture:
        mixture_panel_index: int = len(selected_s_values)
        mixture_row_index: int = mixture_panel_index // num_columns
        mixture_column_index: int = mixture_panel_index % num_columns
        mixture_top_axis: Axes = axes_grid[mixture_row_index * 2][mixture_column_index]
        mixture_bottom_axis: Axes = axes_grid[(mixture_row_index * 2) + 1][
            mixture_column_index
        ]
        ideal_mixture: list[float] = average_distributions(
            ideal_by_s[s_value] for s_value in selected_s_values
        )
        empirical_mixture: list[float] = average_distributions(
            empirical_by_s[s_value] for s_value in selected_s_values
        )
        mixture_tvd: float = compute_tvd(
            reference_distribution=ideal_mixture,
            empirical_distribution=empirical_mixture,
        )
        mixture_max_difference: float = compute_max_absolute_difference(
            reference_distribution=ideal_mixture,
            empirical_distribution=empirical_mixture,
        )
        draw_distribution_panel(
            top_axis=mixture_top_axis,
            bottom_axis=mixture_bottom_axis,
            y_values=y_values,
            ideal_distribution=ideal_mixture,
            empirical_distribution=empirical_mixture,
            panel_title=(
                "equal-weight mixture | "
                f"TVD={mixture_tvd:.4f} | "
                f"max|Delta|={mixture_max_difference:.4f}"
            ),
            simulation_label="simulation",
            ideal_label="finite-Q ideal",
        )

    used_panels: int = len(selected_s_values) + (1 if include_mixture else 0)
    axis_index: int
    total_panel_slots: int = num_rows * num_columns
    for axis_index in range(used_panels, total_panel_slots):
        empty_row_index: int = axis_index // num_columns
        empty_column_index: int = axis_index % num_columns
        axes_grid[empty_row_index * 2][empty_column_index].axis("off")
        axes_grid[(empty_row_index * 2) + 1][empty_column_index].axis("off")

    handles: list[Artist]
    labels: list[str]
    handles, labels = axes_grid[0][0].get_legend_handles_labels()
    if include_mixture:
        mixture_handles: list[Artist]
        mixture_labels: list[str]
        mixture_handles, mixture_labels = mixture_top_axis.get_legend_handles_labels()
        handles.extend(mixture_handles)
        labels.extend(mixture_labels)

    deduplicated_handles: list[Artist] = []
    deduplicated_labels: list[str] = []
    label: str
    handle: object
    for handle, label in zip(handles, labels, strict=True):
        if label not in deduplicated_labels:
            deduplicated_handles.append(handle)
            deduplicated_labels.append(label)

    fig.suptitle(
        "Noiseless Shor Simulation vs Finite-Q Ideal\n"
        f"(n={instance.n}, a={instance.a}, r={instance.r}, m={instance.m}; "
        f"{source_label})",
        fontsize=12.0,
        y=0.985,
    )
    fig.legend(
        deduplicated_handles,
        deduplicated_labels,
        loc="upper center",
        ncol=min(2, len(deduplicated_labels)),
        bbox_to_anchor=(0.5, 0.945),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if show_plot:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Run the validation workflow and save the comparison figure."""

    args: argparse.Namespace = parse_args()
    instance: BenchmarkInstance
    histograms: dict[int, Counter[int]]
    source_label: str
    instance, histograms, source_label = load_or_simulate_histograms(args=args)

    selected_s_values: list[int] = resolve_selected_s(
        selected_s_values=args.s,
        instance=instance,
    )
    plot_comparison(
        instance=instance,
        histograms=histograms,
        selected_s_values=selected_s_values,
        include_mixture=not args.hide_mixture,
        source_label=source_label,
        output_path=args.output,
        show_plot=args.show,
    )
    print(f"Saved validation plot to: {args.output}")


if __name__ == "__main__":
    main()
