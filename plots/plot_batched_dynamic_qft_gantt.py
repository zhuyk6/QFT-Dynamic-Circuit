"""Plot a Gantt chart comparing batched dynamic-QFT execution for two batch sizes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import typer
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib_config import (
    PlotConfig,
    get_latex_figsize,
    plot_context,
)

app: typer.Typer = typer.Typer()
PLOT_DIR: Path = Path(__file__).resolve().parent
PLOT_CONFIG_PATH: Path = PLOT_DIR / "plot_config.toml"
CAPTION_FONT_SIZE: float = 9.0
LEGEND_FONT_SIZE: float = 7.0


@dataclass(frozen=True)
class Stage:
    """One execution stage in a batched dynamic-QFT schedule.

    Args:
        name: Human-readable stage name used in the legend.
        label: Compact text drawn inside the Gantt bars.
        duration: Stage duration in arbitrary time units.
        color: Matplotlib-compatible color.

    Returns:
        Stage description used to build each batch schedule.
    """

    name: str
    label: str
    duration: float
    color: str


@dataclass(frozen=True)
class BatchWindow:
    """Timing interval occupied by one stage on one batch.

    Args:
        batch_index: Zero-based batch index.
        qubit_start: First qubit index included in the batch.
        qubit_stop: Exclusive final qubit index included in the batch.
        stage: Stage metadata.
        start: Stage start time.
        stop: Stage stop time.

    Returns:
        Window metadata used when rendering the Gantt chart.
    """

    batch_index: int
    qubit_start: int
    qubit_stop: int
    stage: Stage
    start: float
    stop: float


def _validate_inputs(
    num_qubits: int,
    batch_size: int,
    qft_duration: float,
    measure_duration: float,
    feed_forward_duration: float,
) -> None:
    """Validate Gantt-chart dimensions and stage durations.

    Args:
        num_qubits: Total number of qubits in the example schedule.
        batch_size: Number of qubits executed in each batch.
        qft_duration: QFT stage duration.
        measure_duration: Measurement stage duration.
        feed_forward_duration: Feed-forward stage duration.

    Returns:
        None.
    """

    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_qubits % batch_size != 0:
        raise ValueError("num_qubits must be divisible by batch_size.")

    durations: tuple[float, ...] = (
        qft_duration,
        measure_duration,
        feed_forward_duration,
    )
    if any(duration <= 0 for duration in durations):
        raise ValueError("All stage durations must be positive.")


def _build_stages(
    qft_duration: float,
    measure_duration: float,
    feed_forward_duration: float,
) -> tuple[Stage, ...]:
    """Build the default QFT, measurement, and feed-forward stage sequence.

    Args:
        qft_duration: QFT stage duration.
        measure_duration: Measurement stage duration.
        feed_forward_duration: Feed-forward stage duration.

    Returns:
        Ordered stage metadata for one batch.
    """

    stages: tuple[Stage, ...] = (
        Stage("QFT", "QFT", qft_duration, "C0"),
        Stage("Measurement", "M", measure_duration, "C1"),
        Stage("Feedforward", "FF", feed_forward_duration, "C2"),
    )
    return stages


def _build_schedule(
    num_qubits: int,
    batch_size: int,
    stages: tuple[Stage, ...],
) -> list[BatchWindow]:
    """Build a serial batched execution schedule.

    Args:
        num_qubits: Total number of qubits in the example schedule.
        batch_size: Number of qubits executed in each batch.
        stages: Ordered stage metadata for one batch.

    Returns:
        List of stage windows for all batches.
    """

    batch_count: int = num_qubits // batch_size
    windows: list[BatchWindow] = []
    batch_start: float = 0.0

    for batch_index in range(batch_count):
        stage_start: float = batch_start
        qubit_start: int = batch_index * batch_size
        qubit_stop: int = qubit_start + batch_size
        active_stages: tuple[Stage, ...] = (
            stages[:-1] if batch_index == batch_count - 1 else stages
        )

        for stage in active_stages:
            stage_stop: float = stage_start + stage.duration
            windows.append(
                BatchWindow(
                    batch_index=batch_index,
                    qubit_start=qubit_start,
                    qubit_stop=qubit_stop,
                    stage=stage,
                    start=stage_start,
                    stop=stage_stop,
                )
            )
            stage_start = stage_stop

        batch_start = stage_start

    return windows


def _get_batch_bounds(
    windows: list[BatchWindow],
    batch_count: int,
) -> tuple[list[float], list[float]]:
    """Return start and stop times for each batch.

    Args:
        windows: List of stage windows for all batches.
        batch_count: Number of batches in the schedule.

    Returns:
        Pair of batch-start and batch-stop time lists.
    """

    starts: list[float] = []
    stops: list[float] = []
    for batch_index in range(batch_count):
        batch_windows: list[BatchWindow] = [
            window for window in windows if window.batch_index == batch_index
        ]
        starts.append(min(window.start for window in batch_windows))
        stops.append(max(window.stop for window in batch_windows))

    return starts, stops


def _draw_bar(
    ax: Axes,
    qubit_index: int,
    start: float,
    stop: float,
    facecolor: str,
    label: str | None = None,
    hatch: str | None = None,
) -> None:
    """Draw one horizontal Gantt bar on a qubit row.

    Args:
        ax: Matplotlib axes to draw on.
        qubit_index: Qubit row index.
        start: Bar start time.
        stop: Bar stop time.
        facecolor: Matplotlib-compatible fill color.
        label: Optional text drawn at the center of the bar.
        hatch: Optional hatch pattern for the bar.

    Returns:
        None.
    """

    bar_height: float = 0.72
    bar_y: float = qubit_index - bar_height / 2.0
    width: float = stop - start
    rectangle: Rectangle = Rectangle(
        (start, bar_y),
        width,
        bar_height,
        facecolor=facecolor,
        edgecolor="white",
        hatch=hatch,
    )
    ax.add_patch(rectangle)


def _draw_waiting_regions(
    ax: Axes,
    num_qubits: int,
    batch_size: int,
    batch_starts: list[float],
) -> None:
    """Draw waiting intervals before each later batch begins.

    Args:
        ax: Matplotlib axes to draw on.
        num_qubits: Total number of qubits in the example schedule.
        batch_size: Number of qubits executed in each batch.
        batch_starts: Start time for each batch.

    Returns:
        None.
    """

    batch_count: int = num_qubits // batch_size
    for batch_index in range(1, batch_count):
        wait_stop: float = batch_starts[batch_index]
        qubit_start: int = batch_index * batch_size
        qubit_stop: int = qubit_start + batch_size

        for qubit_index in range(qubit_start, qubit_stop):
            _draw_bar(
                ax=ax,
                qubit_index=qubit_index,
                start=0.0,
                stop=wait_stop,
                facecolor="#E6E6E6",
                label=None,
                hatch="//",
            )


def _draw_schedule(ax: Axes, windows: list[BatchWindow]) -> None:
    """Draw all scheduled QFT, measurement, and feed-forward windows.

    Args:
        ax: Matplotlib axes to draw on.
        windows: List of stage windows for all batches.

    Returns:
        None.
    """

    for window in windows:
        for qubit_index in range(window.qubit_start, window.qubit_stop):
            _draw_bar(
                ax=ax,
                qubit_index=qubit_index,
                start=window.start,
                stop=window.stop,
                facecolor=window.stage.color,
                label=window.stage.label,
            )


def _draw_batch_guides(
    ax: Axes,
    num_qubits: int,
    batch_size: int,
) -> None:
    """Draw horizontal separators between batches.

    Args:
        ax: Matplotlib axes to draw on.
        num_qubits: Total number of qubits in the example schedule.
        batch_size: Number of qubits executed in each batch.

    Returns:
        None.
    """

    batch_count: int = num_qubits // batch_size
    for batch_index in range(1, batch_count):
        separator_y: float = batch_index * batch_size - 0.5
        ax.axhline(separator_y, color="#BDBDBD", linestyle=":", zorder=0)


def _configure_axes(
    ax: Axes,
    num_qubits: int,
    *,
    show_xlabel: bool = True,
) -> None:
    """Configure labels, limits, and ticks for one Gantt subplot.

    Args:
        ax: Matplotlib axes to configure.
        num_qubits: Total number of qubits in the example schedule.
        show_xlabel: Whether to show the x-axis label (``"Time"``).

    Returns:
        None.
    """

    y_ticks: list[int] = list(range(num_qubits))
    y_labels: list[str] = [f"$q_{{{qubit_index}}}$" for qubit_index in y_ticks]

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(num_qubits - 0.35, -0.65)
    ax.set_xticks([])
    ax.tick_params(axis="x", length=0)
    if show_xlabel:
        ax.set_xlabel("Time")
    ax.set_ylabel("Qubit")

    hidden_spines: tuple[str, str] = ("top", "right")
    for spine_name in hidden_spines:
        ax.spines[spine_name].set_visible(False)


def _draw_gantt_on_axes(
    ax: Axes,
    num_qubits: int,
    batch_size: int,
    qft_duration: float,
    measure_duration: float,
    feed_forward_duration: float,
    *,
    show_xlabel: bool = True,
) -> float:
    """Draw a complete Gantt chart for one batch configuration on given axes.

    Args:
        ax: Matplotlib axes to draw on.
        num_qubits: Total number of qubits in the example schedule.
        batch_size: Number of qubits executed in each batch.
        qft_duration: QFT stage duration.
        measure_duration: Measurement stage duration.
        feed_forward_duration: Feed-forward stage duration.
        show_xlabel: Whether to show the x-axis label on this subplot.

    Returns:
        Total duration of the schedule for this batch configuration.
    """

    _validate_inputs(
        num_qubits=num_qubits,
        batch_size=batch_size,
        qft_duration=qft_duration,
        measure_duration=measure_duration,
        feed_forward_duration=feed_forward_duration,
    )

    stages: tuple[Stage, ...] = _build_stages(
        qft_duration=qft_duration,
        measure_duration=measure_duration,
        feed_forward_duration=feed_forward_duration,
    )
    batch_count: int = num_qubits // batch_size
    windows: list[BatchWindow] = _build_schedule(
        num_qubits=num_qubits,
        batch_size=batch_size,
        stages=stages,
    )
    batch_starts: list[float]
    batch_stops: list[float]
    batch_starts, batch_stops = _get_batch_bounds(
        windows=windows,
        batch_count=batch_count,
    )
    total_duration: float = batch_stops[-1]

    _draw_waiting_regions(
        ax=ax,
        num_qubits=num_qubits,
        batch_size=batch_size,
        batch_starts=batch_starts,
    )
    _draw_schedule(ax=ax, windows=windows)
    _draw_batch_guides(
        ax=ax,
        num_qubits=num_qubits,
        batch_size=batch_size,
    )
    _configure_axes(
        ax=ax,
        num_qubits=num_qubits,
        show_xlabel=show_xlabel,
    )

    return total_duration


def _plot_batched_dynamic_qft_gantt_comparison(
    output_filename: Path,
    batch_sizes: list[int],
    qft_durations: list[float],
    config: PlotConfig,
    num_qubits: int = 6,
    measure_duration: float = 1.0,
    feed_forward_duration: float = 1.0,
) -> None:
    """Plot a Gantt chart comparing batched dynamic-QFT across multiple batch sizes.

    Args:
        output_filename: Output figure path.
        batch_sizes: Batch sizes to compare (e.g. ``[1, 2, 3]``).
        qft_durations: QFT stage duration for each batch size (same length as
            ``batch_sizes``).
        num_qubits: Total number of qubits (must be divisible by every batch size).
        measure_duration: Measurement stage duration (shared).
        feed_forward_duration: Feed-forward stage duration (shared).

    Returns:
        None.
    """

    if len(batch_sizes) != len(qft_durations):
        raise ValueError(
            "batch_sizes and qft_durations must have the same length, "
            f"got {len(batch_sizes)} vs {len(qft_durations)}."
        )
    if len(batch_sizes) < 1:
        raise ValueError("At least one batch size is required.")

    for batch_size in batch_sizes:
        if num_qubits % batch_size != 0:
            raise ValueError(
                f"num_qubits ({num_qubits}) must be divisible by "
                f"batch_size ({batch_size})."
            )

    num_panels: int = len(batch_sizes)
    panel_labels: list[str] = [f"$b = {batch_size}$" for batch_size in batch_sizes]

    figsize: tuple[float, float] = get_latex_figsize(
        config,
        width="column",
        fraction=0.95,
        height_ratio=0.48 * num_panels,
    )
    fig: Figure
    axes_flat: list[Axes]
    fig, axes_array = plt.subplots(
        num_panels, 1, figsize=figsize, sharex=True, squeeze=False
    )
    axes_flat = list(axes_array.flatten())

    max_duration: float = 0.0

    for idx, (batch_size, qft_duration) in enumerate(zip(batch_sizes, qft_durations)):
        ax: Axes = axes_flat[idx]
        total_duration: float = _draw_gantt_on_axes(
            ax=ax,
            num_qubits=num_qubits,
            batch_size=batch_size,
            qft_duration=qft_duration,
            measure_duration=measure_duration,
            feed_forward_duration=feed_forward_duration,
            show_xlabel=(idx == num_panels - 1),
        )
        max_duration = max(max_duration, total_duration)
        ax.text(
            0.5,
            0.96,
            panel_labels[idx],
            transform=ax.transAxes,
            va="top",
            ha="center",
            fontsize=CAPTION_FONT_SIZE,
        )

    axes_flat[-1].set_xlim(0.0, max_duration)

    # Shared legend (stage structure is identical; use first config as reference).
    stages_ref: tuple[Stage, ...] = _build_stages(
        qft_duration=qft_durations[0],
        measure_duration=measure_duration,
        feed_forward_duration=feed_forward_duration,
    )
    legend_handles: list[Patch] = [
        Patch(facecolor=stage.color, label=stage.name) for stage in stages_ref
    ]
    legend_handles.append(Patch(facecolor="#E6E6E6", hatch="//", label="Waiting"))
    axes_flat[-1].legend(
        handles=legend_handles,
        loc="lower right",
        ncols=1,
        fontsize=LEGEND_FONT_SIZE,
        # handlelength=1.0,
        # handletextpad=0.4,
        # borderpad=0.3,
        # columnspacing=0.8,
    )

    output_filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_filename)
    plt.close(fig)


def plot_batched_dynamic_qft_gantt_comparison(
    output_filename: Path,
    batch_sizes: list[int],
    qft_durations: list[float],
    num_qubits: int = 6,
    measure_duration: float = 1.0,
    feed_forward_duration: float = 1.0,
) -> None:
    """Plot a Gantt comparison using scoped Matplotlib configuration."""
    with plot_context(PLOT_CONFIG_PATH, palette="nature") as config:
        _plot_batched_dynamic_qft_gantt_comparison(
            output_filename=output_filename,
            batch_sizes=batch_sizes,
            qft_durations=qft_durations,
            config=config,
            num_qubits=num_qubits,
            measure_duration=measure_duration,
            feed_forward_duration=feed_forward_duration,
        )


@app.command()
def main(
    output: Annotated[Path, typer.Argument(help="Output plot file path")],
    batch_sizes: Annotated[
        list[int],
        typer.Option(
            help="Batch sizes to compare (e.g. --batch-sizes 1 2 3)",
            min=1,
        ),
    ] = [1, 2],
    qft_durations: Annotated[
        list[float],
        typer.Option(
            help="QFT stage duration for each batch size "
            "(same length as --batch-sizes)",
        ),
    ] = [30, 110],
    num_qubits: Annotated[
        int,
        typer.Option(
            help="Total number of qubits in the schedule",
            min=2,
        ),
    ] = 6,
    measure_duration: Annotated[
        float, typer.Option(help="Duration of each measurement stage")
    ] = 500,
    feed_forward_duration: Annotated[
        float, typer.Option(help="Duration of each feed-forward stage")
    ] = 200,
) -> None:
    """Create a Gantt chart comparing batched dynamic-QFT across batch sizes."""

    plot_batched_dynamic_qft_gantt_comparison(
        output_filename=output,
        batch_sizes=batch_sizes,
        qft_durations=qft_durations,
        num_qubits=num_qubits,
        measure_duration=measure_duration,
        feed_forward_duration=feed_forward_duration,
    )


if __name__ == "__main__":
    app()
