"""Plot decoherence exposure versus dynamic-QFT batch size."""

from pathlib import Path
from typing import Annotated, Callable

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib_config import PlotConfig, configure_matplotlib, get_latex_figsize
from numpy.typing import NDArray

PLOT_DIR: Path = Path(__file__).resolve().parent
PLOT_CONFIG: PlotConfig = configure_matplotlib(PLOT_DIR / "plot_config.toml")

app = typer.Typer()


def calc_exposure_components(
    n: int,
    b: int,
    calc_qft_time: Callable[[int], float],
    t_measure: float,
    t_feedforward: float,
) -> tuple[float, float, float, float]:
    """Calculate decoherence exposure components for one batch size.

    Args:
        n (int): Total number of qubits.
        b (int): Batch size. It must divide ``n``.
        calc_qft_time (Callable[[int], float]): Function returning the QFT-block
            duration for a given batch size.
        t_measure (float): Measurement duration.
        t_feedforward (float): Feed-forward duration.

    Returns:
        tuple[float, float, float, float]: Total, QFT, waiting, and measurement
            baseline exposure values.
    """

    assert n % b == 0, "n must be divisible by b"

    num_batches: int = n // b
    t_qft: float = calc_qft_time(b)

    # QFT-block exposure:
    # b * sum_{k=0}^{L-1} (k + 1) * T_qft
    qft_exposure: float = b * t_qft * num_batches * (num_batches + 1) / 2

    # Inter-batch waiting caused by previous measurement + feed-forward rounds:
    # b * sum_{k=0}^{L-1} k * (T_m + T_FF)
    wait_exposure: float = (
        b * (t_measure + t_feedforward) * num_batches * (num_batches - 1) / 2
    )

    # Own measurement duration:
    # b * L * T_m = n * T_m
    # This is independent of b and therefore does not determine the optimum.
    measurement_baseline: float = n * t_measure

    total_exposure: float = qft_exposure + wait_exposure + measurement_baseline

    return total_exposure, qft_exposure, wait_exposure, measurement_baseline


def calc_qft_time(batch_size: int, t_cnot: float) -> float:
    """Calculate the QFT-block duration from the CNOT duration.

    Args:
        batch_size (int): Batch size used by the dynamic-QFT circuit.
        t_cnot (float): Duration of one CNOT gate.

    Returns:
        float: QFT-block duration.
    """
    assert batch_size > 0, "batch_size must be positive"

    depth_cnots: int
    match batch_size:
        case 1:
            depth_cnots = 0
        case 2:
            depth_cnots = 1
        case 3:
            depth_cnots = 5
        case n:
            depth_cnots = 4 * n - 6

    return depth_cnots * t_cnot


def get_valid_batch_sizes(num_qubits: int) -> list[int]:
    """Return nontrivial batch sizes that divide the number of qubits.

    Args:
        num_qubits (int): Total number of qubits.

    Returns:
        list[int]: Divisors of ``num_qubits`` excluding ``num_qubits`` itself.
    """

    batch_sizes: list[int] = [
        batch_size
        for batch_size in range(1, num_qubits + 1)
        if num_qubits % batch_size == 0
    ]
    batch_sizes.pop()
    return batch_sizes


def collect_exposure_series(
    num_qubits: int,
    batch_sizes: list[int],
    t_measure: float,
    t_feedforward: float,
    t_cnot: float,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Collect exposure series for all selected batch sizes.

    Args:
        num_qubits (int): Total number of qubits.
        batch_sizes (list[int]): Batch sizes to evaluate.
        t_measure (float): Measurement duration.
        t_feedforward (float): Feed-forward duration.
        t_cnot (float): Duration of one CNOT gate.

    Returns:
        tuple[NDArray, NDArray, NDArray, NDArray]: Total, QFT,
            waiting, and measurement-baseline series.
    """

    total: list[float] = []
    qft: list[float] = []
    wait: list[float] = []
    meas: list[float] = []

    for batch_size in batch_sizes:
        total_exposure: float
        qft_exposure: float
        wait_exposure: float
        measurement_baseline: float
        total_exposure, qft_exposure, wait_exposure, measurement_baseline = (
            calc_exposure_components(
                n=num_qubits,
                b=batch_size,
                calc_qft_time=lambda b: calc_qft_time(b, t_cnot),
                t_measure=t_measure,
                t_feedforward=t_feedforward,
            )
        )
        total.append(total_exposure)
        qft.append(qft_exposure)
        wait.append(wait_exposure)
        meas.append(measurement_baseline)

        # print(f"b={batch_size}, total_exposure={total_exposure:.2f} ns")
        print(f"{batch_size=}")
        print(f"{total_exposure=:.2f} ns")
        print(f"{qft_exposure=:.2f} ns")
        print(f"{wait_exposure=:.2f} ns")

    return np.array(total), np.array(qft), np.array(wait), np.array(meas)


def draw_decoherence_plot(
    ax: Axes,
    batch_sizes: list[int],
    total: NDArray,
    qft: NDArray,
    wait: NDArray,
    meas: NDArray,
) -> None:
    """Draw decoherence exposure curves onto an axis.

    Args:
        ax (Axes): Axis to draw on.
        batch_sizes (list[int]): Batch-size x coordinates.
        total (NDArray): Total exposure values.
        qft (NDArray): QFT exposure values.
        wait (NDArray): Measurement and feed-forward waiting values.
        meas (NDArray): Measurement baseline values.
    """

    # convert to microseconds for plotting
    total /= 1000
    qft /= 1000
    wait /= 1000
    meas /= 1000

    ax.plot(
        batch_sizes,
        total,
        label="Total exposure",
        marker="o",
    )
    ax.plot(
        batch_sizes,
        qft,
        label="QFT exposure",
        marker="^",
        linestyle="--",
    )
    ax.plot(
        batch_sizes,
        wait,
        label="M + FF Waiting",
        marker="s",
        linestyle="--",
    )
    ax.axhline(
        y=meas[0],
        color="gray",
        linestyle=":",
        label="Measurement",
    )

    # ========================================================
    # # add text annotation and arrows
    # # left: many dynamic rounds
    # point: tuple[float, float] = (batch_sizes[0] + 1, wait[0] + 25)
    # ax.add_patch(
    #     plt.Arrow(
    #         x=point[0],
    #         y=point[1],
    #         dx=-0.6,
    #         dy=-20,
    #         width=0.2,
    #         color="C2",
    #     )
    # )
    # ax.annotate(
    #     "Many dynamic rounds",
    #     xy=point,
    #     xytext=(point[0] - 1.0, point[1] + 5),
    #     fontsize=5,
    #     color="C2",
    # )

    # # right: deep qft blocks
    # point = (batch_sizes[-1] - 1, qft[-1] + 10)
    # ax.add_patch(
    #     plt.Arrow(
    #         x=point[0],
    #         y=point[1],
    #         dx=0.7,
    #         dy=-5,
    #         width=0.2,
    #         color="C1",
    #     )
    # )
    # ax.annotate(
    #     "Deep QFT blocks",
    #     xy=point,
    #     xytext=(point[0] - 3.0, point[1] + 5),
    #     fontsize=5,
    #     color="C1",
    # )

    # ========================================================

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Decoherence exposure (us)")
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.45),
        fontsize=PLOT_CONFIG.latex.caption_font_size_pt - 3,
    )
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)


def plot_decoherence(
    output: Path,
    num_qubits: int,
    t_measure: float,
    t_feedforward: float,
    t_cnot: float,
) -> None:
    """Plot decoherence exposure versus batch size and save the figure.

    Args:
        output (Path): Output plot file path.

    Returns:
        None.
    """

    batch_sizes: list[int] = get_valid_batch_sizes(num_qubits)
    total: NDArray
    qft: NDArray
    wait: NDArray
    meas: NDArray
    total, qft, wait, meas = collect_exposure_series(
        num_qubits=num_qubits,
        batch_sizes=batch_sizes,
        t_measure=t_measure,
        t_feedforward=t_feedforward,
        t_cnot=t_cnot,
    )

    figsize: tuple[float, float] = get_latex_figsize(
        PLOT_CONFIG,
        width="column",
        fraction=0.95,
    )
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=figsize)
    draw_decoherence_plot(
        ax=ax,
        batch_sizes=batch_sizes,
        total=total,
        qft=qft,
        wait=wait,
        meas=meas,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"Saved plot to {output}")


@app.command()
def cli(
    output: Annotated[Path, typer.Argument(help="Output plot file path.")],
    num_qubits: Annotated[int, typer.Option(help="Total number of qubits.")] = 24,
    t_measure: Annotated[float, typer.Option(help="Measurement duration (ns).")] = 500,
    t_feedforward: Annotated[
        float, typer.Option(help="Feed-forward duration (ns).")
    ] = 200,
    t_cnot: Annotated[float, typer.Option(help="CNOT gate duration (ns).")] = 50,
) -> None:
    """Run the default decoherence exposure plotting workflow."""

    plot_decoherence(
        output=output,
        num_qubits=num_qubits,
        t_measure=t_measure,
        t_feedforward=t_feedforward,
        t_cnot=t_cnot,
    )


if __name__ == "__main__":
    app()
