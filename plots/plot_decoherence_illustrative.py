"""Plot decoherence exposure versus dynamic-QFT batch size.
This is an illustrative version of the `plot_decoherence.py` script,
which is used to generate the decoherence exposure plot for the dynamic-QFT circuit.
The main difference is that it does not use the real physical parameters and just uses
a simple model to illustrate the effect of batch size on decoherence exposure.
"""

from pathlib import Path
from typing import Annotated

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


def generate_decoherence_data(
    max_x: int,
    steps: int,
    a1: float,
    b1: float,
    a2: float,
    b2: float,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Generate decoherence exposure data for plotting.

    Args:
        max_x (int): Maximum batch size to consider.
        steps (int): Number of batch sizes to consider.
        a1 (float): Coefficient for QFT exposure.
        b1 (float): Exponent for QFT exposure.
        a2 (float): Coefficient for waiting exposure.
        b2 (float): Exponent for waiting exposure.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
            Arrays of total, QFT, and waiting exposures.
    """

    xs: NDArray[np.float64] = np.linspace(1, max_x, steps)
    qft_exposure: NDArray[np.float64] = a1 * xs + b1  # O(x)
    wait_exposure: NDArray[np.float64] = a2 / xs + b2  # O(1 / x)
    total_exposure: NDArray[np.float64] = qft_exposure + wait_exposure

    return xs, total_exposure, qft_exposure, wait_exposure


def draw_decoherence_plot(
    ax: Axes,
    xs: NDArray[np.float64],
    total: NDArray[np.float64],
    qft: NDArray[np.float64],
    wait: NDArray[np.float64],
    meas: NDArray[np.float64],
) -> None:
    """Draw decoherence exposure curves onto an axis."""

    ax.plot(
        xs,
        total,
        label="Total exposure",
    )
    ax.plot(
        xs,
        qft,
        label=r"QFT exposure $O(b)$",
        linestyle="--",
    )
    ax.plot(
        xs,
        wait,
        label=r"M + FF Waiting $O(1/b)$",
        linestyle="-.",
    )
    ax.plot(
        xs,
        meas,
        label="Measurement",
        linestyle=":",
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

    ax.set_xlabel(r"Batch size $b$")
    ax.set_ylabel("Decoherence exposure")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.1, 1.0),
        fontsize=PLOT_CONFIG.latex.caption_font_size_pt - 3,
    )
    ax.set_ylim(bottom=0)

    # remove x, y ticks and tick labels
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )


def plot_decoherence(
    output: Path,
    max_x: int,
    steps: int,
    a1: float,
    b1: float,
    a2: float,
    b2: float,
    b3: float,
) -> None:
    """Plot decoherence exposure versus batch size and save the figure."""

    xs, total, qft, wait = generate_decoherence_data(
        max_x=max_x,
        steps=steps,
        a1=a1,
        b1=b1,
        a2=a2,
        b2=b2,
    )
    meas = np.full_like(xs, b3)  # constant measurement exposure
    total += meas  # add measurement exposure to total

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
        xs=xs,
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
    max_x: Annotated[int, typer.Option(help="Maximum batch size to consider.")] = 20,
    steps: Annotated[int, typer.Option(help="Number of steps in the plot.")] = 100,
    a1: Annotated[float, typer.Option(help="Coefficient for QFT exposure.")] = 5.0,
    b1: Annotated[float, typer.Option(help="Intercept for QFT exposure.")] = -5.0,
    a2: Annotated[
        float, typer.Option(help="Coefficient for waiting exposure.")
    ] = 100.0,
    b2: Annotated[float, typer.Option(help="Intercept for waiting exposure.")] = 0.0,
    b3: Annotated[
        float, typer.Option(help="Intercept for measurement exposure.")
    ] = 10.0,
) -> None:
    """Run the default decoherence exposure plotting workflow."""

    plot_decoherence(
        output=output,
        max_x=max_x,
        steps=steps,
        a1=a1,
        b1=b1,
        a2=a2,
        b2=b2,
        b3=b3,
    )


if __name__ == "__main__":
    app()
