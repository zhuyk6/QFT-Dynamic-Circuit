from fractions import Fraction
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import typer
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib_config import PlotConfig, get_latex_figsize, plot_context

PLOT_DIR = Path(__file__).resolve().parent
PLOT_CONFIG_PATH: Path = PLOT_DIR / "plot_config.toml"


app = typer.Typer()


def draw_wire(ax: Axes, y: float, x0: float, x1: float):
    ax.plot([x0, x1], [y, y], color="black", linewidth=1.6)


def draw_gate(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    width: float = 0.8,
    height: float = 0.8,
    facecolor: str = "white",
    textcolor: str = "black",
):
    rect = Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        facecolor=facecolor,
        edgecolor="black",
        linewidth=1.6,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=13,
        color=textcolor,
        zorder=4,
    )


def draw_control(ax, x, y, color="black"):
    circ = Circle(
        (x, y), 0.075, facecolor=color, edgecolor=color, linewidth=1.2, zorder=4
    )
    ax.add_patch(circ)


def draw_target(ax, x, y, color="black"):
    circ = Circle(
        (x, y), 0.16, facecolor="white", edgecolor=color, linewidth=1.7, zorder=4
    )
    ax.add_patch(circ)
    ax.plot([x - 0.16, x + 0.16], [y, y], color=color, linewidth=1.5, zorder=5)
    ax.plot([x, x], [y - 0.16, y + 0.16], color=color, linewidth=1.5, zorder=5)


def draw_cnot(ax, x, y_control, y_target, color="black"):
    ax.plot([x, x], [y_control, y_target], color=color, linewidth=1.6, zorder=2)
    draw_control(ax, x, y_control, color=color)
    draw_target(ax, x, y_target, color=color)


def draw_label(ax, x, y, text, color="#8b9cff"):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=15,
        color=color,
        zorder=5,
    )


def draw_slanted_region(ax, points, color="#9ca5ff", alpha=0.65):
    poly = Polygon(
        points, closed=True, facecolor=color, edgecolor="none", alpha=alpha, zorder=0
    )
    ax.add_patch(poly)


def draw_legend(ax: Axes, x: float, y: float):
    ax.add_patch(
        Rectangle(
            (x - 0.8, y - 1.75),
            3.5,
            2.5,
            facecolor="#d3d3d3",
            zorder=0,
        ),
    )
    print(x, y)
    draw_gate(ax, x, y, r"$\theta$", facecolor="black", textcolor="white")
    ax.text(x + 0.6, y, r"$= R_X(\theta)$", ha="left", va="center", fontsize=15)

    draw_gate(ax, x, y - 1, r"$\theta$", facecolor="white", textcolor="black")
    ax.text(x + 0.6, y - 1, r"$= R_Z(\theta)$", ha="left", va="center", fontsize=15)


def _plot_qft_parity(
    output: Annotated[Path, typer.Argument(help="Output plot file path")],
    config: PlotConfig,
) -> None:
    """Draw and save the QFT parity circuit figure."""
    figsize = get_latex_figsize(
        config,
        width="column",
        fraction=0.95,
        height_ratio=0.25,
    )
    scale = 3
    figsize = (figsize[0] * scale, figsize[1] * scale)
    fig, ax = plt.subplots(figsize=figsize)

    ax.axis("off")
    ax.set_aspect("equal")

    # constant parameters
    y_step = 1.0
    x_step = 0.7

    n = 5
    x0, x1 = -1, (6 * n - 1) * x_step
    ys = [y_step * i for i in range(n)]

    # wires
    for y in ys:
        draw_wire(ax, y, x0, x1)

    # initial labels
    for i in range(n):
        draw_label(ax, -1.3, ys[i], f"${i}$")

    # init RZ
    for i in range(n):
        if i == 0:
            draw_gate(ax, 0.0, ys[i], "$H$", facecolor="#d3d3d3")
        else:
            phase = Fraction(0)
            # CP(j, i)
            for j in range(i):
                phase += Fraction(1, (2 ** (i - j + 1)))
            # H(i)
            phase += Fraction(1, 2)
            draw_gate(
                ax,
                0.0,
                ys[i],
                f"$\\frac{{{phase.numerator} \\pi}}{{{phase.denominator}}}$",
                height=0.8,
                width=0.8,
            )

    # PTN
    x_end_ptn: dict[int, float] = {}
    for i in range(n - 1):
        x_start = (1 + i * 6) * x_step

        # PTC(0, n-1-i)
        for j in range(n - 1 - i):
            # DX(j, j+1)
            # qc.cx(j + 1, j)
            # qc.cx(j, j + 1)
            x_pos = x_start + j * 2 * x_step
            draw_cnot(ax, x_pos, ys[j + 1], ys[j])
            draw_cnot(ax, x_pos + x_step, ys[j], ys[j + 1])

        # RZ on physical j: Z{i} Z{i+j+1}
        for j in range(n - 1 - i):
            # qc.rz(-math.pi / 2 ** (j + 2), j)
            x_pos = x_start + (3 + j * 2) * x_step
            draw_gate(
                ax,
                x_pos,
                ys[j],
                f"$\\frac{{-\\pi}}{{{2 ** (j + 2)}}}$",
            )

        # RX on physical 0: RX{i+1}
        # qc.rx(math.pi / 2, 0)
        x_pos = x_start + 5 * x_step
        draw_gate(
            ax,
            x_pos,
            ys[0],
            "$\\frac{{\\pi}}{{2}}$",
            facecolor="black",
            textcolor="white",
        )

        # put parity labels
        for j in range(n - 1 - i):
            x_pos = x_start + (2 + j * 2) * x_step
            draw_label(ax, x_pos - 0.15, ys[j] + 0.15, f"${i}{i + j + 1}$")
        if i == 0:
            draw_label(ax, x_pos - 0.15, ys[j + 1] + 0.15, f"${i}$")
        else:
            draw_label(ax, x_pos - 0.15, ys[j + 1] + 0.15, f"${i - 1}{i}$")

        x_end_ptn[i] = x_pos

        # draw area
        points = [
            (x_start, ys[0]),
            (x_start, ys[1]),
            (x_pos - x_step * 2, ys[n - 1 - i]),
            (x_pos + x_step, ys[n - 1 - i]),
            (x_start + x_step, ys[0]),
        ]
        draw_slanted_region(
            ax,
            points,
        )

    # clean: decode labels
    for idx, i in enumerate(range(n - 1, 1, -1)):
        # qc.cx(i, i - 1, label="decode")
        x_pos = x_end_ptn[idx + 1] + x_step
        draw_cnot(
            ax,
            x_pos,
            ys[i],
            ys[i - 1],
            "purple",
        )
        draw_label(
            ax,
            x_pos + x_step - 0.15,
            ys[i - 1] + 0.15,
            f"${idx + 1}$",
        )
    x_pos += 4 * x_step
    draw_cnot(
        ax,
        x_pos,
        ys[1],
        ys[0],
        "purple",
    )
    draw_label(
        ax,
        x_pos + x_step - 0.15,
        ys[0] + 0.15,
        f"${n - 1}$",
    )

    # final RZ
    x_pos += 2 * x_step
    for i in range(n):
        target = n - 1 - i

        phase = Fraction()
        # CP(i, j)
        for j in range(i + 1, n):
            phase += Fraction(1, (2 ** (j - i + 1)))
        # H(i)
        if i > 0:
            phase += Fraction(1, 2)
        # qc.rz(phase, target)
        draw_gate(
            ax,
            x_pos,
            ys[target],
            f"$\\frac{{{phase.numerator} \\pi}}{{{phase.denominator}}}$",
        )

    # final labels
    x_pos += x_step + 0.2
    for i in range(n):
        draw_label(ax, x_pos, ys[n - 1 - i], f"${i}$")

    # legend
    # draw_legend(ax, x1 + 1.5, ys[n - 1] - 0.5)

    fig.savefig(output)
    plt.close(fig)
    print(f"Saved to {output}")


@app.command()
def main(
    output: Annotated[Path, typer.Argument(help="Output plot file path")],
) -> None:
    """Plot QFT parity using scoped Matplotlib configuration."""
    with plot_context(PLOT_CONFIG_PATH, palette="nature") as config:
        _plot_qft_parity(output=output, config=config)


if __name__ == "__main__":
    app()
