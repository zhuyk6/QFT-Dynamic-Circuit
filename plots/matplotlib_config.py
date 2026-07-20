"""Project-aware, context-local Matplotlib configuration helpers."""

import tomllib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib as mpl
from cycler import cycler
from matplotlib import style as mpl_style
from pydantic import BaseModel, ConfigDict, Field

PT_PER_INCH = 72.27
DEFAULT_STYLE_PATH = Path(__file__).with_name("paper.mplstyle")

WidthName = Literal["column", "text"]


class FrozenConfigModel(BaseModel):
    """Base model for immutable project configuration tables."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class LatexConfig(FrozenConfigModel):
    """Width facts copied from the active LaTeX paper template."""

    column_width_pt: float = Field(gt=0)
    text_width_pt: float = Field(gt=0)


class PlotConfig(FrozenConfigModel):
    """Project metadata and the available, explicitly selected palettes."""

    latex: LatexConfig
    palettes: dict[str, tuple[str, ...]]


def load_plot_config(path: str | Path = "plot_config.toml") -> PlotConfig:
    """Load and validate project plotting configuration from TOML."""

    config_path = Path(path)
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    return PlotConfig.model_validate(raw)


@contextmanager
def plot_context(
    config_path: str | Path = "plot_config.toml",
    *,
    palette: str | None = None,
    style_path: str | Path = DEFAULT_STYLE_PATH,
    rc: Mapping[str, Any] | None = None,
) -> Iterator[PlotConfig]:
    """Apply static style and dynamic choices without leaking global state.

    Select ``palette`` when the figure needs a project-defined color cycle.
    Use ``rc`` only for exceptional context-wide overrides; its values take
    precedence over both the static style and the selected palette.
    """

    config = load_plot_config(config_path)
    dynamic_rc: dict[str, Any] = {}
    if palette is not None:
        dynamic_rc["axes.prop_cycle"] = cycler(color=_resolve_palette(config, palette))
    if rc is not None:
        dynamic_rc.update(rc)

    # Matplotlib's stubs enumerate every valid rcParam key as literals, while
    # this dictionary is assembled dynamically and validated by Matplotlib.
    with mpl_style.context(style_path), mpl.rc_context(cast(Any, dynamic_rc)):
        yield config


def pt_to_inch(pt: float) -> float:
    """Convert TeX points to inches."""

    return pt / PT_PER_INCH


def _get_figsize(
    width_pt: float, fraction: float, height_ratio: float
) -> tuple[float, float]:
    """Return a Matplotlib ``figsize`` tuple in inches."""

    if width_pt <= 0:
        raise ValueError("width_pt must be positive.")
    if fraction <= 0:
        raise ValueError("fraction must be positive.")
    if height_ratio <= 0:
        raise ValueError("height_ratio must be positive.")

    width_in = pt_to_inch(width_pt * fraction)
    return width_in, width_in * height_ratio


def get_latex_figsize(
    config: PlotConfig,
    *,
    width: WidthName,
    fraction: float,
    height_ratio: float,
) -> tuple[float, float]:
    """Return an explicitly specified ``figsize`` for the current figure."""

    return _get_figsize(_resolve_width_pt(config, width), fraction, height_ratio)


def _resolve_palette(config: PlotConfig, palette: str) -> tuple[str, ...]:
    """Return a named palette, with a useful error for invalid choices."""

    try:
        return config.palettes[palette]
    except KeyError as exc:
        choices = ", ".join(sorted(config.palettes))
        raise ValueError(
            f"Unknown palette {palette!r}. Expected one of: {choices}."
        ) from exc


def _resolve_width_pt(config: PlotConfig, width: WidthName) -> float:
    """Resolve a named LaTeX width to points."""

    if width == "column":
        return config.latex.column_width_pt
    if width == "text":
        return config.latex.text_width_pt
    raise ValueError(f"width must be 'column' or 'text', got {width!r}.")
