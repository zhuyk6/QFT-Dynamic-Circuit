"""Project-aware Matplotlib helpers for publication figures."""

import shutil
import tomllib
from pathlib import Path
from typing import Literal

from cycler import cycler
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict, Field

PT_PER_INCH: float = 72.27

WidthName = Literal["column", "text"]
LatexMode = Literal["auto"] | bool


class FrozenConfigModel(BaseModel):
    """Base model for immutable project configuration tables."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class LatexConfig(FrozenConfigModel):
    """Layout facts copied from the active LaTeX paper template."""

    column_width_pt: float = Field(gt=0)
    text_width_pt: float = Field(gt=0)
    caption_font_size_pt: float = Field(gt=0)
    body_font_size_pt: float = Field(gt=0)


class FigureConfig(FrozenConfigModel):
    """Project defaults for figure shape and export."""

    default_width: WidthName = "column"
    default_fraction: float = Field(default=0.95, gt=0)
    default_height_ratio: float = Field(default=0.618, gt=0)
    default_dpi: int = Field(default=300, gt=0)
    default_format: str = "pdf"


class FontConfig(FrozenConfigModel):
    """Project defaults for Matplotlib text rendering."""

    use_latex: LatexMode = "auto"
    family: str = "serif"
    serif: tuple[str, ...] = ("Times",)
    latex_preamble: str = ""


class StyleConfig(FrozenConfigModel):
    """Project defaults for line plots and repeated visual styling."""

    palette: str = "nature"
    line_width: float = Field(default=1.5, gt=0)
    marker_size: float = Field(default=4.0, gt=0)
    marker_edge_width: float = Field(default=1.0, gt=0)
    errorbar_capsize: float = Field(default=3.0, ge=0)


class PlotConfig(FrozenConfigModel):
    """All project-level plotting configuration loaded from TOML."""

    latex: LatexConfig
    figure: FigureConfig = Field(default_factory=FigureConfig)
    font: FontConfig = Field(default_factory=FontConfig)
    style: StyleConfig = Field(default_factory=StyleConfig)
    palettes: dict[str, tuple[str, ...]]


def load_plot_config(path: str | Path = "plot_config.toml") -> PlotConfig:
    """Load project plotting configuration from a TOML file.

    Args:
        path: Path to the project plot configuration file.

    Returns:
        Parsed plotting configuration.
    """

    config_path: Path = Path(path)
    with config_path.open("rb") as file:
        raw: dict[str, object] = tomllib.load(file)

    config: PlotConfig = PlotConfig.model_validate(raw)
    return config


def configure_matplotlib(
    config_path: str | Path = "plot_config.toml",
    *,
    use_latex: LatexMode | None = None,
    palette: str | None = None,
) -> PlotConfig:
    """Load project config and apply publication-oriented rcParams.

    Args:
        config_path: Path to the project plot configuration file.
        use_latex: Optional override for LaTeX text rendering.
        palette: Optional override for the configured color palette.

    Returns:
        Parsed plotting configuration after rcParams are applied.
    """

    config: PlotConfig = load_plot_config(config_path)
    latex_mode: LatexMode = config.font.use_latex if use_latex is None else use_latex
    latex_enabled: bool = _resolve_latex_mode(latex_mode)
    palette_name: str = config.style.palette if palette is None else palette

    base_params: dict[str, object] = {
        "font.size": config.latex.caption_font_size_pt,
        "axes.labelsize": config.latex.caption_font_size_pt,
        "axes.titlesize": config.latex.caption_font_size_pt,
        "xtick.labelsize": max(config.latex.caption_font_size_pt - 1, 1),
        "ytick.labelsize": max(config.latex.caption_font_size_pt - 1, 1),
        "legend.fontsize": max(config.latex.caption_font_size_pt - 1, 1),
        "lines.linewidth": config.style.line_width,
        "lines.markersize": config.style.marker_size,
        "lines.markeredgewidth": config.style.marker_edge_width,
        "errorbar.capsize": config.style.errorbar_capsize,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "standard",
        "savefig.dpi": config.figure.default_dpi,
        "savefig.format": config.figure.default_format,
        "text.usetex": latex_enabled,
        "font.family": config.font.family,
    }

    if config.font.family == "serif":
        base_params["font.serif"] = list(config.font.serif)

    if latex_enabled and config.font.latex_preamble:
        base_params["text.latex.preamble"] = config.font.latex_preamble

    plt.rcParams.update(base_params)
    set_palette(config=config, style=palette_name)
    return config


def pt_to_inch(pt: float) -> float:
    """Convert TeX points to inches.

    Args:
        pt: TeX point value.

    Returns:
        Equivalent length in inches.
    """

    return pt / PT_PER_INCH


def get_figsize(
    width_pt: float, fraction: float, height_ratio: float
) -> tuple[float, float]:
    """Return a Matplotlib ``figsize`` tuple in inches.

    Args:
        width_pt: Full figure width in TeX points.
        fraction: Fraction of the width to occupy.
        height_ratio: Figure height divided by figure width.

    Returns:
        Figure size as ``(width_in, height_in)``.
    """

    if width_pt <= 0:
        raise ValueError("width_pt must be positive.")
    if fraction <= 0:
        raise ValueError("fraction must be positive.")
    if height_ratio <= 0:
        raise ValueError("height_ratio must be positive.")

    width_in: float = pt_to_inch(width_pt * fraction)
    return width_in, width_in * height_ratio


def get_latex_figsize(
    config: PlotConfig,
    *,
    width: WidthName | None = None,
    fraction: float | None = None,
    height_ratio: float | None = None,
) -> tuple[float, float]:
    """Return a ``figsize`` tuple from project LaTeX width defaults.

    Args:
        config: Parsed plotting configuration.
        width: Named LaTeX width to use.
        fraction: Fraction of the named width to occupy.
        height_ratio: Figure height divided by figure width.

    Returns:
        Figure size as ``(width_in, height_in)``.
    """

    width_name: WidthName = config.figure.default_width if width is None else width
    width_pt: float = _resolve_width_pt(config=config, width=width_name)
    return get_figsize(
        width_pt=width_pt,
        fraction=config.figure.default_fraction if fraction is None else fraction,
        height_ratio=config.figure.default_height_ratio
        if height_ratio is None
        else height_ratio,
    )


def set_palette(config: PlotConfig, style: str | None = None) -> None:
    """Set Matplotlib's color cycle to a named palette from project config.

    Args:
        config: Parsed plotting configuration.
        style: Palette name from the config, or the default palette when omitted.

    Raises:
        ValueError: If the requested palette is not configured.
    """

    palette_name: str = config.style.palette if style is None else style
    try:
        colors: tuple[str, ...] = config.palettes[palette_name]
    except KeyError as exc:
        choices: str = ", ".join(sorted(config.palettes))
        raise ValueError(
            f"Unknown palette {palette_name!r}. Expected one of: {choices}."
        ) from exc

    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)


def _resolve_width_pt(config: PlotConfig, width: WidthName) -> float:
    """Resolve a named LaTeX width to points.

    Args:
        config: Parsed plotting configuration.
        width: Named LaTeX width.

    Returns:
        Width in TeX points.
    """

    if width == "column":
        return config.latex.column_width_pt
    if width == "text":
        return config.latex.text_width_pt
    raise ValueError("width must be 'column' or 'text'.")


def _resolve_latex_mode(mode: LatexMode) -> bool:
    """Resolve the configured LaTeX mode to a boolean.

    Args:
        mode: ``"auto"``, ``True``, or ``False``.

    Returns:
        Whether Matplotlib should render text with LaTeX.
    """

    if mode == "auto":
        return shutil.which("latex") is not None
    if isinstance(mode, bool):
        return mode
    raise ValueError("use_latex must be true, false, or 'auto'.")
