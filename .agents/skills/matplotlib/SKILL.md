---
name: matplotlib
description: Create, edit, or review publication-ready Matplotlib figures with scoped styling, LaTeX-derived figure sizes, and explicit per-figure color choices. Use for academic line plots, scatter plots, bar charts, heatmaps, multi-panel figures, and saved paper figures where consistent appearance and no global rcParams pollution matter.
---

# Matplotlib

Use the bundled assets to separate stable paper styling from per-figure choices.

## Set up the target project

- Copy `assets/matplotlib_config.py` and `assets/paper.mplstyle` into the same directory as the plotting scripts.
- Copy `assets/plot_config.toml` to the project root unless the project already has an equivalent file.
- Edit `paper.mplstyle` for stable paper settings such as fonts, font sizes, line widths, markers, layout, export defaults, and `text.usetex`.
- Edit `plot_config.toml` for LaTeX column/text widths and available named palettes.
- Keep `paper.mplstyle` next to `matplotlib_config.py`; its default path is resolved relative to that module.

## Generate a figure

Wrap figure creation and saving in `plot_context`. This applies the style and optional dynamic color cycle only inside the block and restores prior `rcParams` on exit.

```python
from matplotlib import pyplot as plt

from matplotlib_config import get_latex_figsize, plot_context

with plot_context("plot_config.toml", palette="nature") as config:
    figsize = get_latex_figsize(
        config,
        width="column",
        fraction=0.95,
        height_ratio=0.618,
    )
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y, marker="o", label="Method A")
    ax.plot(x, z, marker="s", label="Method B")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()

    fig.savefig("figures/line.pdf")
    plt.close(fig)
```

Supply `width`, `fraction`, and `height_ratio` for every figure. Select a named `palette` when a custom cycle is needed; omit it to retain the cycle from the static style. Use `rc={...}` only for rare, context-wide overrides. Prefer Artist arguments for one-off changes to a single line, legend, or axis.

## Follow plotting rules

- Use native Matplotlib APIs: `plt.subplots`, `plt.subplot_mosaic`, `GridSpec`, or `Figure.subfigures`.
- Keep stable visual defaults out of plotting code; configure them in `paper.mplstyle`.
- Treat figure size, palette selection, labels, layout structure, and data semantics as per-figure choices.
- Prefer the selected color cycle: omit `color`, or use `"C0"`, `"C1"`, and so on.
- Keep figure construction and `fig.savefig(...)` inside `plot_context`.
- Do not call `tight_layout`; constrained layout is enabled by the static style.
- Do not save paper figures with `bbox_inches="tight"`.
- Close saved figures with `plt.close(fig)`.

## Handle LaTeX fonts

Keep LaTeX rendering settings static in `paper.mplstyle`. Leave `text.usetex: False` for a portable default. When exact paper-font matching requires LaTeX, set `text.usetex: True` and configure a supported font there. Add `text.latex.preamble` only when the paper depends on specific LaTeX font/math packages or commands, and verify the required TeX tools and packages are installed.

## Additional examples

Scatter plot:

```python
with plot_context("plot_config.toml", palette="nature") as config:
    figsize = get_latex_figsize(
        config, width="column", fraction=0.95, height_ratio=0.75
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=16, color="C0", label="Samples")
    ax.set(xlabel="Feature 1", ylabel="Feature 2")
    ax.legend()
    fig.savefig("figures/scatter.pdf")
    plt.close(fig)
```

Two-panel figure:

```python
with plot_context("plot_config.toml", palette="binary") as config:
    figsize = get_latex_figsize(
        config, width="text", fraction=0.95, height_ratio=0.45
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    axes[0].plot(x, y1, marker="o", label="A")
    axes[1].plot(x, y2, marker="s", label="B")
    for ax in axes:
        ax.set_xlabel("Step")
        ax.legend()
    axes[0].set_ylabel("Score")
    fig.savefig("figures/two-panel.pdf")
    plt.close(fig)
```
