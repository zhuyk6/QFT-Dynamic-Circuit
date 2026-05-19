---
name: matplotlib
description: Create Matplotlib figures, especially academic or publication-ready plots, using project-level plotting configuration. Use when agent needs to generate, edit, or review Python plotting code with Matplotlib, including line plots, scatter plots, bar charts, heatmaps, multi-panel figures, and saved paper figures.
---

# Matplotlib

Use the bundled configuration files for publication-style defaults.

## Setup

- If the target repo lacks `matplotlib_config.py`, copy `assets/matplotlib_config.py` into the repo. Make sure to place this file
to the same directory as the plotting scripts.
- If the target repo lacks `plot_config.toml`, copy `assets/plot_config.toml` into the repo and ask the user to edit the LaTeX widths, font sizes, palettes, and style defaults.
- Import and configure once near the plotting entrypoint:

```python
from matplotlib import pyplot as plt

from matplotlib_config import configure_matplotlib, get_latex_figsize

config = configure_matplotlib("plot_config.toml")
```

## Plotting Rules

- Compute figure size with `get_latex_figsize(config, ...)`.
- Use native Matplotlib APIs directly: `plt.subplots`, `plt.subplot_mosaic`, `GridSpec`, or `Figure.subfigures`.
- Do not set font sizes in plotting code; use `plot_config.toml`.
- Do not set `linewidth` in `plot`; use `plot_config.toml`.
- Prefer color cycle colors: omit `color`, or use `"C0"`, `"C1"`, etc.
- Do not pass `layout`; constrained layout is configured globally.
- Do not use `tight_layout`.
- Do not save with `bbox_inches="tight"` for paper figures.
- Save with native `fig.savefig(...)`.

## Examples

Line plot:

```python
figsize = get_latex_figsize(config, width="column", fraction=0.95)
fig, ax = plt.subplots(figsize=figsize)

ax.plot(x, y, marker="o", label="Method A")
ax.plot(x, z, marker="s", label="Method B")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()

fig.savefig("figures/line.pdf")
```

Scatter plot:

```python
figsize = get_latex_figsize(config, width="column", height_ratio=0.75)
fig, ax = plt.subplots(figsize=figsize)

ax.scatter(x, y, s=16, color="C0", label="Samples")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

fig.savefig("figures/scatter.pdf")
```

Two-panel figure:

```python
figsize = get_latex_figsize(config, width="text", height_ratio=0.45)
fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

axes[0].plot(x, y1, marker="o", label="A")
axes[1].plot(x, y2, marker="s", label="B")
for ax in axes:
    ax.set_xlabel("Step")
    ax.legend()
axes[0].set_ylabel("Score")

fig.savefig("figures/two_panel.pdf")
```
