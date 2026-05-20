"""Plot process-fidelity benchmark results."""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib_config import PlotConfig, configure_matplotlib, get_latex_figsize

app = typer.Typer()
PLOT_DIR: Path = Path(__file__).resolve().parent
PLOT_CONFIG: PlotConfig = configure_matplotlib(PLOT_DIR / "plot_config.toml")


def _load_benchmark_results(
    results_dir: Path,
) -> dict[int, dict[str, np.ndarray[tuple[int], np.dtype[np.floating]]]]:
    """Aggregate benchmark JSON files into mean/std series per batch size."""
    files = sorted(results_dir.glob("qft*.json"))
    pattern = re.compile(r"^qft(\d+)(?:_(\d+))?\.json$")
    agg: defaultdict[int, defaultdict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for fp in files:
        match = pattern.match(fp.name)
        if match is None:
            continue

        num_qubits: int = int(match.group(1))
        with open(fp, "r") as input_file:
            payload = json.load(input_file)

        fidelity_by_batch_size: dict[str, float] = payload.get(
            "fidelity_by_batch_size", {}
        )

        for batch_key, fid in fidelity_by_batch_size.items():
            batch_size: int = int(batch_key)
            agg[batch_size][num_qubits].append(fid)

    stats: dict[int, dict[str, np.ndarray[tuple[int], np.dtype[np.floating]]]] = {}
    for batch_size, n_dict in sorted(agg.items()):
        n_list: list[int] = sorted(n_dict.keys())
        mean_list: list[float] = []
        std_list: list[float] = []

        for num_qubits in n_list:
            arr = np.array(n_dict[num_qubits], dtype=float)
            mean_list.append(arr.mean())
            std_list.append(arr.std(ddof=0))

        stats[batch_size] = {
            "n": np.array(n_list, dtype=int),
            "mean": np.array(mean_list, dtype=float),
            "std": np.array(std_list, dtype=float),
        }

    return stats


def _snap_to_integer_x(
    points: list[tuple[float, float]],
    x_min: int | None = None,
    x_max: int | None = None,
) -> list[tuple[float, float]]:
    """Keep only one point per integer x after rounding."""
    best: dict[int, tuple[float, float, float]] = {}
    for x_value, y_value in points:
        rounded_x: int = int(round(x_value))
        if x_min is not None and rounded_x < x_min:
            continue
        if x_max is not None and rounded_x > x_max:
            continue

        dist: float = abs(x_value - rounded_x)
        previous = best.get(rounded_x)
        if previous is None:
            best[rounded_x] = (dist, x_value, y_value)
        else:
            prev_dist, _prev_x, prev_y = previous
            if dist < prev_dist or (math.isclose(dist, prev_dist) and y_value > prev_y):
                best[rounded_x] = (dist, x_value, y_value)

    return [(float(x), best[x][2]) for x in sorted(best.keys())]


def _load_baseline(baseline_csv: Path) -> dict[str, list[tuple[float, float]]]:
    """Load baseline curves from the two-row-header CSV export."""
    dataframe = pd.read_csv(baseline_csv, header=[0, 1])
    level0_idx = pd.Index(dataframe.columns.get_level_values(0), dtype="object")
    level1_idx = pd.Index(dataframe.columns.get_level_values(1), dtype="object")
    level0 = level0_idx.to_series().replace(r"^Unnamed:.*$", pd.NA, regex=True).ffill()
    level1 = level1_idx.to_series().replace(r"^Unnamed:.*$", pd.NA, regex=True)
    dataframe.columns = pd.MultiIndex.from_arrays([level0, level1])

    methods = [m for m in dataframe.columns.get_level_values(0).unique()]
    raw_data: dict[str, list[tuple[float, float]]] = {}

    method: str
    for method in methods:
        assert (method, "X") in dataframe.columns or (method, "Y") in dataframe.columns
        x_series = pd.to_numeric(dataframe[(method, "X")], errors="coerce")
        y_series = pd.to_numeric(dataframe[(method, "Y")], errors="coerce")
        mask = x_series.notna() & y_series.notna()
        raw_data[str(method)] = list(
            zip(
                x_series[mask].astype(float).to_list(),
                y_series[mask].astype(float).to_list(),
            )
        )

    return {m: _snap_to_integer_x(data, 2, 40) for m, data in raw_data.items()}


def plot_result(
    results_dir: Path | None,
    baseline_csv: Path | None,
    output_filename: Path,
) -> None:
    """Plot fidelity results from benchmark JSON files and optional baselines."""
    if results_dir is None and baseline_csv is None:
        raise ValueError("Either `results_dir` or `baseline_csv` must be provided.")

    results = _load_benchmark_results(results_dir) if results_dir is not None else None
    baseline = _load_baseline(baseline_csv) if baseline_csv is not None else None

    figsize: tuple[float, float] = get_latex_figsize(
        PLOT_CONFIG,
        width="text",
        fraction=0.95,
        height_ratio=0.62,
    )
    fig, ax = plt.subplots(figsize=figsize)

    if baseline is not None:
        for method, data in baseline.items():
            x_list = [x for x, _ in data]
            y_list = [y for _, y in data]
            color: str = "C0" if method.startswith("unitary") else "C1"
            linestyle: str = "dashed" if method.endswith("no DD") else "-"
            ax.plot(x_list, y_list, label=method, marker="o", color=color, ls=linestyle)

    if results is not None:
        for batch_size in sorted(results.keys()):
            x = results[batch_size]["n"]
            y = results[batch_size]["mean"]
            yerr = results[batch_size]["std"]
            ax.errorbar(x, y, yerr=yerr, marker="x", label=f"batch size = {batch_size}", color=f"C{batch_size + 1}")

    ax.set_xlim(2, 12 if baseline is None else 40)
    ax.set_ylim(0, 1)
    ax.legend()

    output_filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_filename)


@app.command()
def main(
    output: Annotated[Path, typer.Argument(help="Output plot file path")],
    results_dir: Annotated[
        Path | None, typer.Option(help="Directory of benchmark JSON files")
    ] = None,
    baseline_csv: Annotated[
        Path | None, typer.Option(help="Baseline CSV file path")
    ] = None,
) -> None:
    """Plot dynamic-QFT fidelity results."""
    plot_result(results_dir, baseline_csv, output)


if __name__ == "__main__":
    app()
