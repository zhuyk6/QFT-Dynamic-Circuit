"""Plot process fidelity versus qubit count from a directory of datasets."""

from collections import defaultdict
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib_config import get_latex_figsize, plot_context
from plot_fidelity import ProcessFidelitySeries, build_process_fidelity_series

from qft_dynamic.experiment_data import (
    ExperimentDataset,
    MetadataValue,
    counts_to_probabilities,
)

app = typer.Typer()
PLOT_DIR: Path = Path(__file__).resolve().parent
PLOT_CONFIG_PATH: Path = PLOT_DIR / "plot_config.toml"


def _load_scaling_results(
    input_dir: Path,
) -> dict[int, dict[str, list[float] | list[int]]]:
    """Aggregate process fidelity from all counter datasets in a directory.

    Args:
        input_dir: Directory whose JSON files each contain one qubit scale of a
            process-fidelity experiment.

    Returns:
        Mean and standard deviation by batch size and qubit count.

    Raises:
        ValueError: If the directory has no JSON datasets or contains a dataset
            from another experiment family.
    """

    files: list[Path] = sorted(input_dir.glob("*.json"))
    if not files:
        raise ValueError(f"input directory contains no JSON files: {input_dir}")
    agg: defaultdict[int, defaultdict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for dataset_path in files:
        dataset: ExperimentDataset = ExperimentDataset.load(dataset_path)
        if dataset.experiment_type != "process_fidelity":
            raise ValueError(
                f"dataset {dataset_path} has experiment_type="
                f"{dataset.experiment_type!r}, expected 'process_fidelity'"
            )
        probability_dataset = counts_to_probabilities(dataset)
        series_list: list[ProcessFidelitySeries] = build_process_fidelity_series(
            probability_dataset
        )
        for group, series in zip(probability_dataset.groups, series_list, strict=True):
            batch_value: MetadataValue | None = group.attributes.get("batch_size")
            if isinstance(batch_value, bool) or not isinstance(batch_value, int):
                raise ValueError(f"group {group.name!r} requires integer batch_size")
            agg[batch_value][dataset.num_qubits].append(series.process_fidelity)

    stats: dict[int, dict[str, list[float] | list[int]]] = {}
    for batch_size, n_dict in sorted(agg.items()):
        n_list: list[int] = sorted(n_dict.keys())
        n_fidelities = [np.array(n_dict[n], dtype=np.float64) for n in n_list]
        n_fidelity_means = [float(np.mean(fidelities)) for fidelities in n_fidelities]
        n_fidelity_stds = [float(np.std(fidelities)) for fidelities in n_fidelities]

        stats[batch_size] = {
            "n": n_list,
            "fidelity_means": n_fidelity_means,
            "fidelity_stds": n_fidelity_stds,
        }

    return stats


def plot_result(
    input_dir: Path,
    output_filename: Path,
) -> None:
    """Plot aggregated fidelity scaling from counter dataset JSON files."""

    results = _load_scaling_results(input_dir)

    with plot_context(PLOT_CONFIG_PATH, palette="nature") as config:
        figsize: tuple[float, float] = get_latex_figsize(
            config,
            width="column",
            fraction=0.95,
            height_ratio=0.62,
        )
        fig, ax = plt.subplots(figsize=figsize)

        for color_index, batch_size in enumerate(sorted(results)):
            x = results[batch_size]["n"]
            y = results[batch_size]["fidelity_means"]
            err = results[batch_size]["fidelity_stds"]
            ax.errorbar(
                x,
                y,
                yerr=err,
                marker="x",
                label=f"batch size = {batch_size}",
                color=f"C{color_index}",
            )

        all_num_qubits: list[int] = sorted(
            {
                int(num_qubits)
                for result in results.values()
                for num_qubits in result["n"]
            }
        )
        ax.set_xlim(all_num_qubits[0] - 0.5, all_num_qubits[-1] + 0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Number of qubits")
        ax.set_ylabel("Process fidelity")
        ax.legend()

        output_filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_filename)
        plt.close(fig)


@app.command()
def main(
    input_dir: Annotated[
        Path,
        typer.Argument(help="Directory of process-fidelity counter dataset JSON files"),
    ],
    output: Annotated[Path, typer.Argument(help="Output plot file path")],
) -> None:
    """Plot dynamic-QFT fidelity scaling across dataset files."""

    plot_result(input_dir, output)


if __name__ == "__main__":
    app()
