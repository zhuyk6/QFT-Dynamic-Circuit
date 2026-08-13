"""Plot measurement-encoding benchmark results."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib_config import get_latex_figsize, plot_context

from qft_dynamic.experiment_data import ExperimentDataset, MetadataValue
from qft_dynamic.tools.data_process import calc_tvd

app = typer.Typer()
PLOT_DIR: Path = Path(__file__).resolve().parent
PLOT_CONFIG_PATH: Path = PLOT_DIR / "plot_config.toml"


def plot_result(
    results_filename: Path,
    savefig_filename: Path,
) -> None:
    """Plot TVD derived from a measurement-encoding count dataset."""

    dataset: ExperimentDataset = ExperimentDataset.load(results_filename)
    if dataset.experiment_type != "circular_state_qft":
        raise ValueError("measurement-encoding plot requires circular_state_qft data")
    ideal_probabilities: dict[int, float] = {
        state << (dataset.num_qubits - 2): 0.25 for state in range(4)
    }
    dict_tvd_batch_method: dict[int, dict[str, float]] = {}
    for group in dataset.groups:
        batch_value: MetadataValue | None = group.attributes.get("batch_size")
        method_value: MetadataValue | None = group.attributes.get("method")
        if isinstance(batch_value, bool) or not isinstance(batch_value, int):
            raise ValueError(f"group {group.name!r} requires integer batch_size")
        if not isinstance(method_value, str):
            raise ValueError(f"group {group.name!r} requires string method")
        dict_tvd_batch_method.setdefault(batch_value, {})[method_value] = calc_tvd(
            ideal_probabilities,
            group.runs[0].counts,
        )

    batch_sizes = sorted(dict_tvd_batch_method.keys())
    methods = ["base", "enc_perfect", "enc_modified"]
    x = np.arange(len(batch_sizes))
    width = 0.2

    with plot_context(PLOT_CONFIG_PATH, palette="nature") as config:
        figsize: tuple[float, float] = get_latex_figsize(
            config,
            width="column",
            fraction=0.95,
            height_ratio=0.75,
        )
        fig, ax = plt.subplots(figsize=figsize)
        for index, method in enumerate(methods):
            method_values = [
                dict_tvd_batch_method[batch_size][method] for batch_size in batch_sizes
            ]
            ax.bar(x + index * width, method_values, width, label=method)

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("TVD")
        ax.set_title("TVD for Different Batch Sizes and Encode Methods")
        ax.set_xticks(x + width)
        ax.set_xticklabels(map(str, batch_sizes))
        ax.legend()

        savefig_filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savefig_filename)
        plt.close(fig)


@app.command()
def main(
    results: Annotated[Path, typer.Argument(help="Counter ExperimentDataset JSON")],
    output: Annotated[Path, typer.Argument(help="Output plot file path")],
) -> None:
    """Plot measurement-encoding benchmark results."""
    plot_result(results, output)


if __name__ == "__main__":
    app()
