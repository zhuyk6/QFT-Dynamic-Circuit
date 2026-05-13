"""Plot measurement-encoding benchmark results."""

import json
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer()


def plot_result(
    results_filename: Path,
    savefig_filename: Path,
) -> None:
    """Plot the measurement-encoding benchmark JSON as grouped bars."""
    with open(results_filename, "r") as input_file:
        raw_dict: dict[str, dict[str, float]] = json.load(input_file)
    dict_tvd_batch_method = {int(k): v for k, v in raw_dict.items()}

    batch_sizes = sorted(dict_tvd_batch_method.keys())
    methods = ["base", "enc perfect", "enc modify"]
    x = np.arange(len(batch_sizes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
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


@app.command()
def main(
    results: Annotated[Path, typer.Argument(help="Benchmark results JSON file")],
    output: Annotated[Path, typer.Argument(help="Output plot file path")],
) -> None:
    """Plot measurement-encoding benchmark results."""
    plot_result(results, output)


if __name__ == "__main__":
    app()
