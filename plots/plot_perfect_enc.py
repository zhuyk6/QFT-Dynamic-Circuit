"""Plot measurement-encoding benchmark results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    method: str
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


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for measurement-encoding plotting."""

    parser = argparse.ArgumentParser(
        description="Plot measurement-encoding benchmark results."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()
    plot_result(args.results, args.output)


if __name__ == "__main__":
    main()
