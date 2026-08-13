"""Normalize manifest-described physical data as a logical counter dataset."""

from pathlib import Path
from typing import Annotated

import typer

from qft_dynamic.experiment_data import (
    ExperimentDataset,
    LogicalCountsGroup,
    load_experiment_dataset,
)

app: typer.Typer = typer.Typer()


@app.command()
def main(
    manifest: Annotated[Path, typer.Argument(help="Experiment manifest TOML")],
    output: Annotated[
        Path, typer.Argument(help="Output counter ExperimentDataset JSON")
    ],
) -> None:
    """Normalize physical NPZ files as logical integer counters.

    Args:
        manifest: Experiment manifest TOML path.
        output: Destination for the canonical logical counter dataset.
    """

    dataset: ExperimentDataset = load_experiment_dataset(manifest)
    dataset.save(output)
    print(f"Saved normalized logical counters to: {output}")

    group: LogicalCountsGroup
    for group in dataset.groups:
        shot_counts: list[int] = [run.num_shots for run in group.runs]
        print(
            f"{group.name}: runs={len(group.runs)}, "
            f"shot counts={sorted(set(shot_counts))}"
        )


if __name__ == "__main__":
    app()
