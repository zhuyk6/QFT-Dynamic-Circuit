"""Normalize manifest-described physical experiment data."""

from pathlib import Path
from typing import Annotated

import typer

from qft_dynamic.experiment_data import (
    ExperimentDataset,
    ExperimentManifest,
    LogicalCountsGroup,
    ProbabilityExperimentDataset,
    counts_to_probabilities,
    load_experiment_dataset,
    mitigate_probabilities,
)

app: typer.Typer = typer.Typer()


@app.command()
def main(
    manifest: Annotated[Path, typer.Argument(help="Experiment manifest TOML")],
    probability_output: Annotated[
        Path,
        typer.Option(help="Normalized probability JSON"),
    ],
    counter_output: Annotated[
        Path | None,
        typer.Option(help="Optional normalized counter JSON"),
    ] = None,
    mitigation_output: Annotated[
        Path | None,
        typer.Option(help="Optional readout-mitigated probability JSON"),
    ] = None,
) -> None:
    """Normalize physical NPZ files and optionally mitigate readout error.

    Args:
        manifest: Experiment manifest TOML path.
        probability_output: Destination for the normalized probability dataset.
        counter_output: Optional destination for the measured logical counters.
        mitigation_output: Optional destination for a separate mitigated
            probability dataset.
    """

    dataset: ExperimentDataset = load_experiment_dataset(manifest)

    group: LogicalCountsGroup
    for group in dataset.groups:
        shot_counts: list[int] = [run.num_shots for run in group.runs]
        print(
            f"{group.name}: runs={len(group.runs)}, "
            f"shot counts={sorted(set(shot_counts))}"
        )
    probability_dataset: ProbabilityExperimentDataset = counts_to_probabilities(dataset)
    probability_dataset.save(probability_output)
    print(f"Saved normalized probabilities to: {probability_output}")

    if counter_output is not None:
        dataset.save(counter_output)
        print(f"Saved normalized logical counters to: {counter_output}")

    if mitigation_output is not None:
        manifest_model: ExperimentManifest = ExperimentManifest.load(manifest)
        mitigated_dataset: ProbabilityExperimentDataset = mitigate_probabilities(
            dataset=probability_dataset,
            manifest=manifest_model,
        )
        mitigated_dataset.save(mitigation_output)
        print(f"Saved readout-mitigated probabilities to: {mitigation_output}")


if __name__ == "__main__":
    app()
