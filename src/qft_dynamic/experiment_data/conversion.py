"""Conversions between common experiment-data representations."""

from .types import (
    ExperimentDataset,
    LogicalCountsGroup,
    LogicalCountsRun,
    LogicalProbabilitiesGroup,
    LogicalProbabilitiesRun,
    ProbabilityExperimentDataset,
)


def counts_to_probabilities(
    dataset: ExperimentDataset,
) -> ProbabilityExperimentDataset:
    """Normalize logical integer counters into probability distributions.

    Args:
        dataset: Logical integer counters to normalize independently per run.

    Returns:
        A probability dataset with the same groups, attributes, and run metadata.
    """

    probability_groups: list[LogicalProbabilitiesGroup] = []
    counts_group: LogicalCountsGroup
    for counts_group in dataset.groups:
        probability_runs: list[LogicalProbabilitiesRun] = []
        counts_run: LogicalCountsRun
        for counts_run in counts_group.runs:
            num_shots: int = counts_run.num_shots
            if num_shots <= 0:
                raise ValueError(
                    f"cannot normalize empty counter from {counts_run.source_file!r}"
                )
            probabilities: dict[int, float] = {
                state: count / num_shots for state, count in counts_run.counts.items()
            }
            probability_runs.append(
                LogicalProbabilitiesRun(
                    source_file=counts_run.source_file,
                    metadata=counts_run.metadata,
                    num_shots=num_shots,
                    probabilities=probabilities,
                )
            )
        probability_groups.append(
            LogicalProbabilitiesGroup(
                name=counts_group.name,
                attributes=counts_group.attributes,
                runs=probability_runs,
            )
        )
    return ProbabilityExperimentDataset(
        schema_version=dataset.schema_version,
        dataset_id=dataset.dataset_id,
        experiment_type=dataset.experiment_type,
        num_qubits=dataset.num_qubits,
        bit_order=dataset.bit_order,
        groups=probability_groups,
    )
