"""Plot detailed QFT process-fidelity experiment data."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import typer
from matplotlib_config import get_latex_figsize, plot_context

from qft_dynamic.experiment_data import (
    ExperimentDataset,
    LogicalProbabilitiesGroup,
    LogicalProbabilitiesRun,
    MetadataScalar,
    MetadataValue,
    ProbabilityExperimentDataset,
    counts_to_probabilities,
)

app: typer.Typer = typer.Typer()
PLOT_DIR: Path = Path(__file__).resolve().parent
PLOT_CONFIG_PATH: Path = PLOT_DIR / "plot_config.toml"


def _load_probabilities(
    counter_dataset: Path | None,
    probability_dataset: Path | None,
) -> ProbabilityExperimentDataset:
    """Load exactly one supported CLI dataset representation.

    Args:
        counter_dataset: Counter dataset to convert to probabilities in memory.
        probability_dataset: Probability dataset to use directly.

    Returns:
        Logical probabilities ready for fidelity analysis.

    Raises:
        typer.BadParameter: If neither input or both inputs are provided.
    """

    if (counter_dataset is None) == (probability_dataset is None):
        raise typer.BadParameter(
            "provide exactly one of --counter-dataset or --probability-dataset"
        )
    if counter_dataset is not None:
        return counts_to_probabilities(ExperimentDataset.load(counter_dataset))
    if probability_dataset is not None:
        return ProbabilityExperimentDataset.load(probability_dataset)
    raise typer.BadParameter(
        "provide exactly one of --counter-dataset or --probability-dataset"
    )


@dataclass(frozen=True)
class ProcessFidelitySeries:
    """Per-target and aggregate process-fidelity data for one group."""

    name: str
    target_k: npt.NDArray[np.int64]
    target_probability: npt.NDArray[np.float64]
    target_standard_error: npt.NDArray[np.float64]
    mean_target_probability: float
    process_fidelity: float
    sampling_mode: Literal["exact", "sample"]


@dataclass(frozen=True)
class PrefixSuccessSeries:
    """Mean completed-output prefix success for one group."""

    name: str
    completed_output_qubits: npt.NDArray[np.int64]
    mean_success_probability: npt.NDArray[np.float64]
    target_sem: npt.NDArray[np.float64]


@dataclass(frozen=True)
class PooledProbabilityDistribution:
    """Shot-weighted probability distribution for repeated target runs."""

    probabilities: dict[int, float]
    num_shots: int


def _validate_process_fidelity_dataset(
    dataset: ProbabilityExperimentDataset,
) -> None:
    """Require the experiment type consumed by this plotting script."""

    if dataset.experiment_type != "process_fidelity":
        raise ValueError(
            "plot_fidelity requires experiment_type='process_fidelity', "
            f"got {dataset.experiment_type!r}"
        )


def _target_k(run: LogicalProbabilitiesRun) -> int:
    """Read the integer process-fidelity target from run metadata."""

    target_value: MetadataValue | None = run.metadata.get("k")
    if isinstance(target_value, bool) or not isinstance(target_value, (int, str)):
        raise ValueError(
            f"run {run.run_id!r} must have integer-compatible 'k' metadata"
        )
    try:
        return int(target_value)
    except ValueError as error:
        raise ValueError(
            f"run {run.run_id!r} has non-integer k={target_value!r}"
        ) from error


def _pool_probabilities_by_target(
    group: LogicalProbabilitiesGroup,
) -> dict[int, PooledProbabilityDistribution]:
    """Pool repeated target runs using their shot counts as weights."""

    weighted_probabilities: dict[int, dict[int, float]] = {}
    shots_by_target: dict[int, int] = {}
    run: LogicalProbabilitiesRun
    for run in group.runs:
        target_k: int = _target_k(run)
        target_weights: dict[int, float] = weighted_probabilities.setdefault(
            target_k, {}
        )
        state: int
        probability: float
        for state, probability in run.probabilities.items():
            target_weights[state] = (
                target_weights.get(state, 0.0) + probability * run.num_shots
            )
        shots_by_target[target_k] = shots_by_target.get(target_k, 0) + run.num_shots

    return {
        target_k: PooledProbabilityDistribution(
            probabilities={
                state: weighted_probability / shots_by_target[target_k]
                for state, weighted_probability in target_weights.items()
            },
            num_shots=shots_by_target[target_k],
        )
        for target_k, target_weights in weighted_probabilities.items()
    }


def _probabilities_by_target(
    group: LogicalProbabilitiesGroup,
    num_qubits: int,
) -> dict[int, PooledProbabilityDistribution]:
    """Return pooled probabilities after validating target indices."""

    probabilities_by_target: dict[int, PooledProbabilityDistribution] = (
        _pool_probabilities_by_target(group)
    )
    unexpected_targets: list[int] = sorted(
        target_k
        for target_k in probabilities_by_target
        if target_k < 0 or target_k >= 1 << num_qubits
    )
    if unexpected_targets:
        raise ValueError(
            f"group {group.name!r} has target indices outside "
            f"0..{(1 << num_qubits) - 1}: {unexpected_targets}"
        )
    return probabilities_by_target


def _sampling_mode(
    dataset: ProbabilityExperimentDataset,
    group: LogicalProbabilitiesGroup,
    observed_targets: set[int],
) -> Literal["exact", "sample"]:
    """Resolve the estimator from simulation metadata or target coverage.

    Simulation datasets explicitly record ``sampling_mode``. Physical datasets
    do not require that attribute, so a complete target sweep selects the exact
    formula and an incomplete sweep selects the sampled estimator.
    """

    configured_mode: MetadataValue | None = dataset.attributes.get("sampling_mode")
    if configured_mode is not None:
        if not isinstance(configured_mode, str) or configured_mode not in {
            "exact",
            "sample",
        }:
            raise ValueError(
                "dataset attribute 'sampling_mode' must be 'exact' or 'sample', "
                f"got {configured_mode!r}"
            )
        mode: Literal["exact", "sample"] = (
            "exact" if configured_mode == "exact" else "sample"
        )
    else:
        expected_targets: set[int] = set(range(1 << dataset.num_qubits))
        mode = "exact" if observed_targets == expected_targets else "sample"

    if mode == "exact":
        expected_targets = set(range(1 << dataset.num_qubits))
        if observed_targets != expected_targets:
            missing_targets: list[int] = sorted(expected_targets - observed_targets)
            raise ValueError(
                f"group {group.name!r} declares exact sampling but does not contain "
                f"a complete target sweep; missing={missing_targets}"
            )
    elif len(observed_targets) < 2:
        raise ValueError(
            f"group {group.name!r} requires at least two distinct targets for "
            "sampled process fidelity"
        )
    return mode


def _process_fidelity(
    probabilities: list[float],
    sampling_mode: Literal["exact", "sample"],
) -> float:
    """Calculate exact or unbiased sampled process fidelity."""

    num_targets: int = len(probabilities)
    mean_sqrt: float = sum(math.sqrt(value) for value in probabilities) / num_targets
    if sampling_mode == "exact":
        return mean_sqrt**2
    return (num_targets / (num_targets - 1.0)) * mean_sqrt**2 - (
        sum(probabilities) / (num_targets * (num_targets - 1.0))
    )


def _completed_output_qubits(
    group: LogicalProbabilitiesGroup,
    num_qubits: int,
) -> list[int]:
    """Read and validate plot-specific completed-output prefix sizes."""

    raw_stages: MetadataValue | None = group.attributes.get("completed_output_qubits")
    if not isinstance(raw_stages, list):
        raise ValueError(
            f"group {group.name!r} must define the list attribute "
            "'completed_output_qubits' for the stages plot"
        )

    stages: list[int] = []
    value: MetadataScalar
    for value in raw_stages:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"group {group.name!r} has non-integer completed-output stage {value!r}"
            )
        stages.append(value)
    if stages != sorted(set(stages)):
        raise ValueError(
            f"group {group.name!r} completed-output stages must be unique and sorted"
        )
    if any(stage < 1 or stage > num_qubits for stage in stages):
        raise ValueError(
            f"group {group.name!r} completed-output stages must lie in 1..{num_qubits}"
        )
    return stages


def build_process_fidelity_series(
    dataset: ProbabilityExperimentDataset,
) -> list[ProcessFidelitySeries]:
    """Calculate per-target probabilities and process fidelity.

    Repeated source files carrying the same ``k`` metadata are pooled using
    their shot counts as weights. The dataset ``sampling_mode`` attribute
    selects the simulation estimator. Without that attribute, complete target
    coverage selects the exact formula and incomplete coverage selects the
    unbiased sampled formula.

    Args:
        dataset: Normalized process-fidelity experiment data.

    Returns:
        One target-probability and aggregate series per experiment group.
    """

    _validate_process_fidelity_dataset(dataset)
    series_list: list[ProcessFidelitySeries] = []
    group: LogicalProbabilitiesGroup
    for group in dataset.groups:
        probabilities_by_target: dict[int, PooledProbabilityDistribution] = (
            _probabilities_by_target(
                group=group,
                num_qubits=dataset.num_qubits,
            )
        )
        target_values: list[int] = sorted(probabilities_by_target)
        sampling_mode: Literal["exact", "sample"] = _sampling_mode(
            dataset=dataset,
            group=group,
            observed_targets=set(target_values),
        )
        probabilities: list[float] = []
        standard_errors: list[float] = []
        target_k: int
        for target_k in target_values:
            distribution: PooledProbabilityDistribution = probabilities_by_target[
                target_k
            ]
            probability: float = distribution.probabilities.get(target_k, 0.0)
            probabilities.append(probability)
            standard_errors.append(
                math.sqrt(probability * (1.0 - probability) / distribution.num_shots)
            )

        series_list.append(
            ProcessFidelitySeries(
                name=group.name,
                target_k=np.asarray(target_values, dtype=np.int64),
                target_probability=np.asarray(probabilities, dtype=np.float64),
                target_standard_error=np.asarray(standard_errors, dtype=np.float64),
                mean_target_probability=float(np.mean(probabilities)),
                process_fidelity=_process_fidelity(probabilities, sampling_mode),
                sampling_mode=sampling_mode,
            )
        )
    return series_list


def build_prefix_success_series(
    dataset: ProbabilityExperimentDataset,
) -> list[PrefixSuccessSeries]:
    """Calculate mean MSB-prefix success at configured completed-output stages.

    Args:
        dataset: Normalized process-fidelity experiment data.

    Returns:
        One completed-output prefix series per experiment group.
    """

    _validate_process_fidelity_dataset(dataset)
    series_list: list[PrefixSuccessSeries] = []
    group: LogicalProbabilitiesGroup
    for group in dataset.groups:
        probabilities_by_target: dict[int, PooledProbabilityDistribution] = (
            _probabilities_by_target(
                group=group,
                num_qubits=dataset.num_qubits,
            )
        )
        target_values: list[int] = sorted(probabilities_by_target)
        stages: list[int] = _completed_output_qubits(
            group=group,
            num_qubits=dataset.num_qubits,
        )
        stage_means: list[float] = []
        stage_sems: list[float] = []
        stage: int
        for stage in stages:
            shift: int = dataset.num_qubits - stage
            probabilities: list[float] = []
            target_k: int
            for target_k in target_values:
                distribution: PooledProbabilityDistribution = probabilities_by_target[
                    target_k
                ]
                target_prefix: int = target_k >> shift
                success_probability: float = sum(
                    probability
                    for state, probability in distribution.probabilities.items()
                    if state >> shift == target_prefix
                )
                probabilities.append(success_probability)
            values: npt.NDArray[np.float64] = np.asarray(
                probabilities, dtype=np.float64
            )
            stage_means.append(float(values.mean()))
            stage_sems.append(
                float(values.std(ddof=1) / math.sqrt(values.size))
                if values.size > 1
                else 0.0
            )
        series_list.append(
            PrefixSuccessSeries(
                name=group.name,
                completed_output_qubits=np.asarray(stages, dtype=np.int64),
                mean_success_probability=np.asarray(stage_means, dtype=np.float64),
                target_sem=np.asarray(stage_sems, dtype=np.float64),
            )
        )
    return series_list


def plot_overview(
    dataset: ProbabilityExperimentDataset,
    output_path: Path,
) -> list[ProcessFidelitySeries]:
    """Plot overlaid target probabilities and process-fidelity bars."""

    series_list: list[ProcessFidelitySeries] = build_process_fidelity_series(dataset)
    with plot_context(PLOT_CONFIG_PATH, palette="nature") as config:
        figsize: tuple[float, float] = get_latex_figsize(
            config,
            width="text",
            fraction=0.95,
            height_ratio=0.46,
        )
        figure, axes = plt.subplots(
            1,
            2,
            figsize=figsize,
            width_ratios=[2.2, 1.0],
        )
        target_axis, summary_axis = axes

        index: int
        series: ProcessFidelitySeries
        for index, series in enumerate(series_list):
            target_axis.errorbar(
                series.target_k,
                series.target_probability,
                yerr=series.target_standard_error,
                marker="o",
                markersize=3,
                linewidth=1,
                capsize=2,
                color=f"C{index}",
                label=series.name,
            )
        target_axis.set_xlabel("Target $k$")
        target_axis.set_ylabel(r"$P(\mathrm{output}=k)$")
        target_axis.set_ylim(0.0, 1.0)
        target_axis.legend()

        positions: npt.NDArray[np.float64] = np.arange(
            len(series_list), dtype=np.float64
        )
        summary_axis.bar(
            positions,
            [series.process_fidelity for series in series_list],
            width=0.62,
            color=[f"C{index}" for index in range(len(series_list))],
        )
        summary_axis.set_xticks(
            positions,
            [series.name for series in series_list],
            rotation=20,
            ha="right",
        )
        summary_axis.set_ylabel("Process fidelity")
        summary_axis.set_ylim(0.0, 1.0)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path)
        plt.close(figure)
    return series_list


def plot_target_probability_panels(
    dataset: ProbabilityExperimentDataset,
    output_path: Path,
    combined: bool,
) -> list[ProcessFidelitySeries]:
    """Reproduce the physical-team per-group target-probability panels."""

    series_list: list[ProcessFidelitySeries] = build_process_fidelity_series(dataset)
    with plot_context(PLOT_CONFIG_PATH, palette="nature") as config:
        axes: list[plt.Axes]
        if combined:
            figsize: tuple[float, float] = get_latex_figsize(
                config,
                width="column",
                fraction=0.95,
                height_ratio=0.8,
            )
            figure, ax = plt.subplots(figsize=figsize)
            axes = [ax] * len(series_list)
        else:
            figsize: tuple[float, float] = get_latex_figsize(
                config,
                width="column",
                fraction=0.95,
                height_ratio=0.30 * len(series_list),
            )
            figure, ax = plt.subplots(
                len(series_list),
                1,
                figsize=figsize,
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            axes = ax.flatten().tolist()

        for index, series in enumerate(series_list):
            axis = axes[index]
            axis.errorbar(
                series.target_k,
                series.target_probability,
                yerr=series.target_standard_error,
                label=series.name,
                marker="o",
                markersize=1,
                linewidth=1,
                capsize=0.5,
                color=f"C{index}",
            )
            axis.set_ylabel(r"$p_k$")
            axis.set_ylim(0.0, 1.0)

            if not combined:
                axis.set_title(
                    f"{series.name}: process fidelity={series.process_fidelity:.4f}"
                )
        axes[-1].set_xlabel("Target $k$")

        if combined:
            axes[0].legend()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path)
        plt.close(figure)
    return series_list


def plot_prefix_success(
    dataset: ProbabilityExperimentDataset,
    output_path: Path,
) -> list[PrefixSuccessSeries]:
    """Reproduce completed-output prefix success versus prefix size."""

    series_list: list[PrefixSuccessSeries] = build_prefix_success_series(dataset)
    with plot_context(PLOT_CONFIG_PATH, palette="nature") as config:
        figsize: tuple[float, float] = get_latex_figsize(
            config,
            width="column",
            fraction=0.95,
            height_ratio=0.78,
        )
        figure, axis = plt.subplots(figsize=figsize)
        index: int
        series: PrefixSuccessSeries
        for index, series in enumerate(series_list):
            axis.errorbar(
                series.completed_output_qubits,
                series.mean_success_probability,
                yerr=series.target_sem,
                marker="o",
                capsize=3,
                color=f"C{index}",
                label=series.name,
            )
        all_stages: list[int] = sorted(
            {
                int(stage)
                for series in series_list
                for stage in series.completed_output_qubits
            }
        )
        axis.set_xticks(all_stages)
        axis.set_xlabel("Completed logical output qubits")
        axis.set_ylabel("Mean prefix success probability")
        axis.set_ylim(0.0, 1.0)
        axis.legend()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path)
        plt.close(figure)
    return series_list


def plot_process_fidelity_summary(
    dataset: ProbabilityExperimentDataset,
    output_path: Path,
) -> list[ProcessFidelitySeries]:
    """Plot one process-fidelity bar per experiment group."""

    series_list: list[ProcessFidelitySeries] = build_process_fidelity_series(dataset)
    with plot_context(PLOT_CONFIG_PATH, palette="nature") as config:
        figsize: tuple[float, float] = get_latex_figsize(
            config,
            width="column",
            fraction=0.95,
            height_ratio=0.75,
        )
        figure, axis = plt.subplots(figsize=figsize)
        positions: npt.NDArray[np.float64] = np.arange(
            len(series_list), dtype=np.float64
        )
        bars = axis.bar(
            positions,
            [series.process_fidelity for series in series_list],
            width=0.62,
            color=[f"C{index}" for index in range(len(series_list))],
        )
        axis.set_xticks(positions, [series.name for series in series_list])
        axis.set_ylabel("Process fidelity")
        axis.set_ylim(0.0, 1.0)
        for bar, series in zip(bars, series_list, strict=True):
            axis.annotate(
                f"{series.process_fidelity:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path)
        plt.close(figure)
    return series_list


def _print_process_fidelity(series_list: list[ProcessFidelitySeries]) -> None:
    """Print aggregate metrics for plotted experiment groups."""

    series: ProcessFidelitySeries
    for series in series_list:
        print(
            f"{series.name}: mean target probability="
            f"{series.mean_target_probability:.6f}, "
            f"process fidelity={series.process_fidelity:.6f} "
            f"({series.sampling_mode})"
        )


@app.command()
def overview(
    output_path: Annotated[Path, typer.Argument(help="Output figure path")],
    counter_dataset: Annotated[
        Path | None,
        typer.Option("--counter-dataset", help="Counter ExperimentDataset JSON"),
    ] = None,
    probability_dataset: Annotated[
        Path | None,
        typer.Option("--probability-dataset", help="ProbabilityExperimentDataset JSON"),
    ] = None,
) -> None:
    """Plot overlaid target probabilities and process-fidelity summary."""

    series_list: list[ProcessFidelitySeries] = plot_overview(
        dataset=_load_probabilities(counter_dataset, probability_dataset),
        output_path=output_path,
    )
    _print_process_fidelity(series_list)


@app.command()
def targets(
    output_path: Annotated[Path, typer.Argument(help="Output figure path")],
    counter_dataset: Annotated[
        Path | None,
        typer.Option("--counter-dataset", help="Counter ExperimentDataset JSON"),
    ] = None,
    probability_dataset: Annotated[
        Path | None,
        typer.Option("--probability-dataset", help="ProbabilityExperimentDataset JSON"),
    ] = None,
    combined: Annotated[
        bool, typer.Option(help="Combine all groups into one panel")
    ] = False,
) -> None:
    """Plot one target-probability panel per experiment group."""

    series_list: list[ProcessFidelitySeries] = plot_target_probability_panels(
        dataset=_load_probabilities(counter_dataset, probability_dataset),
        output_path=output_path,
        combined=combined,
    )
    _print_process_fidelity(series_list)


@app.command()
def stages(
    output_path: Annotated[Path, typer.Argument(help="Output figure path")],
    counter_dataset: Annotated[
        Path | None,
        typer.Option("--counter-dataset", help="Counter ExperimentDataset JSON"),
    ] = None,
    probability_dataset: Annotated[
        Path | None,
        typer.Option("--probability-dataset", help="ProbabilityExperimentDataset JSON"),
    ] = None,
) -> None:
    """Plot configured completed-output prefix success curves."""

    plot_prefix_success(
        dataset=_load_probabilities(counter_dataset, probability_dataset),
        output_path=output_path,
    )


@app.command()
def summary(
    output_path: Annotated[Path, typer.Argument(help="Output figure path")],
    counter_dataset: Annotated[
        Path | None,
        typer.Option("--counter-dataset", help="Counter ExperimentDataset JSON"),
    ] = None,
    probability_dataset: Annotated[
        Path | None,
        typer.Option("--probability-dataset", help="ProbabilityExperimentDataset JSON"),
    ] = None,
) -> None:
    """Plot process fidelity for every experiment group."""

    series_list: list[ProcessFidelitySeries] = plot_process_fidelity_summary(
        dataset=_load_probabilities(counter_dataset, probability_dataset),
        output_path=output_path,
    )
    _print_process_fidelity(series_list)


if __name__ == "__main__":
    app()
