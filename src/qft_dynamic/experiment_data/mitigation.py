"""Independent readout mitigation for normalized experiment counts."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .calibration import (
    PhysicalQubitReadoutCalibration,
    ReadoutCalibrationFile,
)
from .manifest import ExperimentManifest, ExperimentManifestGroup
from .types import (
    LogicalProbabilitiesGroup,
    LogicalProbabilitiesRun,
    MetadataValue,
    ProbabilityExperimentDataset,
)


def _assignment_matrix(
    calibrations: Sequence[PhysicalQubitReadoutCalibration],
) -> npt.NDArray[np.float64]:
    """Build the MSB-first tensor-product readout assignment matrix."""

    matrix: npt.NDArray[np.float64] = np.array([[1.0]], dtype=np.float64)
    calibration: PhysicalQubitReadoutCalibration
    for calibration in calibrations:
        single_qubit_matrix: npt.NDArray[np.float64] = np.array(
            [
                [calibration.p0_given_0, 1.0 - calibration.p1_given_1],
                [1.0 - calibration.p0_given_0, calibration.p1_given_1],
            ],
            dtype=np.float64,
        )
        matrix = np.kron(matrix, single_qubit_matrix)
    return matrix


def _mitigate_probability_distribution(
    probabilities: Mapping[int, float],
    assignment_matrix: npt.NDArray[np.float64],
) -> dict[int, float]:
    """Apply assignment-matrix inversion to one probability distribution."""

    num_states: int = assignment_matrix.shape[0]
    measured_probabilities: npt.NDArray[np.float64] = np.asarray(
        [probabilities.get(state, 0.0) for state in range(num_states)],
        dtype=np.float64,
    )
    mitigated_probabilities: npt.NDArray[np.float64]
    mitigated_probabilities, *_unused = np.linalg.lstsq(
        assignment_matrix,
        measured_probabilities,
        rcond=None,
    )
    mitigated_probabilities = np.clip(mitigated_probabilities, 0.0, None)
    probability_sum: float = float(mitigated_probabilities.sum())
    if probability_sum <= 0.0:
        raise ValueError("readout mitigation produced no positive probability mass")
    mitigated_probabilities /= probability_sum
    return {
        state: float(probability)
        for state, probability in enumerate(mitigated_probabilities)
    }


def _logical_calibrations(
    group: ExperimentManifestGroup,
    calibration_file: ReadoutCalibrationFile,
) -> list[PhysicalQubitReadoutCalibration]:
    """Order physical calibration values by logical output position."""

    logical_physical_qubits: list[int] = [
        group.physical_qubits[position] for position in group.logical_from_physical
    ]
    missing_qubits: list[int] = [
        physical_qubit
        for physical_qubit in logical_physical_qubits
        if physical_qubit not in calibration_file.qubits
    ]
    if missing_qubits:
        raise ValueError(
            f"readout calibration for group {group.name!r} is missing physical "
            f"qubits {missing_qubits}"
        )
    return [
        calibration_file.qubits[physical_qubit]
        for physical_qubit in logical_physical_qubits
    ]


def _resolve_calibration_path(
    group: LogicalProbabilitiesGroup,
    manifest: ExperimentManifest,
) -> Path:
    """Resolve a calibration filename stored in generic group attributes."""

    calibration_value: MetadataValue | None = group.attributes.get(
        "readout_calibration_file"
    )
    if not isinstance(calibration_value, str):
        raise ValueError(
            f"group {group.name!r} attributes must contain a string "
            "readout_calibration_file"
        )
    return manifest.resolve_path(Path(calibration_value))


def mitigate_probabilities(
    dataset: ProbabilityExperimentDataset,
    manifest: ExperimentManifest,
) -> ProbabilityExperimentDataset:
    """Apply readout mitigation to logical probability distributions.

    Args:
        dataset: Logical probability distributions to mitigate.
        manifest: Manifest providing each group's physical-to-logical mapping.

    Returns:
        A probability dataset with mitigated run distributions.
    """

    dataset_groups: dict[str, LogicalProbabilitiesGroup] = {
        group.name: group for group in dataset.groups
    }
    mitigated_groups: list[LogicalProbabilitiesGroup] = []
    manifest_group: ExperimentManifestGroup
    for manifest_group in manifest.groups:
        dataset_group: LogicalProbabilitiesGroup | None = dataset_groups.get(
            manifest_group.name
        )
        if dataset_group is None:
            raise ValueError(
                f"normalized dataset is missing manifest group {manifest_group.name!r}"
            )
        calibration_path: Path = _resolve_calibration_path(
            group=dataset_group,
            manifest=manifest,
        )
        calibration_file: ReadoutCalibrationFile = ReadoutCalibrationFile.load(
            calibration_path
        )
        calibrations: list[PhysicalQubitReadoutCalibration] = _logical_calibrations(
            group=manifest_group,
            calibration_file=calibration_file,
        )
        assignment_matrix: npt.NDArray[np.float64] = _assignment_matrix(calibrations)
        mitigated_runs: list[LogicalProbabilitiesRun] = []
        run: LogicalProbabilitiesRun
        for run in dataset_group.runs:
            num_shots: int = run.num_shots
            mitigated_runs.append(
                LogicalProbabilitiesRun(
                    source_file=run.source_file,
                    metadata=run.metadata,
                    num_shots=num_shots,
                    probabilities=_mitigate_probability_distribution(
                        probabilities=run.probabilities,
                        assignment_matrix=assignment_matrix,
                    ),
                )
            )
        mitigated_groups.append(
            LogicalProbabilitiesGroup(
                name=dataset_group.name,
                attributes=dataset_group.attributes,
                runs=mitigated_runs,
            )
        )
    return ProbabilityExperimentDataset(
        schema_version=dataset.schema_version,
        dataset_id=dataset.dataset_id,
        experiment_type=dataset.experiment_type,
        num_qubits=dataset.num_qubits,
        bit_order=dataset.bit_order,
        groups=mitigated_groups,
    )
