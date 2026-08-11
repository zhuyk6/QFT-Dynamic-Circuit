"""Load vendor NPZ bit arrays into the common logical-counter structure."""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .manifest import ExperimentManifest, ExperimentManifestGroup
from .types import ExperimentDataset, LogicalCountsGroup, LogicalCountsRun

type BitMatrix = npt.NDArray[np.uint8]


def _is_binary_vector(array: npt.NDArray[np.generic]) -> bool:
    """Return whether an array is a nonempty one-dimensional bit vector."""

    unique_values: npt.NDArray[np.generic] = np.unique(array)
    return (
        array.ndim == 1
        and array.size > 0
        and set(unique_values.tolist()).issubset({0, 1})
    )


def _find_bit_key(archive: np.lib.npyio.NpzFile, physical_qubit: int) -> str:
    """Find the unique classified-bit array for one physical qubit."""

    qubit_name: str = f"Q{physical_qubit}"
    candidates: list[str] = []
    key: str
    for key in archive.files:
        if key.endswith("_iq"):
            continue
        if key != qubit_name and not key.startswith(f"{qubit_name}_"):
            continue
        array: npt.NDArray[np.generic] = np.asarray(archive[key])
        if _is_binary_vector(array):
            candidates.append(key)

    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one classified-bit key for {qubit_name}, "
            f"got {candidates}"
        )
    return candidates[0]


def load_logical_counter(
    input_path: Path,
    group: ExperimentManifestGroup,
) -> Counter[int]:
    """Convert one physical NPZ file to an MSB-first logical counter.

    Args:
        input_path: NPZ file containing per-qubit classified shot arrays.
        group: Manifest group containing the physical and logical mapping.

    Returns:
        Counts keyed by logical integer output.
    """

    physical_arrays: list[npt.NDArray[np.uint8]] = []
    with np.load(input_path, allow_pickle=False) as archive:
        physical_qubit: int
        for physical_qubit in group.physical_qubits:
            bit_key: str = _find_bit_key(archive, physical_qubit)
            bit_array: npt.NDArray[np.uint8] = np.asarray(
                archive[bit_key], dtype=np.uint8
            ).reshape(-1)
            physical_arrays.append(bit_array)

    shot_lengths: set[int] = {array.size for array in physical_arrays}
    if len(shot_lengths) != 1:
        lengths: list[int] = [array.size for array in physical_arrays]
        raise ValueError(f"inconsistent shot counts in {input_path}: {lengths}")

    physical_bits: BitMatrix = np.column_stack(physical_arrays)
    logical_bits: BitMatrix = physical_bits[:, group.logical_from_physical]
    num_qubits: int = logical_bits.shape[1]
    weights: npt.NDArray[np.int64] = np.left_shift(
        np.int64(1), np.arange(num_qubits - 1, -1, -1, dtype=np.int64)
    )
    logical_indices: npt.NDArray[np.int64] = logical_bits @ weights
    return Counter(int(value) for value in logical_indices)


def _load_group(
    raw_data_dir: Path,
    group: ExperimentManifestGroup,
    filename_metadata_regex: str,
) -> LogicalCountsGroup:
    """Load all files in one group and extract their filename metadata."""

    metadata_pattern: re.Pattern[str] = re.compile(filename_metadata_regex)
    if not metadata_pattern.groupindex:
        raise ValueError("filename_metadata_regex must define at least one named group")
    runs: list[LogicalCountsRun] = []
    input_path: Path
    for input_path in sorted(raw_data_dir.glob(group.file_glob)):
        match: re.Match[str] | None = metadata_pattern.search(input_path.stem)
        if match is None:
            raise ValueError(f"cannot parse metadata from filename: {input_path.name}")
        metadata: dict[str, str] = {
            key: value for key, value in match.groupdict().items() if value is not None
        }
        runs.append(
            LogicalCountsRun(
                counts=load_logical_counter(input_path=input_path, group=group),
                source_file=input_path.name,
                metadata=metadata,
            )
        )

    if not runs:
        raise FileNotFoundError(
            f"no physical data files match {raw_data_dir / group.file_glob}"
        )
    return LogicalCountsGroup(
        name=group.name,
        attributes=dict(group.attributes),
        runs=runs,
    )


def load_experiment_dataset(manifest_path: Path) -> ExperimentDataset:
    """Load every manifest group as logical integer counters.

    This function performs no metric calculation, mitigation, aggregation, or
    target-coverage validation. Those operations belong to the caller.

    Args:
        manifest_path: Path to the experiment-data manifest.

    Returns:
        Common logical-counter dataset ready for caller-defined processing.
    """

    manifest: ExperimentManifest = ExperimentManifest.load(manifest_path)
    raw_data_dir: Path = manifest.resolve_raw_data_dir()
    if not raw_data_dir.is_dir():
        raise FileNotFoundError(f"raw data directory does not exist: {raw_data_dir}")
    groups: list[LogicalCountsGroup] = [
        _load_group(
            raw_data_dir=raw_data_dir,
            group=group,
            filename_metadata_regex=manifest.filename_metadata_regex,
        )
        for group in manifest.groups
    ]
    return ExperimentDataset(
        schema_version=1,
        dataset_id=manifest.dataset_id,
        experiment_type=manifest.experiment_type,
        num_qubits=manifest.num_qubits,
        bit_order=manifest.bit_order,
        groups=groups,
    )
