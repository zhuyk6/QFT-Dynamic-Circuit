"""Tests for manifest-driven physical experiment data ingestion."""

from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from qft_dynamic.experiment_data import (
    ExperimentDataset,
    ExperimentManifest,
    ExperimentManifestGroup,
    LogicalCountsGroup,
    LogicalCountsRun,
    ProbabilityExperimentDataset,
    ReadoutCalibrationFile,
    counts_to_probabilities,
    load_experiment_dataset,
    load_logical_counter,
    mitigate_probabilities,
)


def _write_npz(input_path: Path, first: list[int], second: list[int]) -> None:
    """Write a minimal vendor-shaped two-qubit NPZ fixture."""

    first_bits: npt.NDArray[np.int32] = np.asarray(first, dtype=np.int32)
    second_bits: npt.NDArray[np.int32] = np.asarray(second, dtype=np.int32)
    iq_values: npt.NDArray[np.complex128] = np.zeros(
        first_bits.size, dtype=np.complex128
    )
    np.savez(
        input_path,
        Q10_0=first_bits,
        Q10_0_iq=iq_values,
        Q11_0=second_bits,
        Q11_0_iq=iq_values,
    )


def _group(logical_from_physical: list[int]) -> ExperimentManifestGroup:
    """Build a two-qubit test manifest group."""

    return ExperimentManifestGroup(
        name="example",
        file_glob="*k*.npz",
        physical_qubits=[10, 11],
        logical_from_physical=logical_from_physical,
        attributes={"batch_size": 1},
    )


def test_load_logical_counter_applies_mapping_before_msb_conversion(
    tmp_path: Path,
) -> None:
    """Logical mapping should be applied before integer conversion."""

    input_path: Path = tmp_path / "run_k0.npz"
    _write_npz(input_path, first=[0, 0, 0, 1], second=[0, 0, 1, 1])

    counts: Counter[int] = load_logical_counter(
        input_path=input_path,
        group=_group(logical_from_physical=[1, 0]),
    )

    assert counts == Counter({0: 2, 2: 1, 3: 1})


def test_repository_manifest_only_describes_data_loading() -> None:
    """The repository manifest should contain mapping, not metric choices."""

    repository_root: Path = Path(__file__).resolve().parents[1]
    manifest_path: Path = repository_root / "configs/experiments/qft_fidelity_6q.toml"

    manifest: ExperimentManifest = ExperimentManifest.load(manifest_path)

    assert manifest.groups[2].logical_from_physical == [0, 2, 1, 3, 5, 4]
    assert manifest.groups[2].attributes == {
        "batch_size": 3,
        "completed_output_qubits": [3, 6],
        "readout_calibration_file": "readout_calibration_batch13.toml",
    }
    assert not hasattr(manifest.groups[2], "readout_calibration_file")
    assert manifest.experiment_type == "process_fidelity"
    assert not hasattr(manifest, "protocol")
    assert not hasattr(manifest, "target_k_min")


def test_repository_example_produces_documented_logical_counters() -> None:
    """The checked-in physical example should verify logical remapping."""

    repository_root: Path = Path(__file__).resolve().parents[1]
    manifest_path: Path = repository_root / "data/experiments_example/manifest.toml"
    expected_counts: list[Counter[int]] = [
        Counter({0: 6, 1: 1, 2: 1}),
        Counter({1: 6, 2: 1, 3: 1}),
        Counter({2: 6, 3: 1, 0: 1}),
        Counter({3: 6, 0: 1, 1: 1}),
    ]

    dataset: ExperimentDataset = load_experiment_dataset(manifest_path)
    runs: list[LogicalCountsRun] = dataset.groups[0].runs

    assert dataset.experiment_type == "process_fidelity"
    assert [run.metadata for run in runs] == [
        {"k": "00"},
        {"k": "01"},
        {"k": "02"},
        {"k": "03"},
    ]
    assert [run.counts for run in runs] == expected_counts


def test_counts_convert_to_generic_probabilities() -> None:
    """Counter normalization should preserve structure without adding semantics."""

    counts: ExperimentDataset = ExperimentDataset(
        dataset_id="probability-conversion",
        experiment_type="example",
        num_qubits=2,
        bit_order="msb_first",
        groups=[
            LogicalCountsGroup(
                name="example",
                attributes={"purpose": "conversion"},
                runs=[
                    LogicalCountsRun(
                        source_file="run_k0.npz",
                        metadata={"k": "0"},
                        counts=Counter({0: 6, 1: 2}),
                    )
                ],
            )
        ],
    )

    probabilities: ProbabilityExperimentDataset = counts_to_probabilities(counts)

    assert probabilities.groups[0].attributes == {"purpose": "conversion"}
    assert probabilities.groups[0].runs[0].metadata == {"k": "0"}
    assert probabilities.groups[0].runs[0].num_shots == 8
    assert probabilities.groups[0].runs[0].probabilities == {0: 0.75, 1: 0.25}
    assert not hasattr(probabilities, "mitigation_method")


def test_repository_example_produces_mitigated_probabilities(
    tmp_path: Path,
) -> None:
    """The example calibration should produce a serializable probability dataset."""

    repository_root: Path = Path(__file__).resolve().parents[1]
    manifest_path: Path = repository_root / "data/experiments_example/manifest.toml"
    manifest: ExperimentManifest = ExperimentManifest.load(manifest_path)
    counts: ExperimentDataset = load_experiment_dataset(manifest_path)
    probabilities: ProbabilityExperimentDataset = counts_to_probabilities(counts)

    mitigated: ProbabilityExperimentDataset = mitigate_probabilities(
        dataset=probabilities,
        manifest=manifest,
    )
    output_path: Path = tmp_path / "mitigated.json"
    mitigated.save(output_path)
    restored: ProbabilityExperimentDataset = ProbabilityExperimentDataset.load(
        output_path
    )

    assert restored == mitigated
    assert mitigated.groups[0].attributes["readout_calibration_file"] == (
        "readout_calibration.toml"
    )
    assert not hasattr(mitigated, "mitigation_method")
    assert not hasattr(mitigated.groups[0], "calibration_file")
    for run in mitigated.groups[0].runs:
        assert run.num_shots == 8
        assert sum(run.probabilities.values()) == pytest.approx(1.0)
        assert set(run.probabilities) == {0, 1, 2, 3}


def test_mitigation_reorders_calibration_from_physical_to_logical(
    tmp_path: Path,
) -> None:
    """A swapped logical mapping should also swap calibration matrix factors."""

    manifest_path: Path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        """
schema_version = 1
dataset_id = "mitigation-order"
experiment_type = "example"
num_qubits = 2
bit_order = "msb_first"
raw_data_dir = "."
filename_metadata_regex = '_k(?P<k>\\d+)'

[[groups]]
name = "example"
file_glob = "*.npz"
physical_qubits = [10, 11]
logical_from_physical = [1, 0]

[groups.attributes]
readout_calibration_file = "readout.toml"
""".strip(),
        encoding="utf-8",
    )
    calibration_path: Path = tmp_path / "readout.toml"
    calibration_path.write_text(
        """
schema_version = 1

[qubits.10]
p0_given_0 = 0.8
p1_given_1 = 0.9

[qubits.11]
p0_given_0 = 1.0
p1_given_1 = 1.0
""".strip(),
        encoding="utf-8",
    )
    manifest: ExperimentManifest = ExperimentManifest.load(manifest_path)
    dataset: ExperimentDataset = ExperimentDataset(
        dataset_id="mitigation-order",
        experiment_type="example",
        num_qubits=2,
        bit_order="msb_first",
        groups=[
            LogicalCountsGroup(
                name="example",
                attributes={"readout_calibration_file": "readout.toml"},
                runs=[
                    LogicalCountsRun(
                        source_file="run_k0.npz",
                        metadata={"k": "0"},
                        counts=Counter({0: 80, 1: 20}),
                    )
                ],
            )
        ],
    )

    probability_dataset: ProbabilityExperimentDataset = counts_to_probabilities(dataset)
    mitigated: ProbabilityExperimentDataset = mitigate_probabilities(
        dataset=probability_dataset,
        manifest=manifest,
    )
    probabilities: dict[int, float] = mitigated.groups[0].runs[0].probabilities

    assert probabilities[0] == pytest.approx(1.0)
    assert probabilities[1] == pytest.approx(0.0, abs=1e-12)
    assert probabilities[2] == pytest.approx(0.0, abs=1e-12)
    assert probabilities[3] == pytest.approx(0.0, abs=1e-12)


def test_readout_calibration_rejects_invalid_probability(tmp_path: Path) -> None:
    """Calibration probabilities must remain within the unit interval."""

    calibration_path: Path = tmp_path / "invalid.toml"
    calibration_path.write_text(
        """
schema_version = 1

[qubits.10]
p0_given_0 = 1.1
p1_given_1 = 0.9
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        ReadoutCalibrationFile.load(calibration_path)


def test_manifest_resolves_relative_and_absolute_paths(tmp_path: Path) -> None:
    """A loaded manifest should retain its source for centralized path resolution."""

    manifest_path: Path = tmp_path / "configs" / "manifest.toml"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        """
schema_version = 1
dataset_id = "path-resolution"
experiment_type = "example"
num_qubits = 1
bit_order = "msb_first"
raw_data_dir = "../raw"
filename_metadata_regex = '_k(?P<k>\\d+)'

[[groups]]
name = "example"
file_glob = "*.npz"
physical_qubits = [10]
logical_from_physical = [0]
""".strip(),
        encoding="utf-8",
    )
    manifest: ExperimentManifest = ExperimentManifest.load(manifest_path)
    absolute_path: Path = tmp_path / "absolute-calibration.toml"

    assert manifest.source_path == manifest_path.resolve()
    assert manifest.resolve_path(Path("calibration.toml")) == (
        manifest_path.parent / "calibration.toml"
    )
    assert manifest.resolve_path(absolute_path) == absolute_path
    assert manifest.resolve_raw_data_dir() == manifest_path.parent / "../raw"
    assert "source_path" not in manifest.model_dump()


def test_loader_accepts_caller_defined_partial_run_collection(tmp_path: Path) -> None:
    """Loading should not impose complete target-sweep requirements."""

    _write_npz(tmp_path / "run_k7.npz", first=[0, 1], second=[1, 1])
    manifest_path: Path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        """
schema_version = 1
dataset_id = "partial"
experiment_type = "partial"
num_qubits = 2
bit_order = "msb_first"
raw_data_dir = "."
filename_metadata_regex = '_k(?P<k>\\d+)'

[[groups]]
name = "example"
file_glob = "*.npz"
physical_qubits = [10, 11]
logical_from_physical = [0, 1]
""".strip(),
        encoding="utf-8",
    )

    dataset: ExperimentDataset = load_experiment_dataset(manifest_path)

    assert dataset.experiment_type == "partial"
    assert dataset.groups[0].runs[0].metadata == {"k": "7"}
    assert dataset.groups[0].runs[0].counts == Counter({1: 1, 3: 1})


def test_experiment_dataset_json_round_trip(tmp_path: Path) -> None:
    """The common model should round-trip counters through JSON directly."""

    _write_npz(tmp_path / "run_k7.npz", first=[0, 1], second=[1, 1])
    manifest_path: Path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        """
schema_version = 1
dataset_id = "serialization"
experiment_type = "example"
num_qubits = 2
bit_order = "msb_first"
raw_data_dir = "."
filename_metadata_regex = '_k(?P<k>\\d+)'

[[groups]]
name = "example"
file_glob = "*.npz"
physical_qubits = [10, 11]
logical_from_physical = [0, 1]
""".strip(),
        encoding="utf-8",
    )
    dataset: ExperimentDataset = load_experiment_dataset(manifest_path)
    output_path: Path = tmp_path / "normalized.json"

    dataset.save(output_path)
    restored_dataset: ExperimentDataset = ExperimentDataset.load(output_path)

    serialized_text: str = output_path.read_text(encoding="utf-8")
    assert restored_dataset == dataset
    assert isinstance(restored_dataset.groups[0].runs[0].counts, Counter)
    assert restored_dataset.groups[0].runs[0].num_shots == 2
    assert '"counts"' in serialized_text
    assert '"k": "7"' in serialized_text
    assert "run_id" not in serialized_text
    assert "num_shots" not in serialized_text
    assert "process_fidelity" not in serialized_text
    assert "tvd" not in serialized_text


def test_source_files_can_share_the_same_extracted_metadata(tmp_path: Path) -> None:
    """Repeated experiment parameters should remain distinct source runs."""

    _write_npz(tmp_path / "first_k7.npz", first=[0], second=[0])
    _write_npz(tmp_path / "second_k7.npz", first=[1], second=[1])
    manifest_path: Path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        """
schema_version = 1
dataset_id = "repeated"
experiment_type = "example"
num_qubits = 2
bit_order = "msb_first"
raw_data_dir = "."
filename_metadata_regex = '_k(?P<k>\\d+)'

[[groups]]
name = "example"
file_glob = "*.npz"
physical_qubits = [10, 11]
logical_from_physical = [0, 1]
""".strip(),
        encoding="utf-8",
    )

    dataset: ExperimentDataset = load_experiment_dataset(manifest_path)

    assert len(dataset.groups[0].runs) == 2
    assert [run.metadata for run in dataset.groups[0].runs] == [
        {"k": "7"},
        {"k": "7"},
    ]
    assert len({run.source_file for run in dataset.groups[0].runs}) == 2
