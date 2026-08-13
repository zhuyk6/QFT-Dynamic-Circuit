"""Tests for manifest-driven physical experiment data ingestion."""

from collections import Counter
from pathlib import Path
from textwrap import dedent

import numpy as np
import numpy.typing as npt
import pytest

from qft_dynamic.experiment_data import (
    ExperimentDataset,
    ExperimentManifest,
    ExperimentManifestGroup,
    LogicalCountsGroup,
    LogicalCountsRun,
    LogicalProbabilitiesRun,
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


def _write_logical_states_npz(input_path: Path, states: list[int]) -> None:
    """Write logical states using the fixture's swapped physical-bit layout."""

    _write_npz(
        input_path=input_path,
        first=[state & 1 for state in states],
        second=[state >> 1 for state in states],
    )


@pytest.fixture
def physical_experiment_manifest(tmp_path: Path) -> Path:
    """Create a complete manifest-driven physical experiment in a temp directory."""

    raw_data_dir: Path = tmp_path / "raw"
    raw_data_dir.mkdir()
    logical_states_by_target: list[list[int]] = [
        [0, 0, 0, 0, 0, 0, 1, 2],
        [1, 1, 1, 1, 1, 1, 2, 3],
        [2, 2, 2, 2, 2, 2, 3, 0],
        [3, 3, 3, 3, 3, 3, 0, 1],
    ]
    for target_k, states in enumerate(logical_states_by_target):
        _write_logical_states_npz(
            raw_data_dir / f"run_k{target_k:02d}.npz",
            states=states,
        )

    calibration_path: Path = tmp_path / "readout.toml"
    calibration_path.write_text(
        dedent(
            """
            schema_version = 1

            [qubits.10]
            p0_given_0 = 1.0
            p1_given_1 = 1.0

            [qubits.11]
            p0_given_0 = 1.0
            p1_given_1 = 1.0
            """
        ).strip(),
        encoding="utf-8",
    )
    manifest_path: Path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        dedent(
            r"""
            schema_version = 1
            dataset_id = "physical-fixture"
            experiment_type = "process_fidelity"
            num_qubits = 2
            bit_order = "msb_first"
            raw_data_dir = "raw"
            filename_metadata_regex = '_k(?P<k>\d+)'

            [attributes]
            campaign = "temporary-test"

            [[groups]]
            name = "swapped-readout"
            file_glob = "run_k*.npz"
            physical_qubits = [10, 11]
            logical_from_physical = [1, 0]

            [groups.attributes]
            batch_size = 1
            completed_output_qubits = [1, 2]
            purpose = "test physical-to-logical output remapping"
            readout_calibration_file = "readout.toml"
            """
        ).strip(),
        encoding="utf-8",
    )
    return manifest_path


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


def test_manifest_only_describes_data_loading(
    physical_experiment_manifest: Path,
) -> None:
    """A manifest should contain loading context without metric choices."""

    manifest: ExperimentManifest = ExperimentManifest.load(physical_experiment_manifest)

    assert manifest.groups[0].logical_from_physical == [1, 0]
    assert manifest.groups[0].attributes == {
        "batch_size": 1,
        "completed_output_qubits": [1, 2],
        "purpose": "test physical-to-logical output remapping",
        "readout_calibration_file": "readout.toml",
    }
    assert manifest.attributes == {"campaign": "temporary-test"}
    assert not hasattr(manifest.groups[0], "readout_calibration_file")
    assert manifest.experiment_type == "process_fidelity"
    assert not hasattr(manifest, "protocol")
    assert not hasattr(manifest, "target_k_min")


def test_physical_fixture_produces_logical_counters(
    physical_experiment_manifest: Path,
) -> None:
    """Temporary physical files should be normalized with logical remapping."""

    expected_counts: list[Counter[int]] = [
        Counter({0: 6, 1: 1, 2: 1}),
        Counter({1: 6, 2: 1, 3: 1}),
        Counter({2: 6, 3: 1, 0: 1}),
        Counter({3: 6, 0: 1, 1: 1}),
    ]

    dataset: ExperimentDataset = load_experiment_dataset(physical_experiment_manifest)
    runs: list[LogicalCountsRun] = dataset.groups[0].runs

    assert dataset.experiment_type == "process_fidelity"
    assert dataset.schema_version == 2
    assert dataset.producer == "physical_npz"
    assert dataset.attributes == {"campaign": "temporary-test"}
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
        producer="test",
        attributes={"campaign": 3},
        groups=[
            LogicalCountsGroup(
                name="example",
                attributes={"purpose": "conversion"},
                runs=[
                    LogicalCountsRun(
                        run_id="run-0",
                        source_ref="run_k0.npz",
                        metadata={"k": 0},
                        counts=Counter({0: 6, 1: 2}),
                    )
                ],
            )
        ],
    )

    probabilities: ProbabilityExperimentDataset = counts_to_probabilities(counts)

    assert probabilities.groups[0].attributes == {"purpose": "conversion"}
    assert probabilities.producer == "test"
    assert probabilities.attributes == {"campaign": 3}
    assert probabilities.groups[0].runs[0].run_id == "run-0"
    assert probabilities.groups[0].runs[0].source_ref == "run_k0.npz"
    assert probabilities.groups[0].runs[0].metadata == {"k": 0}
    assert probabilities.groups[0].runs[0].num_shots == 8
    assert probabilities.groups[0].runs[0].probabilities == {0: 0.75, 1: 0.25}
    assert not hasattr(probabilities, "mitigation_method")


def test_physical_fixture_produces_mitigated_probabilities(
    physical_experiment_manifest: Path,
    tmp_path: Path,
) -> None:
    """Temporary calibration data should produce a serializable dataset."""

    manifest: ExperimentManifest = ExperimentManifest.load(physical_experiment_manifest)
    counts: ExperimentDataset = load_experiment_dataset(physical_experiment_manifest)
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
        "readout.toml"
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
        dedent(
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
            """
        ).strip(),
        encoding="utf-8",
    )

    calibration_path: Path = tmp_path / "readout.toml"
    calibration_path.write_text(
        dedent(
            """
            schema_version = 1

            [qubits.10]
            p0_given_0 = 0.8
            p1_given_1 = 0.9

            [qubits.11]
            p0_given_0 = 1.0
            p1_given_1 = 1.0
            """
        ).strip(),
        encoding="utf-8",
    )
    manifest: ExperimentManifest = ExperimentManifest.load(manifest_path)
    dataset: ExperimentDataset = ExperimentDataset(
        dataset_id="mitigation-order",
        experiment_type="example",
        num_qubits=2,
        bit_order="msb_first",
        producer="test",
        groups=[
            LogicalCountsGroup(
                name="example",
                attributes={"readout_calibration_file": "readout.toml"},
                runs=[
                    LogicalCountsRun(
                        run_id="run-0",
                        source_ref="run_k0.npz",
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
        dedent(
            """
            schema_version = 1

            [qubits.10]
            p0_given_0 = 1.1
            p1_given_1 = 0.9
            """
        ).strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        ReadoutCalibrationFile.load(calibration_path)


def test_manifest_resolves_relative_and_absolute_paths(tmp_path: Path) -> None:
    """A loaded manifest should retain its source for centralized path resolution."""

    manifest_path: Path = tmp_path / "configs" / "manifest.toml"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        dedent(
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
            """
        ).strip(),
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
        dedent(
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
            """
        ).strip(),
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
        dedent(
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
            """
        ).strip(),
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
    assert restored_dataset.schema_version == 2
    assert restored_dataset.producer == "physical_npz"
    assert '"counts"' in serialized_text
    assert '"k": "7"' in serialized_text
    assert '"run_id": "run_k7"' in serialized_text
    assert '"source_ref": "run_k7.npz"' in serialized_text
    assert "num_shots" not in serialized_text
    assert "process_fidelity" not in serialized_text
    assert "tvd" not in serialized_text


def test_source_files_can_share_the_same_extracted_metadata(tmp_path: Path) -> None:
    """Repeated experiment parameters should remain distinct source runs."""

    _write_npz(tmp_path / "first_k7.npz", first=[0], second=[0])
    _write_npz(tmp_path / "second_k7.npz", first=[1], second=[1])
    manifest_path: Path = tmp_path / "manifest.toml"
    manifest_path.write_text(
        dedent(
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
            """
        ).strip(),
        encoding="utf-8",
    )

    dataset: ExperimentDataset = load_experiment_dataset(manifest_path)

    assert len(dataset.groups[0].runs) == 2
    assert [run.metadata for run in dataset.groups[0].runs] == [
        {"k": "7"},
        {"k": "7"},
    ]
    assert len({run.run_id for run in dataset.groups[0].runs}) == 2
    assert len({run.source_ref for run in dataset.groups[0].runs}) == 2


@pytest.mark.parametrize(
    ("counts", "error_match"),
    [
        (Counter(), "counts must not be empty"),
        (Counter({0: 0}), "positive counts"),
        (Counter({-1: 1}), "non-negative states"),
    ],
)
def test_logical_counts_run_rejects_invalid_counts(
    counts: Counter[int],
    error_match: str,
) -> None:
    """Count runs should reject values that cannot represent measured shots."""

    with pytest.raises(ValueError, match=error_match):
        LogicalCountsRun(run_id="invalid", counts=counts)


def test_experiment_dataset_rejects_duplicate_runs_and_out_of_range_states() -> None:
    """Dataset validation should enforce run identity and logical output width."""

    duplicate_runs: list[LogicalCountsRun] = [
        LogicalCountsRun(run_id="same", counts=Counter({0: 1})),
        LogicalCountsRun(run_id="same", counts=Counter({1: 1})),
    ]
    with pytest.raises(ValueError, match="run_id values must be unique"):
        LogicalCountsGroup(name="duplicate", runs=duplicate_runs)

    with pytest.raises(ValueError, match="logical outputs must be smaller than 4"):
        ExperimentDataset(
            dataset_id="invalid-state",
            experiment_type="example",
            num_qubits=2,
            bit_order="msb_first",
            producer="test",
            groups=[
                LogicalCountsGroup(
                    name="example",
                    runs=[LogicalCountsRun(run_id="run", counts=Counter({4: 1}))],
                )
            ],
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        {},
        {0: -0.1, 1: 1.1},
        {0: float("nan"), 1: 1.0},
        {0: 0.4, 1: 0.5},
    ],
)
def test_logical_probabilities_run_rejects_invalid_distributions(
    probabilities: dict[int, float],
) -> None:
    """Probability runs should contain finite normalized probability mass."""

    with pytest.raises(ValueError):
        LogicalProbabilitiesRun(
            run_id="invalid",
            num_shots=10,
            probabilities=probabilities,
        )
