"""Source-independent data structures for logical measurement results."""

import math
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

type MetadataScalar = str | int | float | bool
type MetadataValue = MetadataScalar | list[MetadataScalar]


class LogicalCountsRun(BaseModel):
    """Logical integer counts from one fixed-configuration measurement run.

    Args:
        run_id: Stable identifier that is unique within the containing group.
        source_ref: Optional reference to the raw artifact used to construct the run.
        metadata: Run-specific parameters and provenance.
        counts: Measurement counts keyed by logical integer output.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    source_ref: str | None = Field(default=None, min_length=1)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    counts: Counter[int]

    @field_validator("counts")
    @classmethod
    def validate_counts(cls, counts: Counter[int]) -> Counter[int]:
        """Require a nonempty counter containing positive shot counts."""

        if not counts:
            raise ValueError("counts must not be empty")
        invalid_states: list[int] = [
            state for state, count in counts.items() if state < 0 or count <= 0
        ]
        if invalid_states:
            raise ValueError(
                "counts must use non-negative states and positive counts; "
                f"invalid states={invalid_states}"
            )
        return counts

    @property
    def num_shots(self) -> int:
        """Return the number of shots represented by the counter."""

        return sum(self.counts.values())


class LogicalCountsGroup(BaseModel):
    """A named collection of logical-count runs sharing one configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    attributes: dict[str, MetadataValue] = Field(default_factory=dict)
    runs: list[LogicalCountsRun] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_run_ids(self) -> "LogicalCountsGroup":
        """Require run identifiers to be unique within the group."""

        run_ids: list[str] = [run.run_id for run in self.runs]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError(f"run_id values must be unique in group {self.name!r}")
        return self


class ExperimentDataset(BaseModel):
    """Logical measurement counters normalized from any acquisition source.

    Args:
        schema_version: Normalized dataset schema version.
        dataset_id: Stable identifier for this collection of measurements.
        experiment_type: Experiment family interpreted by downstream analysis.
        num_qubits: Width of every logical integer output.
        bit_order: Convention used to encode logical bits as integers.
        producer: Source adapter or acquisition system that created the dataset.
        attributes: Dataset-wide configuration and provenance.
        groups: Measurement groups sharing the dataset-level contract.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    dataset_id: str = Field(min_length=1)
    experiment_type: str = Field(min_length=1)
    num_qubits: int = Field(gt=0)
    bit_order: Literal["msb_first"]
    producer: str = Field(min_length=1)
    attributes: dict[str, MetadataValue] = Field(default_factory=dict)
    groups: list[LogicalCountsGroup] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dataset(self) -> "ExperimentDataset":
        """Validate group uniqueness and logical output ranges."""

        group_names: list[str] = [group.name for group in self.groups]
        if len(set(group_names)) != len(group_names):
            raise ValueError("group names must be unique")

        num_states: int = 1 << self.num_qubits
        invalid_outputs: list[str] = [
            f"{group.name}/{run.run_id}:{state}"
            for group in self.groups
            for run in group.runs
            for state in run.counts
            if state >= num_states
        ]
        if invalid_outputs:
            raise ValueError(
                f"logical outputs must be smaller than {num_states}; "
                f"invalid outputs={invalid_outputs}"
            )
        return self

    def save(self, output_path: Path) -> None:
        """Save the normalized experiment dataset as JSON.

        Args:
            output_path: Destination JSON file.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, input_path: Path) -> "ExperimentDataset":
        """Load a normalized experiment dataset from JSON.

        Args:
            input_path: Source JSON file.

        Returns:
            Validated normalized experiment dataset.
        """

        return cls.model_validate_json(input_path.read_text(encoding="utf-8"))


class LogicalProbabilitiesRun(BaseModel):
    """Logical probabilities derived from one measurement run.

    Args:
        run_id: Stable identifier that is unique within the containing group.
        source_ref: Optional reference to the raw artifact behind the run.
        metadata: Run-specific parameters and provenance.
        num_shots: Shot count used to estimate the probabilities.
        probabilities: Probability mass keyed by logical integer output.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    source_ref: str | None = Field(default=None, min_length=1)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    num_shots: int = Field(gt=0)
    probabilities: dict[int, float]

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(
        cls, probabilities: dict[int, float]
    ) -> dict[int, float]:
        """Require a normalized distribution over non-negative states."""

        if not probabilities:
            raise ValueError("probabilities must not be empty")
        invalid_states: list[int] = [
            state
            for state, probability in probabilities.items()
            if state < 0 or not math.isfinite(probability) or probability < 0.0
        ]
        if invalid_states:
            raise ValueError(
                "probabilities must use non-negative states and finite, non-negative "
                f"values; invalid states={invalid_states}"
            )
        probability_sum: float = sum(probabilities.values())
        if not math.isclose(probability_sum, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"probabilities must sum to one, got {probability_sum}")
        return probabilities


class LogicalProbabilitiesGroup(BaseModel):
    """A named collection of logical distributions sharing one configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    attributes: dict[str, MetadataValue] = Field(default_factory=dict)
    runs: list[LogicalProbabilitiesRun] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_run_ids(self) -> "LogicalProbabilitiesGroup":
        """Require run identifiers to be unique within the group."""

        run_ids: list[str] = [run.run_id for run in self.runs]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError(f"run_id values must be unique in group {self.name!r}")
        return self


class ProbabilityExperimentDataset(BaseModel):
    """Logical probability distributions derived from a measurement dataset."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    dataset_id: str = Field(min_length=1)
    experiment_type: str = Field(min_length=1)
    num_qubits: int = Field(gt=0)
    bit_order: Literal["msb_first"]
    producer: str = Field(min_length=1)
    attributes: dict[str, MetadataValue] = Field(default_factory=dict)
    groups: list[LogicalProbabilitiesGroup] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dataset(self) -> "ProbabilityExperimentDataset":
        """Validate group uniqueness and logical output ranges."""

        group_names: list[str] = [group.name for group in self.groups]
        if len(set(group_names)) != len(group_names):
            raise ValueError("group names must be unique")

        num_states: int = 1 << self.num_qubits
        invalid_outputs: list[str] = [
            f"{group.name}/{run.run_id}:{state}"
            for group in self.groups
            for run in group.runs
            for state in run.probabilities
            if state >= num_states
        ]
        if invalid_outputs:
            raise ValueError(
                f"logical outputs must be smaller than {num_states}; "
                f"invalid outputs={invalid_outputs}"
            )
        return self

    def save(self, output_path: Path) -> None:
        """Save the probability experiment dataset as JSON.

        Args:
            output_path: Destination JSON file.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, input_path: Path) -> "ProbabilityExperimentDataset":
        """Load a probability experiment dataset from JSON.

        Args:
            input_path: Source JSON file.

        Returns:
            Validated logical probability dataset.
        """

        return cls.model_validate_json(input_path.read_text(encoding="utf-8"))
