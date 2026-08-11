"""Minimal common data structures for logical experiment counts."""

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type MetadataScalar = str | int | float | bool
type MetadataValue = MetadataScalar | list[MetadataScalar]


class LogicalCountsRun(BaseModel):
    """Logical integer counts from one physical experiment run.

    Args:
        counts: Measurement counts keyed by logical integer output.
        source_file: Physical data filename used to construct the counts.
        metadata: Named values extracted from the source filename.
    """

    model_config = ConfigDict(extra="forbid")

    source_file: str = Field(min_length=1)
    metadata: dict[str, str]
    counts: Counter[int]

    @property
    def num_shots(self) -> int:
        """Return the number of shots represented by the counter."""

        return sum(self.counts.values())


class LogicalCountsGroup(BaseModel):
    """A named collection of logical-count runs sharing one readout mapping."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    attributes: dict[str, MetadataValue]
    runs: list[LogicalCountsRun]


class ExperimentDataset(BaseModel):
    """Logical counters loaded from one experiment-data manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    experiment_type: str = Field(min_length=1)
    num_qubits: int = Field(gt=0)
    bit_order: Literal["msb_first"]
    groups: list[LogicalCountsGroup]

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
    """Logical probabilities for one physical experiment run."""

    model_config = ConfigDict(extra="forbid")

    source_file: str = Field(min_length=1)
    metadata: dict[str, str]
    num_shots: int = Field(gt=0)
    probabilities: dict[int, float]


class LogicalProbabilitiesGroup(BaseModel):
    """A named collection of logical probability distributions."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    attributes: dict[str, MetadataValue]
    runs: list[LogicalProbabilitiesRun]


class ProbabilityExperimentDataset(BaseModel):
    """Logical probability distributions derived from experiment data."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    experiment_type: str = Field(min_length=1)
    num_qubits: int = Field(gt=0)
    bit_order: Literal["msb_first"]
    groups: list[LogicalProbabilitiesGroup]

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
