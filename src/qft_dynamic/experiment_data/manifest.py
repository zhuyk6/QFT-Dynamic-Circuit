"""Validated manifest schema for converting physical data to logical counts."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import MetadataValue


class ExperimentManifestGroup(BaseModel):
    """Files and qubit mapping shared by one group of experiment runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    file_glob: str = Field(min_length=1)
    physical_qubits: list[int] = Field(min_length=1)
    logical_from_physical: list[int] = Field(min_length=1)
    attributes: dict[str, MetadataValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_qubit_mapping(self) -> "ExperimentManifestGroup":
        """Validate the physical-qubit list and logical column permutation."""

        num_qubits: int = len(self.physical_qubits)
        if len(set(self.physical_qubits)) != num_qubits:
            raise ValueError("physical_qubits must contain unique indices")
        expected_positions: list[int] = list(range(num_qubits))
        if sorted(self.logical_from_physical) != expected_positions:
            raise ValueError(
                "logical_from_physical must be a permutation of physical column "
                f"positions {expected_positions}"
            )
        return self


class ExperimentManifest(BaseModel):
    """Top-level instructions for loading physical experiment files."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    source_path: Path = Field(exclude=True)
    dataset_id: str = Field(min_length=1)
    experiment_type: str = Field(min_length=1)
    num_qubits: int = Field(gt=0)
    bit_order: Literal["msb_first"]
    raw_data_dir: Path
    filename_metadata_regex: str = Field(min_length=1)
    attributes: dict[str, MetadataValue] = Field(default_factory=dict)
    groups: list[ExperimentManifestGroup] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_groups(self) -> "ExperimentManifest":
        """Validate group names and qubit counts across the dataset."""

        group_names: list[str] = [group.name for group in self.groups]
        if len(set(group_names)) != len(group_names):
            raise ValueError("group names must be unique")
        mismatched_groups: list[str] = [
            group.name
            for group in self.groups
            if len(group.physical_qubits) != self.num_qubits
        ]
        if mismatched_groups:
            raise ValueError(
                "group physical_qubits length does not match num_qubits: "
                f"{mismatched_groups}"
            )
        return self

    @classmethod
    def load(cls, manifest_path: Path) -> "ExperimentManifest":
        """Load and validate a TOML experiment-data manifest.

        Args:
            manifest_path: Path to the TOML manifest.

        Returns:
            Validated experiment manifest.
        """

        with manifest_path.open("rb") as input_file:
            manifest_data: dict[str, object] = tomllib.load(input_file)
        manifest_data["source_path"] = manifest_path.resolve()
        return cls.model_validate(manifest_data)

    def resolve_path(self, path: Path) -> Path:
        """Resolve a path relative to the source manifest.

        Args:
            path: Absolute path or path relative to the manifest directory.

        Returns:
            The input absolute path, or the resolved manifest-relative path.
        """

        if path.is_absolute():
            return path
        return self.source_path.parent / path

    def resolve_raw_data_dir(self) -> Path:
        """Resolve the physical-data directory relative to the manifest."""

        return self.resolve_path(self.raw_data_dir)
