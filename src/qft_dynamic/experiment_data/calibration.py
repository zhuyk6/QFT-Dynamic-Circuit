"""Readout calibration file models."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PhysicalQubitReadoutCalibration(BaseModel):
    """Independent assignment fidelities for one physical qubit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    p0_given_0: float = Field(ge=0.0, le=1.0)
    p1_given_1: float = Field(ge=0.0, le=1.0)


class ReadoutCalibrationFile(BaseModel):
    """Physical-qubit readout calibration loaded from TOML."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    qubits: dict[int, PhysicalQubitReadoutCalibration]

    @classmethod
    def load(cls, input_path: Path) -> "ReadoutCalibrationFile":
        """Load and validate a readout calibration TOML file.

        Args:
            input_path: Calibration TOML path.

        Returns:
            Validated physical-qubit calibration values.
        """

        with input_path.open("rb") as input_file:
            calibration_data: dict[str, object] = tomllib.load(input_file)
        return cls.model_validate(calibration_data)
