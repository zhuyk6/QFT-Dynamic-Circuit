"""Pydantic schemas for Shor benchmark serialization."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from shor_benchmark.types import BenchmarkInstance, CombinedCurveResult


class SimulationMetadataModel(BaseModel):
    """Serializable metadata for one simulation run."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int
    num_shots: int
    gate_error: bool
    readout_error: bool
    thermal_relaxation: bool


class HistogramFileModel(BaseModel):
    """Serializable schema for per-s histogram files."""

    model_config = ConfigDict(extra="forbid")

    instance: BenchmarkInstance
    simulation: SimulationMetadataModel
    histograms: dict[int, dict[int, int]]

    @model_validator(mode="after")
    def validate_histograms(self) -> "HistogramFileModel":
        """Ensure histogram keys cover exactly all phase labels."""

        expected_keys: list[int] = list(range(self.instance.r))
        observed_keys: list[int] = sorted(self.histograms.keys())
        if observed_keys != expected_keys:
            raise ValueError(
                f"histogram phase labels must be exactly {expected_keys}, "
                f"got {observed_keys}"
            )
        s_value: int
        histogram: dict[int, int]
        for s_value, histogram in self.histograms.items():
            if not histogram:
                raise ValueError(f"histogram for s={s_value} is empty")
        return self

    def save(self, output_path: Path) -> None:
        """Save the histogram file to disk as JSON."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, input_path: Path) -> "HistogramFileModel":
        """Load and validate a histogram file from disk."""

        return cls.model_validate_json(input_path.read_text(encoding="utf-8"))


class StrictBenchmarkResultFileModel(BaseModel):
    """Serializable schema for strict benchmark output JSON."""

    model_config = ConfigDict(extra="forbid")

    instance: BenchmarkInstance
    k_list: list[int]
    m_mc: int
    seed: int
    result: CombinedCurveResult
    simulation_histogram_file: str | None = None
