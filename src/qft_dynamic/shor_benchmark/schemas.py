"""Pydantic schemas for Shor benchmark serialization."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .types import BenchmarkInstance, CombinedCurveResult


class StrictBenchmarkResultFileModel(BaseModel):
    """Serializable schema for strict benchmark output JSON."""

    model_config = ConfigDict(extra="forbid")

    instance: BenchmarkInstance
    k_list: list[int]
    m_mc: int
    seed: int
    result: CombinedCurveResult
    experiment_dataset_files: list[Path]
