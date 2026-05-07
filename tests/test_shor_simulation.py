"""Unit tests for Shor histogram simulation utilities."""

import json
from collections import Counter
from pathlib import Path

import pytest
from qiskit.quantum_info import Statevector

from bench_shor_strict import run_strict_benchmark
from qft_dynamic.shor_benchmark.samplers import HistogramSampler
from qft_dynamic.shor_benchmark.schemas import (
    HistogramFileModel,
    StrictBenchmarkResultFileModel,
)
from qft_dynamic.shor_benchmark.simulation import (
    prepare_forward_qft_phase_state,
    save_histograms,
    simulate_histograms_for_instance,
)
from qft_dynamic.shor_benchmark.types import BenchmarkInstance
from qft_dynamic.tools.build_circuits import qft_unitary


def test_phase_state_matches_swapless_qft_convention_for_aligned_case() -> None:
    """Prepared phase states should map to one peak under swapless forward QFT."""

    instance: BenchmarkInstance = BenchmarkInstance(n=15, a=2, r=4, m=2)

    s_value: int
    for s_value in range(instance.r):
        preparation_circuit = prepare_forward_qft_phase_state(
            instance=instance,
            s=s_value,
        )
        qft_circuit = qft_unitary(instance.m, measure=False)
        full_circuit = preparation_circuit.compose(qft_circuit)
        assert full_circuit is not None
        probabilities = Statevector.from_instruction(full_circuit).probabilities_dict()

        expected_bitstring: str = format(s_value, f"0{instance.m}b")
        assert probabilities[expected_bitstring] == pytest.approx(1.0)


def test_simulate_histograms_for_small_instance() -> None:
    """Simulation should produce one non-empty histogram per phase label."""

    instance: BenchmarkInstance = BenchmarkInstance(n=15, a=2, r=4, m=2)

    histograms: dict[int, Counter[int]] = simulate_histograms_for_instance(
        instance=instance,
        batch_size=2,
        num_shots=32,
        gate_error=False,
        readout_error=False,
        thermal_relaxation=False,
    )

    assert sorted(histograms.keys()) == [0, 1, 2, 3]
    s_value: int
    for s_value in range(instance.r):
        assert sum(histograms[s_value].values()) == 32
        assert histograms[s_value]


def test_histogram_sampler_and_benchmark_can_load_histogram_file(
    tmp_path: Path,
) -> None:
    """HistogramSampler and strict benchmark should load histograms from disk."""

    instance: BenchmarkInstance = BenchmarkInstance(n=15, a=2, r=4, m=2)
    histograms: dict[int, Counter[int]] = {
        0: Counter({0: 8}),
        1: Counter({1: 8}),
        2: Counter({2: 8}),
        3: Counter({3: 8}),
    }
    histogram_path: Path = tmp_path / "shor_histograms.json"

    save_histograms(
        instance=instance,
        histograms=histograms,
        output_path=histogram_path,
        batch_size=2,
        num_shots=8,
        gate_error=False,
        readout_error=False,
        thermal_relaxation=False,
    )

    sampler: HistogramSampler = HistogramSampler.from_file(
        histogram_path=histogram_path,
        instance=instance,
    )
    assert sampler.histograms[1][1] == 8

    histogram_file: HistogramFileModel = HistogramFileModel.load(histogram_path)
    assert histogram_file.instance == instance

    result = run_strict_benchmark(
        instance=instance,
        k_list=[1, 2],
        m_mc=16,
        seed=7,
        histogram_path=histogram_path,
    )
    assert result.simulation is not None
    assert sorted(result.simulation.metrics_by_k.keys()) == [1, 2]


def test_strict_benchmark_output_schema_round_trip(tmp_path: Path) -> None:
    """Strict benchmark output JSON should validate with the Pydantic schema."""

    benchmark_output: Path = tmp_path / "strict_output.json"
    benchmark_output.write_text(
        json.dumps(
            {
                "instance": {"n": 15, "a": 2, "r": 4, "m": 2},
                "k_list": [1, 2],
                "m_mc": 16,
                "seed": 7,
                "result": {
                    "ideal": {
                        "metrics_by_k": {
                            "1": {
                                "p_ord_strict": 0.5,
                                "p_wrong": 0.0,
                                "p_null": 0.5,
                            }
                        }
                    },
                    "uniform": {
                        "metrics_by_k": {
                            "1": {
                                "p_ord_strict": 0.25,
                                "p_wrong": 0.0,
                                "p_null": 0.75,
                            }
                        }
                    },
                    "arithmetic": {"p_ord_strict_by_k": {"1": 0.5}},
                    "simulation": {
                        "metrics_by_k": {
                            "1": {
                                "p_ord_strict": 0.75,
                                "p_wrong": 0.0,
                                "p_null": 0.25,
                            }
                        }
                    },
                },
                "simulation_histogram_file": "/tmp/example.json",
            }
        ),
        encoding="utf-8",
    )

    loaded: StrictBenchmarkResultFileModel = (
        StrictBenchmarkResultFileModel.model_validate_json(
            benchmark_output.read_text(encoding="utf-8")
        )
    )

    assert loaded.instance.q == 4
    assert loaded.result.simulation is not None
