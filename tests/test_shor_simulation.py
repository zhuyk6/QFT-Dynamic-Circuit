"""Unit tests for Shor histogram simulation utilities."""

import json
from collections import Counter
from pathlib import Path

import pytest
from qiskit.quantum_info import Statevector

from qft_dynamic.experiment_data import (
    ExperimentDataset,
    LogicalCountsGroup,
    LogicalCountsRun,
)
from qft_dynamic.shor_benchmark.samplers import HistogramSampler
from qft_dynamic.shor_benchmark.schemas import StrictBenchmarkResultFileModel
from qft_dynamic.shor_benchmark.simulation import (
    prepare_forward_qft_phase_state,
    simulate_dataset_for_instance,
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
        full_circuit = preparation_circuit.compose(qft_circuit, inplace=False)
        probabilities = Statevector.from_instruction(full_circuit).probabilities_dict()

        expected_bitstring: str = format(s_value, f"0{instance.m}b")
        assert probabilities[expected_bitstring] == pytest.approx(1.0)


def test_simulate_dataset_for_small_instance() -> None:
    """Simulation should produce one non-empty run per phase label."""

    instance: BenchmarkInstance = BenchmarkInstance(n=15, a=2, r=4, m=2)

    dataset: ExperimentDataset = simulate_dataset_for_instance(
        instance=instance,
        batch_size=1,
        num_shots=32,
        gate_error=False,
        readout_error=False,
        thermal_relaxation=False,
    )

    assert dataset.experiment_type == "shor_order_finding"
    assert dataset.producer == "qiskit_aer"
    assert dataset.attributes == {"n": 15, "a": 2, "r": 4, "m": 2}
    assert [run.metadata for run in dataset.groups[0].runs] == [
        {"s": 0},
        {"s": 1},
        {"s": 2},
        {"s": 3},
    ]
    for s_value, run in enumerate(dataset.groups[0].runs):
        assert run.num_shots == 32
        assert run.counts == Counter({s_value: 32})


def test_histogram_sampler_can_load_experiment_dataset(
    tmp_path: Path,
) -> None:
    """HistogramSampler should consume the common logical dataset format."""

    instance: BenchmarkInstance = BenchmarkInstance(n=15, a=2, r=4, m=2)
    dataset: ExperimentDataset = ExperimentDataset(
        dataset_id="shor-test",
        experiment_type="shor_order_finding",
        num_qubits=2,
        bit_order="msb_first",
        producer="test",
        attributes={"n": 15, "a": 2, "r": 4, "m": 2},
        groups=[
            LogicalCountsGroup(
                name="example",
                runs=[
                    LogicalCountsRun(
                        run_id=f"s-{s_value}",
                        metadata={"s": s_value},
                        counts=Counter({s_value: 8}),
                    )
                    for s_value in range(4)
                ],
            )
        ],
    )
    dataset_path: Path = tmp_path / "shor-dataset.json"
    dataset.save(dataset_path)

    sampler: HistogramSampler = HistogramSampler.from_dataset_file(
        dataset_path=dataset_path,
        instance=instance,
    )
    assert sampler.histograms[1][1] == 8


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
                    "experiments": [
                        {
                            "metrics_by_k": {
                                "1": {
                                    "p_ord_strict": 0.75,
                                    "p_wrong": 0.0,
                                    "p_null": 0.25,
                                }
                            }
                        }
                    ],
                },
                "experiment_dataset_files": ["/tmp/example.json"],
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
    assert len(loaded.result.experiments) == 1
    assert len(loaded.experiment_dataset_files) == 1
