"""Simulate measurement-encoding variants as logical count datasets."""

import warnings
from pathlib import Path
from typing import Annotated

import typer
from qiskit import QuantumCircuit
from tqdm import tqdm

from qft_dynamic.experiment_data import (
    ExperimentDataset,
    LogicalCountsGroup,
    LogicalCountsRun,
    MetadataValue,
)
from qft_dynamic.tools.build_backend import load_hardware_config
from qft_dynamic.tools.build_circuits import prepare_circular_state_circuit
from qft_dynamic.tools.config import resolve_shor_benchmark_paths
from qft_dynamic.tools.simulation import (
    NoiseModelConfig,
    _build_line_backend,
    _transpile_for_simulation,
    build_sampler,
    build_tiled_qft_circuit,
    compose_with_layout,
    sample_counts,
)
from qft_dynamic.tools.transpile import add_delay_before_measurement

app = typer.Typer()


def setup_warnings() -> None:
    """Suppress noisy Qiskit warnings."""

    warnings.filterwarnings("ignore", module="qiskit")


def simulate_encoding_group(
    num_qubits: int,
    batch_size: int,
    method: str,
    num_shots: int,
    delay_time: float | None = None,
    prob_meas1_prep0: float | None = None,
    prob_meas0_prep1: float | None = None,
) -> LogicalCountsGroup:
    """Simulate one measurement-encoding configuration.

    Args:
        num_qubits: Logical circuit width.
        batch_size: Dynamic-QFT tile size.
        method: Stable measurement-encoding method label.
        num_shots: Number of simulation shots.
        delay_time: Optional delay before measurement in seconds.
        prob_meas1_prep0: Optional modified ``P(1|0)`` readout error.
        prob_meas0_prep1: Optional modified ``P(0|1)`` readout error.

    Returns:
        Dataset group containing the full logical count distribution.
    """

    resolved_paths = resolve_shor_benchmark_paths()
    hardware_config = load_hardware_config(resolved_paths.hardware_config_path)
    if delay_time is not None:
        hardware_config["prob_meas0_prep1"] = prob_meas0_prep1 or 0.0
        hardware_config["prob_meas1_prep0"] = prob_meas1_prep0 or 0.0
    backend = _build_line_backend(
        num_qubits=num_qubits, hardware_config=hardware_config
    )
    circuit: QuantumCircuit = build_tiled_qft_circuit(
        num_qubits=num_qubits,
        batch_size=batch_size,
        t_feed_forward=hardware_config["t_feed_forward"],
        opt_circuits_path=resolved_paths.opt_circuits_path,
    )
    if delay_time is not None:
        circuit = add_delay_before_measurement(circuit, delay_time)
    transpiled_circuit: QuantumCircuit = _transpile_for_simulation(circuit, backend)
    total_circuit: QuantumCircuit = compose_with_layout(
        transpiled_circuit=transpiled_circuit,
        prepare_circuit=prepare_circular_state_circuit(num_qubits, r=4),
    )
    sampler = build_sampler(backend=backend, noise_config=NoiseModelConfig())
    attributes: dict[str, MetadataValue] = {
        "batch_size": batch_size,
        "method": method,
    }
    if delay_time is not None:
        attributes["delay_time"] = delay_time
        attributes["prob_meas1_prep0"] = prob_meas1_prep0 or 0.0
        attributes["prob_meas0_prep1"] = prob_meas0_prep1 or 0.0
    return LogicalCountsGroup(
        name=f"batch-{batch_size}-{method}",
        attributes=attributes,
        runs=[
            LogicalCountsRun(
                run_id="repeat-0",
                metadata={"repeat": 0},
                counts=sample_counts(total_circuit, sampler, num_shots),
            )
        ],
    )


def simulate_encoding_dataset(
    num_qubits: int,
    batch_sizes: list[int],
    delay_time: float,
    num_shots: int,
    prob_meas1_prep0: float,
    prob_meas0_prep1: float,
) -> ExperimentDataset:
    """Simulate all measurement-encoding and batch-size configurations."""

    groups: list[LogicalCountsGroup] = []
    for batch_size in tqdm(batch_sizes):
        groups.extend(
            [
                simulate_encoding_group(
                    num_qubits=num_qubits,
                    batch_size=batch_size,
                    method="base",
                    num_shots=num_shots,
                ),
                simulate_encoding_group(
                    num_qubits=num_qubits,
                    batch_size=batch_size,
                    method="enc_perfect",
                    num_shots=num_shots,
                    delay_time=delay_time,
                ),
                simulate_encoding_group(
                    num_qubits=num_qubits,
                    batch_size=batch_size,
                    method="enc_modified",
                    num_shots=num_shots,
                    delay_time=delay_time,
                    prob_meas1_prep0=prob_meas1_prep0,
                    prob_meas0_prep1=prob_meas0_prep1,
                ),
            ]
        )
    return ExperimentDataset(
        dataset_id=f"qft-measurement-encoding-{num_qubits}q",
        experiment_type="circular_state_qft",
        num_qubits=num_qubits,
        bit_order="msb_first",
        producer="qiskit_aer",
        attributes={"input_state": "circular", "period": 4},
        groups=groups,
    )


@app.command()
def main(
    output: Annotated[
        Path, typer.Argument(help="Output counter ExperimentDataset JSON")
    ],
    num_qubits: Annotated[int, typer.Option(help="Number of qubits")] = 12,
    batch_sizes: Annotated[list[int], typer.Option(help="Batch sizes")] = [1, 2, 3],
    delay_time: Annotated[
        float, typer.Option(help="Delay before measurement in seconds")
    ] = 100e-9,
    num_shots: Annotated[int, typer.Option(help="Shots per configuration")] = 10**5,
    prob_meas1_prep0: Annotated[
        float, typer.Option(help="Modified probability P(1|0)")
    ] = 0.001,
    prob_meas0_prep1: Annotated[
        float, typer.Option(help="Modified probability P(0|1)")
    ] = 0.002,
) -> None:
    """Simulate measurement encoding and save full logical counts."""

    setup_warnings()
    dataset: ExperimentDataset = simulate_encoding_dataset(
        num_qubits=num_qubits,
        batch_sizes=batch_sizes,
        delay_time=delay_time,
        num_shots=num_shots,
        prob_meas1_prep0=prob_meas1_prep0,
        prob_meas0_prep1=prob_meas0_prep1,
    )
    dataset.save(output)
    print(f"Saved simulated logical counts to: {output}")


if __name__ == "__main__":
    app()
