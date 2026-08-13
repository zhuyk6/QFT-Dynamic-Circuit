r"""Simulate dynamic-QFT process-fidelity logical measurement datasets."""

import math
import random
import warnings
from pathlib import Path
from typing import Annotated, Literal

import typer
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import Sampler
from tqdm import tqdm

from qft_dynamic.experiment_data import (
    ExperimentDataset,
    LogicalCountsGroup,
    LogicalCountsRun,
)
from qft_dynamic.tools.config import resolve_shor_benchmark_paths
from qft_dynamic.tools.simulation import (
    NoiseModelConfig,
    build_qft_simulation_context,
    build_sampler,
    compose_with_layout,
    sample_counts,
)

app = typer.Typer()


def setup_warnings() -> None:
    """Suppress noisy Qiskit warnings."""

    warnings.filterwarnings("ignore", module="qiskit")


def prepare_sigma_k_star(num_qubits: int, k: int) -> QuantumCircuit:
    """Prepare the process-fidelity input state indexed by ``k``.

    Args:
        num_qubits: Logical circuit width.
        k: Target basis-state index.

    Returns:
        State-preparation circuit for ``sigma_k_star``.
    """

    preparation = QuantumCircuit(num_qubits)
    preparation.h(range(num_qubits))
    for qubit_index in range(num_qubits):
        theta: float = -math.pi * k / (2**qubit_index)
        preparation.rz(theta, qubit_index)
    preparation.barrier()
    return preparation


def select_targets(
    num_qubits: int,
    mode: Literal["exact", "sample"],
    num_samples: int,
    seed: int | None,
) -> list[int]:
    """Select exact or sampled process-fidelity target indices."""

    num_states: int = 1 << num_qubits
    if mode == "exact":
        return list(range(num_states))
    if num_samples < 2:
        raise ValueError("num_samples must be at least two")
    if num_samples > num_states:
        raise ValueError("num_samples must not exceed the number of logical states")
    return random.Random(seed).sample(range(num_states), num_samples)


def simulate_process_fidelity_group(
    num_qubits: int,
    batch_size: int,
    target_k_values: list[int],
    num_shots: int,
    noise_config: NoiseModelConfig,
) -> LogicalCountsGroup:
    """Simulate full logical counts for one batch-size configuration.

    Args:
        num_qubits: Logical circuit width.
        batch_size: Dynamic-QFT tile size.
        target_k_values: Process-fidelity input indices to simulate.
        num_shots: Shots per target index.
        noise_config: Enabled Aer noise components.

    Returns:
        One dataset group containing a run per target index.
    """

    resolved_paths = resolve_shor_benchmark_paths()
    context = build_qft_simulation_context(
        num_qubits=num_qubits,
        batch_size=batch_size,
        hardware_config_path=resolved_paths.hardware_config_path,
        opt_circuits_path=resolved_paths.opt_circuits_path,
    )
    sampler: Sampler = build_sampler(
        backend=context.backend,
        noise_config=noise_config,
    )
    runs: list[LogicalCountsRun] = []
    for target_k in target_k_values:
        total_circuit: QuantumCircuit = compose_with_layout(
            transpiled_circuit=context.transpiled_qft,
            prepare_circuit=prepare_sigma_k_star(num_qubits, target_k),
        )
        counts = sample_counts(total_circuit, sampler, num_shots)
        runs.append(
            LogicalCountsRun(
                run_id=f"k-{target_k}",
                metadata={"k": target_k},
                counts=counts,
            )
        )

    return LogicalCountsGroup(
        name=f"batch-{batch_size}",
        attributes={
            "batch_size": batch_size,
            "gate_error": noise_config.gate_error,
            "readout_error": noise_config.readout_error,
            "thermal_relaxation": noise_config.thermal_relaxation,
        },
        runs=runs,
    )


def simulate_process_fidelity_dataset(
    num_qubits: int,
    batch_sizes: list[int],
    mode: Literal["exact", "sample"],
    num_shots: int,
    num_samples: int,
    seed: int | None,
    noise_config: NoiseModelConfig,
) -> ExperimentDataset:
    """Simulate process-fidelity counts for every requested batch size."""

    target_k_values: list[int] = select_targets(
        num_qubits=num_qubits,
        mode=mode,
        num_samples=num_samples,
        seed=seed,
    )
    groups: list[LogicalCountsGroup] = [
        simulate_process_fidelity_group(
            num_qubits=num_qubits,
            batch_size=batch_size,
            target_k_values=target_k_values,
            num_shots=num_shots,
            noise_config=noise_config,
        )
        for batch_size in tqdm(batch_sizes)
        if num_qubits % batch_size == 0
    ]
    return ExperimentDataset(
        dataset_id=f"qft-process-fidelity-{num_qubits}q",
        experiment_type="process_fidelity",
        num_qubits=num_qubits,
        bit_order="msb_first",
        producer="qiskit_aer",
        attributes={
            "sampling_mode": mode,
            "num_samples": len(target_k_values),
            **({"seed": seed} if seed is not None else {}),
        },
        groups=groups,
    )


@app.command()
def main(
    output: Annotated[
        Path, typer.Argument(help="Output counter ExperimentDataset JSON")
    ],
    num_qubits: Annotated[int, typer.Option(help="Number of qubits")] = 12,
    batch_sizes: Annotated[list[int], typer.Option(help="Batch sizes")] = [1, 2, 3],
    mode: Annotated[
        Literal["exact", "sample"],
        typer.Option(help="Target-index selection mode"),
    ] = "sample",
    num_shots: Annotated[int, typer.Option(help="Shots per target circuit")] = 10**4,
    num_samples: Annotated[int, typer.Option(help="Sampled target count")] = 20,
    seed: Annotated[int | None, typer.Option(help="Target-selection seed")] = None,
    gate_error: Annotated[bool, typer.Option(help="Enable gate error")] = True,
    readout_error: Annotated[bool, typer.Option(help="Enable readout error")] = True,
    thermal_relaxation: Annotated[
        bool, typer.Option(help="Enable thermal relaxation")
    ] = True,
) -> None:
    """Simulate and save process-fidelity logical counts."""

    setup_warnings()
    dataset: ExperimentDataset = simulate_process_fidelity_dataset(
        num_qubits=num_qubits,
        batch_sizes=batch_sizes,
        mode=mode,
        num_shots=num_shots,
        num_samples=num_samples,
        seed=seed,
        noise_config=NoiseModelConfig(
            gate_error=gate_error,
            readout_error=readout_error,
            thermal_relaxation=thermal_relaxation,
        ),
    )
    dataset.save(output)
    print(f"Saved simulated logical counts to: {output}")


if __name__ == "__main__":
    app()
