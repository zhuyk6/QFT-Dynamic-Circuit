"""Generate logical measurement datasets for dynamic-QFT batch-size studies."""

import warnings
from itertools import product
from pathlib import Path
from typing import Annotated, Callable, cast

import typer
from qiskit import QuantumCircuit
from tqdm import tqdm

from qft_dynamic.experiment_data import (
    ExperimentDataset,
    LogicalCountsGroup,
    LogicalCountsRun,
)
from qft_dynamic.tools.build_circuits import prepare_circular_state_circuit
from qft_dynamic.tools.config import resolve_shor_benchmark_paths
from qft_dynamic.tools.simulation import (
    NoiseModelConfig,
    build_qft_simulation_context,
    build_sampler,
    compose_with_layout,
    estimate_tiled_qft_runtime,
    sample_counts,
)

app = typer.Typer()
type NoiseTag = tuple[bool, bool, bool]


def _all_noise_tags() -> list[NoiseTag]:
    """Return all combinations of the three Aer noise toggles."""

    return [cast(NoiseTag, tag) for tag in product([False, True], repeat=3)]


def _prepare_ghz_circuit(num_qubits: int) -> QuantumCircuit:
    """Prepare a GHZ state on the requested number of qubits."""

    preparation = QuantumCircuit(num_qubits)
    preparation.h(0)
    for qubit_index in range(1, num_qubits):
        preparation.cx(0, qubit_index)
    preparation.barrier()
    return preparation


def _prepare_circular_circuit(num_qubits: int) -> QuantumCircuit:
    """Prepare the period-four circular state used by batch-size studies."""

    return prepare_circular_state_circuit(num_qubits, r=4)


def simulate_batch_group(
    batch_size: int,
    num_qubits: int,
    noise_tag: NoiseTag,
    prepare_circuit: Callable[[int], QuantumCircuit],
    num_shots: int,
) -> LogicalCountsGroup:
    """Simulate one batch-size and noise configuration as logical counts."""

    resolved_paths = resolve_shor_benchmark_paths()
    context = build_qft_simulation_context(
        num_qubits=num_qubits,
        batch_size=batch_size,
        hardware_config_path=resolved_paths.hardware_config_path,
        opt_circuits_path=resolved_paths.opt_circuits_path,
    )
    gate_error, readout_error, thermal_relaxation = noise_tag
    sampler = build_sampler(
        backend=context.backend,
        noise_config=NoiseModelConfig(
            gate_error=gate_error,
            readout_error=readout_error,
            thermal_relaxation=thermal_relaxation,
        ),
    )
    total_circuit: QuantumCircuit = compose_with_layout(
        transpiled_circuit=context.transpiled_qft,
        prepare_circuit=prepare_circuit(num_qubits),
    )
    noise_name: str = (
        f"g{int(gate_error)}-r{int(readout_error)}-t{int(thermal_relaxation)}"
    )
    return LogicalCountsGroup(
        name=f"batch-{batch_size}-{noise_name}",
        attributes={
            "batch_size": batch_size,
            "gate_error": gate_error,
            "readout_error": readout_error,
            "thermal_relaxation": thermal_relaxation,
        },
        runs=[
            LogicalCountsRun(
                run_id="repeat-0",
                metadata={"repeat": 0},
                counts=sample_counts(total_circuit, sampler, num_shots),
            )
        ],
    )


def simulate_batch_dataset(
    num_qubits: int,
    batch_sizes: list[int],
    noise_tags: list[NoiseTag],
    input_state: str,
    prepare_circuit: Callable[[int], QuantumCircuit],
    num_shots: int,
) -> ExperimentDataset:
    """Simulate a batch-size/noise sweep for one logical output width."""

    groups: list[LogicalCountsGroup] = []
    for noise_tag in tqdm(noise_tags):
        for batch_size in batch_sizes:
            if num_qubits % batch_size != 0:
                continue
            groups.append(
                simulate_batch_group(
                    batch_size=batch_size,
                    num_qubits=num_qubits,
                    noise_tag=noise_tag,
                    prepare_circuit=prepare_circuit,
                    num_shots=num_shots,
                )
            )
    return ExperimentDataset(
        dataset_id=f"qft-batch-sweep-{input_state}-{num_qubits}q",
        experiment_type=f"{input_state}_state_qft",
        num_qubits=num_qubits,
        bit_order="msb_first",
        producer="qiskit_aer",
        attributes={"input_state": input_state},
        groups=groups,
    )


def calculate_runtime(batch_size: int, num_qubits: int) -> float:
    """Calculate transpiled runtime for a batch size and circuit width."""

    resolved_paths = resolve_shor_benchmark_paths()
    return estimate_tiled_qft_runtime(
        num_qubits=num_qubits,
        batch_size=batch_size,
        hardware_config_path=resolved_paths.hardware_config_path,
        opt_circuits_path=resolved_paths.opt_circuits_path,
        unit="s",
        unroll_dynamic_circuit=True,
    )


def setup_warnings() -> None:
    """Suppress noisy Qiskit warnings."""

    warnings.filterwarnings("ignore", module="qiskit")


def _save_state_dataset(
    output: Path,
    num_qubits: int,
    batch_sizes: list[int],
    num_shots: int,
    input_state: str,
) -> None:
    """Simulate and save one input-state batch-size sweep."""

    prepare_circuit: Callable[[int], QuantumCircuit]
    if input_state == "circular":
        prepare_circuit = _prepare_circular_circuit
    elif input_state == "ghz":
        prepare_circuit = _prepare_ghz_circuit
    else:
        raise ValueError(f"unsupported input state: {input_state}")
    dataset: ExperimentDataset = simulate_batch_dataset(
        num_qubits=num_qubits,
        batch_sizes=batch_sizes,
        noise_tags=_all_noise_tags(),
        input_state=input_state,
        prepare_circuit=prepare_circuit,
        num_shots=num_shots,
    )
    dataset.save(output)
    print(f"Saved simulated logical counts to: {output}")


@app.command()
def run_circular(
    output: Annotated[
        Path, typer.Argument(help="Output counter ExperimentDataset JSON")
    ],
    num_qubits: Annotated[int, typer.Option(help="Number of qubits")] = 12,
    batch_sizes: Annotated[list[int], typer.Option(help="Batch sizes")] = [1, 2, 3],
    num_shots: Annotated[int, typer.Option(help="Shots per configuration")] = 10**5,
) -> None:
    """Simulate a circular-state batch-size/noise sweep."""

    setup_warnings()
    _save_state_dataset(output, num_qubits, batch_sizes, num_shots, "circular")


@app.command()
def run_ghz(
    output: Annotated[
        Path, typer.Argument(help="Output counter ExperimentDataset JSON")
    ],
    num_qubits: Annotated[int, typer.Option(help="Number of qubits")] = 12,
    batch_sizes: Annotated[list[int], typer.Option(help="Batch sizes")] = [1, 2, 3],
    num_shots: Annotated[int, typer.Option(help="Shots per configuration")] = 10**5,
) -> None:
    """Simulate a GHZ-state batch-size/noise sweep."""

    setup_warnings()
    _save_state_dataset(output, num_qubits, batch_sizes, num_shots, "ghz")


@app.command()
def run_time(
    num_qubits: Annotated[int, typer.Option(help="Number of qubits")] = 12,
    batch_sizes: Annotated[list[int], typer.Option(help="Batch sizes")] = [1, 2, 3],
) -> None:
    """Print estimated runtime for each compatible batch size."""

    setup_warnings()
    runtime_by_batch: dict[int, float] = {
        batch_size: calculate_runtime(batch_size, num_qubits)
        for batch_size in batch_sizes
        if num_qubits % batch_size == 0
    }
    print(runtime_by_batch)


@app.command()
def run_counts_vs_num_qubits(
    output_dir: Annotated[Path, typer.Argument(help="Output dataset directory")],
    min_num_qubits: Annotated[int, typer.Option(help="Minimum number of qubits")] = 2,
    max_num_qubits: Annotated[int, typer.Option(help="Maximum number of qubits")] = 12,
    batch_sizes: Annotated[list[int], typer.Option(help="Batch sizes")] = [1, 2, 3],
    num_shots: Annotated[int, typer.Option(help="Shots per configuration")] = 10**5,
) -> None:
    """Save one circular-state dataset per logical output width."""

    setup_warnings()
    for num_qubits in range(min_num_qubits, max_num_qubits + 1):
        dataset: ExperimentDataset = simulate_batch_dataset(
            num_qubits=num_qubits,
            batch_sizes=batch_sizes,
            noise_tags=[(True, True, True)],
            input_state="circular",
            prepare_circuit=_prepare_circular_circuit,
            num_shots=num_shots,
        )
        dataset.save(output_dir / f"circular-{num_qubits}q.json")


if __name__ == "__main__":
    app()
