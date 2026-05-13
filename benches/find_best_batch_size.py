"""Benchmark utilities for evaluating different batch sizes.

This script evaluates batch sizes using:
- the minimum running time
- the minimum TVD (or maximum fidelity)
"""

import pickle
import warnings
from collections import Counter
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from pprint import pprint
from typing import Annotated, Callable

import numpy as np
import typer
from qiskit import QuantumCircuit
from tqdm import tqdm

from qft_dynamic.tools.build_circuits import prepare_circular_state_circuit
from qft_dynamic.tools.config import resolve_shor_benchmark_paths
from qft_dynamic.tools.data_process import calc_tvd
from qft_dynamic.tools.simulation import (
    NoiseModelConfig,
    build_qft_simulation_context,
    build_sampler,
    compose_with_layout,
    estimate_tiled_qft_runtime,
    sample_counts,
)

app = typer.Typer()


def _all_sampler_tags() -> list[tuple[bool, bool, bool]]:
    """Return all combinations of the three noise toggles."""
    sampler_tag_list: list[tuple[bool, bool, bool]] = []
    for tag in product([False, True], repeat=3):
        sampler_tag_list.append(tag)  # type: ignore
    return sampler_tag_list


def _save_pickle_with_optional_suffix(
    payload: object,
    output_filename: Path,
    auto_suffix: bool = True,
) -> Path:
    """Save a pickle payload, optionally auto-incrementing the filename."""
    filename: Path = output_filename
    if auto_suffix:
        counter: int = 0
        while filename.exists():
            counter += 1
            filename = filename.with_name(
                f"{output_filename.stem}_{counter}{output_filename.suffix}"
            )

    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as output_file:
        pickle.dump(payload, output_file)
    print(f"Saved results to: {filename}")
    return filename


def calculate_runtime(
    batch_size: int,
    num_qubits: int,
) -> float:
    """Calculate the runtime for a given batch size.

    Args:
        batch_size (int): The batch size to use for the circuit.
        num_qubits (int): The total number of qubits in the circuit.
    """
    resolved_paths = resolve_shor_benchmark_paths()

    duration: float = estimate_tiled_qft_runtime(
        num_qubits=num_qubits,
        batch_size=batch_size,
        hardware_config_path=resolved_paths.hardware_config_path,
        opt_circuits_path=resolved_paths.opt_circuits_path,
        unit="s",
        unroll_dynamic_circuit=True,
    )
    return duration


def calculate_metric(
    batch_size: int,
    num_qubits: int,
    sampler_tag: tuple[bool, bool, bool],
    prepare_circ_fn: Callable[[int], QuantumCircuit],
    metric_fn: Callable[[Mapping[int, int]], float],
    num_shots: int = 10**4,
) -> float:
    """Calculate the metric for a given batch size and noise model.

    Args:
        batch_size (int): The batch size to use for the circuit.
        num_qubits (int): The total number of qubits in the circuit.
        sampler_tag (tuple[bool, bool, bool]): A tuple indicating which noise components to include (gate_error, readout_error, thermal_relaxation).
        prepare_circ_fn (Callable[[int], QuantumCircuit]): A function that takes num_qubits and returns a QuantumCircuit that prepares the desired state.
        metric_fn (Callable[[dict[int, int]], float]): A function that takes the noisy counts and calculates the desired metric (e.g., TVD or fidelity).
        num_shots (int): The number of shots to use for sampling. Default is 10^4.
    """
    resolved_paths = resolve_shor_benchmark_paths()

    context = build_qft_simulation_context(
        num_qubits=num_qubits,
        batch_size=batch_size,
        hardware_config_path=resolved_paths.hardware_config_path,
        opt_circuits_path=resolved_paths.opt_circuits_path,
    )

    gate_error, readout_error, thermal_relaxation = sampler_tag
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
        prepare_circuit=prepare_circ_fn(num_qubits),
    )

    noisy_counts: Counter[int] = sample_counts(
        circuit=total_circuit,
        sampler=sampler,
        num_shots=num_shots,
    )
    metric: float = metric_fn(noisy_counts)
    return metric


def run_circular_state(
    num_qubits: int,
    batch_size_list: list[int],
    sampler_tag_list: list[tuple[bool, bool, bool]],
    num_shots: int = 10**4,
) -> dict[str, dict[int, float]]:
    """Input circular state (r=4) and calculate TVD for different batch sizes and noise models."""
    # Record TVD results for each (sampler_key, batch_size)
    dict_sampler_batch_tvd: dict[str, dict[int, float]] = {}

    def metric_tvd(noisy_counts: Mapping[int, int]) -> float:
        ideal_prob = {k << (num_qubits - 2): 1 / 4 for k in range(4)}
        return calc_tvd(ideal_prob, noisy_counts)

    for gate_error, readout_error, thermal_relaxation_error in tqdm(sampler_tag_list):
        key = (
            f"g{int(gate_error)}-r{int(readout_error)}-t{int(thermal_relaxation_error)}"
        )
        print(f"Running for sampler {key}")

        dict_sampler_batch_tvd[key] = {}

        for batch_size in batch_size_list:
            tvd = calculate_metric(
                batch_size=batch_size,
                num_qubits=num_qubits,
                sampler_tag=(gate_error, readout_error, thermal_relaxation_error),
                prepare_circ_fn=lambda n: prepare_circular_state_circuit(n, r=4),
                metric_fn=metric_tvd,
                num_shots=num_shots,
            )

            dict_sampler_batch_tvd[key][batch_size] = tvd
            print(f"  Batch Size {batch_size}: TVD = {tvd:.6f}")

        print(f"Finished sampler {key}\n")

    return dict_sampler_batch_tvd


def run_ghz_state(
    num_qubits: int,
    batch_size_list: list[int],
    sampler_tag_list: list[tuple[bool, bool, bool]],
    num_shots: int = 10**4,
) -> dict[str, dict[int, float]]:
    """Input GHZ state and calculate TVD for different batch sizes and noise models."""
    # Record TVD results for each (sampler_key, batch_size)
    dict_sampler_batch_tvd: dict[str, dict[int, float]] = {}

    def prepare_ghz_circuit(num_qubits: int) -> QuantumCircuit:
        prepare_circ = QuantumCircuit(num_qubits)
        prepare_circ.h(0)
        for i in range(1, num_qubits):
            prepare_circ.cx(0, i)
        prepare_circ.barrier()
        return prepare_circ

    def metric_tvd(noisy_counts: Mapping[int, int]) -> float:
        N = 2**num_qubits
        ideal_prob = {k: (1 + np.cos(2 * np.pi * k / N)) / N for k in range(N)}
        return calc_tvd(ideal_prob, noisy_counts)

    for gate_error, readout_error, thermal_relaxation_error in tqdm(sampler_tag_list):
        key = (
            f"g{int(gate_error)}-r{int(readout_error)}-t{int(thermal_relaxation_error)}"
        )
        print(f"Running for sampler {key}")

        dict_sampler_batch_tvd[key] = {}

        for batch_size in batch_size_list:
            tvd = calculate_metric(
                batch_size=batch_size,
                num_qubits=num_qubits,
                sampler_tag=(gate_error, readout_error, thermal_relaxation_error),
                prepare_circ_fn=prepare_ghz_circuit,
                metric_fn=metric_tvd,
                num_shots=num_shots,
            )

            dict_sampler_batch_tvd[key][batch_size] = tvd
            print(f"  Batch Size {batch_size}: TVD = {tvd:.6f}")

        print(f"Finished sampler {key}\n")

    return dict_sampler_batch_tvd


def setup_warnings() -> None:
    """Suppress noisy Qiskit warnings."""
    warnings.filterwarnings("ignore", module="qiskit")


@app.command()
def run_circular(
    output: Annotated[Path, typer.Argument(help="Output pickle file path")],
    num_qubits: Annotated[int, typer.Option(help="Number of qubits")] = 12,
    batch_sizes: Annotated[
        list[int],
        typer.Option(
            help="Batch sizes",
        ),
    ] = [1, 2, 3],
    num_shots: Annotated[int, typer.Option(help="Number of shots per circuit")] = 10**5,
    auto_suffix: Annotated[
        bool, typer.Option(help="Auto-increment output path if it exists")
    ] = True,
) -> None:
    """Run circular-state TVD benchmark for all noise combinations."""
    setup_warnings()
    result = run_circular_state(
        num_qubits=num_qubits,
        batch_size_list=batch_sizes,
        sampler_tag_list=_all_sampler_tags(),
        num_shots=num_shots,
    )
    pprint(result)
    _save_pickle_with_optional_suffix(
        payload=result,
        output_filename=output,
        auto_suffix=auto_suffix,
    )


@app.command()
def run_ghz(
    output: Annotated[Path, typer.Argument(help="Output pickle file path")],
    num_qubits: Annotated[int, typer.Option(help="Number of qubits")] = 12,
    batch_sizes: Annotated[
        list[int],
        typer.Option(
            help="Batch sizes",
        ),
    ] = [1, 2, 3],
    num_shots: Annotated[int, typer.Option(help="Number of shots per circuit")] = 10**5,
    auto_suffix: Annotated[
        bool, typer.Option(help="Auto-increment output path if it exists")
    ] = True,
) -> None:
    """Run GHZ-state TVD benchmark for all noise combinations."""
    setup_warnings()
    result = run_ghz_state(
        num_qubits=num_qubits,
        batch_size_list=batch_sizes,
        sampler_tag_list=_all_sampler_tags(),
        num_shots=num_shots,
    )
    pprint(result)
    _save_pickle_with_optional_suffix(
        payload=result,
        output_filename=output,
        auto_suffix=auto_suffix,
    )


@app.command()
def run_time(
    num_qubits: Annotated[int, typer.Option(help="Number of qubits")] = 12,
    batch_sizes: Annotated[
        list[int],
        typer.Option(
            help="Batch sizes",
        ),
    ] = [1, 2, 3],
) -> None:
    """Print estimated runtime for each batch size."""
    runtime_dict: dict[int, float] = {}
    for batch_size in batch_sizes:
        runtime_dict[batch_size] = calculate_runtime(
            batch_size=batch_size,
            num_qubits=num_qubits,
        )
    print("Runtime for different batch sizes:")
    pprint(runtime_dict)


@app.command()
def run_tvd_vs_num_qubits(
    output: Annotated[Path, typer.Argument(help="Output pickle file path")],
    min_num_qubits: Annotated[int, typer.Option(help="Minimum number of qubits")] = 2,
    max_num_qubits: Annotated[int, typer.Option(help="Maximum number of qubits")] = 12,
    batch_sizes: Annotated[
        list[int],
        typer.Option(
            help="Batch sizes",
        ),
    ] = [1, 2, 3],
    num_shots: Annotated[int, typer.Option(help="Number of shots per circuit")] = 10**5,
    auto_suffix: Annotated[
        bool, typer.Option(help="Auto-incrementing output path if file exists")
    ] = True,
) -> None:
    """Run circular-state TVD benchmark versus the number of qubits."""
    setup_warnings()

    dict_batch_num_tvd: dict[int, dict[int, float]] = {}
    for num_qubits in tqdm(range(min_num_qubits, max_num_qubits + 1)):
        for batch_size in batch_sizes:
            if num_qubits % batch_size != 0:
                continue

            tvd: float = calculate_metric(
                batch_size=batch_size,
                num_qubits=num_qubits,
                sampler_tag=(True, True, True),
                prepare_circ_fn=lambda n: prepare_circular_state_circuit(n, r=4),
                metric_fn=lambda counts: calc_tvd(
                    {k << (num_qubits - 2): 1 / 4 for k in range(4)},
                    counts,
                ),
                num_shots=num_shots,
            )

            if batch_size not in dict_batch_num_tvd:
                dict_batch_num_tvd[batch_size] = {}
            dict_batch_num_tvd[batch_size][num_qubits] = tvd

    pprint(dict_batch_num_tvd)
    _save_pickle_with_optional_suffix(
        payload=dict_batch_num_tvd,
        output_filename=output,
        auto_suffix=auto_suffix,
    )


if __name__ == "__main__":
    app()
