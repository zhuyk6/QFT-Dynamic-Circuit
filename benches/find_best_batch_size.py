"""This script evaluates the performance of different batch sizes.

This script will try to find the best batch size which has:
- the minimum running time
- the minimum TVD (or maximum fidelity)
"""

import pickle
from collections import Counter
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from pprint import pprint
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from qiskit import QuantumCircuit

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

    for gate_error, readout_error, thermal_relaxation_error in sampler_tag_list:
        key = (
            f"g{int(gate_error)}-r{int(readout_error)}-t{int(thermal_relaxation_error)}"
        )
        print(f"Running for sampelr {key}")

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

    for gate_error, readout_error, thermal_relaxation_error in sampler_tag_list:
        key = (
            f"g{int(gate_error)}-r{int(readout_error)}-t{int(thermal_relaxation_error)}"
        )
        print(f"Running for sampelr {key}")

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


def main_circular():
    sampler_tag_list: list[tuple[bool, bool, bool]] = []
    for tag in product([False, True], repeat=3):
        sampler_tag_list.append(tag)  # type: ignore

    dict_sampler_batch_tvd = run_circular_state(
        12,
        [1, 2, 3],
        sampler_tag_list,
        10**5,
    )

    pprint(dict_sampler_batch_tvd)

    filename = (
        Path.cwd() / "data" / "opt_circuits" / "circular_state_tvd_sampler_batch.pkl"
    )
    # if filename already exists, add suffix number to avoid overwriting
    counter = 0
    while filename.exists():
        counter += 1
        filename = (
            Path.cwd()
            / "data"
            / "opt_circuits"
            / f"circular_state_tvd_sampler_batch_{counter}.pkl"
        )
    with open(filename, "wb") as f_out:
        pickle.dump(dict_sampler_batch_tvd, f_out)


def main_ghz():
    sampler_tag_list: list[tuple[bool, bool, bool]] = []
    for tag in product([False, True], repeat=3):
        sampler_tag_list.append(tag)  # type: ignore

    dict_sampler_batch_tvd = run_ghz_state(
        12,
        [1, 2, 3],
        sampler_tag_list,
        10**5,
    )

    pprint(dict_sampler_batch_tvd)

    filename = Path.cwd() / "data" / "opt_circuits" / "ghz_state_tvd_sampler_batch.pkl"
    # if filename already exists, add suffix number to avoid overwriting
    counter = 0
    while filename.exists():
        counter += 1
        filename = (
            Path.cwd()
            / "data"
            / "opt_circuits"
            / f"ghz_state_tvd_sampler_batch_{counter}.pkl"
        )

    with open(filename, "wb") as f_out:
        pickle.dump(dict_sampler_batch_tvd, f_out)


def main_time():
    batch_size_list = [1, 2, 3]
    runtime_dict: dict[int, float] = {}
    for batch_size in batch_size_list:
        runtime = calculate_runtime(batch_size, 12)
        runtime_dict[batch_size] = runtime

    print("Runtime for different batch sizes:")
    pprint(runtime_dict)


def main_tvd_vs_num_qubits():
    """Calculate TVD vs num_qubits for different batch size."""

    min_num_qubits = 2
    max_num_qubits = 12
    batch_size_list = [1, 2, 3]

    dict_batch_num_tvd: dict[int, dict[int, float]] = {}

    for num_qubits in range(min_num_qubits, max_num_qubits + 1):
        for batch_size in batch_size_list:
            # skip if num_qubits is not multiple of batch_size
            if num_qubits % batch_size != 0:
                continue

            tvd = calculate_metric(
                batch_size=batch_size,
                num_qubits=num_qubits,
                sampler_tag=(True, True, True),
                prepare_circ_fn=lambda n: prepare_circular_state_circuit(n, r=4),
                metric_fn=lambda counts: calc_tvd(
                    {k << (num_qubits - 2): 1 / 4 for k in range(4)}, counts
                ),
                num_shots=10**5,
            )

            if batch_size not in dict_batch_num_tvd:
                dict_batch_num_tvd[batch_size] = {}
            dict_batch_num_tvd[batch_size][num_qubits] = tvd

    print("TVD for different batch sizes and number of qubits:")
    pprint(dict_batch_num_tvd)

    # save results to disk
    filename = (
        Path.cwd() / "data" / "opt_circuits" / "circular_state_tvd_batch_num_qubits.pkl"
    )
    # if filename already exists, add suffix
    counter = 0
    while filename.exists():
        counter += 1
        filename = (
            Path.cwd()
            / "data"
            / "opt_circuits"
            / f"circular_state_tvd_batch_num_qubits_{counter}.pkl"
        )
    with open(filename, "wb") as f_out:
        pickle.dump(dict_batch_num_tvd, f_out)


def plot_tvd_vs_batch_size_for_different_samplers(filename: Path, savefile: Path):
    # Load data from disk
    with open(filename, "rb") as f_in:
        dict_sampler_batch_tvd: dict[str, dict[int, float]] = pickle.load(f_in)

    # Plot TVD vs Batch Size for different samplers
    # Each sampler is a subfig
    fig, axes = plt.subplots(4, 2, figsize=(10, 10))

    # Add gap between subplots
    fig.subplots_adjust(wspace=0.3, hspace=1.0)

    for i, (sampler_key, batch_tvd) in enumerate(dict_sampler_batch_tvd.items()):
        ax: Axes = axes[i // 2, i % 2]
        batch_sizes = sorted(batch_tvd.keys())
        tvd_values = [batch_tvd[bs] for bs in batch_sizes]
        ax.plot(batch_sizes, tvd_values, marker="o")

        ax.set_ylim(0.0, 0.5)

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("TVD")
        ax.set_title(f"Sampler: {sampler_key}")

        # set x ticks to be batch sizes
        ax.set_xticks(batch_sizes)

    fig.savefig(savefile)


def plot_tvd_vs_num_qubits_for_different_batch_sizes(filename: Path, savefile: Path):
    # Load data from disk
    with open(filename, "rb") as f_in:
        dict_batch_num_tvd: dict[int, dict[int, float]] = pickle.load(f_in)

    # Plot TVD vs Number of Qubits for different batch sizes
    fig, ax = plt.subplots(figsize=(8, 6))

    batch_size_list = list(dict_batch_num_tvd.keys())

    markers = ["o", "s", "^", "D", "v", "P", "*"]
    linestyles = ["-", "--", "-.", ":"]

    for i, batch_size in enumerate(batch_size_list):
        dict_num_tvd = dict_batch_num_tvd[batch_size]
        num_qubits = sorted(dict_num_tvd.keys())
        tvd_values = [dict_num_tvd[nq] for nq in num_qubits]

        ax.plot(
            num_qubits,
            tvd_values,
            marker=markers[i],
            linestyle=linestyles[i],
            label=f"Batch Size {batch_size}",
        )

    ax.legend()
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("TVD")

    fig.savefig(savefile)


if __name__ == "__main__":
    # main_circular()
    # main_ghz()
    # main_time()

    # main_tvd_vs_num_qubits()

    plot_tvd_vs_num_qubits_for_different_batch_sizes(
        Path.cwd()
        / "data"
        / "opt_circuits"
        / "circular_state_tvd_batch_num_qubits.pkl",
        Path.cwd()
        / "data"
        / "opt_circuits"
        / "circular_state_tvd_batch_num_qubits.png",
    )

    # plot_results(
    #     Path.cwd() / "data" / "opt_circuits" / "circular_state_tvd_sampler_batch.pkl",
    #     Path.cwd() / "data" / "opt_circuits" / "circular_state_tvd_sampler_batch.png",
    # )
