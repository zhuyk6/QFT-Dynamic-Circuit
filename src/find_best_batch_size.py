"""This script evaluates the performance of different batch sizes.

This script will try to find the best batch size which has:
- the minimum running time
- the minimum TVD (or maximum fidelity)
"""

import pickle
from itertools import product
from pathlib import Path
from pprint import pprint
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from qiskit import QuantumCircuit, generate_preset_pass_manager, qpy
from qiskit.transpiler import CouplingMap, PassManager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import Sampler
from qiskit_ibm_runtime.transpiler.passes.scheduling import (
    ASAPScheduleAnalysis,
    PadDelay,
)

from tools.build_backend import build_backend, load_hardware_config
from tools.build_circuits import (
    prepare_circular_state_circuit,
    tile_transpiled_circuit,
)
from tools.data_process import calc_tvd
from tools.transpile import unroll_if_true


def calculate_runtime(
    batch_size: int,
    num_qubits: int,
) -> float:
    """Calculate the runtime for a given batch size.

    Args:
        batch_size (int): The batch size to use for the circuit.
        num_qubits (int): The total number of qubits in the circuit.
    """
    assert num_qubits % batch_size == 0, "num_qubits must be multiple of batch_size"

    # Load optimized circuit
    filename = Path.cwd() / "data" / "opt_circuits" / f"qft{batch_size}.qpy"
    with open(filename, "rb") as f_in:
        circ_opt: QuantumCircuit = qpy.load(f_in)[0]

    # build backend with given parameters
    hardware_config = load_hardware_config(
        Path.cwd() / "data" / "量子院" / "hardware.toml"
    )

    coupling_map = CouplingMap.from_line(num_qubits)
    backend = build_backend(coupling_map, hardware_config)

    # Transpilation PassManager
    pm: PassManager = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        initial_layout=list(range(num_qubits)),
        routing_method="none",
    )
    durations = backend.target.durations()
    pm.scheduling = PassManager(  # type: ignore
        [
            ASAPScheduleAnalysis(durations),
            PadDelay(durations),
        ]
    )

    # Build large circuit by tiling
    sub_circ = circ_opt.copy()
    num_tiles = num_qubits // batch_size
    tiling_pattern = [
        [i + j * batch_size for i in range(batch_size)] for j in range(num_tiles)
    ]

    large_circuit = tile_transpiled_circuit(
        sub_circ,
        tiling_pattern,
        hardware_config["t_feed_forward"],
    )
    layout = large_circuit.layout

    # make dynamic circuit to static circuit by unrolling
    static_circuit = unroll_if_true(large_circuit)

    # Transpile large circuit
    total_circuit = pm.run(static_circuit)
    total_circuit._layout = layout
    assert isinstance(total_circuit, QuantumCircuit)
    assert total_circuit.layout is not None

    # Calculate running time
    duration = total_circuit.estimate_duration(target=backend.target, unit="s")
    return duration


def calculate_metric(
    batch_size: int,
    num_qubits: int,
    sampler_tag: tuple[bool, bool, bool],
    prepare_circ_fn: Callable[[int], QuantumCircuit],
    metric_fn: Callable[[dict[int, int]], float],
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
    assert num_qubits % batch_size == 0, "num_qubits must be multiple of batch_size"

    # Load optimized circuit
    filename = Path.cwd() / "data" / "opt_circuits" / f"qft{batch_size}.qpy"
    with open(filename, "rb") as f_in:
        circ_opt: QuantumCircuit = qpy.load(f_in)[0]

    # build backend with given parameters
    hardware_config = load_hardware_config(
        Path.cwd() / "data" / "量子院" / "hardware.toml"
    )

    coupling_map = CouplingMap.from_line(num_qubits)
    backend = build_backend(coupling_map, hardware_config)

    # build sampler with given noise model
    gate_error, readout_error, thermal_relaxation = sampler_tag
    noise_model = NoiseModel.from_backend(
        backend,
        gate_error=gate_error,
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
    )
    sampler = Sampler(mode=AerSimulator(noise_model=noise_model))

    # Transpilation PassManager
    pm: PassManager = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        initial_layout=list(range(num_qubits)),
        routing_method="none",
    )
    durations = backend.target.durations()
    pm.scheduling = PassManager(  # type: ignore
        [
            ASAPScheduleAnalysis(durations),
            PadDelay(durations),
        ]
    )

    # Build large circuit by tiling
    sub_circ = circ_opt.copy()
    num_tiles = num_qubits // batch_size
    tiling_pattern = [
        [i + j * batch_size for i in range(batch_size)] for j in range(num_tiles)
    ]

    large_circuit = tile_transpiled_circuit(
        sub_circ,
        tiling_pattern,
        hardware_config["t_feed_forward"],
    )
    layout = large_circuit.layout

    # Transpile large circuit
    total_circuit = pm.run(large_circuit)
    total_circuit._layout = layout
    assert isinstance(total_circuit, QuantumCircuit)
    assert total_circuit.layout is not None

    # Build total circuit: prepare + large
    map_logical_to_physical = {
        vq._index: pq
        for pq, vq in total_circuit.layout.initial_layout.get_physical_bits().items()
    }
    map_bits = [map_logical_to_physical[i] for i in range(total_circuit.num_qubits)]
    total_circuit.compose(
        prepare_circ_fn(num_qubits), qubits=map_bits, front=True, inplace=True
    )

    # Run sampler and calculate TVD
    result = sampler.run([total_circuit], shots=num_shots).result()
    counts = result[0].data["c"].get_counts()

    num_qubits = large_circuit.num_qubits
    noisy_counts: dict[int, int] = {int(k, base=2): v for k, v in counts.items()}

    metric = metric_fn(noisy_counts)

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

    def metric_tvd(noisy_counts: dict[int, int]) -> float:
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

    def metric_tvd(noisy_counts: dict[int, int]) -> float:
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
