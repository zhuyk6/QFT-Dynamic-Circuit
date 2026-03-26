"""This script benchmarks the effect of measurement encoding.

Here we compare two methods:
- Original circuit: consider three kinds of noise.
- Delayed circuit: using a delay for measurement encoding, consider a modified readout error.
"""

import json
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import Sampler

from tools.build_backend import build_backend, load_hardware_config
from tools.build_circuits import (
    prepare_circular_state_circuit,
    tile_transpiled_circuit,
)
from tools.data_process import calc_tvd
from tools.transpile import add_delay_before_measurement, generate_pass_manager

ROOT: Path = Path.cwd()


def build_circuit(
    num_qubits: int,
    batch_size: int,
    t_feed_forward: float,
) -> QuantumCircuit:
    """Build the large circuit by tiling the optimized circuit.

    Args:
        num_qubits (int): number of qubits in the large circuit.
        batch_size (int): number of qubits in the optimized circuit (tile size).
        t_feed_forward (float): feed forward time for dynamic circuit, the unit is seconds.

    Returns:
        QuantumCircuit: the large circuit built by tiling the optimized circuit.
    """
    assert num_qubits % batch_size == 0, "num_qubits must be multiple of batch_size"

    # Load optimized circuit
    filename = ROOT / "data" / "opt_circuits" / f"qft{batch_size}.qpy"
    with open(filename, "rb") as f_in:
        circ_opt: QuantumCircuit = qpy.load(f_in)[0]

    # Build large circuit by tiling
    sub_circ = circ_opt.copy()
    num_tiles = num_qubits // batch_size
    tiling_pattern = [
        [i + j * batch_size for i in range(batch_size)] for j in range(num_tiles)
    ]

    large_circuit = tile_transpiled_circuit(
        sub_circ,
        tiling_pattern,
        t_feed_forward,
    )

    return large_circuit


def benchmark(
    num_qubits: int,
    batch_size: int,
    delay_time: float | None = None,
    num_shots: int = 10**5,
    prob_meas1_prep0: float | None = None,
    prob_meas0_prep1: float | None = None,
) -> float:
    """The benchmark function for measurement encoding.

    - If `delay_time` is None, the original circuit is used and no delay is added.
    - If `delay_time` is not None, a delay is added before measurement and the readout error
    is modified according to `prob_meas1_prep0` and `prob_meas0_prep1`.
    If these two parameters are None, the readout error is ignored (set to 0).

    Args:
        num_qubits (int): number of qubits in the large circuit.
        batch_size (int): number of qubits in the optimized circuit (tile size).
        delay_time (float | None, optional): delay time before measurement, the unit is seconds. Defaults to None.
        num_shots (int, optional): number of shots for simulation. Defaults to 10**5.
        prob_meas1_prep0 (float | None, optional): the modified probability of measuring 1 when preparing 0. Defaults to None.
        prob_meas0_prep1 (float | None, optional): the modified probability of measuring 0 when preparing 1. Defaults to None.

    Returns:
        float: TVD between ideal and noisy distributions.
    """
    # Load hardware config
    hardware_config = load_hardware_config(ROOT / "data" / "hardware.toml")
    if delay_time is not None:
        if prob_meas1_prep0 is not None and prob_meas0_prep1 is not None:
            hardware_config["prob_meas0_prep1"] = prob_meas0_prep1
            hardware_config["prob_meas1_prep0"] = prob_meas1_prep0
        else:
            hardware_config["prob_meas0_prep1"] = 0.0
            hardware_config["prob_meas1_prep0"] = 0.0

    # Build backend
    coupling_map = CouplingMap.from_line(num_qubits)
    backend = build_backend(coupling_map, hardware_config)

    # define circuit
    circuit = build_circuit(num_qubits, batch_size, hardware_config["t_feed_forward"])

    # add delay before measurement
    if delay_time is not None:
        circuit = add_delay_before_measurement(circuit, delay_time)

    # transpile
    pm = generate_pass_manager(backend)
    transpiled_circuit = pm.run(circuit)

    # Build total circuit: prepare + transpiled
    def build_total_circuit(transpiled_circuit: QuantumCircuit) -> QuantumCircuit:
        assert transpiled_circuit.layout is not None, (
            "Transpiled circuit must have layout information"
        )

        prepare_circuit = prepare_circular_state_circuit(num_qubits, r=4)

        map_logical_to_physical = {
            vq._index: pq
            for pq, vq in transpiled_circuit.layout.initial_layout.get_physical_bits().items()
        }
        map_bits = [
            map_logical_to_physical[i] for i in range(transpiled_circuit.num_qubits)
        ]
        total_circuit = transpiled_circuit.compose(
            prepare_circuit, qubits=map_bits, front=True, inplace=False
        )
        assert total_circuit is not None
        return total_circuit

    total_circuit = build_total_circuit(transpiled_circuit)

    # define sampler
    sampler = Sampler(
        mode=AerSimulator(
            noise_model=NoiseModel.from_backend(
                backend=backend,
                gate_error=True,
                readout_error=True,
                thermal_relaxation=True,
            )
        )
    )

    # define metrics
    def metric_tvd(noisy_counts: dict[int, int]) -> float:
        ideal_prob = {k << (num_qubits - 2): 1 / 4 for k in range(4)}
        return calc_tvd(ideal_prob, noisy_counts)

    # Run sampler and calculate metrics
    def simulate_and_metric(total_circuit: QuantumCircuit, sampler: Sampler) -> float:
        result = sampler.run([total_circuit], shots=num_shots).result()
        counts = result[0].data["c"].get_counts()
        noisy_counts: dict[int, int] = {int(k, base=2): v for k, v in counts.items()}
        tvd = metric_tvd(noisy_counts)
        return tvd

    tvd = simulate_and_metric(total_circuit, sampler)

    return tvd


def main():
    num_qubits = 12

    delay_time = 100e-9  # 100 ns

    # modified readout error
    prob_meas1_prep0 = 0.001
    prob_meas0_prep1 = 0.002

    # save results
    dict_tvd_batch_method: dict[int, dict[str, float]] = {}

    batch_size_list = [1, 2, 3]
    for batch_size in batch_size_list:
        if batch_size not in dict_tvd_batch_method:
            dict_tvd_batch_method[batch_size] = {}

        dict_tvd_batch_method[batch_size]["base"] = benchmark(
            num_qubits=num_qubits,
            batch_size=batch_size,
            delay_time=None,
        )
        dict_tvd_batch_method[batch_size]["enc perfect"] = benchmark(
            num_qubits=num_qubits,
            batch_size=batch_size,
            delay_time=delay_time,
        )
        dict_tvd_batch_method[batch_size]["enc modify"] = benchmark(
            num_qubits=num_qubits,
            batch_size=batch_size,
            delay_time=delay_time,
            prob_meas0_prep1=prob_meas0_prep1,
            prob_meas1_prep0=prob_meas1_prep0,
        )

    print("TVD results:")
    pprint(dict_tvd_batch_method)

    # save results to disk
    filename = ROOT / "data" / "opt_circuits" / "bench_enc.json"
    # if file exists, increase suffix
    counter = 0
    while filename.exists():
        counter += 1
        filename = ROOT / "data" / "opt_circuits" / f"bench_enc_{counter}.json"
    with open(filename, "w") as f_out:
        json.dump(dict_tvd_batch_method, f_out, indent=4)


def plot_result(
    results_filename: Path,
    savefig_filename: Path,
) -> None:
    # Load results
    with open(results_filename, "r") as f_in:
        dict_tvd_batch_method: dict[int, dict[str, float]] = json.load(f_in)

    # Plot results
    # Using histogram:
    # - each group of bars corresponds to a batch size
    # - each bar in a group corresponds to a method (base, enc perfect, enc modify)
    batch_sizes = sorted(dict_tvd_batch_method.keys())
    methods = ["base", "enc perfect", "enc modify"]

    x = np.arange(len(batch_sizes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        method_values = [
            dict_tvd_batch_method[batch_size][method] for batch_size in batch_sizes
        ]
        ax.bar(x + i * width, method_values, width, label=method)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("TVD")
    ax.set_title("TVD for Different Batch Sizes and Encode Methods")
    ax.set_xticks(x + width)
    ax.set_xticklabels(map(str, batch_sizes))
    ax.legend()

    fig.savefig(savefig_filename)


if __name__ == "__main__":
    print(f"Project root: {ROOT}")
    # main()

    plot_result(
        ROOT / "data" / "opt_circuits" / "bench_enc.json",
        ROOT / "data" / "opt_circuits" / "bench_enc.png",
    )
