"""This script benchmarks the effect of measurement encoding.

Here we compare two methods:
- Original circuit: consider three kinds of noise.
- Delayed circuit: using a delay for measurement encoding, consider a modified readout error.
"""

import argparse
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

# Root path of the project, used for loading data files.
ROOT = Path.cwd()


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


def run_benchmark_suite(
    num_qubits: int,
    batch_size_list: list[int],
    delay_time: float,
    num_shots: int,
    prob_meas1_prep0: float,
    prob_meas0_prep1: float,
    output_filename: Path,
    auto_suffix: bool = True,
) -> Path:
    """Run all benchmark variants and save results to disk.

    Returns:
        Path: the actual output file path.
    """
    dict_tvd_batch_method: dict[int, dict[str, float]] = {}

    for batch_size in batch_size_list:
        if batch_size not in dict_tvd_batch_method:
            dict_tvd_batch_method[batch_size] = {}

        dict_tvd_batch_method[batch_size]["base"] = benchmark(
            num_qubits=num_qubits,
            batch_size=batch_size,
            delay_time=None,
            num_shots=num_shots,
        )
        dict_tvd_batch_method[batch_size]["enc perfect"] = benchmark(
            num_qubits=num_qubits,
            batch_size=batch_size,
            delay_time=delay_time,
            num_shots=num_shots,
        )
        dict_tvd_batch_method[batch_size]["enc modify"] = benchmark(
            num_qubits=num_qubits,
            batch_size=batch_size,
            delay_time=delay_time,
            num_shots=num_shots,
            prob_meas0_prep1=prob_meas0_prep1,
            prob_meas1_prep0=prob_meas1_prep0,
        )

    print("TVD results:")
    pprint(dict_tvd_batch_method)

    # save results to disk
    filename = output_filename
    if auto_suffix:
        # If file exists, increase suffix.
        counter = 0
        while filename.exists():
            counter += 1
            filename = filename.with_name(f"{filename.stem}_{counter}{filename.suffix}")

    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f_out:
        json.dump(dict_tvd_batch_method, f_out, indent=4)

    print(f"Saved benchmark results to: {filename}")
    return filename


def plot_result(
    results_filename: Path,
    savefig_filename: Path,
) -> None:
    # Load results
    with open(results_filename, "r") as f_in:
        raw_dict: dict[str, dict[str, float]] = json.load(f_in)
    dict_tvd_batch_method = {int(k): v for k, v in raw_dict.items()}

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

    savefig_filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savefig_filename)
    print(f"Saved figure to: {savefig_filename}")


def parse_batch_sizes(value: str) -> list[int]:
    """Parse comma-separated batch sizes such as "1,2,3"."""
    try:
        parsed = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid batch size list: {value}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("Batch size list cannot be empty")
    if any(v <= 0 for v in parsed):
        raise argparse.ArgumentTypeError("Batch sizes must be positive integers")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark and plot measurement-encoding results."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # "run" command for running benchmark and saving JSON results
    run_parser = subparsers.add_parser("run", help="Run benchmark and save JSON")
    run_parser.add_argument("--num-qubits", type=int, default=12)
    run_parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=[1, 2, 3],
        help='Comma-separated batch sizes, e.g. "1,2,3"',
    )
    run_parser.add_argument(
        "--delay-time",
        type=float,
        default=100e-9,
        help="Delay time before measurement in seconds",
    )
    run_parser.add_argument("--num-shots", type=int, default=10**5)
    run_parser.add_argument("--prob-meas1-prep0", type=float, default=0.001)
    run_parser.add_argument("--prob-meas0-prep1", type=float, default=0.002)
    run_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path",
    )
    run_parser.add_argument(
        "--no-auto-suffix",
        action="store_true",
        help="Overwrite output path instead of auto-incrementing suffix",
    )

    # "plot" command for plotting figure from JSON results
    plot_parser = subparsers.add_parser("plot", help="Plot benchmark JSON as figure")
    plot_parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Input JSON results file path",
    )
    plot_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output figure file path",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    match args.command:
        case "run":
            run_benchmark_suite(
                num_qubits=args.num_qubits,
                batch_size_list=args.batch_sizes,
                delay_time=args.delay_time,
                num_shots=args.num_shots,
                prob_meas1_prep0=args.prob_meas1_prep0,
                prob_meas0_prep1=args.prob_meas0_prep1,
                output_filename=args.output,
                auto_suffix=not args.no_auto_suffix,
            )
        case "plot":
            plot_result(args.results, args.output)
        case cmd:
            raise ValueError(f"Unknown command: {cmd}")


if __name__ == "__main__":
    print(f"Current working directory: {ROOT}")
    main()
