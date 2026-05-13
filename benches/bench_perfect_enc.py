"""This script benchmarks the effect of measurement encoding.

Here we compare two methods:
- Original circuit: consider three kinds of noise.
- Delayed circuit: using a delay for measurement encoding, consider a modified readout error.
"""

import argparse
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from pprint import pprint

from qiskit import QuantumCircuit
from tqdm import tqdm

from qft_dynamic.tools.build_backend import load_hardware_config
from qft_dynamic.tools.build_circuits import prepare_circular_state_circuit
from qft_dynamic.tools.config import resolve_shor_benchmark_paths
from qft_dynamic.tools.data_process import calc_tvd
from qft_dynamic.tools.simulation import (
    NoiseModelConfig,
    _build_line_backend,
    _transpile_for_simulation,
    build_sampler,
    build_tiled_qft_circuit,
    compose_with_layout,
    sample_counts,
)
from qft_dynamic.tools.transpile import (
    add_delay_before_measurement,
)


def setup_warnings():
    warnings.filterwarnings("ignore", module="qiskit")


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
    resolved_paths = resolve_shor_benchmark_paths()

    hardware_config = load_hardware_config(resolved_paths.hardware_config_path)
    if delay_time is not None:
        if prob_meas1_prep0 is not None and prob_meas0_prep1 is not None:
            hardware_config["prob_meas0_prep1"] = prob_meas0_prep1
            hardware_config["prob_meas1_prep0"] = prob_meas1_prep0
        else:
            hardware_config["prob_meas0_prep1"] = 0.0
            hardware_config["prob_meas1_prep0"] = 0.0
    backend = _build_line_backend(
        num_qubits=num_qubits,
        hardware_config=hardware_config,
    )

    circuit: QuantumCircuit = build_tiled_qft_circuit(
        num_qubits=num_qubits,
        batch_size=batch_size,
        t_feed_forward=hardware_config["t_feed_forward"],
        opt_circuits_path=resolved_paths.opt_circuits_path,
    )

    # add delay before measurement
    if delay_time is not None:
        circuit = add_delay_before_measurement(circuit, delay_time)

    transpiled_circuit: QuantumCircuit = _transpile_for_simulation(
        circuit=circuit,
        backend=backend,
    )
    total_circuit: QuantumCircuit = compose_with_layout(
        transpiled_circuit=transpiled_circuit,
        prepare_circuit=prepare_circular_state_circuit(num_qubits, r=4),
    )

    sampler = build_sampler(
        backend=backend,
        noise_config=NoiseModelConfig(
            gate_error=True,
            readout_error=True,
            thermal_relaxation=True,
        ),
    )

    # define metrics
    def metric_tvd(noisy_counts: dict[int, int]) -> float:
        ideal_prob = {k << (num_qubits - 2): 1 / 4 for k in range(4)}
        return calc_tvd(ideal_prob, noisy_counts)

    # Run sampler and calculate metrics
    def simulate_and_metric(total_circuit: QuantumCircuit) -> float:
        noisy_counts: Counter[int] = sample_counts(
            circuit=total_circuit,
            sampler=sampler,
            num_shots=num_shots,
        )
        tvd = metric_tvd(noisy_counts)
        return tvd

    tvd = simulate_and_metric(total_circuit)

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
    dict_tvd_batch_method: defaultdict[int, dict[str, float]] = defaultdict(dict)

    for batch_size in tqdm(batch_size_list):
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
        description="Benchmark measurement-encoding results."
    )
    parser.add_argument("--num-qubits", type=int, default=12)
    parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=[1, 2, 3],
        help='Comma-separated batch sizes, e.g. "1,2,3"',
    )
    parser.add_argument(
        "--delay-time",
        type=float,
        default=100e-9,
        help="Delay time before measurement in seconds",
    )
    parser.add_argument("--num-shots", type=int, default=10**5)
    parser.add_argument("--prob-meas1-prep0", type=float, default=0.001)
    parser.add_argument("--prob-meas0-prep1", type=float, default=0.002)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--no-auto-suffix",
        action="store_true",
        help="Overwrite output path instead of auto-incrementing suffix",
    )
    return parser


def main() -> None:
    setup_warnings()

    parser = build_parser()
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
