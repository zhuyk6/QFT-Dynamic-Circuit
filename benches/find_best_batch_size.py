"""Benchmark utilities for evaluating different batch sizes.

This script evaluates batch sizes using:
- the minimum running time
- the minimum TVD (or maximum fidelity)
"""

import argparse
import pickle
import warnings
from collections import Counter
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from pprint import pprint
from typing import Callable

import numpy as np
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


def parse_batch_sizes(value: str) -> list[int]:
    """Parse comma-separated batch sizes such as ``"1,2,3"``."""

    try:
        parsed: list[int] = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid batch size list: {value}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("Batch size list cannot be empty")
    if any(v <= 0 for v in parsed):
        raise argparse.ArgumentTypeError("Batch sizes must be positive integers")
    return parsed


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

    for gate_error, readout_error, thermal_relaxation_error in tqdm(sampler_tag_list):
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


def setup_warnings():
    warnings.filterwarnings("ignore", module="qiskit")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for batch-size benchmarks."""

    parser = argparse.ArgumentParser(
        description="Benchmark different batch sizes for dynamic-QFT workloads."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    circular_parser = subparsers.add_parser(
        "run-circular",
        help="Run circular-state TVD benchmark for all noise combinations",
    )
    circular_parser.add_argument("--num-qubits", type=int, default=12)
    circular_parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=[1, 2, 3],
        help='Comma-separated batch sizes, e.g. "1,2,3"',
    )
    circular_parser.add_argument("--num-shots", type=int, default=10**5)
    circular_parser.add_argument("--output", type=Path, required=True)
    circular_parser.add_argument(
        "--no-auto-suffix",
        action="store_true",
        help="Overwrite output path instead of auto-incrementing suffix",
    )

    ghz_parser = subparsers.add_parser(
        "run-ghz",
        help="Run GHZ-state TVD benchmark for all noise combinations",
    )
    ghz_parser.add_argument("--num-qubits", type=int, default=12)
    ghz_parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=[1, 2, 3],
        help='Comma-separated batch sizes, e.g. "1,2,3"',
    )
    ghz_parser.add_argument("--num-shots", type=int, default=10**5)
    ghz_parser.add_argument("--output", type=Path, required=True)
    ghz_parser.add_argument(
        "--no-auto-suffix",
        action="store_true",
        help="Overwrite output path instead of auto-incrementing suffix",
    )

    runtime_parser = subparsers.add_parser(
        "run-time",
        help="Print estimated runtime for each batch size",
    )
    runtime_parser.add_argument("--num-qubits", type=int, default=12)
    runtime_parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=[1, 2, 3],
        help='Comma-separated batch sizes, e.g. "1,2,3"',
    )

    tvd_parser = subparsers.add_parser(
        "run-tvd-vs-num-qubits",
        help="Run circular-state TVD benchmark versus the number of qubits",
    )
    tvd_parser.add_argument("--min-num-qubits", type=int, default=2)
    tvd_parser.add_argument("--max-num-qubits", type=int, default=12)
    tvd_parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=[1, 2, 3],
        help='Comma-separated batch sizes, e.g. "1,2,3"',
    )
    tvd_parser.add_argument("--num-shots", type=int, default=10**5)
    tvd_parser.add_argument("--output", type=Path, required=True)
    tvd_parser.add_argument(
        "--no-auto-suffix",
        action="store_true",
        help="Overwrite output path instead of auto-incrementing suffix",
    )

    return parser


def main() -> None:
    """CLI entry point for batch-size benchmarks."""
    setup_warnings()

    parser = build_parser()
    args = parser.parse_args()

    match args.command:
        case "run-circular":
            result = run_circular_state(
                num_qubits=args.num_qubits,
                batch_size_list=args.batch_sizes,
                sampler_tag_list=_all_sampler_tags(),
                num_shots=args.num_shots,
            )
            pprint(result)
            _save_pickle_with_optional_suffix(
                payload=result,
                output_filename=args.output,
                auto_suffix=not args.no_auto_suffix,
            )
        case "run-ghz":
            result = run_ghz_state(
                num_qubits=args.num_qubits,
                batch_size_list=args.batch_sizes,
                sampler_tag_list=_all_sampler_tags(),
                num_shots=args.num_shots,
            )
            pprint(result)
            _save_pickle_with_optional_suffix(
                payload=result,
                output_filename=args.output,
                auto_suffix=not args.no_auto_suffix,
            )
        case "run-time":
            runtime_dict: dict[int, float] = {}
            batch_size: int
            for batch_size in args.batch_sizes:
                runtime_dict[batch_size] = calculate_runtime(
                    batch_size=batch_size,
                    num_qubits=args.num_qubits,
                )
            print("Runtime for different batch sizes:")
            pprint(runtime_dict)
        case "run-tvd-vs-num-qubits":
            min_num_qubits: int = args.min_num_qubits
            max_num_qubits: int = args.max_num_qubits
            batch_size_list: list[int] = args.batch_sizes

            dict_batch_num_tvd: dict[int, dict[int, float]] = {}
            for num_qubits in tqdm(range(min_num_qubits, max_num_qubits + 1)):
                for batch_size in batch_size_list:
                    if num_qubits % batch_size != 0:
                        continue

                    tvd: float = calculate_metric(
                        batch_size=batch_size,
                        num_qubits=num_qubits,
                        sampler_tag=(True, True, True),
                        prepare_circ_fn=lambda n: prepare_circular_state_circuit(
                            n, r=4
                        ),
                        metric_fn=lambda counts: calc_tvd(
                            {k << (num_qubits - 2): 1 / 4 for k in range(4)},
                            counts,
                        ),
                        num_shots=args.num_shots,
                    )

                    if batch_size not in dict_batch_num_tvd:
                        dict_batch_num_tvd[batch_size] = {}
                    dict_batch_num_tvd[batch_size][num_qubits] = tvd

            pprint(dict_batch_num_tvd)
            _save_pickle_with_optional_suffix(
                payload=dict_batch_num_tvd,
                output_filename=args.output,
                auto_suffix=not args.no_auto_suffix,
            )
        case command:
            raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
