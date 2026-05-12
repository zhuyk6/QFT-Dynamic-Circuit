r"""Benchmark process fidelity for dynamic QFT.

Process fidelity is defined as:
    F = [ (1 / 2^n) * sum_{k=0}^{2^n-1} sqrt( Pr(k | QFT_tilde(sigma_k_star)) ) ]^2

where
    sigma_k_star = ( \otimes_{l=0}^{n-1} Rz(-pi * k / 2^l, l) ) H^{\otimes n} |0>

For large n, a sampled estimator is used:
    F ~= (m/(m-1)) * [ (1/m) * sum_l sqrt(p_l) ]^2 - (1/(m*(m-1))) * sum_l p_l
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Literal
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import Sampler

from qft_dynamic.tools.config import resolve_shor_benchmark_paths
from qft_dynamic.tools.simulation import (
    NoiseModelConfig,
    build_qft_simulation_context,
    build_sampler,
    compose_with_layout,
    sample_counts,
)


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


def prepare_sigma_k_star(num_qubits: int, k: int) -> QuantumCircuit:
    """Prepare sigma_k_star = (prod_l Rz(-pi*k/2^l, l)) H^n |0>."""
    prep = QuantumCircuit(num_qubits)
    prep.h(range(num_qubits))
    for i in range(num_qubits):
        theta = -math.pi * k / (2**i)
        prep.rz(theta, i)
    prep.barrier()
    return prep


def probability_of_k(
    num_qubits: int,
    k: int,
    transpiled_qft: QuantumCircuit,
    sampler: Sampler,
    num_shots: int,
) -> float:
    """Estimate Pr(k | QFT_tilde(sigma_k_star)) by sampling."""
    prep = prepare_sigma_k_star(num_qubits, k)
    total_circuit = compose_with_layout(transpiled_qft, prep)

    counts: Counter[int] = sample_counts(
        circuit=total_circuit,
        sampler=sampler,
        num_shots=num_shots,
    )
    p_k = counts[k] / num_shots
    return p_k


def process_fidelity_exact(
    num_qubits: int,
    transpiled_qft: QuantumCircuit,
    sampler: Sampler,
    num_shots: int,
) -> float:
    """Exact fidelity by enumerating all k.

    The formula is
        F = [(1/N) * sum_{k=0}^{N-1} sqrt(Pr(k | QFT_tilde(sigma_k_star)))]^2
    """
    n_states: int = 2**num_qubits
    s = 0.0
    for k in range(n_states):
        p_k = probability_of_k(
            num_qubits=num_qubits,
            k=k,
            transpiled_qft=transpiled_qft,
            sampler=sampler,
            num_shots=num_shots,
        )
        s += math.sqrt(p_k)

    fidelity = (s / n_states) ** 2
    return fidelity


def process_fidelity_sampled(
    num_qubits: int,
    transpiled_qft: QuantumCircuit,
    sampler: Sampler,
    num_shots: int,
    num_samples: int,
    seed: int | None = None,
) -> float:
    """Sampled unbiased estimator of process fidelity.

    The formula is
        F = [(m/(m-1)) * (mean sqrt(p_k))^2] - [sum(p_k) / (m*(m-1))]
    """
    if num_samples < 2:
        raise ValueError("num_samples must be >= 2 for sampled estimator")

    if num_samples > 2**num_qubits:
        raise ValueError("num_samples must be <= 2**num_qubits for sampled estimator")

    rng = random.Random(seed)
    n_states: int = 2**num_qubits
    sampled_k = rng.sample(range(n_states), num_samples)

    p_values: list[float] = []
    sqrt_p_values: list[float] = []
    for k in sampled_k:
        p_k = probability_of_k(
            num_qubits=num_qubits,
            k=k,
            transpiled_qft=transpiled_qft,
            sampler=sampler,
            num_shots=num_shots,
        )
        p_values.append(p_k)
        sqrt_p_values.append(math.sqrt(p_k))

    m = num_samples
    mean_sqrt_p = sum(sqrt_p_values) / m
    sum_p = sum(p_values)

    fidelity: float = (m / (m - 1.0)) * (mean_sqrt_p**2) - (sum_p / (m * (m - 1.0)))
    return fidelity


def benchmark_process_fidelity(
    num_qubits: int,
    batch_size: int,
    mode: Literal["exact", "sample"],
    num_shots: int = 10**4,
    num_samples: int = 20,
    seed: int | None = None,
    noise_config: NoiseModelConfig | None = None,
) -> float:
    """Run process-fidelity benchmark for one batch size."""
    assert num_qubits % batch_size == 0, "num_qubits must be multiple of batch_size"

    resolved_paths = resolve_shor_benchmark_paths()

    context = build_qft_simulation_context(
        num_qubits=num_qubits,
        batch_size=batch_size,
        hardware_config_path=resolved_paths.hardware_config_path,
        opt_circuits_path=resolved_paths.opt_circuits_path,
    )
    sampler = build_sampler(
        backend=context.backend,
        noise_config=noise_config,
    )

    match mode:
        case "exact":
            return process_fidelity_exact(
                num_qubits=num_qubits,
                transpiled_qft=context.transpiled_qft,
                sampler=sampler,
                num_shots=num_shots,
            )
        case "sample":
            return process_fidelity_sampled(
                num_qubits=num_qubits,
                transpiled_qft=context.transpiled_qft,
                sampler=sampler,
                num_shots=num_shots,
                num_samples=num_samples,
                seed=seed,
            )
        case other:
            raise ValueError(f"Unknown mode: {other}")


def run_benchmark_suite(
    num_qubits: int,
    batch_size_list: list[int],
    mode: Literal["exact", "sample"],
    num_shots: int,
    num_samples: int,
    seed: int | None,
    output_filename: Path,
    auto_suffix: bool = True,
    noise_config: NoiseModelConfig | None = None,
) -> Path:
    """Run fidelity benchmark for all batch sizes and save JSON."""
    results: dict[int, float] = {}

    for batch_size in batch_size_list:
        fidelity = benchmark_process_fidelity(
            num_qubits=num_qubits,
            batch_size=batch_size,
            mode=mode,
            num_shots=num_shots,
            num_samples=num_samples,
            seed=seed,
            noise_config=noise_config,
        )
        results[batch_size] = fidelity
        print(f"Batch Size {batch_size}: process fidelity = {fidelity:.6f}")

    # If `output_filename` already exists, append a suffix to avoid overwriting
    filename = output_filename
    if auto_suffix:
        counter = 0
        while filename.exists():
            counter += 1
            filename = filename.with_name(
                f"{output_filename.stem}_{counter}{output_filename.suffix}"
            )

    if noise_config is None:
        noise_config = NoiseModelConfig()

    gate_error, readout_error, thermal_relaxation = noise_config

    payload = {
        "num_qubits": num_qubits,
        "mode": mode,
        "num_shots": num_shots,
        "num_samples": num_samples,
        "seed": seed,
        "noise": {
            "gate_error": gate_error,
            "readout_error": readout_error,
            "thermal_relaxation": thermal_relaxation,
        },
        "fidelity_by_batch_size": results,
    }

    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f_out:
        json.dump(payload, f_out, indent=4)

    print(f"Saved benchmark results to: {filename}")
    return filename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark process fidelity for dynamic QFT."
    )
    parser.add_argument("--num-qubits", type=int, default=12)
    parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=[1, 2, 3],
        help='Comma-separated batch sizes, e.g. "1,2,3"',
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["exact", "sample"],
        default="sample",
        help="exact: enumerate all k; sample: Monte Carlo estimator",
    )
    parser.add_argument("--num-shots", type=int, default=10**4)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--gate-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--readout-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--thermal-relaxation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser = build_parser()
    args = parser.parse_args()
    run_benchmark_suite(
        num_qubits=args.num_qubits,
        batch_size_list=args.batch_sizes,
        mode=args.mode,
        num_shots=args.num_shots,
        num_samples=args.num_samples,
        seed=args.seed,
        output_filename=args.output,
        auto_suffix=not args.no_auto_suffix,
        noise_config=NoiseModelConfig(
            gate_error=args.gate_error,
            readout_error=args.readout_error,
            thermal_relaxation=args.thermal_relaxation,
        ),
    )


if __name__ == "__main__":
    main()
