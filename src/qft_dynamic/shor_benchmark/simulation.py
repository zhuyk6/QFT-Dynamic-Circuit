"""Qiskit simulation workflow for Shor logical measurement datasets."""

from math import pi

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import Sampler

from qft_dynamic.experiment_data import (
    ExperimentDataset,
    LogicalCountsGroup,
    LogicalCountsRun,
)
from qft_dynamic.tools.config import BenchmarkPaths, resolve_shor_benchmark_paths
from qft_dynamic.tools.simulation import (
    NoiseModelConfig,
    build_qft_simulation_context,
    build_sampler,
    compose_with_layout,
    sample_counts,
)

from .types import BenchmarkInstance


def prepare_forward_qft_phase_state(
    instance: BenchmarkInstance,
    s: int,
) -> QuantumCircuit:
    """Prepare the phase state for a forward-QFT benchmark circuit.

    The benchmark document is written in inverse-QFT convention. Because the
    available circuit here implements forward QFT, the prepared state uses the
    conjugate phase and the swapless circuit's reversed wire significance.

    Args:
        instance: Benchmark instance.
        s: Phase label in ``[0, r - 1]``.

    Returns:
        State-preparation circuit on ``instance.m`` qubits.

    Raises:
        ValueError: If ``s`` is outside the instance phase-label range.
    """

    if not (0 <= s < instance.r):
        raise ValueError("s must satisfy 0 <= s < r")

    preparation_circuit: QuantumCircuit = QuantumCircuit(instance.m)
    for qubit_index in range(instance.m):
        phase_weight: int = 2 ** (instance.m - 1 - qubit_index)
        phase_angle: float = (-2.0 * pi * s * phase_weight) / instance.r
        preparation_circuit.h(qubit_index)
        preparation_circuit.rz(phase_angle, qubit_index)
    preparation_circuit.barrier()
    return preparation_circuit


def simulate_dataset_for_instance(
    instance: BenchmarkInstance,
    batch_size: int,
    num_shots: int,
    gate_error: bool = True,
    readout_error: bool = True,
    thermal_relaxation: bool = True,
    resource_paths: BenchmarkPaths | None = None,
) -> ExperimentDataset:
    """Simulate one logical-count run per phase label.

    Args:
        instance: Benchmark instance.
        batch_size: Tile size of the optimized QFT block.
        num_shots: Number of shots per phase label.
        gate_error: Whether to include gate error in the noise model.
        readout_error: Whether to include readout error in the noise model.
        thermal_relaxation: Whether to include thermal relaxation in the noise model.
        resource_paths: Optional resolved resource paths.

    Returns:
        Source-independent logical measurement dataset. Each run stores its
        phase label in integer metadata under ``s``.

    Raises:
        ValueError: If ``num_shots`` is not positive.
    """

    if num_shots <= 0:
        raise ValueError("num_shots must be positive")

    resolved_paths: BenchmarkPaths = resource_paths or resolve_shor_benchmark_paths()
    context = build_qft_simulation_context(
        num_qubits=instance.m,
        batch_size=batch_size,
        hardware_config_path=resolved_paths.hardware_config_path,
        opt_circuits_path=resolved_paths.opt_circuits_path,
    )
    sampler: Sampler = build_sampler(
        backend=context.backend,
        noise_config=NoiseModelConfig(
            gate_error=gate_error,
            readout_error=readout_error,
            thermal_relaxation=thermal_relaxation,
        ),
    )

    runs: list[LogicalCountsRun] = []
    for s_value in range(instance.r):
        prepare_circuit: QuantumCircuit = prepare_forward_qft_phase_state(
            instance=instance,
            s=s_value,
        )
        total_circuit: QuantumCircuit = compose_with_layout(
            transpiled_circuit=context.transpiled_qft,
            prepare_circuit=prepare_circuit,
        )
        runs.append(
            LogicalCountsRun(
                run_id=f"s-{s_value}",
                metadata={"s": s_value},
                counts=sample_counts(
                    circuit=total_circuit,
                    sampler=sampler,
                    num_shots=num_shots,
                ),
            )
        )

    return ExperimentDataset(
        dataset_id=(
            f"shor-n{instance.n}-a{instance.a}-r{instance.r}-m{instance.m}"
            f"-batch{batch_size}"
        ),
        experiment_type="shor_order_finding",
        num_qubits=instance.m,
        bit_order="msb_first",
        producer="qiskit_aer",
        attributes={
            "n": instance.n,
            "a": instance.a,
            "r": instance.r,
            "m": instance.m,
        },
        groups=[
            LogicalCountsGroup(
                name=f"batch-{batch_size}",
                attributes={
                    "batch_size": batch_size,
                    "gate_error": gate_error,
                    "readout_error": readout_error,
                    "thermal_relaxation": thermal_relaxation,
                },
                runs=runs,
            )
        ],
    )
