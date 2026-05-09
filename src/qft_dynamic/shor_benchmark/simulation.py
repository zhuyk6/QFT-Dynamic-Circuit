"""Qiskit-based simulation utilities for the Shor strict benchmark."""

from collections import Counter
from math import pi
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import Sampler

from qft_dynamic.tools.config import BenchmarkPaths, resolve_shor_benchmark_paths
from qft_dynamic.tools.simulation import (
    NoiseModelConfig,
    build_qft_simulation_context,
    build_sampler,
    compose_with_layout,
    sample_counts,
)

from .schemas import HistogramFileModel, SimulationMetadataModel
from .types import BenchmarkInstance


def prepare_forward_qft_phase_state(
    instance: BenchmarkInstance,
    s: int,
) -> QuantumCircuit:
    """Prepare the phase state for a forward-QFT benchmark circuit.

    The benchmark document is written in inverse-QFT convention. Because the
    available circuit here implements forward QFT, we prepare the complex
    conjugate phase state, which flips the phase sign.

    Args:
        instance: Benchmark instance.
        s: Phase label in [0, r - 1].

    Returns:
        State-preparation circuit on ``instance.m`` qubits.
    """

    if not (0 <= s < instance.r):
        raise ValueError("s must satisfy 0 <= s < r")

    preparation_circuit: QuantumCircuit = QuantumCircuit(instance.m)
    qubit_index: int
    for qubit_index in range(instance.m):
        # The forward-QFT implementation omits the terminal SWAP, so logical
        # significance is reversed across the wire order. Prepare the conjugate
        # phase state in that reversed significance convention.
        phase_weight: int = 2 ** (instance.m - 1 - qubit_index)
        phase_angle: float = (-2.0 * pi * s * phase_weight) / (instance.r)
        preparation_circuit.h(qubit_index)
        preparation_circuit.rz(phase_angle, qubit_index)
    preparation_circuit.barrier()
    return preparation_circuit


def simulate_histograms_for_instance(
    instance: BenchmarkInstance,
    batch_size: int,
    num_shots: int,
    gate_error: bool = True,
    readout_error: bool = True,
    thermal_relaxation: bool = True,
    resource_paths: BenchmarkPaths | None = None,
) -> dict[int, Counter[int]]:
    """Simulate one histogram per phase label s for a benchmark instance.

    Args:
        instance: Benchmark instance.
        batch_size: Tile size of the optimized QFT block.
        num_shots: Number of shots per phase label.
        gate_error: Whether to include gate error in the noise model.
        readout_error: Whether to include readout error in the noise model.
        thermal_relaxation: Whether to include thermal relaxation in the noise model.
        resource_paths: Optional resolved resource paths.

    Returns:
        Mapping from phase label s to histogram over decoded integers y.
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

    histograms: dict[int, Counter[int]] = {}
    s_value: int
    for s_value in range(instance.r):
        prepare_circuit: QuantumCircuit = prepare_forward_qft_phase_state(
            instance=instance,
            s=s_value,
        )
        total_circuit: QuantumCircuit = compose_with_layout(
            transpiled_circuit=context.transpiled_qft,
            prepare_circuit=prepare_circuit,
        )
        counts = sample_counts(
            circuit=total_circuit,
            sampler=sampler,
            num_shots=num_shots,
        )

        histograms[s_value] = counts

    return histograms


def save_histograms(
    instance: BenchmarkInstance,
    histograms: dict[int, Counter[int]],
    output_path: Path,
    batch_size: int,
    num_shots: int,
    gate_error: bool,
    readout_error: bool,
    thermal_relaxation: bool,
) -> None:
    """Serialize simulated histograms to JSON on disk.

    Args:
        instance: Benchmark instance.
        histograms: Histogram data indexed by phase label s.
        output_path: Output JSON path.
        batch_size: Tile size used during simulation.
        num_shots: Number of shots per phase label.
        gate_error: Whether gate error was enabled.
        readout_error: Whether readout error was enabled.
        thermal_relaxation: Whether thermal relaxation was enabled.
    """

    histogram_file: HistogramFileModel = HistogramFileModel(
        instance=instance,
        simulation=SimulationMetadataModel(
            batch_size=batch_size,
            num_shots=num_shots,
            gate_error=gate_error,
            readout_error=readout_error,
            thermal_relaxation=thermal_relaxation,
        ),
        histograms={
            s_value: dict(histogram)
            for s_value, histogram in sorted(histograms.items())
        },
    )
    histogram_file.save(output_path)


def load_histogram_file(input_path: Path) -> HistogramFileModel:
    """Load and validate a histogram JSON file.

    Args:
        input_path: Histogram JSON path.

    Returns:
        Validated histogram file model.
    """

    histogram_file: HistogramFileModel = HistogramFileModel.load(input_path)
    return histogram_file
