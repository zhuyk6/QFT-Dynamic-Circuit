"""Qiskit-based simulation utilities for the Shor strict benchmark."""

from collections import Counter
from math import pi
from pathlib import Path

from qiskit import QuantumCircuit, qpy
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import Sampler

from qft_dynamic.tools.build_backend import build_backend, load_hardware_config
from qft_dynamic.tools.build_circuits import tile_transpiled_circuit
from qft_dynamic.tools.transpile import generate_pass_manager

from .config import ShorBenchmarkPaths, resolve_shor_benchmark_paths
from .schemas import HistogramFileModel, SimulationMetadataModel
from .types import BenchmarkInstance


def build_qft_circuit(
    num_qubits: int,
    batch_size: int,
    t_feed_forward: float,
    opt_circuits_path: Path,
) -> QuantumCircuit:
    """Build the tiled forward-QFT circuit used in simulation.

    Args:
        num_qubits: Total number of qubits.
        batch_size: Tile size of the optimized QFT block.
        t_feed_forward: Feed-forward duration in seconds.
        opt_circuits_path: Directory containing optimized QFT tiles.

    Returns:
        Tiled forward-QFT circuit with measurements.
    """

    if num_qubits % batch_size != 0:
        raise ValueError("num_qubits must be a multiple of batch_size")

    filename: Path = opt_circuits_path / f"qft{batch_size}.qpy"
    with filename.open("rb") as input_file:
        optimized_circuit: QuantumCircuit = qpy.load(input_file)[0]

    num_tiles: int = num_qubits // batch_size
    tiling_pattern: list[list[int]] = [
        [index + tile_index * batch_size for index in range(batch_size)]
        for tile_index in range(num_tiles)
    ]
    large_circuit: QuantumCircuit = tile_transpiled_circuit(
        optimized_circuit.copy(),
        tiling_pattern,
        t_feed_forward,
    )
    return large_circuit


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


def compose_with_layout(
    transpiled_qft: QuantumCircuit,
    prepare_circuit: QuantumCircuit,
) -> QuantumCircuit:
    """Compose state preparation in front of a transpiled QFT circuit.

    Args:
        transpiled_qft: Transpiled forward-QFT circuit.
        prepare_circuit: Logical state-preparation circuit.

    Returns:
        Full circuit with state preparation composed before QFT.
    """

    if transpiled_qft.layout is None:
        raise ValueError("transpiled circuit must carry layout information")

    logical_to_physical: dict[int, int] = {
        virtual_qubit._index: physical_qubit
        for physical_qubit, virtual_qubit in transpiled_qft.layout.initial_layout.get_physical_bits().items()
    }
    mapped_qubits: list[int] = [
        logical_to_physical[index] for index in range(transpiled_qft.num_qubits)
    ]
    total_circuit = transpiled_qft.compose(
        prepare_circuit,
        qubits=mapped_qubits,
        front=True,
        inplace=False,
    )
    assert total_circuit is not None
    return total_circuit


def _decode_bitstring_to_y(bitstring: str) -> int:
    """Decode a measured bitstring into the benchmark integer y.

    Args:
        bitstring: Measured computational basis string.

    Returns:
        Decoded integer y.
    """

    # Bit-order convention used throughout this repository:
    #
    # - logical input wires are ordered as q0, q1, ..., q_{m-1}
    # - q0 is the most significant logical bit at the QFT input
    # - the forward-QFT circuit omits terminal SWAPs, so after measurement q0
    #   becomes the least significant output bit
    #
    # The circuit measures q_i into c_i. Qiskit count strings are displayed as
    # c_{m-1} ... c_1 c_0, i.e. from the most significant classical bit to the
    # least significant classical bit. Under the swapless-QFT convention above,
    # this display order already matches the benchmark integer y directly, so
    # we must not reverse the bitstring again here.
    return int(bitstring, base=2)


def simulate_histograms_for_instance(
    instance: BenchmarkInstance,
    batch_size: int,
    num_shots: int,
    gate_error: bool = True,
    readout_error: bool = True,
    thermal_relaxation: bool = True,
    resource_paths: ShorBenchmarkPaths | None = None,
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

    resolved_paths: ShorBenchmarkPaths = (
        resource_paths or resolve_shor_benchmark_paths()
    )
    hardware_config = load_hardware_config(resolved_paths.hardware_config_path)
    coupling_map: CouplingMap = CouplingMap.from_line(instance.m)
    backend = build_backend(coupling_map, hardware_config)

    qft_circuit: QuantumCircuit = build_qft_circuit(
        num_qubits=instance.m,
        batch_size=batch_size,
        t_feed_forward=hardware_config["t_feed_forward"],
        opt_circuits_path=resolved_paths.opt_circuits_path,
    )
    transpiled_qft: QuantumCircuit = generate_pass_manager(backend).run(qft_circuit)

    noise_model: NoiseModel = NoiseModel.from_backend(
        backend=backend,
        gate_error=gate_error,
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
    )
    sampler: Sampler = Sampler(mode=AerSimulator(noise_model=noise_model))

    histograms: dict[int, Counter[int]] = {}
    s_value: int
    for s_value in range(instance.r):
        prepare_circuit: QuantumCircuit = prepare_forward_qft_phase_state(
            instance=instance,
            s=s_value,
        )
        total_circuit: QuantumCircuit = compose_with_layout(
            transpiled_qft=transpiled_qft,
            prepare_circuit=prepare_circuit,
        )
        result = sampler.run([total_circuit], shots=num_shots).result()
        counts: dict[str, int] = result[0].data["c"].get_counts()

        decoded_histogram: Counter[int] = Counter()
        bitstring: str
        count: int
        for bitstring, count in counts.items():
            y_value: int = _decode_bitstring_to_y(bitstring=bitstring)
            decoded_histogram[y_value] += count

        histograms[s_value] = decoded_histogram

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
            s_value: {
                y_value: int(count) for y_value, count in sorted(histogram.items())
            }
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
