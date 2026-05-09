"""Shared Qiskit simulation utilities for dynamic-QFT workflows."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from qiskit import QuantumCircuit, qpy
from qiskit.primitives import BitArray, DataBin, PrimitiveResult, SamplerPubResult
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import Sampler

from .build_backend import HardwareConfig, build_backend, load_hardware_config
from .build_circuits import tile_transpiled_circuit
from .transpile import generate_pass_manager, unroll_if_true


class NoiseModelConfig(NamedTuple):
    """Configuration flags for the Aer noise model.

    Args:
        gate_error: Whether to include gate error.
        readout_error: Whether to include readout error.
        thermal_relaxation: Whether to include thermal relaxation.
    """

    gate_error: bool = True
    readout_error: bool = True
    thermal_relaxation: bool = True


@dataclass(frozen=True)
class QftSimulationContext:
    """Reusable simulation objects for one tiled QFT instance.

    Args:
        hardware_config: Loaded hardware configuration.
        backend: Backend built for a line coupling map of the requested size.
        qft_circuit: Untanspiled tiled QFT circuit.
        transpiled_qft: Transpiled tiled QFT circuit.
    """

    hardware_config: HardwareConfig
    backend: GenericBackendV2
    qft_circuit: QuantumCircuit
    transpiled_qft: QuantumCircuit


def build_tiled_qft_circuit(
    num_qubits: int,
    batch_size: int,
    t_feed_forward: float,
    opt_circuits_path: Path,
) -> QuantumCircuit:
    """Build a tiled dynamic-QFT circuit from an optimized QPY tile.

    NOTE:
        This tiling pattern is trivial linearly. If the sub-QFT has 3 qubits
        and the total circuit has 6 qubits, the tiling pattern is [[0, 1, 2], [3, 4, 5]].

    Args:
        num_qubits: Total number of qubits in the large circuit.
        batch_size: Tile size of the optimized QFT block.
        t_feed_forward: Feed-forward duration in seconds.
        opt_circuits_path: Directory containing optimized QFT ``.qpy`` files.

    Returns:
        QuantumCircuit: The tiled dynamic-QFT circuit.

    Raises:
        ValueError: If ``num_qubits`` is not a multiple of ``batch_size``.
    """

    if num_qubits % batch_size != 0:
        raise ValueError("num_qubits must be a multiple of batch_size")

    # Load optimized circuit
    filename: Path = opt_circuits_path / f"qft{batch_size}.qpy"
    with filename.open("rb") as input_file:
        optimized_circuit: QuantumCircuit = qpy.load(input_file)[0]

    # Build tiling pattern: linear tiling
    num_tiles: int = num_qubits // batch_size
    tiling_pattern: list[list[int]] = [
        [index + tile_index * batch_size for index in range(batch_size)]
        for tile_index in range(num_tiles)
    ]

    tiled_circuit: QuantumCircuit = tile_transpiled_circuit(
        optimized_circuit.copy(),
        tiling_pattern,
        t_feed_forward,
    )
    return tiled_circuit


def build_line_backend_from_paths(
    num_qubits: int,
    hardware_config_path: Path,
) -> tuple[HardwareConfig, GenericBackendV2]:
    """Build a line-topology backend from a hardware configuration file.

    Args:
        num_qubits: Number of qubits on the backend.
        hardware_config_path: Path to the hardware TOML file.

    Returns:
        tuple[HardwareConfig, GenericBackendV2]: Loaded hardware config and backend.
    """

    hardware_config: HardwareConfig = load_hardware_config(hardware_config_path)
    backend: GenericBackendV2 = _build_line_backend(
        num_qubits=num_qubits,
        hardware_config=hardware_config,
    )
    return hardware_config, backend


def _build_line_backend(
    num_qubits: int,
    hardware_config: HardwareConfig,
) -> GenericBackendV2:
    """Build a line-topology backend from an in-memory hardware config.

    Args:
        num_qubits: Number of qubits on the backend.
        hardware_config: Loaded hardware configuration.

    Returns:
        GenericBackendV2: Backend built for a line coupling map.
    """

    coupling_map: CouplingMap = CouplingMap.from_line(num_qubits)
    backend: GenericBackendV2 = build_backend(coupling_map, hardware_config)
    return backend


def _transpile_for_simulation(
    circuit: QuantumCircuit,
    backend: GenericBackendV2,
) -> QuantumCircuit:
    """Transpile a circuit using the shared dynamic-circuit pass manager.

    Args:
        circuit: Circuit to transpile.
        backend: Target backend.

    Returns:
        QuantumCircuit: Transpiled circuit.
    """

    transpiled_circuit: QuantumCircuit = generate_pass_manager(backend).run(circuit)
    return transpiled_circuit


def build_qft_simulation_context(
    num_qubits: int,
    batch_size: int,
    hardware_config_path: Path,
    opt_circuits_path: Path,
) -> QftSimulationContext:
    """Build the shared backend and QFT circuits for one simulation setup.

    Args:
        num_qubits: Total number of qubits in the large circuit.
        batch_size: Tile size of the optimized QFT block.
        hardware_config_path: Path to the hardware TOML file.
        opt_circuits_path: Directory containing optimized QFT ``.qpy`` files.

    Returns:
        QftSimulationContext: Shared simulation objects for this setup.
    """

    hardware_config: HardwareConfig
    backend: GenericBackendV2
    hardware_config, backend = build_line_backend_from_paths(
        num_qubits=num_qubits,
        hardware_config_path=hardware_config_path,
    )
    qft_circuit: QuantumCircuit = build_tiled_qft_circuit(
        num_qubits=num_qubits,
        batch_size=batch_size,
        t_feed_forward=hardware_config["t_feed_forward"],
        opt_circuits_path=opt_circuits_path,
    )
    transpiled_qft: QuantumCircuit = _transpile_for_simulation(
        circuit=qft_circuit,
        backend=backend,
    )
    return QftSimulationContext(
        hardware_config=hardware_config,
        backend=backend,
        qft_circuit=qft_circuit,
        transpiled_qft=transpiled_qft,
    )


def build_sampler(
    backend: GenericBackendV2,
    noise_config: NoiseModelConfig | None = None,
) -> Sampler:
    """Build an Aer-backed sampler for a backend and noise configuration.

    Args:
        backend: Target backend.
        noise_config: Optional noise-model settings. Defaults to all enabled.

    Returns:
        Sampler: Aer-backed sampler.
    """

    resolved_noise_config: NoiseModelConfig = noise_config or NoiseModelConfig()
    noise_model: NoiseModel = NoiseModel.from_backend(
        backend=backend,
        gate_error=resolved_noise_config.gate_error,
        readout_error=resolved_noise_config.readout_error,
        thermal_relaxation=resolved_noise_config.thermal_relaxation,
    )
    sampler: Sampler = Sampler(mode=AerSimulator(noise_model=noise_model))
    return sampler


def compose_with_layout(
    transpiled_circuit: QuantumCircuit,
    prepare_circuit: QuantumCircuit,
) -> QuantumCircuit:
    """Compose a preparation circuit in front of a transpiled circuit.

    Args:
        transpiled_circuit: Transpiled circuit carrying layout information.
        prepare_circuit: Logical state-preparation circuit.

    Returns:
        QuantumCircuit: Combined circuit.
    """

    if transpiled_circuit.layout is None:
        raise ValueError("transpiled circuit must carry layout information")

    logical_to_physical: dict[int, int] = {
        virtual_qubit._index: physical_qubit
        for physical_qubit, virtual_qubit in transpiled_circuit.layout.initial_layout.get_physical_bits().items()
    }
    mapped_qubits: list[int] = [
        logical_to_physical[index] for index in range(transpiled_circuit.num_qubits)
    ]
    total_circuit = transpiled_circuit.compose(
        prepare_circuit,
        qubits=mapped_qubits,
        front=True,
        inplace=False,
    )
    assert total_circuit is not None
    return total_circuit


def sample_counts(
    circuit: QuantumCircuit,
    sampler: Sampler,
    num_shots: int,
    register_name: str = "c",
) -> Counter[int]:
    """Run a sampler and return bitstring counts.

    Args:
        circuit: Circuit to execute.
        sampler: Sampler to use.
        num_shots: Number of shots.
        register_name: Name of the classical register in the result payload.

    Returns:
        dict[str, int]: Bitstring counts.
    """

    result: PrimitiveResult[SamplerPubResult] = sampler.run(
        [circuit], shots=num_shots
    ).result()
    data_bin: DataBin = result[0].data
    bit_array: BitArray = data_bin[register_name]
    counts: dict[int, int] = bit_array.get_int_counts()
    return Counter(counts)


def estimate_tiled_qft_runtime(
    num_qubits: int,
    batch_size: int,
    hardware_config_path: Path,
    opt_circuits_path: Path,
    unit: str = "s",
    unroll_dynamic_circuit: bool = False,
) -> float:
    """Estimate the runtime of a tiled QFT circuit after transpilation.

    Args:
        num_qubits: Total number of qubits in the large circuit.
        batch_size: Tile size of the optimized QFT block.
        hardware_config_path: Path to the hardware TOML file.
        opt_circuits_path: Directory containing optimized QFT ``.qpy`` files.
        unit: Duration unit passed to Qiskit.
        unroll_dynamic_circuit: Whether to unroll ``if`` blocks before transpiling.

    Returns:
        float: Estimated circuit duration.
    """

    hardware_config: HardwareConfig
    backend: GenericBackendV2
    hardware_config, backend = build_line_backend_from_paths(
        num_qubits=num_qubits,
        hardware_config_path=hardware_config_path,
    )
    qft_circuit: QuantumCircuit = build_tiled_qft_circuit(
        num_qubits=num_qubits,
        batch_size=batch_size,
        t_feed_forward=hardware_config["t_feed_forward"],
        opt_circuits_path=opt_circuits_path,
    )
    circuit_to_transpile: QuantumCircuit = qft_circuit
    if unroll_dynamic_circuit:
        circuit_to_transpile = unroll_if_true(qft_circuit)

    transpiled_circuit: QuantumCircuit = _transpile_for_simulation(
        circuit=circuit_to_transpile,
        backend=backend,
    )
    duration: float = transpiled_circuit.estimate_duration(
        target=backend.target,
        unit=unit,
    )
    return duration
