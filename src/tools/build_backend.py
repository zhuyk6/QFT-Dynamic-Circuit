from pathlib import Path
from typing import NotRequired, TypedDict

import tomllib
from qiskit.circuit import Delay, IfElseOp, Parameter
from qiskit.circuit.library import (
    CZGate,
    Measure,
    RZGate,
    SXGate,
    XGate,
)
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import (
    CouplingMap,
    InstructionProperties,
    QubitProperties,
    Target,
)


class HardwareConfig(TypedDict):
    """TypedDict describing quantum hardware timing, frequency and error parameters."""

    dt_in_sec: float
    """Time resolution used by the backend, in seconds."""

    t1: float
    """Energy relaxation time T1 of the qubit, in seconds."""

    t2: float
    """Coherence time T2 (dephasing) of the qubit, in seconds."""

    frequency: float
    """Qubit operating frequency, in hertz."""

    t_single_gate: float
    """Duration of a single-qubit gate, in seconds."""

    t_cz_gate: float
    """Duration of a CZ (controlled-Z) two-qubit gate, in seconds."""

    t_iswap_gate: float
    """Duration of an iSWAP two-qubit gate, in seconds."""

    t_measure: float
    """Duration of a measurement operation, in seconds."""

    t_feed_forward: float
    """Time required for feed-forward operations or classical processing, in seconds."""

    e_single_gate: float
    """Error rate (probability) for single-qubit gates."""

    e_two_gate: float
    """Error rate (probability) for two-qubit gates."""

    e_measure: float
    """Measurement error probability (overall)."""

    prob_meas1_prep0: NotRequired[float]
    """Probability of measuring '1' when the qubit was prepared in the '0' state (false positive/readout bias)."""

    prob_meas0_prep1: NotRequired[float]
    """Probability of measuring '0' when the qubit was prepared in the '1' state (false negative/readout bias)."""


def load_hardware_config(file_path: Path) -> HardwareConfig:
    with open(file_path, "rb") as f:
        config = tomllib.load(f)

    return config  # type: ignore


def _build_target(coupling_map: CouplingMap, hardware_config: HardwareConfig):
    n_qubits = coupling_map.size()

    # load hardware parameters from `hardware_config`
    dt_in_sec = hardware_config["dt_in_sec"]
    t1 = hardware_config["t1"]
    t2 = hardware_config["t2"]
    frequency = hardware_config["frequency"]

    t_single_gate = hardware_config["t_single_gate"]
    t_cz_gate = hardware_config["t_cz_gate"]
    t_measure = hardware_config["t_measure"]

    e_single_gate = hardware_config["e_single_gate"]
    e_two_gate = hardware_config["e_two_gate"]

    e_measure = hardware_config["e_measure"]

    target = Target(
        description=f"{n_qubits}-qubit target with dynamic circuit support",
        num_qubits=n_qubits,
        dt=dt_in_sec,
        qubit_properties=[
            QubitProperties(t1=t1, t2=t2, frequency=frequency) for _ in range(n_qubits)
        ],
    )

    # Add single-qubit gates
    sx_props = InstructionProperties(duration=t_single_gate, error=e_single_gate)
    target.add_instruction(SXGate(), {(q,): sx_props for q in range(n_qubits)})

    rz_props = InstructionProperties(duration=0.0, error=0.0)  # virtual Rz gate
    target.add_instruction(
        RZGate(Parameter("theta")), {(q,): rz_props for q in range(n_qubits)}
    )

    # x_props = InstructionProperties(duration=2 * t_single_gate, error=2 * e_single_gate)
    # target.add_instruction(
    #     XGate(),
    #     {(q,): x_props for q in range(n_qubits)},
    # )

    # Add two-qubit gates
    cz_props = InstructionProperties(duration=t_cz_gate, error=e_two_gate)
    target.add_instruction(
        CZGate(), {edge: cz_props for edge in coupling_map.get_edges()}
    )

    # AerSimulator can't support iSWAP gate yet, so comment it out
    # iswap_props = InstructionProperties(duration=t_iswap_gate, error=e_two_gate)
    # target.add_instruction(iSwapGate(), {edge: iswap_props for edge in coupling_map.get_edges()})

    # Add measurement
    measure_props = InstructionProperties(duration=t_measure, error=e_measure)
    if "prob_meas1_prep0" in hardware_config and "prob_meas0_prep1" in hardware_config:
        measure_props.prob_meas1_prep0 = hardware_config["prob_meas1_prep0"]
        measure_props.prob_meas0_prep1 = hardware_config["prob_meas0_prep1"]
    target.add_instruction(Measure(), {(q,): measure_props for q in range(n_qubits)})

    # Add control flow
    target.add_instruction(IfElseOp, name="if_else")

    # Add delay
    target.add_instruction(
        Delay(Parameter("t")),
        {(q,): InstructionProperties(duration=0.0, error=0.0) for q in range(n_qubits)},
    )

    return target


def build_backend(
    coupling_map: CouplingMap, hardware_config: HardwareConfig
) -> GenericBackendV2:
    target = _build_target(coupling_map, hardware_config)

    basis_gates = list(set(target.operation_names) - {"if_else"})

    backend = GenericBackendV2(
        num_qubits=target.num_qubits,
        coupling_map=coupling_map,
        control_flow=True,
        basis_gates=basis_gates,
    )
    backend._target = target  # replace the target with given `target`

    return backend


if __name__ == "__main__":
    from pprint import pprint

    config_path = Path("hardware.toml")
    hardware_config = load_hardware_config(config_path)
    print("Loaded hardware configuration:")
    pprint(hardware_config)

    coupling_map = CouplingMap.from_grid(2, 3)
    backend = build_backend(coupling_map, hardware_config)

    assert backend.num_qubits == 6

    print(f"Backend's basis gates: {backend._basis_gates}")
