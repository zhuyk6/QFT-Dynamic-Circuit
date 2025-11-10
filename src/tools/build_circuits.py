import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.passes import RemoveBarriers

from .build_backend import HardwareConfig

import logging

logger = logging.getLogger(__name__)


def qft_unitary(n: int, /, measure: bool = False) -> QuantumCircuit:
    """
    生成一个n-qubit的标准酉变换QFT线路。
    """
    circuit = QuantumCircuit(n, name=f"QFT({n})")

    # 对每个量子比特应用Hadamard门和受控旋转门
    for qubit in range(n):
        circuit.h(qubit)

        for control_qubit in range(qubit + 1, n):
            # Qiskit的受控相位门CP(theta, ctrl, target)
            k = control_qubit - qubit
            circuit.cp(2 * np.pi / (2 ** (k + 1)), control_qubit, qubit)

        # 为了可视化，分隔每一步
        if qubit < n - 1:
            circuit.barrier()

    if measure:
        c_reg = ClassicalRegister(n, "C")
        circuit.add_register(c_reg)

        circuit.barrier()
        circuit.measure(range(n), c_reg)

    return circuit


def qft_dynamic_batched(
    n: int, batch_size: int, /, hardware_config: HardwareConfig
) -> QuantumCircuit:
    """Build a batched dynamic QFT circuit.

    Args:
        n (int): Number of qubits.
        batch_size (int): Size of each batch.
        hardware_config (HardwareConfig): Config of the hardware. This is used to get the feed forward time.

    Returns:
        QuantumCircuit: The dynamic QFT circuit.
    """
    q_reg = QuantumRegister(n, "Q")
    c_reg = ClassicalRegister(n, "C")
    circuit = QuantumCircuit(
        q_reg, c_reg, name=f"Dynamic_QFT_Batched({n},{batch_size})"
    )

    num_batch = n // batch_size
    last_batch_size = n % batch_size

    logger.debug(
        f"{n} qubits divided into {num_batch}x{batch_size} batches, last batch size: {last_batch_size}"
    )

    for i in range(num_batch):
        start_qubit = i * batch_size
        end_qubit = (i + 1) * batch_size

        # build QFT for this batch
        circ_qft = RemoveBarriers()(qft_unitary(end_qubit - start_qubit))
        circuit.compose(circ_qft, inplace=True, qubits=range(start_qubit, end_qubit))

        # measure this batch
        circuit.barrier()
        for q in range(start_qubit, end_qubit):
            circuit.measure(q, q)
        circuit.barrier()

        # feed-forward delay
        if end_qubit < n:
            circuit.delay(
                duration=hardware_config["t_feed_forward"],
                qarg=range(end_qubit, n),
                unit="s",
            )

        # apply conditional phase rotations
        for q in range(start_qubit, end_qubit):
            with circuit.if_test((circuit.clbits[q], 1)):
                for target_bit in range(end_qubit, n):
                    k = target_bit - q
                    circuit.p(2 * np.pi / (2 ** (k + 1)), target_bit)

    # last batch
    if last_batch_size > 0:
        start_qubit = num_batch * batch_size
        end_qubit = n

        circ_qft = RemoveBarriers()(qft_unitary(end_qubit - start_qubit))
        circuit.compose(circ_qft, inplace=True, qubits=range(start_qubit, end_qubit))

        circuit.barrier()
        for q in range(start_qubit, end_qubit):
            circuit.measure(q, q)
        circuit.barrier()

    return circuit


def qft_dynamic_batched_with_measurement_encoding(
    n: int,
    batch_size: int,
    /,
    hardware_config: HardwareConfig,
) -> QuantumCircuit:
    q_data = QuantumRegister(n, "Q")
    c_data = ClassicalRegister(n, "C")
    circuit = QuantumCircuit(
        q_data,
        c_data,
        name=f"Dynamic_QFT_Batched({n},{batch_size}) with Measurement Encoding",
    )

    num_batch = n // batch_size
    last_batch_size = n % batch_size

    logger.debug(
        f"{n} qubits divided into {num_batch}x{batch_size} batches, last batch size: {last_batch_size}"
    )

    q_ancilla = QuantumRegister(n, "M")
    c_ancilla = ClassicalRegister(n, "C_anc")
    circuit.add_register(q_ancilla)
    circuit.add_register(c_ancilla)

    for i in range(num_batch):
        start_qubit = i * batch_size
        end_qubit = (i + 1) * batch_size

        circ_qft = RemoveBarriers()(qft_unitary(end_qubit - start_qubit))

        circuit.compose(circ_qft, inplace=True, qubits=range(start_qubit, end_qubit))

        circuit.barrier()
        for q in range(start_qubit, end_qubit):
            circuit.cx(q, q_ancilla[q])
            circuit.x(q_ancilla[q])
            circuit.measure(q, q)
            circuit.measure(q_ancilla[q], c_ancilla[q])
        circuit.barrier()

        if end_qubit < n:
            circuit.delay(
                duration=hardware_config["t_feed_forward"],
                qarg=range(end_qubit, n),
                unit="s",
            )

        for q in range(start_qubit, end_qubit):
            with circuit.if_test((circuit.clbits[q], 1)):
                for target_bit in range(end_qubit, n):
                    k = target_bit - q
                    circuit.p(2 * np.pi / (2 ** (k + 1)), target_bit)

    # last batch
    if last_batch_size > 0:
        start_qubit = num_batch * batch_size
        end_qubit = n

        circ_qft = RemoveBarriers()(qft_unitary(end_qubit - start_qubit))

        circuit.compose(circ_qft, inplace=True, qubits=range(start_qubit, end_qubit))

        circuit.barrier()
        for q in range(start_qubit, end_qubit):
            circuit.measure(q, q)
        circuit.barrier()

    return circuit


def prepare_circular_state_circuit(n: int, r: int) -> QuantumCircuit:
    """Prepare a circular state |ψ⟩ = 1/√m ∑|j x r⟩ for j=0 to m-1, where m = 2^n / r.
    But considering no-SWAP in QFT, the state is actually |ψ⟩ = 1/√m ∑|rev(j x r)⟩.

    Args:
        n (int): Number of qubits.

    Returns:
        QuantumCircuit: The circuit preparing the circular state.
    """

    assert r > 0, "r should be positive"

    assert (2**n) % r == 0, "r should divide 2**n"

    k: int = r.bit_length() - 1

    circuit = QuantumCircuit(n, name=f"Circular_State_n{n}_r{r}")

    # Apply H gates on the lower n - k qubits.
    # The circular state is then prepared due to the bit-reversal in QFT.
    for i in range(n - k):
        circuit.h(i)

    circuit.barrier()

    return circuit
