import logging

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import Layout, TranspileLayout
from qiskit.transpiler.passes import RemoveBarriers

from .build_backend import HardwareConfig

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

            circuit.barrier()

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
    assert batch_size >= 1, "Batch size should be at least 1"

    # data registers
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

    # ancilla registers
    q_ancilla = QuantumRegister(n, "M")
    c_ancilla = ClassicalRegister(n, "C_anc")
    circuit.add_register(q_ancilla)
    circuit.add_register(c_ancilla)

    # build the circuit batch by batch
    for i in range(num_batch):
        # this batch: [start_qubit, end_qubit)
        start_qubit = i * batch_size
        end_qubit = (i + 1) * batch_size

        # build QFT for this batch
        circ_qft = RemoveBarriers()(qft_unitary(end_qubit - start_qubit))
        circuit.compose(circ_qft, inplace=True, qubits=range(start_qubit, end_qubit))

        # measurement encoding for this batch
        circuit.barrier()
        for q in range(start_qubit, end_qubit):
            circuit.cx(q_data[q], q_ancilla[q])
            circuit.x(q_ancilla[q])
            circuit.measure(q_data[q], c_data[q])
            circuit.measure(q_ancilla[q], c_ancilla[q])
        circuit.barrier()

        # feed-forward delay
        if end_qubit < n:
            circuit.delay(
                duration=hardware_config["t_feed_forward"],
                qarg=range(end_qubit, n),
                unit="s",
            )

            for q in range(start_qubit, end_qubit):
                # if c_data[q] == 1 and c_ancilla[q] == 0
                with circuit.if_test((c_data[q], 1)):
                    with circuit.if_test((c_ancilla[q], 0)):
                        for target_bit in range(end_qubit, n):
                            k = target_bit - q
                            circuit.p(2 * np.pi / (2 ** (k + 1)), q_data[target_bit])

            circuit.barrier()

    # last batch if batch_size cannot divide n
    if last_batch_size > 0:
        start_qubit = num_batch * batch_size
        end_qubit = n

        circ_qft = RemoveBarriers()(qft_unitary(end_qubit - start_qubit))
        circuit.compose(circ_qft, inplace=True, qubits=range(start_qubit, end_qubit))

        circuit.barrier()
        for q in range(start_qubit, end_qubit):
            circuit.cx(q_data[q], q_ancilla[q])
            circuit.x(q_ancilla[q])
            circuit.measure(q_data[q], c_data[q])
            circuit.measure(q_ancilla[q], c_ancilla[q])
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


def prepare_qft_dag_on_computation_basis(n: int, k: int) -> QuantumCircuit:
    """Prepare the state `QFT^dag |k>`, where `|k>` is the computation basis.

    Args:
        n (int): Number of qubits.
        k (int): Computation basis state index.

    Returns:
        QuantumCircuit: The circuit preparing the state `QFT^dag |k>`.
    """
    circ = QuantumCircuit(n, name=f"QFT_dag_on_|{k}>")

    # 1. Apply H gates on all qubits
    for qubit in range(n):
        circ.h(qubit)

    # 2. Apply Rz(- 2 * pi * k / 2^l) on qubit l = 1, ..., n
    for q in range(n):
        angle = -2 * np.pi * k / 2 ** (q + 1)
        circ.rz(angle, q)

    return circ


def tile_transpiled_circuit(
    sub_circuit: QuantumCircuit,
    tiling_pattern: list[list[int]],
    t_feed_forward: float,
) -> QuantumCircuit:
    """Tile a transpiled sub-circuit across a larger circuit according to a specified tiling pattern,
    adding feed-forward logic between tiles.

    Args:
        sub_circuit (QuantumCircuit): The transpiled sub-circuit to be tiled.
        tiling_pattern (list[list[int]]): The pattern specifying how the sub-circuit tiles are arranged in the larger circuit.
            Each inner list represents the physical qubit indices for a tile.
        t_feed_forward (float): The duration of the feed-forward delay between tiles.

    Returns:
        QuantumCircuit: The tiled large circuit with feed-forward delays added.
    """

    assert len(sub_circuit.qregs) == 1, (
        "Expected a single quantum register in sub-circuit."
    )

    source_tile = list(range(sub_circuit.num_qubits))  # physical qubits of sub-circuit
    assert all(len(tile) == len(source_tile) for tile in tiling_pattern), (
        "All target tiles must have the same size as the source tile."
    )

    source_set = set(source_tile)
    tile_size = len(source_tile)

    # Layout message of sub-circuit
    if sub_circuit.layout is None:
        # Generate a trivial layout if none exists
        layout_dict = {vq: i for i, vq in enumerate(sub_circuit.qubits)}
        layout = Layout(layout_dict)
        sub_circuit._layout = TranspileLayout(
            initial_layout=layout,
            input_qubit_mapping=layout_dict,
            final_layout=None,
        )
    assert sub_circuit.layout is not None, "Sub-circuit layout is None."
    sub_physical_to_logical: dict[int, int] = {
        phys_q: sub_circuit.layout.initial_layout[phys_q]._index
        for phys_q in source_tile
    }
    sub_logical_to_physical: dict[int, int] = {
        vq: pq for pq, vq in sub_physical_to_logical.items()
    }

    num_qubits_large = tile_size * len(tiling_pattern)
    large_circuit = QuantumCircuit(num_qubits_large, num_qubits_large)

    # Tile the sub-circuit across the large circuit
    for i, target_tile in enumerate(tiling_pattern):
        # Create the mapping: source physical qubits -> target physical qubits
        qubit_map: dict[int, int] = {
            src_q: tgt_q for src_q, tgt_q in zip(source_tile, target_tile)
        }

        # 1. Iterate through each instruction in the optimized subcircuit
        for instruction in sub_circuit.data:
            op = instruction.operation
            qargs = instruction.qubits
            cargs = instruction.clbits

            original_indices = [q._index for q in qargs]

            # CRITICAL: Only copy instructions that are fully contained within the source tile.
            # This filters out any stray SWAPs the transpiler might have added elsewhere.
            if not source_set.issuperset(original_indices):
                continue

            # Use the map to find the new physical qubit indices
            new_indices = [qubit_map[idx] for idx in original_indices]

            # Get the actual Qubit objects from the large circuit
            new_qubits = [large_circuit.qubits[idx] for idx in new_indices]

            # Remap classical bits if necessary
            new_clbits = [large_circuit.clbits[c._index + i * tile_size] for c in cargs]

            # Append the instruction to the large circuit at the new location
            large_circuit.append(op, qargs=new_qubits, cargs=new_clbits)

        # 2. Add feed-forward logic between tiles
        if i < len(tiling_pattern) - 1:
            large_circuit.barrier()

            # Add delays for all qubits
            large_circuit.delay(
                duration=t_feed_forward,
                qarg=large_circuit.qubits,
                unit="s",
            )

            # Add feed-forward phase gates
            for c in range(tile_size):
                # logical qubit index: c'th in i'th tile
                idx1 = i * tile_size + c
                with large_circuit.if_test((large_circuit.clbits[idx1], 1)):
                    for j in range(i + 1, len(tiling_pattern)):
                        # map from source to target: all physical
                        qubit_map = {
                            src_q: tgt_q
                            for src_q, tgt_q in zip(source_tile, tiling_pattern[j])
                        }

                        for k in range(tile_size):
                            # logical qubit index: k'th in j'th tile
                            idx2 = j * tile_size + k
                            phase = np.pi / 2 ** (idx2 - idx1)

                            # find physical qubit of logical idx2
                            target_physical = qubit_map[sub_logical_to_physical[k]]
                            large_circuit.rz(
                                phase, large_circuit.qubits[target_physical]
                            )

            large_circuit.barrier()

    # Reconstruct the layout for the large circuit
    virtual_qreg: QuantumRegister = large_circuit.qregs[0]
    initial_mapping_dict = {}
    for i, target_tile in enumerate(tiling_pattern):
        # Create the mapping: source physical qubits -> target physical qubits
        qubit_map = {src_q: tgt_q for src_q, tgt_q in zip(source_tile, target_tile)}

        for k in range(tile_size):
            # These are the logical qubits in the new circuit (0, 1, 2, ..., 11)
            large_logical_qubit_index = i * tile_size + k
            virtual_qubit = virtual_qreg[large_logical_qubit_index]

            # These are the logical qubits in the sub_circuit (0, 1, 2)
            sub_logical_qubit_index = k

            # Find path: new logical -> old logical k -> old physical p_src -> new physical p_tgt
            source_physical_qubit = sub_logical_to_physical[sub_logical_qubit_index]
            target_physical_qubit = qubit_map[source_physical_qubit]

            # Map the new virtual qubit to the final physical qubit index
            initial_mapping_dict[virtual_qubit] = target_physical_qubit

    initial_layout = Layout(initial_mapping_dict)
    large_circuit._layout = TranspileLayout(
        initial_layout=initial_layout,
        input_qubit_mapping={q: i for i, q in enumerate(virtual_qreg)},
        final_layout=None,
    )
    return large_circuit
