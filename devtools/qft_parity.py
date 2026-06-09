# %%
import math
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from tqdm import tqdm

from qft_dynamic.tools.build_circuits import qft_unitary


# %%
def create_qft_circuit(n: int, with_measure: bool = False) -> QuantumCircuit:
    assert n > 0
    qc = QuantumCircuit(n, n)

    if n == 1:
        qc.h(0)
        if with_measure:
            qc.measure(0, 0)
        return qc

    # init RZ
    for i in range(n):
        if i == 0:
            qc.h(0)
        else:
            phase = 0.0
            # CP(j, i)
            for j in range(i):
                phase += math.pi / (2 ** (i - j + 1))
            # H(i)
            if i < n - 1:
                phase += math.pi / 2
            qc.rz(phase, i)

    qc.barrier()

    # PTN
    for i in range(n - 1):
        # PTC(0, n-1-i)
        for j in range(n - 1 - i):
            # DX(j, j+1)
            qc.cx(j + 1, j)
            qc.cx(j, j + 1)

        # RZ on physical j: Z{i} Z{i+j+1}
        for j in range(n - 1 - i):
            qc.rz(-math.pi / 2 ** (j + 2), j)

        # RX on physical 0: RX{i+1}
        if i < n - 2:
            qc.rx(math.pi / 2, 0)

    # clean: decode labels
    for i in range(n - 1, 0, -1):
        qc.cx(i, i - 1, label="decode")

    qc.barrier()

    # final RZ
    for i in range(n):
        target = n - 1 - i
        if i == n - 1:
            qc.h(target)
        else:
            phase = 0.0
            # CP(i, j)
            for j in range(i + 1, n):
                phase += math.pi / (2 ** (j - i + 1))
            # H(i)
            if i > 0:
                phase += math.pi / 2
            qc.rz(phase, target)

    qc.barrier()

    # measure
    if with_measure:
        for i in range(n):
            qc.measure(i, n - 1 - i)

    return qc


qc = create_qft_circuit(3)
qc.draw("mpl", fold=-1, reverse_bits=True)


# %%
def check_unitary_equiv(qc_list: list[QuantumCircuit]) -> bool:
    assert len(qc_list) > 1
    op_list = [Operator(qc) for qc in qc_list]
    op0 = op_list[0]
    for op in op_list[1:]:
        if not op0.equiv(op):
            return False
    return True


# %%
# Check the correctness of the qft parity circuit
for n in range(1, 10):
    qc_1 = create_qft_circuit(n, False)

    qft = qft_unitary(n, do_swap=True)

    assert check_unitary_equiv([qc_1, qft])
print("All passed!")


# %%
def create_qft_circuit2(
    n: int,
    /,
    do_decode: bool = True,
    keep_final_rz: bool = True,
    with_measure: bool = False,
) -> QuantumCircuit:
    assert n > 0
    qc = QuantumCircuit(n, n)

    if n == 1:
        qc.h(0)
        return qc

    # init RZ
    for i in range(n):
        if i == 0:
            qc.h(0)
        else:
            phase = 0.0
            # CP(j, i)
            for j in range(i):
                phase += math.pi / (2 ** (i - j + 1))
            # H(i)
            phase += math.pi / 2
            qc.rz(phase, i)

    qc.barrier()

    # PTN
    for i in range(n - 1):
        # PTC(0, n-1-i)
        for j in range(n - 1 - i):
            # DX(j, j+1)
            qc.cx(j + 1, j)
            qc.cx(j, j + 1)

        # RZ on physical j: Z{i} Z{i+j+1}
        for j in range(n - 1 - i):
            qc.rz(-math.pi / 2 ** (j + 2), j)

        # RX on physical 0: RX{i+1}
        qc.rx(math.pi / 2, 0)

    # clean: decode labels
    if do_decode:
        for i in range(n - 1, 0, -1):
            qc.cx(i, i - 1, label="decode")

    qc.barrier()

    # final RZ
    if keep_final_rz:
        for i in range(n):
            target = n - 1 - i

            phase = 0.0
            # CP(i, j)
            for j in range(i + 1, n):
                phase += math.pi / (2 ** (j - i + 1))
            # H(i)
            if i > 0:
                phase += math.pi / 2
            qc.rz(phase, target)

    # measure
    if with_measure:
        for i in range(n):
            qc.measure(i, n - 1 - i)

    return qc


qc = create_qft_circuit2(4, keep_final_rz=True)
qc.draw("mpl", fold=-1, reverse_bits=True)
# %%

for n in tqdm(range(1, 10)):
    qft = qft_unitary(n, do_swap=True)
    qc_1 = create_qft_circuit(n)
    qc_2 = create_qft_circuit2(n, keep_final_rz=True)
    assert check_unitary_equiv([qc_1, qft, qc_2])


# %%
def create_qft_without_decode(n: int) -> QuantumCircuit:
    assert n > 0
    qc = QuantumCircuit(n, n)

    if n == 1:
        qc.h(0)
        qc.measure(0, 0)
        return qc

    # init RZ
    for i in range(n):
        if i == 0:
            qc.h(0)
        else:
            phase = 0.0
            # CP(j, i)
            for j in range(i):
                phase += math.pi / (2 ** (i - j + 1))
            # H(i)
            if i < n - 1:
                phase += math.pi / 2
            qc.rz(phase, i)

    qc.barrier()

    # PTN
    for i in range(n - 1):
        # PTC(0, n-1-i)
        for j in range(n - 1 - i):
            # DX(j, j+1)
            qc.cx(j + 1, j)
            qc.cx(j, j + 1)

        # RZ on physical j: Z{i} Z{i+j+1}
        for j in range(n - 1 - i):
            qc.rz(-math.pi / 2 ** (j + 2), j)

        # RX on physical 0: RX{i+1}
        if i < n - 2:
            qc.rx(math.pi / 2, 0)

    # last H
    qc.h(0)

    qc.barrier()

    # measure
    for i in range(n):
        qc.measure(i, n - 1 - i)

    return qc


qc = create_qft_without_decode(3)
qc.draw("mpl", fold=-1, reverse_bits=True)


# %%
def get_measurement_distribution(
    qc: QuantumCircuit,
    initial_state: Statevector | None = None,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    qc_no_measure = qc.remove_final_measurements(inplace=False)
    assert qc_no_measure is not None

    if initial_state is None:
        initial_state = Statevector.from_int(0, 2**qc.num_qubits)

    final_state = initial_state.evolve(qc_no_measure)
    probs = np.abs(final_state.probabilities())
    return probs


def decode_probability_distribution(
    num_qubits: int,
    probs: np.ndarray[tuple[int], np.dtype[np.float64]],
    decode_fn: Callable[[int], int] = lambda x: x,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    assert len(probs) == 2**num_qubits

    decoded_probs = np.zeros(2**num_qubits, dtype=np.float64)
    for i in range(2**num_qubits):
        decoded_probs[decode_fn(i)] += probs[i]
    return decoded_probs


def qft_parity_decode(x: int, n: int) -> int:
    if n == 1:
        return x

    bitstring = f"{x:0{n}b}"
    decoded_bitstring: list[str] = []

    decoded_bitstring.append(bitstring[0])
    for i in range(1, n - 1):
        acc = 0
        for j in range(i + 1):
            acc ^= int(bitstring[j])
        decoded_bitstring.append(str(acc))

    decoded_bitstring.append(bitstring[-1])
    return int("".join(reversed(decoded_bitstring)), 2)


f"{
    qft_parity_decode(
        0b01010,
        5,
    ):05b}"

# %%
n = 6
qft = qft_unitary(n, do_swap=True)
qc_undecode = create_qft_without_decode(n)


for i in tqdm(range(2**n)):
    state = Statevector.from_int(i, 2**n)
    probs0 = get_measurement_distribution(qft, state)
    probs1 = get_measurement_distribution(qc_undecode, state)
    probs1 = decode_probability_distribution(
        n, probs1, lambda x: qft_parity_decode(x, n)
    )

    assert np.allclose(probs0, probs1)

print("All passed!")


# %%
def calc_positions(
    n: int,
) -> None:
    assert n > 0
    qc = QuantumCircuit(n, n)

    if n == 1:
        qc.h(0)
        return

    # init RZ
    for i in range(n):
        if i == 0:
            qc.h(0)
        else:
            phase = 0.0
            # CP(j, i)
            for j in range(i):
                phase += math.pi / (2 ** (i - j + 1))
            # H(i)
            phase += math.pi / 2
            qc.rz(phase, i)

    qc.barrier()

    # PTN
    for i in range(n - 1):
        # PTC(0, n-1-i)
        for j in range(n - 1 - i):
            # DX(j, j+1)
            qc.cx(j + 1, j)
            qc.cx(j, j + 1)

        # RZ on physical j: Z{i} Z{i+j+1}
        for j in range(n - 1 - i):
            qc.rz(-math.pi / 2 ** (j + 2), j)

        # RX on physical 0: RX{i+1}
        qc.rx(math.pi / 2, 0)

    # clean: decode labels
    for i in range(n - 1, 0, -1):
        qc.cx(i, i - 1, label="decode")

    qc.barrier()

    # final RZ
    for i in range(n):
        target = n - 1 - i

        phase = 0.0
        # CP(i, j)
        for j in range(i + 1, n):
            phase += math.pi / (2 ** (j - i + 1))
        # H(i)
        if i > 0:
            phase += math.pi / 2
        qc.rz(phase, target)
