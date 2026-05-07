from qiskit import QuantumCircuit
from qiskit.circuit import Delay, IfElseOp
from qiskit.circuit.library import Measure
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit_ibm_runtime.transpiler.passes.scheduling import (
    ASAPScheduleAnalysis,
    PadDelay,
)


class UnrollIfTrue(TransformationPass):
    """
    A transpiler pass that unrolls all IfElseOp blocks,
    keeping only the 'true' branch.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run the pass on the DAG.

        Args:
            dag: The directed acyclic graph to transform.
        Returns:
            A new DAG with IfElseOp nodes replaced by their true-body content.
        """
        # 遍历DAG中的所有操作节点
        for node in dag.op_nodes():
            # 只关心 IfElseOp 类型的节点
            if isinstance(node.op, IfElseOp):
                # 提取 'true' 分支的线路
                # op.blocks 是一个元组 (true_circuit, false_circuit)
                true_body_circuit = node.op.blocks[0]

                # 如果true分支是空的，就直接删除if节点
                if len(true_body_circuit) == 0:
                    dag.remove_op_node(node)
                    continue

                # 将 'true' 分支的线路转换成它自己的DAG
                true_body_dag = circuit_to_dag(true_body_circuit)
                dag.substitute_node_with_dag(node, true_body_dag)

        return dag


def unroll_if_true(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Unroll all IfElseOp blocks in the given circuit,
    keeping only the 'true' branch.

    Args:
        circuit: The quantum circuit to transform.
    Returns:
        A new quantum circuit with IfElseOp nodes replaced by their true-body content.
    """

    pm = PassManager(UnrollIfTrue())
    return pm.run(circuit)


def generate_pass_manager(backend: GenericBackendV2) -> PassManager:
    """Generate the pass manager that supports dynamic circuit.

    It will perform the following transformations:
    - Decompose gates
    - No routing
    - Initial layout is just 0, 1, 2, ...
    - Schedule the circuit with:
        - ASAP scheduling
        - Padding delays according to the backend's timing.

    Args:
        backend (GenericBackendV2): Given backend.

    Returns:
        PassManager: The generated pass manager.
    """
    num_qubits = backend.num_qubits

    pm: PassManager = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        initial_layout=list(range(num_qubits)),
        routing_method="none",
    )
    durations = backend.target.durations()
    pm.scheduling = PassManager(  # type: ignore
        [
            ASAPScheduleAnalysis(durations),
            PadDelay(durations),
        ]
    )

    return pm


class DelayMeasurement(TransformationPass):
    """A transpiler pass that inserts a Delay right before each Measure."""

    def __init__(self, delay_time: float):
        super().__init__()
        self.delay_time = delay_time

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # Build a new DAG and copy all ops in topological order.
        # For each Measure op, insert Delay on its qubit(s) first.
        new_dag = dag.copy_empty_like()

        for node in dag.topological_op_nodes():
            if isinstance(node.op, Measure):
                for qarg in node.qargs:
                    new_dag.apply_operation_back(
                        Delay(self.delay_time, unit="s"),
                        qargs=(qarg,),
                        check=True,
                    )

            new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)

        return new_dag


def add_delay_before_measurement(
    circuit: QuantumCircuit, delay_time: float
) -> QuantumCircuit:
    """Add a delay before each measurement in the given circuit."""
    pm = PassManager([DelayMeasurement(delay_time)])
    return pm.run(circuit)
