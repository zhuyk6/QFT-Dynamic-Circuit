from qiskit import QuantumCircuit
from qiskit.circuit import IfElseOp
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass


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
