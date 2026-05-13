"""Run Qiskit simulation and save per-s histograms for Shor strict benchmark."""

import logging
import warnings
from pathlib import Path
from typing import Annotated

import typer

from qft_dynamic.shor_benchmark.simulation import (
    save_histograms,
    simulate_histograms_for_instance,
)
from qft_dynamic.shor_benchmark.types import BenchmarkInstance

app = typer.Typer()
logger = logging.getLogger(__name__)


def setup_logging(log_file: str | Path, verbose: bool = False):
    """Setup logging config."""
    level = logging.DEBUG if verbose else logging.INFO

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        filename=log_path,
        filemode="a",
    )

    # suppress third-party logging level
    noisy_libs = [
        "qiskit",
        "qiskit_aer",
        "stevedore",
        "base_primitive",
        "base_primitive._run",
    ]
    for lib_name in noisy_libs:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(logging.CRITICAL + 1)
        lib_logger.handlers = []
        lib_logger.propagate = False


def setup_warnings() -> None:
    """Suppress noisy Qiskit warnings."""
    warnings.filterwarnings("ignore", module="qiskit")


@app.command()
def main(
    n: Annotated[int, typer.Argument(help="Modulus N")],
    a: Annotated[int, typer.Argument(help="Base a")],
    r: Annotated[int, typer.Argument(help="Order r")],
    m: Annotated[int, typer.Argument(help="Control qubit count")],
    batch_size: Annotated[
        int, typer.Argument(help="Tile size of the optimized QFT block")
    ],
    output: Annotated[
        Path, typer.Argument(help="Output JSON path for simulated histograms")
    ],
    num_shots: Annotated[
        int, typer.Option(help="Simulation shots per phase label s")
    ] = 4096,
    gate_error: Annotated[bool, typer.Option(help="Enable gate error")] = True,
    readout_error: Annotated[bool, typer.Option(help="Enable readout error")] = True,
    thermal_relaxation: Annotated[
        bool, typer.Option(help="Enable thermal relaxation")
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="Logging DEBUG mode"),
    ] = False,
) -> None:
    """Simulate per-s histograms for the Shor strict benchmark."""
    setup_logging("logs/shor-simulation.log", verbose)
    setup_warnings()

    logger.info("Main Start...")
    logger.debug("args: n=%s a=%s r=%s m=%s batch_size=%s", n, a, r, m, batch_size)

    instance: BenchmarkInstance = BenchmarkInstance(n, a, r, m)
    histograms = simulate_histograms_for_instance(
        instance=instance,
        batch_size=batch_size,
        num_shots=num_shots,
        gate_error=gate_error,
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
    )
    save_histograms(
        instance=instance,
        histograms=histograms,
        output_path=output,
        batch_size=batch_size,
        num_shots=num_shots,
        gate_error=gate_error,
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
    )
    print(f"Saved simulated histograms to: {output}")

    logger.info("Main end.")


if __name__ == "__main__":
    app()
