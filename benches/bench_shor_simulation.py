"""Run Qiskit simulation and save per-s histograms for Shor strict benchmark."""

import logging
import warnings
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, Field

from qft_dynamic.shor_benchmark.simulation import (
    save_histograms,
    simulate_histograms_for_instance,
)
from qft_dynamic.shor_benchmark.types import BenchmarkInstance

app = typer.Typer()
logger = logging.getLogger(__name__)


def setup_logging(log_file: str | Path, verbose: bool = False) -> None:
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


def main(
    instance: BenchmarkInstance,
    batch_size: int,
    output: Path,
    num_shots: int,
    gate_error: bool,
    readout_error: bool,
    thermal_relaxation: bool,
    verbose: bool,
) -> None:
    """Simulate per-s histograms for the Shor strict benchmark."""
    setup_logging("logs/shor-simulation.log", verbose)
    setup_warnings()

    logger.info("Main Start...")
    logger.debug(
        "args: instance=%s batch_size=%s output=%s num_shots=%s gate_error=%s readout_error=%s thermal_relaxation=%s",
        instance,
        batch_size,
        output,
        num_shots,
        gate_error,
        readout_error,
        thermal_relaxation,
    )

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


@app.command("intput")
def cli_manual_input(
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
    instance = BenchmarkInstance(n, a, r, m)
    main(
        instance=instance,
        batch_size=batch_size,
        output=output,
        num_shots=num_shots,
        gate_error=gate_error,
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
        verbose=verbose,
    )


class InstanceModel(BaseModel):
    """Pydantic model for benchmark instance in JSON file."""

    generator: str | None
    n: int
    a: int
    r: int
    m: int
    order_factors: list[int] = Field(default_factory=list)


@app.command("file")
def cli_file_input(
    input_file: Annotated[
        Path, typer.Argument(help="Input JSON path for benchmark instances")
    ],
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
    """Simulate per-s histograms for the Shor strict benchmark from a JSON file."""
    instance_data = InstanceModel.model_validate_json(input_file.read_text())
    instance = BenchmarkInstance(
        n=instance_data.n,
        a=instance_data.a,
        r=instance_data.r,
        m=instance_data.m,
    )

    main(
        instance=instance,
        batch_size=batch_size,
        output=output,
        num_shots=num_shots,
        gate_error=gate_error,
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
        verbose=verbose,
    )


if __name__ == "__main__":
    app()
