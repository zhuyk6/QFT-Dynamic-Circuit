"""Run Qiskit simulation and save per-s histograms for Shor strict benchmark."""

import argparse
import logging
import warnings
from pathlib import Path

from shor_benchmark.simulation import save_histograms, simulate_histograms_for_instance
from shor_benchmark.types import BenchmarkInstance

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


def setup_warnings():
    warnings.filterwarnings("ignore", module="qiskit")


def main() -> None:
    """CLI entry point for histogram simulation."""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Simulate per-s histograms for the Shor strict benchmark."
    )
    parser.add_argument("--n", type=int, required=True, help="Modulus N")
    parser.add_argument("--a", type=int, required=True, help="Base a")
    parser.add_argument("--r", type=int, required=True, help="Order r")
    parser.add_argument("--m", type=int, required=True, help="Control qubit count")
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Tile size of the optimized QFT block",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=4096,
        help="Simulation shots per phase label s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path for simulated histograms",
    )
    parser.add_argument(
        "--disable-gate-error",
        action="store_true",
        help="Disable gate error in the Aer noise model",
    )
    parser.add_argument(
        "--disable-readout-error",
        action="store_true",
        help="Disable readout error in the Aer noise model",
    )
    parser.add_argument(
        "--disable-thermal-relaxation",
        action="store_true",
        help="Disable thermal relaxation in the Aer noise model",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Logging DEBUG mode",
    )
    args: argparse.Namespace = parser.parse_args()

    setup_logging("logs/shor-simulation.log", args.verbose)
    setup_warnings()

    logger.info("Main Start...")
    logger.debug(args)

    instance: BenchmarkInstance = BenchmarkInstance(
        n=args.n,
        a=args.a,
        r=args.r,
        m=args.m,
    )
    gate_error: bool = not args.disable_gate_error
    readout_error: bool = not args.disable_readout_error
    thermal_relaxation: bool = not args.disable_thermal_relaxation

    histograms = simulate_histograms_for_instance(
        instance=instance,
        batch_size=args.batch_size,
        num_shots=args.num_shots,
        gate_error=gate_error,
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
    )
    save_histograms(
        instance=instance,
        histograms=histograms,
        output_path=args.output,
        batch_size=args.batch_size,
        num_shots=args.num_shots,
        gate_error=gate_error,
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
    )
    print(f"Saved simulated histograms to: {args.output}")

    logger.info("Main end.")


if __name__ == "__main__":
    main()
