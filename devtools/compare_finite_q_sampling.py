"""Visualize finite-Q ideal sampling distributions for two exact methods.

This script compares the following distributions for one benchmark instance:

- the closed-form finite-Q ideal probability P(y | s)
- empirical samples from the explicit-enumeration sampler
- empirical samples from the bitwise semiclassical-IQFT sampler

The goal is visual inspection rather than automated testing.
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt

from qft_dynamic.shor_benchmark.samplers import (
    FiniteQIdealSampler,
    finite_q_ideal_probability,
)
from qft_dynamic.shor_benchmark.types import BenchmarkInstance


def estimate_empirical_distribution(
    sampler: FiniteQIdealSampler,
    s: int,
    num_samples: int,
    seed: int,
) -> list[float]:
    """Estimate the output distribution for one fixed phase label.

    Args:
        sampler: Finite-Q ideal sampler.
        s: Phase label in [0, r - 1].
        num_samples: Number of Monte Carlo samples.
        seed: Random seed for reproducibility.

    Returns:
        Empirical probability vector over y in [0, Q - 1].
    """

    rng: random.Random = random.Random(seed)
    counts: list[int] = [0 for _ in range(sampler.instance.q)]

    sample_index: int
    for sample_index in range(num_samples):
        _ignored_sample_index: int = sample_index
        sampled_y: int = sampler.sample_y(s=s, rng=rng)
        counts[sampled_y] += 1

    empirical: list[float] = [count / num_samples for count in counts]
    return empirical


def closed_form_distribution(
    instance: BenchmarkInstance,
    s: int,
) -> list[float]:
    """Compute the exact finite-Q ideal distribution for one fixed s.

    Args:
        instance: Benchmark instance.
        s: Phase label in [0, r - 1].

    Returns:
        Exact probability vector over y in [0, Q - 1].
    """

    distribution: list[float] = [
        finite_q_ideal_probability(y=y, s=s, instance=instance)
        for y in range(instance.q)
    ]
    return distribution


def plot_distributions(
    instance: BenchmarkInstance,
    s: int,
    exact_distribution: list[float],
    enumerate_distribution: list[float],
    bitwise_distribution: list[float],
    num_samples: int,
    output_path: Path,
    show_plot: bool,
) -> None:
    """Plot and save the three distributions for visual comparison.

    Args:
        instance: Benchmark instance.
        s: Phase label in [0, r - 1].
        exact_distribution: Closed-form finite-Q ideal probabilities.
        enumerate_distribution: Empirical distribution from enumeration method.
        bitwise_distribution: Empirical distribution from bitwise method.
        num_samples: Number of samples used for each empirical distribution.
        output_path: Output path for the figure.
        show_plot: Whether to display the plot interactively.
    """

    y_values: list[int] = list(range(instance.q))
    figure_width: float = max(10.0, min(18.0, instance.q / 4.0))
    fig, ax = plt.subplots(figsize=(figure_width, 6.0))

    ax.plot(
        y_values,
        exact_distribution,
        color="black",
        linewidth=2.0,
        label="closed-form",
    )
    ax.scatter(
        y_values,
        enumerate_distribution,
        color="#d55e00",
        s=22.0,
        alpha=0.8,
        label="enumerate",
    )
    ax.scatter(
        y_values,
        bitwise_distribution,
        color="#0072b2",
        s=18.0,
        alpha=0.8,
        marker="x",
        label="bitwise",
    )

    ax.set_title(
        "Finite-Q Ideal Sampling Comparison\n"
        f"(n={instance.n}, a={instance.a}, r={instance.r}, m={instance.m}, "
        f"s={s}, samples={num_samples})"
    )
    ax.set_xlabel("y")
    ax.set_ylabel("Probability")
    ax.grid(alpha=0.25)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)

    if show_plot:
        plt.show()

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the comparison script.

    Returns:
        Parsed argument namespace.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Compare enumerate and bitwise finite-Q ideal samplers."
    )
    parser.add_argument("--n", type=int, default=21, help="Modulus N")
    parser.add_argument("--a", type=int, default=2, help="Base a")
    parser.add_argument("--r", type=int, default=6, help="Order r")
    parser.add_argument("--m", type=int, default=6, help="Control qubit count")
    parser.add_argument(
        "--s",
        type=int,
        default=1,
        help="Phase label s to visualize",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30000,
        help="Monte Carlo samples for each empirical distribution",
    )
    parser.add_argument(
        "--seed-enumerate",
        type=int,
        default=11,
        help="Random seed for enumeration-based sampling",
    )
    parser.add_argument(
        "--seed-bitwise",
        type=int,
        default=29,
        help="Random seed for bitwise sampling",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output image path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively in addition to saving it",
    )
    return parser.parse_args()


def main() -> None:
    """Run the finite-Q sampling comparison and render a plot."""

    args: argparse.Namespace = parse_args()
    instance: BenchmarkInstance = BenchmarkInstance(
        n=args.n,
        a=args.a,
        r=args.r,
        m=args.m,
    )

    if not (0 <= args.s < instance.r):
        raise ValueError("s must satisfy 0 <= s < r")
    if args.num_samples <= 0:
        raise ValueError("num-samples must be positive")

    enumerate_sampler: FiniteQIdealSampler = FiniteQIdealSampler(
        instance=instance,
        sample_method="enumerate",
    )
    bitwise_sampler: FiniteQIdealSampler = FiniteQIdealSampler(
        instance=instance,
        sample_method="bitwise",
    )

    exact_distribution: list[float] = closed_form_distribution(
        instance=instance,
        s=args.s,
    )
    enumerate_distribution: list[float] = estimate_empirical_distribution(
        sampler=enumerate_sampler,
        s=args.s,
        num_samples=args.num_samples,
        seed=args.seed_enumerate,
    )
    bitwise_distribution: list[float] = estimate_empirical_distribution(
        sampler=bitwise_sampler,
        s=args.s,
        num_samples=args.num_samples,
        seed=args.seed_bitwise,
    )

    plot_distributions(
        instance=instance,
        s=args.s,
        exact_distribution=exact_distribution,
        enumerate_distribution=enumerate_distribution,
        bitwise_distribution=bitwise_distribution,
        num_samples=args.num_samples,
        output_path=args.output,
        show_plot=args.show,
    )

    print(f"Saved comparison plot to: {args.output}")


if __name__ == "__main__":
    main()
