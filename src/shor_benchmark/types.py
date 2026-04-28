"""Data models for the Shor strict benchmark."""

from dataclasses import dataclass
from math import gcd


@dataclass(frozen=True)
class BenchmarkInstance:
    """Shor benchmark instance.

    Args:
        n: The modulus used in order finding.
        a: The base integer with gcd(a, n) == 1.
        r: The multiplicative order of a modulo n.
        m: Number of control qubits for QFT/IQFT.
    """

    n: int
    a: int
    r: int
    m: int

    def __post_init__(self) -> None:
        """Validate benchmark-instance invariants.

        Raises:
            ValueError: If the instance parameters do not define a valid Shor
                order-finding benchmark instance.
        """

        if self.n <= 1:
            raise ValueError("n must be greater than 1")
        if not (1 < self.a < self.n):
            raise ValueError("a must satisfy 1 < a < n")
        if gcd(self.a, self.n) != 1:
            raise ValueError("a and n must be coprime")
        if self.r <= 0:
            raise ValueError("r must be positive")
        if self.m <= 0:
            raise ValueError("m must be positive")
        if pow(self.a, self.r, self.n) != 1:
            raise ValueError("r must satisfy a^r == 1 (mod n)")

    @property
    def q(self) -> int:
        """Return Q = 2^m."""
        q_value: int = 2**self.m
        return q_value


@dataclass(frozen=True)
class StrictMetrics:
    """Strict benchmark metrics for one K value.

    Args:
        p_ord_strict: Probability that strict post-processing returns the true order.
        p_wrong: Probability that strict post-processing returns an incorrect order.
        p_null: Probability that strict post-processing rejects with null output.
    """

    p_ord_strict: float
    p_wrong: float
    p_null: float


@dataclass(frozen=True)
class ArithmeticStrictPoint:
    """Arithmetic-ideal strict metric for one K value.

    Args:
        p_ord_strict: Closed-form strict success probability.
    """

    p_ord_strict: float
