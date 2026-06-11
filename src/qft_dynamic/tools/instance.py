"""Big instance generators for Shor order-finding benchmarks.

Implements three families of known-order instance generators as specified in
``docs/plan-shor-instance.md`` and ``docs/correctness-shor-instance.md``:

1. **Prime-order instance** — simplest large-order sanity test.
2. **Composite-order instance** — tests partial denominator recovery and LCM
   aggregation.
3. **RSA-style instance** — composite modulus ``N = p * q``, known global
   order, and guaranteed Shor-style gcd factorization success.
"""

import logging
import random
from math import gcd
from typing import Protocol, runtime_checkable

import sympy

from qft_dynamic.shor_benchmark.types import BenchmarkInstance

logger: logging.Logger = logging.getLogger(__name__)


def is_probable_prime(x: int) -> bool:
    """Test whether ``x`` is a probable prime.

    Uses sympy's ``isprime``, which is deterministic for ``x < 2^64`` and
    probabilistic (Miller-Rabin with many bases) for larger values.

    Args:
        x: Integer to test.

    Returns:
        ``True`` if ``x`` is a probable prime.
    """

    return bool(sympy.isprime(x))


def random_prime(bits: int) -> int:
    """Generate a random prime with exactly ``bits`` bits.

    Args:
        bits: Number of bits for the prime. Must be at least 2.

    Returns:
        A random prime with exactly ``bits`` bits.
    """

    while True:
        n: int = random.getrandbits(bits)
        n |= 1 << (bits - 1)
        n |= 1
        if is_probable_prime(n):
            return n


def find_prime_of_form(
    d: int,
    k_start: int,
    k_step: int,
    k_max: int | None = None,
) -> tuple[int, int]:
    """Search for a prime ``p = k * d + 1``.

    Args:
        d: Target divisor.
        k_start: Starting value for ``k``.
        k_step: Increment for ``k``.
        k_max: Optional upper bound on ``k`` (inclusive).

    Returns:
        A tuple ``(p, k)`` where ``p`` is prime and ``p = k * d + 1``.

    Raises:
        ValueError: If no prime is found so that ``k <= k_max``.
    """

    k: int = k_start
    while True:
        if k_max is not None and k > k_max:
            raise ValueError(
                f"No prime of form k*{d}+1 found for k in [{k_start}, {k_max}]"
            )
        p: int = k * d + 1
        if is_probable_prime(p):
            logger.debug("Found prime p=%d = %d*%d + 1", p, k, d)
            return p, k
        k += k_step


def crt_pair(a_p: int, p: int, a_q: int, q: int) -> int:
    """Combine two residues via the Chinese Remainder Theorem.

    Compute ``a`` modulo ``N = p * q`` such that::

        a ≡ a_p (mod p)
        a ≡ a_q (mod q)

    Args:
        a_p: Residue modulo ``p``.
        p: First modulus.
        a_q: Residue modulo ``q``.
        q: Second modulus.

    Returns:
        Combined residue modulo ``p * q``.
    """

    N: int = p * q
    inv_q_mod_p: int = pow(q, -1, p)
    inv_p_mod_q: int = pow(p, -1, q)
    a: int = (a_p * q * inv_q_mod_p + a_q * p * inv_p_mod_q) % N
    return a


@runtime_checkable
class InstanceGenerator(Protocol):
    """Protocol for Shor benchmark instance generators."""

    def generate(self) -> BenchmarkInstance:
        """Generate a benchmark instance with known ground-truth order.

        Returns:
            A valid ``BenchmarkInstance``.
        """
        ...


class PrimeOrderGenerator:
    """Generate a prime-order benchmark instance.

    Args:
        order_bits: Bit length of the target (prime) order ``r``.
        slack: Extra qubit count added when computing ``m``.
        k_max: Upper bound on ``k`` during prime-form search.
        max_attempts: Maximum attempts before raising an error.
    """

    def __init__(
        self,
        order_bits: int,
        slack: int = 2,
        k_max: int | None = None,
        max_attempts: int = 20,
    ) -> None:
        self.order_bits: int = order_bits
        self.slack: int = slack
        self.k_max: int | None = k_max
        self.max_attempts: int = max_attempts

    def generate(self) -> BenchmarkInstance:
        """Generate a prime-order instance.

        Returns:
            A ``BenchmarkInstance`` with prime order ``r`` and modulus
            ``N = p``.
        """

        for attempt in range(self.max_attempts):
            logger.debug("Prime-order attempt %d/%d", attempt + 1, self.max_attempts)
            try:
                return self._generate_one()
            except ValueError as exc:
                logger.debug("Attempt %d failed: %s", attempt + 1, exc)
                continue
        raise RuntimeError(
            f"Failed to generate prime-order instance in {self.max_attempts} attempts"
        )

    def _generate_one(self) -> BenchmarkInstance:
        r: int = random_prime(self.order_bits)

        p: int
        k: int
        p, k = find_prime_of_form(
            d=r,
            k_start=2,
            k_step=2,
            k_max=self.k_max,
        )

        h: int
        a: int
        while True:
            h = random.randint(2, p - 2)
            a = pow(h, k, p)
            if a != 1:
                break

        N: int = p
        m: int = 2 * N.bit_length() + self.slack

        assert gcd(a, N) == 1
        assert pow(a, r, N) == 1
        assert a != 1

        return BenchmarkInstance(n=N, a=a, r=r, m=m)


class CompositeOrderGenerator:
    """Generate a composite-order benchmark instance with known factorization.

    Args:
        factorization: Factorization of the target order as
            ``[(prime, exponent), ...]``.
        slack: Extra qubit count added when computing ``m``.
        k_max: Upper bound on ``k`` during prime-form search.
        max_attempts: Maximum attempts before raising an error.
    """

    def __init__(
        self,
        factorization: list[tuple[int, int]],
        slack: int = 2,
        k_max: int | None = None,
        max_attempts: int = 20,
    ) -> None:
        self.factorization: list[tuple[int, int]] = factorization
        self.slack: int = slack
        self.k_max: int | None = k_max
        self.max_attempts: int = max_attempts

    def generate(self) -> BenchmarkInstance:
        """Generate a composite-order instance.

        Returns:
            A ``BenchmarkInstance`` with composite order ``r`` and modulus
            ``N = p``.
        """

        for attempt in range(self.max_attempts):
            logger.debug(
                "Composite-order attempt %d/%d", attempt + 1, self.max_attempts
            )
            try:
                return self._generate_one()
            except ValueError as exc:
                logger.debug("Attempt %d failed: %s", attempt + 1, exc)
                continue
        raise RuntimeError(
            f"Failed to generate composite-order instance in {self.max_attempts} attempts"
        )

    def _generate_one(self) -> BenchmarkInstance:
        r: int = 1
        distinct_ells: list[int] = []
        for ell, exp in self.factorization:
            r *= ell**exp
            distinct_ells.append(ell)

        if r % 2 == 0:
            k_start: int = 1
            k_step: int = 1
        else:
            k_start = 2
            k_step = 2

        p: int
        k: int
        p, k = find_prime_of_form(
            d=r,
            k_start=k_start,
            k_step=k_step,
            k_max=self.k_max,
        )

        a: int
        while True:
            h: int = random.randint(2, p - 2)
            a = pow(h, k, p)

            ok: bool = True
            for ell in distinct_ells:
                if pow(a, r // ell, p) == 1:
                    ok = False
                    break
            if ok:
                break

        N: int = p
        m: int = 2 * N.bit_length() + self.slack

        assert gcd(a, N) == 1
        assert pow(a, r, N) == 1
        for ell in distinct_ells:
            assert pow(a, r // ell, N) != 1

        return BenchmarkInstance(n=N, a=a, r=r, m=m)


class RsaStyleGenerator:
    """Generate an RSA-style benchmark instance with composite modulus.

    The modulus is ``N = p * q`` with known ground-truth order
    ``r = 2 * s * t``. The classical gcd factoring step is guaranteed to
    succeed: ``gcd(a^(r/2) ± 1, N)`` recovers ``p`` and ``q``.

    Args:
        order_bits: Approximate bit length of the global order ``r``.
        slack: Extra qubit count added when computing ``m``.
        k_max: Upper bound on ``k`` during prime-form search.
        max_attempts: Maximum attempts before raising an error.
    """

    def __init__(
        self,
        order_bits: int,
        slack: int = 2,
        k_max: int | None = None,
        max_attempts: int = 20,
    ) -> None:
        self.order_bits: int = order_bits
        self.slack: int = slack
        self.k_max: int | None = k_max
        self.max_attempts: int = max_attempts

    def generate(self) -> BenchmarkInstance:
        """Generate an RSA-style instance.

        Returns:
            A ``BenchmarkInstance`` with ``N = p * q`` and ground-truth
            order ``r = 2 * s * t``.
        """

        for attempt in range(self.max_attempts):
            logger.debug("RSA-style attempt %d/%d", attempt + 1, self.max_attempts)
            try:
                return self._generate_one()
            except ValueError as exc:
                logger.debug("Attempt %d failed: %s", attempt + 1, exc)
                continue
        raise RuntimeError(
            f"Failed to generate RSA-style instance in {self.max_attempts} attempts"
        )

    def _generate_one(self) -> BenchmarkInstance:
        s_bits: int = (self.order_bits - 1) // 2
        t_bits: int = self.order_bits - 1 - s_bits

        s: int
        t: int
        while True:
            s = random_prime(s_bits)
            t = random_prime(t_bits)
            if s != t and s % 2 == 1 and t % 2 == 1:
                break

        p: int
        k_p: int
        p, k_p = find_prime_of_form(
            d=s,
            k_start=2,
            k_step=2,
            k_max=self.k_max,
        )

        a_p: int
        while True:
            h_p: int = random.randint(2, p - 2)
            a_p = pow(h_p, k_p, p)
            if a_p != 1:
                break

        q: int
        k_q: int
        q, k_q = find_prime_of_form(
            d=2 * t,
            k_start=1,
            k_step=1,
            k_max=self.k_max,
        )

        while q == p:
            q, k_q = find_prime_of_form(
                d=2 * t,
                k_start=k_q + 1,
                k_step=1,
                k_max=self.k_max,
            )

        b_q: int
        while True:
            h_q: int = random.randint(2, q - 2)
            b_q = pow(h_q, (q - 1) // t, q)
            if b_q != 1:
                break

        a_q: int = (-b_q) % q

        N: int = p * q
        a: int = crt_pair(a_p, p, a_q, q)
        r: int = 2 * s * t
        m: int = 2 * N.bit_length() + self.slack

        assert pow(a_p, s, p) == 1
        assert a_p != 1
        assert pow(a_q, 2 * t, q) == 1
        assert pow(a_q, t, q) == q - 1

        assert gcd(a, N) == 1
        assert pow(a, r, N) == 1
        assert pow(a, r // 2, N) not in (1, N - 1)

        g_minus: int = gcd(pow(a, r // 2, N) - 1, N)
        g_plus: int = gcd(pow(a, r // 2, N) + 1, N)

        assert g_minus in (p, q)
        assert g_plus in (p, q)
        assert g_minus * g_plus == N

        return BenchmarkInstance(n=N, a=a, r=r, m=m)


__all__: list[str] = [
    "CompositeOrderGenerator",
    "InstanceGenerator",
    "PrimeOrderGenerator",
    "RsaStyleGenerator",
    "crt_pair",
    "find_prime_of_form",
    "is_probable_prime",
    "random_prime",
]
