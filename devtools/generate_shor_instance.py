"""CLI for generating Shor benchmark instances with known ground-truth order.

Supports three generator families:

- ``prime``: prime-order instance
- ``composite``: composite-order instance with known factorization
- ``rsa``: RSA-style instance with composite modulus

The output is a JSON object with keys ``n``, ``a``, ``r``, ``m``.

Examples::

    python devtools/generate_shor_instance.py prime --order-bits 128
    python devtools/generate_shor_instance.py composite --factorization 2:8,3:1,5:1,7:1
    python devtools/generate_shor_instance.py rsa --order-bits 128
"""

from pathlib import Path

import typer
from pydantic import BaseModel

from qft_dynamic.shor_benchmark.types import BenchmarkInstance
from qft_dynamic.tools.instance import (
    CompositeOrderGenerator,
    PrimeOrderGenerator,
    RsaStyleGenerator,
    random_prime,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Generate a Shor benchmark instance with known ground-truth order.",
)


def parse_factorization(spec: str) -> list[tuple[int, int]]:
    """Parse a factorization specification string.

    Format: ``p1:e1,p2:e2,...`` (e.g. ``2:8,3:1,5:1,7:1``).

    Args:
        spec: Factorization specification.

    Returns:
        List of ``(prime, exponent)`` tuples.

    Raises:
        ValueError: If the specification is malformed.
    """

    factors: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid factor '{part}': expected format prime:exponent")
        prime_str: str
        exp_str: str
        prime_str, exp_str = part.split(":", 1)
        factors.append((int(prime_str), int(exp_str)))
    return factors


class Payload(BaseModel):
    generator: str
    n: int
    a: int
    r: int
    m: int


@app.command("prime")
def generate_prime(
    order_bits: int,
    savepath: Path,
    slack: int = 2,
    k_max: int = 1000,
    max_attempts: int = 20,
) -> None:
    """Generate a prime-order instance (r is prime)."""
    generator: PrimeOrderGenerator = PrimeOrderGenerator(
        order_bits=order_bits,
        slack=slack,
        k_max=k_max,
        max_attempts=max_attempts,
    )
    instance: BenchmarkInstance = generator.generate()

    payload = Payload(
        generator="prime",
        n=instance.n,
        a=instance.a,
        r=instance.r,
        m=instance.m,
    )
    savepath.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved to {savepath}")


@app.command("composite")
def generate_composite(
    order_bits: int,
    num_primes: int,
    savepath: Path,
    slack: int = 2,
    k_max: int = 1000,
    max_attempts: int = 20,
) -> None:
    """Generate a composite-order instance (r = 2 * p1 * ... * pn)."""
    factors = [2]
    cnt = 0
    while True:
        p = random_prime(order_bits)
        if p in factors:
            continue
        cnt += 1
        factors.append(p)
        if cnt == num_primes:
            break

    generator = CompositeOrderGenerator(
        factorization=[(p, 1) for p in factors],
        slack=slack,
        k_max=k_max,
        max_attempts=max_attempts,
    )
    instance: BenchmarkInstance = generator.generate()

    payload = Payload(
        generator="composite",
        n=instance.n,
        a=instance.a,
        r=instance.r,
        m=instance.m,
    )
    savepath.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved to {savepath}")


@app.command("rsa")
def generate_rsa(
    order_bits: int,
    savepath: Path,
    slack: int = 2,
    k_max: int = 1000,
    max_attempts: int = 20,
) -> None:
    """Generate an RSA-style instance (N = p * q, r = 2 * s * t)."""
    generator = RsaStyleGenerator(
        order_bits=order_bits,
        slack=slack,
        k_max=k_max,
        max_attempts=max_attempts,
    )
    instance: BenchmarkInstance = generator.generate()

    payload = Payload(
        generator="rsa",
        n=instance.n,
        a=instance.a,
        r=instance.r,
        m=instance.m,
    )
    savepath.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved to {savepath}")


if __name__ == "__main__":
    app()
