"""Unit tests for Shor benchmark data models."""

import pytest

from qft_dynamic.shor_benchmark.types import BenchmarkInstance


def test_benchmark_instance_accepts_valid_order_finding_instance() -> None:
    """BenchmarkInstance should accept a valid Shor order-finding instance."""

    instance: BenchmarkInstance = BenchmarkInstance(n=15, a=2, r=4, m=4)

    assert instance.q == 16


def test_benchmark_instance_rejects_non_coprime_a_and_n() -> None:
    """BenchmarkInstance should reject bases that are not coprime with n."""

    with pytest.raises(ValueError, match="coprime"):
        BenchmarkInstance(n=15, a=3, r=4, m=4)


def test_benchmark_instance_rejects_invalid_order_r() -> None:
    """BenchmarkInstance should reject r when a^r is not 1 modulo n."""

    with pytest.raises(ValueError, match=r"a\^r == 1 \(mod n\)"):
        BenchmarkInstance(n=15, a=2, r=3, m=4)


def test_benchmark_instance_rejects_non_positive_m() -> None:
    """BenchmarkInstance should reject non-positive control-register sizes."""

    with pytest.raises(ValueError, match="m must be positive"):
        BenchmarkInstance(n=15, a=2, r=4, m=0)
