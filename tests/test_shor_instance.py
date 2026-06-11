"""Unit tests for Shor benchmark instance generators."""

from math import gcd

import pytest

from qft_dynamic.shor_benchmark.types import BenchmarkInstance
from qft_dynamic.tools.instance import (
    CompositeOrderGenerator,
    InstanceGenerator,
    PrimeOrderGenerator,
    RsaStyleGenerator,
    crt_pair,
    find_prime_of_form,
    is_probable_prime,
    random_prime,
)


class TestHelperFunctions:
    """Tests for public and internal helper functions."""

    def test_is_probable_prime_recognizes_small_primes(self) -> None:
        assert is_probable_prime(2) is True
        assert is_probable_prime(3) is True
        assert is_probable_prime(5) is True
        assert is_probable_prime(7) is True
        assert is_probable_prime(11) is True
        assert is_probable_prime(8191) is True

    def test_is_probable_prime_rejects_small_composites(self) -> None:
        assert is_probable_prime(1) is False
        assert is_probable_prime(4) is False
        assert is_probable_prime(6) is False
        assert is_probable_prime(9) is False

    def test_random_prime_generates_prime_with_correct_bit_length(self) -> None:
        prime: int = random_prime(8)
        assert prime.bit_length() == 8
        assert is_probable_prime(prime)

    def test_random_prime_multiple_invocations_produce_primes(self) -> None:
        for _ in range(10):
            prime: int = random_prime(6)
            assert prime.bit_length() == 6
            assert is_probable_prime(prime)

    def test_find_prime_of_form_finds_valid_prime(self) -> None:
        p: int
        k: int
        p, k = find_prime_of_form(d=7, k_start=2, k_step=2, k_max=1000)
        assert is_probable_prime(p)
        assert p == k * 7 + 1
        assert k % 2 == 0

    def test_find_prime_of_form_with_odd_d_and_even_k(self) -> None:
        p: int
        k: int
        p, k = find_prime_of_form(d=5, k_start=2, k_step=2, k_max=1000)
        assert is_probable_prime(p)
        assert p == k * 5 + 1

    def test_find_prime_of_form_raises_when_not_found(self) -> None:
        with pytest.raises(ValueError, match="No prime of form"):
            find_prime_of_form(d=7, k_start=2, k_step=2, k_max=2)

    def test_crt_pair_basic(self) -> None:
        a: int = crt_pair(1, 3, 2, 5)
        assert a == 7
        assert a % 3 == 1
        assert a % 5 == 2

    def test_crt_pair_larger_moduli(self) -> None:
        a_p: int = 3
        p: int = 7
        a_q: int = 5
        q: int = 11
        a: int = crt_pair(a_p, p, a_q, q)
        assert 0 <= a < p * q
        assert a % p == a_p
        assert a % q == a_q


class TestPrimeOrderGenerator:
    """Tests for the prime-order instance generator."""

    def test_generator_is_instance_of_protocol(self) -> None:
        generator: PrimeOrderGenerator = PrimeOrderGenerator(order_bits=4, k_max=1000)
        assert isinstance(generator, InstanceGenerator)

    def test_generates_valid_benchmark_instance(self) -> None:
        generator: PrimeOrderGenerator = PrimeOrderGenerator(
            order_bits=5,
            slack=2,
            k_max=10000,
        )
        instance: BenchmarkInstance = generator.generate()

        assert is_probable_prime(instance.r)
        assert instance.a != 1
        assert pow(instance.a, instance.r, instance.n) == 1

    def test_produces_different_instances_on_repeated_calls(self) -> None:
        generator: PrimeOrderGenerator = PrimeOrderGenerator(
            order_bits=6,
            k_max=50000,
        )
        instance1: BenchmarkInstance = generator.generate()
        instance2: BenchmarkInstance = generator.generate()

        assert instance1 != instance2

    def test_m_value_reasonable(self) -> None:
        generator: PrimeOrderGenerator = PrimeOrderGenerator(
            order_bits=5,
            slack=3,
            k_max=10000,
        )
        instance: BenchmarkInstance = generator.generate()
        expected_m: int = 2 * instance.n.bit_length() + 3
        assert instance.m == expected_m

    def test_order_is_exact_for_prime_r(self) -> None:
        generator: PrimeOrderGenerator = PrimeOrderGenerator(
            order_bits=5,
            slack=2,
            k_max=10000,
        )
        instance: BenchmarkInstance = generator.generate()

        assert is_probable_prime(instance.r)
        assert pow(instance.a, instance.r, instance.n) == 1
        assert instance.a != 1

    def test_raises_after_max_attempts(self) -> None:
        generator: PrimeOrderGenerator = PrimeOrderGenerator(
            order_bits=4,
            k_max=1,
            max_attempts=2,
        )
        with pytest.raises(RuntimeError, match="Failed to generate"):
            generator.generate()


class TestCompositeOrderGenerator:
    """Tests for the composite-order instance generator."""

    def test_generator_is_instance_of_protocol(self) -> None:
        generator: CompositeOrderGenerator = CompositeOrderGenerator(
            factorization=[(2, 1), (3, 1), (5, 1)],
            k_max=10000,
        )
        assert isinstance(generator, InstanceGenerator)

    def test_generates_valid_instance_with_exact_order(self) -> None:
        generator: CompositeOrderGenerator = CompositeOrderGenerator(
            factorization=[(2, 1), (3, 1), (5, 1)],
            slack=2,
            k_max=10000,
        )
        instance: BenchmarkInstance = generator.generate()

        assert instance.r == 30
        assert pow(instance.a, instance.r, instance.n) == 1

        for ell in [2, 3, 5]:
            assert pow(instance.a, instance.r // ell, instance.n) != 1

    def test_generates_instance_with_odd_composite_order(self) -> None:
        generator: CompositeOrderGenerator = CompositeOrderGenerator(
            factorization=[(3, 1), (5, 1)],
            slack=2,
            k_max=100000,
        )
        instance: BenchmarkInstance = generator.generate()

        assert instance.r == 15
        assert pow(instance.a, instance.r, instance.n) == 1

        for ell in [3, 5]:
            assert pow(instance.a, instance.r // ell, instance.n) != 1

    def test_m_value_consistent(self) -> None:
        generator: CompositeOrderGenerator = CompositeOrderGenerator(
            factorization=[(2, 1), (3, 1)],
            slack=4,
            k_max=10000,
        )
        instance: BenchmarkInstance = generator.generate()
        expected_m: int = 2 * instance.n.bit_length() + 4
        assert instance.m == expected_m

    def test_raises_after_max_attempts(self) -> None:
        generator: CompositeOrderGenerator = CompositeOrderGenerator(
            factorization=[(2, 1), (3, 1), (5, 1)],
            k_max=0,
            max_attempts=2,
        )
        with pytest.raises(RuntimeError, match="Failed to generate"):
            generator.generate()


class TestRsaStyleGenerator:
    """Tests for the RSA-style instance generator."""

    def test_generator_is_instance_of_protocol(self) -> None:
        generator: RsaStyleGenerator = RsaStyleGenerator(
            order_bits=10,
            k_max=100000,
        )
        assert isinstance(generator, InstanceGenerator)

    def test_generates_valid_instance_with_r_divisible_by_2(self) -> None:
        generator: RsaStyleGenerator = RsaStyleGenerator(
            order_bits=10,
            slack=2,
            k_max=100000,
        )
        instance: BenchmarkInstance = generator.generate()

        assert instance.r > 0
        assert instance.r % 2 == 0

        assert pow(instance.a, instance.r, instance.n) == 1

    def test_gcd_step_recovers_nontrivial_factors(self) -> None:
        generator: RsaStyleGenerator = RsaStyleGenerator(
            order_bits=10,
            slack=2,
            k_max=100000,
        )
        instance: BenchmarkInstance = generator.generate()

        half_order: int = instance.r // 2
        g_minus: int = gcd(pow(instance.a, half_order, instance.n) - 1, instance.n)
        g_plus: int = gcd(pow(instance.a, half_order, instance.n) + 1, instance.n)

        assert 1 < g_minus < instance.n
        assert 1 < g_plus < instance.n
        assert g_minus * g_plus == instance.n

    def test_half_order_not_congruent_to_one_or_minus_one(self) -> None:
        generator: RsaStyleGenerator = RsaStyleGenerator(
            order_bits=10,
            slack=2,
            k_max=100000,
        )
        instance: BenchmarkInstance = generator.generate()

        half_pow: int = pow(instance.a, instance.r // 2, instance.n)
        assert half_pow != 1
        assert half_pow != instance.n - 1

    def test_m_value_consistent(self) -> None:
        generator: RsaStyleGenerator = RsaStyleGenerator(
            order_bits=10,
            slack=4,
            k_max=100000,
        )
        instance: BenchmarkInstance = generator.generate()
        expected_m: int = 2 * instance.n.bit_length() + 4
        assert instance.m == expected_m

    def test_raises_after_max_attempts(self) -> None:
        generator: RsaStyleGenerator = RsaStyleGenerator(
            order_bits=8,
            k_max=1,
            max_attempts=2,
        )
        with pytest.raises(RuntimeError, match="Failed to generate"):
            generator.generate()
