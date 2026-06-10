# Plan: Big Instance Generation for Depolarized Shor Order-Finding Benchmark

## Scope

Generate three families of known-order benchmark instances:

1. **Prime-order instance**: simplest sanity test.
2. **Composite-order instance**: tests partial denominator recovery and LCM aggregation.
3. **RSA-style instance**: composite modulus `N = p*q`, known global order, and guaranteed Shor-style gcd factorization success.

ECC-style discrete-log instances are out of scope.

---

## Common conventions

A generated instance should provide:

```python
BenchmarkInstance(
    n=N,
    a=a,
    r=r,
    m=m,
)
```

Recommended `m` rule:

```python
m = 2 * N.bit_length() + slack
```
where set `slack` default to `2`.

### Required helper functions

Implement these first:

```python
is_probable_prime(x) -> bool
random_prime(bits) -> int
find_prime_of_form(d, k_start=1, k_step=1, k_max=None) -> tuple[p, k]
crt_pair(ap, p, aq, q) -> int
factorization_to_distinct_primes(factorization) -> list[int]
```

`find_prime_of_form(d)` searches for:

```python
p = k * d + 1
```

with `p` prime.

Parity handling:

```python
if d is odd:
    k_start = 2
    k_step = 2
else:
    k_start = 1
    k_step = 1
```

This avoids even `p > 2`.

### Output verification

Every generator should perform final assertions:

```python
assert 1 < a < N
assert gcd(a, N) == 1
assert pow(a, r, N) == 1
```

For exact order, also verify:

```python
for ell in distinct_prime_factors_of_r:
    assert pow(a, r // ell, N) != 1
```

For RSA-style instances, use the RSA-specific checks listed below.

---

## 1. Prime-order instance

### Purpose

Use this as the simplest large-order sanity test. It is appropriate for checking whether the benchmark can handle a large denominator, such as a 256-bit order.

This case is intentionally easy: a single successful shot often recovers the whole order when noise is low.

### Inputs

```python
order_bits: int      # e.g. 256
slack: int           # e.g. 2
k_max: int | None    # optional search limit
```

### Construction

1. Generate a prime `r` with `order_bits` bits:

   ```python
   r = random_prime(order_bits)
   ```

2. Search for prime:

   ```python
   p = k * r + 1
   ```

   Since `r` is odd, use even `k`:

   ```python
   k = 2, 4, 6, ...
   ```

3. Sample random `h` in `F_p*`:

   ```python
   h = random integer in [2, p - 2]
   ```

4. Set:

   ```python
   a = pow(h, k, p)
   ```

5. Accept if:

   ```python
   a != 1
   ```

6. Set:

   ```python
   N = p
   m = 2 * N.bit_length() + slack
   ```

7. Return:

   ```python
   BenchmarkInstance(n=N, a=a, r=r, m=m)
   ```

### Pseudocode

```python
def generate_prime_order_instance(order_bits=256, slack=2, k_max=None):
    r = random_prime(order_bits)

    p, k = find_prime_of_form(
        d=r,
        k_start=2,
        k_step=2,
        k_max=k_max,
    )

    while True:
        h = random_integer(2, p - 2)
        a = pow(h, k, p)
        if a != 1:
            break

    N = p
    m = 2 * N.bit_length() + slack

    assert pow(a, r, N) == 1
    assert a != 1

    return BenchmarkInstance(n=N, a=a, r=r, m=m)
```

### Complexity and success probability

Let `b = order_bits`.

| Step | Expected cost |
|---|---:|
| Generate prime `r` | heuristic `O(b)` primality tests |
| Find `p = k*r + 1` | heuristic `O(log p)` primality tests |
| Sample `a` | expected 1 trial |
| Modular exponentiation | polynomial in `log p` |

Random acceptance probability for `a`:

```text
Pr[a != 1] = 1 - 1/r
```

For 256-bit prime `r`, failure probability is about `2^-256`.

---

## 2. Composite-order instance

### Purpose

Use this to stress-test denominator recovery and LCM aggregation. Unlike prime `r`, a single ideal sample may recover only a proper divisor of `r`.

### Recommended shape of `r`

Use a composite `r` with known factorization, for example:

```text
r = 2^e * q1 * q2 * q3
```

where:

```text
e >= 1
q1, q2, q3 are distinct odd primes
```

Example parameterization:

```python
order_bits = 256
small_power_two = 8
num_large_prime_factors = 3
```

Choose the bit sizes of `q1, q2, q3` so that the final `r` has approximately `order_bits` bits.

### Inputs

```python
factorization: list[tuple[int, int]]
# Example: [(2, e), (q1, 1), (q2, 1), (q3, 1)]

slack: int
k_max: int | None
```

### Construction

1. Compute:

   ```python
   r = product(ell ** exp for ell, exp in factorization)
   distinct_ells = [ell for ell, exp in factorization]
   ```

2. Search for prime:

   ```python
   p = k * r + 1
   ```

   If `r` is even, `k` can be any positive integer. If `r` is odd, use even `k`.

3. Repeatedly sample:

   ```python
   h = random integer in [2, p - 2]
   a = pow(h, k, p)
   ```

4. Accept if exact-order test passes:

   ```python
   for ell in distinct_ells:
       pow(a, r // ell, p) != 1
   ```

5. Set:

   ```python
   N = p
   m = 2 * N.bit_length() + slack
   ```

6. Return:

   ```python
   BenchmarkInstance(n=N, a=a, r=r, m=m)
   ```

### Pseudocode

```python
def generate_composite_order_instance(factorization, slack=2, k_max=None):
    r = 1
    distinct_ells = []

    for ell, exp in factorization:
        r *= ell ** exp
        distinct_ells.append(ell)

    if r % 2 == 0:
        k_start, k_step = 1, 1
    else:
        k_start, k_step = 2, 2

    p, k = find_prime_of_form(
        d=r,
        k_start=k_start,
        k_step=k_step,
        k_max=k_max,
    )

    while True:
        h = random_integer(2, p - 2)
        a = pow(h, k, p)

        ok = True
        for ell in distinct_ells:
            if pow(a, r // ell, p) == 1:
                ok = False
                break

        if ok:
            break

    N = p
    m = 2 * N.bit_length() + slack

    assert pow(a, r, N) == 1
    for ell in distinct_ells:
        assert pow(a, r // ell, N) != 1

    return BenchmarkInstance(n=N, a=a, r=r, m=m)
```

### Complexity and success probability

Let `b = r.bit_length()`.

| Step | Expected cost |
|---|---:|
| Build `r` from known factorization | negligible |
| Find `p = k*r + 1` | heuristic `O(log p)` primality tests |
| Sample exact-order `a` | geometric trials |
| Each exact-order test | `omega(r)` modular exponentiations, where `omega(r)` is the number of distinct prime factors |

Random acceptance probability:

```text
Pr[ord_p(a) = r] = phi(r) / r
                 = product over ell | r of (1 - 1/ell)
```

For the recommended shape:

```text
r = 2^e * q1 * q2 * q3
```

with large `q_i`, this is approximately:

```text
1/2
```

So expected rejection-sampling trials are approximately `2`.

---

## 3. RSA-style instance

### Purpose

Use this to test composite-modulus order finding and the Shor-style classical gcd step.

This generator constructs:

```text
N = p * q
r = ord_N(a)
```

and guarantees:

```python
gcd(a^(r//2) - 1, N) gives one factor
gcd(a^(r//2) + 1, N) gives the other factor
```

### Recommended shape

Choose two distinct odd primes:

```text
s, t
```

Use local orders:

```text
r_p = s
r_q = 2*t
```

Then the global order is:

```text
r = lcm(s, 2*t) = 2*s*t
```

For an approximately `order_bits`-bit global order, choose:

```python
s_bits = floor((order_bits - 1) / 2)
t_bits = ceil((order_bits - 1) / 2)
```

### Inputs

```python
order_bits: int      # e.g. 256
slack: int           # e.g. 2
k_max: int | None
```

### Construction

#### Step A: Generate local target orders

1. Generate distinct odd primes:

   ```python
   s = random_prime(s_bits)
   t = random_prime(t_bits)
   assert s != t
   ```

2. Set:

   ```python
   r_p = s
   r_q = 2 * t
   r = 2 * s * t
   ```

#### Step B: Construct `p` and local element `a_p`

1. Search for prime:

   ```python
   p = k_p * s + 1
   ```

   Since `s` is odd, use even `k_p`.

2. Sample:

   ```python
   h_p = random integer in [2, p - 2]
   a_p = pow(h_p, k_p, p)
   ```

3. Accept if:

   ```python
   a_p != 1
   ```

#### Step C: Construct `q` and local element `a_q`

1. Search for prime:

   ```python
   q = k_q * (2*t) + 1
   ```

   Here `k_q` can start at `1` and increment by `1`.

2. Construct an element `b_q` of order `t`:

   ```python
   h_q = random integer in [2, q - 2]
   b_q = pow(h_q, (q - 1) // t, q)
   ```

3. Accept if:

   ```python
   b_q != 1
   ```

4. Set:

   ```python
   a_q = (-b_q) % q
   ```

   This gives local order:

   ```text
   ord_q(a_q) = 2*t
   ```

#### Step D: CRT combine

Compute:

```python
N = p * q
```

Use CRT to construct `a` satisfying:

```text
a ≡ a_p mod p
a ≡ a_q mod q
```

Implementation formula:

```python
inv_q_mod_p = pow(q, -1, p)
inv_p_mod_q = pow(p, -1, q)

a = (a_p * q * inv_q_mod_p + a_q * p * inv_p_mod_q) % N
```

Set:

```python
m = 2 * N.bit_length() + slack
```

Return:

```python
BenchmarkInstance(n=N, a=a, r=r, m=m)
```

Optionally also return trapdoor factors for testing:

```python
metadata = {
    "p": p,
    "q": q,
    "s": s,
    "t": t,
    "k_p": k_p,
    "k_q": k_q,
}
```

### Pseudocode

```python
def generate_rsa_style_instance(order_bits=256, slack=2, k_max=None):
    s_bits = (order_bits - 1) // 2
    t_bits = order_bits - 1 - s_bits

    while True:
        s = random_prime(s_bits)
        t = random_prime(t_bits)
        if s != t and s % 2 == 1 and t % 2 == 1:
            break

    # p side: local order s
    p, k_p = find_prime_of_form(
        d=s,
        k_start=2,
        k_step=2,
        k_max=k_max,
    )

    while True:
        h_p = random_integer(2, p - 2)
        a_p = pow(h_p, k_p, p)
        if a_p != 1:
            break

    # q side: local order 2*t
    q, k_q = find_prime_of_form(
        d=2 * t,
        k_start=1,
        k_step=1,
        k_max=k_max,
    )

    while q == p:
        q, k_q = find_prime_of_form(
            d=2 * t,
            k_start=k_q + 1,
            k_step=1,
            k_max=k_max,
        )

    while True:
        h_q = random_integer(2, q - 2)
        b_q = pow(h_q, (q - 1) // t, q)
        if b_q != 1:
            break

    a_q = (-b_q) % q

    N = p * q

    inv_q_mod_p = pow(q, -1, p)
    inv_p_mod_q = pow(p, -1, q)
    a = (a_p * q * inv_q_mod_p + a_q * p * inv_p_mod_q) % N

    r = 2 * s * t
    m = 2 * N.bit_length() + slack

    # local checks
    assert pow(a_p, s, p) == 1
    assert a_p != 1
    assert pow(a_q, 2 * t, q) == 1
    assert pow(a_q, t, q) == q - 1

    # global checks
    assert gcd(a, N) == 1
    assert pow(a, r, N) == 1
    assert pow(a, r // 2, N) not in (1, N - 1)

    g_minus = gcd(pow(a, r // 2, N) - 1, N)
    g_plus = gcd(pow(a, r // 2, N) + 1, N)

    assert g_minus in (p, q)
    assert g_plus in (p, q)
    assert g_minus * g_plus == N

    return BenchmarkInstance(n=N, a=a, r=r, m=m)
```

### Complexity and success probability

Let `b = order_bits`.

| Step | Expected cost |
|---|---:|
| Generate `s`, `t` | heuristic `O(b)` primality tests total |
| Find `p = k_p*s + 1` | heuristic `O(log p)` primality tests |
| Find `q = k_q*(2*t) + 1` | heuristic `O(log q)` primality tests |
| Sample `a_p` | expected 1 trial |
| Sample `b_q` | expected 1 trial |
| CRT combine | polynomial in `log N` |
| Final verification | constant number of modular exponentiations and gcds |

Sampling acceptance probabilities:

```text
Pr[a_p accepted] = 1 - 1/s
Pr[b_q accepted] = 1 - 1/t
```

For cryptographic-size `s` and `t`, both are essentially `1`.

Prime-search success for `p = k*d + 1` is heuristic. In practice, enumerate `k` up to a large cap such as:

```python
k_max = 100_000
```

If no prime is found, regenerate the target prime(s) and retry.

---

## Recommended test matrix

Use three tiers:

| Tier | Generator | Suggested order bits | Purpose |
|---|---|---:|---|
| 1 | Prime-order | 128, 256, 512 | Large denominator sanity test |
| 2 | Composite-order | 128, 256, 512 | LCM aggregation stress test |
| 3 | RSA-style | 128, 256, 512 | Composite modulus + gcd factoring step |

For runtime-sensitive Monte Carlo tests, start with:

```text
order_bits = 128
slack = 2
```

Then scale to:

```text
order_bits = 256
```

after correctness is confirmed.

---

## Failure handling policy

Use bounded randomized generation:

```python
for attempt in range(max_attempts):
    try to generate instance
    if success:
        return instance
raise RuntimeError("failed to generate instance under current bounds")
```

Recommended defaults:

```python
k_max = 100_000
max_attempts = 20
slack = 8
```

If prime search fails repeatedly:

1. Regenerate `r`, or regenerate `s, t` for RSA-style.
2. Increase `k_max`.
3. Reduce the number of small factors in composite `r` if exact-order sampling becomes slow.

---

## Minimal implementation order

1. Implement primality and random-prime utilities.
2. Implement `find_prime_of_form(d)`.
3. Implement prime-order generator.
4. Implement composite-order generator.
5. Implement CRT helper.
6. Implement RSA-style generator.
7. Add verification tests for all three generators.
8. Add benchmark integration wrapper that outputs `BenchmarkInstance` plus optional metadata.
