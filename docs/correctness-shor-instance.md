# 已知 Order Big Instance 构造的数学正确性说明

## 范围

本文解释 `plan.md` 中三类 instance 生成方案的数学正确性：

1. **prime-order instance**：最简单的素数阶实例；
2. **composite-order instance**：已知分解的合数阶实例；
3. **RSA-style instance**：复合模数 `N = p*q`，已知全局 order，并保证 Shor factoring 的 gcd 步骤成功。

目标都是构造：

```python
BenchmarkInstance(n=N, a=a, r=r, m=m)
```

并保证：

\[
\operatorname{ord}_N(a)=r.
\]

其中 `m` 是采样和后处理参数，不参与数论正确性证明。

---

## 0. 公共数学事实

### 0.1 Order 定义

当 \(\gcd(a,N)=1\) 时，定义：

\[
\operatorname{ord}_N(a)
\]

为最小正整数 \(r\)，使得：

\[
a^r\equiv1\pmod N.
\]

---

### 0.2 素数模数下的投影构造

设目标阶为 \(d\)。如果找到素数：

\[
p=k d+1,
\]

则：

\[
p-1=kd.
\]

任取非零元素 \(h\in\mathbb F_p^*\)，令：

\[
a=h^k\bmod p.
\]

则：

\[
a^d=(h^k)^d=h^{kd}=h^{p-1}\equiv1\pmod p.
\]

因此：

\[
\operatorname{ord}_p(a)\mid d.
\]

这是所有素数模数构造的核心。它不需要寻找原根，也不需要分解 \(k\)。

---

### 0.3 Exact-order 判定

假设已经知道：

\[
\operatorname{ord}_p(a)\mid d.
\]

并且 \(d\) 的不同素因子为：

\[
\ell_1,\ell_2,\dots,\ell_t.
\]

那么：

\[
\operatorname{ord}_p(a)=d
\]

当且仅当：

\[
\forall \ell_i\mid d,\qquad a^{d/\ell_i}\not\equiv1\pmod p.
\]

代码层面的检查是：

```python
assert pow(a, d, p) == 1
for ell in distinct_prime_factors(d):
    assert pow(a, d // ell, p) != 1
```

在本构造中，`pow(a, d, p) == 1` 通常由构造自动保证，核心检查是所有 `d // ell` 指数。

---

### 0.4 Exact-order 判定为什么正确

令：

\[
e=\operatorname{ord}_p(a).
\]

已知：

\[
e\mid d.
\]

如果 \(e=d\)，那么对任意素因子 \(\ell\mid d\)，都有：

\[
e\nmid d/\ell,
\]

所以：

\[
a^{d/\ell}\not\equiv1\pmod p.
\]

反过来，如果 \(e<d\)，由于 \(e\mid d\)，可写成：

\[
d=e u,
\qquad u>1.
\]

取任意素因子 \(\ell\mid u\)，则：

\[
\frac d\ell=e\cdot\frac u\ell
\]

仍然是 \(e\) 的倍数，因此：

\[
a^{d/\ell}\equiv1\pmod p.
\]

所以只要所有 \(a^{d/\ell}\neq1\) 检查都通过，就必然有：

\[
\operatorname{ord}_p(a)=d.
\]

---

## 1. Prime-order instance

### 构造

选择素数目标 order：

\[
r.
\]

搜索素数：

\[
p=k r+1.
\]

随机取非零元素：

\[
h\in\mathbb F_p^*.
\]

令：

\[
a=h^k\bmod p.
\]

接受条件：

\[
a\neq1.
\]

最后设置：

\[
N=p.
\]

---

### 正确性

因为：

\[
p-1=kr,
\]

所以：

\[
a^r=(h^k)^r=h^{kr}=h^{p-1}\equiv1\pmod p.
\]

因此：

\[
\operatorname{ord}_p(a)\mid r.
\]

由于 \(r\) 是素数，\(\operatorname{ord}_p(a)\) 只能是：

\[
1\quad\text{或}\quad r.
\]

接受条件 \(a\neq1\) 排除了 order 为 \(1\) 的情况，因此：

\[
\operatorname{ord}_p(a)=r.
\]

也就是：

\[
\operatorname{ord}_N(a)=r.
\]

---

### 成功概率

如果 \(h\) 从整个 \(\mathbb F_p^*\) 中均匀采样，则映射：

\[
h\mapsto h^k
\]

把 \(\mathbb F_p^*\) 均匀映射到阶为 \(r\) 的子群中。该子群中只有一个元素是 \(1\)。因此：

\[
\Pr[a=1]=\frac1r,
\]

\[
\Pr[\text{accept}]=1-\frac1r.
\]

对于 256-bit 素数 \(r\)，失败概率约为：

\[
2^{-256}.
\]

如果实现中从 `[2, p - 2]` 采样而不是从整个 \(\mathbb F_p^*\) 采样，精确概率会有极小变化，但在密码学大小下可以忽略。

---

## 2. Composite-order instance

### 构造

选择合数目标 order，并且已知其因式分解：

\[
r=\prod_i \ell_i^{e_i}.
\]

搜索素数：

\[
p=k r+1.
\]

随机取非零元素：

\[
h\in\mathbb F_p^*.
\]

令：

\[
a=h^k\bmod p.
\]

接受条件：

\[
\forall \ell_i\mid r,\qquad a^{r/\ell_i}\not\equiv1\pmod p.
\]

最后设置：

\[
N=p.
\]

---

### 正确性

由于：

\[
p-1=kr,
\]

有：

\[
a^r=(h^k)^r=h^{kr}=h^{p-1}\equiv1\pmod p.
\]

因此：

\[
\operatorname{ord}_p(a)\mid r.
\]

接受条件正是 Section 0.3 中的 exact-order 判定：

\[
\forall \ell_i\mid r,\qquad a^{r/\ell_i}\not\equiv1\pmod p.
\]

所以：

\[
\operatorname{ord}_p(a)=r.
\]

也就是：

\[
\operatorname{ord}_N(a)=r.
\]

---

### 需要知道哪些因式分解

对于 composite \(r\)，必须知道 \(r\) 的不同素因子：

\[
\ell_i\mid r.
\]

这些因子用于 exact-order 检查。

不需要知道 \(k\) 的因式分解。构造只需要 \(k\) 的数值，用来计算：

\[
a=h^k\bmod p.
\]

---

### 成功概率

如果 \(h\) 从 \(\mathbb F_p^*\) 中均匀采样，则 \(a=h^k\) 在阶为 \(r\) 的子群中均匀分布。

阶为 \(r\) 的循环群中，exact order 为 \(r\) 的元素数量是：

\[
\varphi(r).
\]

因此：

\[
\Pr[\operatorname{ord}_p(a)=r]
=\frac{\varphi(r)}r
=\prod_{\ell\mid r}\left(1-\frac1\ell\right).
\]

例如推荐形状：

\[
r=2^e q_1q_2q_3,
\]

其中 \(q_i\) 是大奇素数，则：

\[
\frac{\varphi(r)}r
=\frac12
\left(1-\frac1{q_1}\right)
\left(1-\frac1{q_2}\right)
\left(1-\frac1{q_3}\right)
\approx\frac12.
\]

所以期望重采样次数约为 \(2\)。

---

## 3. RSA-style instance

### 构造概览

选择两个不同的奇素数：

\[
s,t.
\]

构造复合模数：

\[
N=pq.
\]

目标是让局部 order 满足：

\[
\operatorname{ord}_p(a_p)=s,
\]

\[
\operatorname{ord}_q(a_q)=2t.
\]

CRT 合并后，全局 order 为：

\[
r=\operatorname{ord}_N(a)=\operatorname{lcm}(s,2t)=2st.
\]

该构造还会保证 Shor factoring 的 classical gcd 步骤成功。

---

### 3.1 构造 \(p\) 侧局部元素

搜索素数：

\[
p=k_p s+1.
\]

随机取非零元素：

\[
h_p\in\mathbb F_p^*.
\]

令：

\[
a_p=h_p^{k_p}\bmod p.
\]

接受条件：

\[
a_p\neq1.
\]

由于 \(s\) 是素数，根据 prime-order case：

\[
\operatorname{ord}_p(a_p)=s.
\]

---

### 3.2 构造 \(q\) 侧局部元素

搜索素数：

\[
q=k_q(2t)+1.
\]

先构造一个 order 为 \(t\) 的元素。随机取非零元素：

\[
h_q\in\mathbb F_q^*.
\]

令：

\[
b_q=h_q^{(q-1)/t}\bmod q.
\]

则：

\[
b_q^t=h_q^{q-1}\equiv1\pmod q.
\]

因此：

\[
\operatorname{ord}_q(b_q)\mid t.
\]

由于 \(t\) 是素数，接受条件：

\[
b_q\neq1
\]

保证：

\[
\operatorname{ord}_q(b_q)=t.
\]

然后令：

\[
a_q=-b_q\bmod q.
\]

因为 \(t\) 是奇数：

\[
a_q^t=(-b_q)^t=-b_q^t\equiv-1\pmod q.
\]

同时：

\[
a_q^{2t}\equiv1\pmod q.
\]

所以：

\[
\operatorname{ord}_q(a_q)=2t.
\]

也可以用 exact-order test 表述为：

\[
a_q^t\not\equiv1\pmod q,
\]

\[
a_q^2\not\equiv1\pmod q.
\]

因此 \(a_q\) 的阶确实是 \(2t\)。

---

### 3.3 CRT 合并

通过 CRT 构造 \(a\bmod N\)，其中：

\[
N=pq,
\]

并满足：

\[
a\equiv a_p\pmod p,
\]

\[
a\equiv a_q\pmod q.
\]

由于 \(a_p\neq0\pmod p\)，且 \(a_q\neq0\pmod q\)，所以：

\[
\gcd(a,N)=1.
\]

---

### 3.4 全局 order 正确性

由 CRT：

\[
a^e\equiv1\pmod N
\]

当且仅当：

\[
a^e\equiv1\pmod p
\]

且：

\[
a^e\equiv1\pmod q.
\]

这等价于：

\[
s\mid e
\]

且：

\[
2t\mid e.
\]

满足这两个条件的最小正整数是：

\[
\operatorname{lcm}(s,2t).
\]

由于 \(s,t\) 是不同奇素数：

\[
\operatorname{lcm}(s,2t)=2st.
\]

因此：

\[
\operatorname{ord}_N(a)=2st.
\]

所以 ground-truth order 是：

\[
r=2st.
\]

---

### 3.5 GCD factoring 步骤正确性

Shor factoring 的 classical step 使用：

\[
a^{r/2}=a^{st}.
\]

模 \(p\) 下，由于 \(\operatorname{ord}_p(a_p)=s\)：

\[
a^{st}\equiv a_p^{st}=(a_p^s)^t\equiv1\pmod p.
\]

模 \(q\) 下，由于 \(a_q^t\equiv-1\pmod q\)：

\[
a^{st}\equiv a_q^{st}=(a_q^t)^s\equiv(-1)^s\equiv-1\pmod q.
\]

这里用到了 \(s\) 是奇数。

因此：

\[
a^{r/2}\equiv1\pmod p,
\]

\[
a^{r/2}\equiv-1\pmod q.
\]

所以：

\[
p\mid a^{r/2}-1,
\qquad
q\nmid a^{r/2}-1.
\]

于是：

\[
\gcd(a^{r/2}-1,N)=p.
\]

同理：

\[
q\mid a^{r/2}+1,
\qquad
p\nmid a^{r/2}+1.
\]

于是：

\[
\gcd(a^{r/2}+1,N)=q.
\]

所以 RSA-style generator 不仅生成了已知 order 的 instance，还保证 classical gcd factorization step 成功。

---

### 3.6 成功概率

\(p\) 侧采样接受概率：

\[
\Pr[a_p\neq1]=1-\frac1s.
\]

\(q\) 侧采样接受概率：

\[
\Pr[b_q\neq1]=1-\frac1t.
\]

对于密码学大小的 \(s,t\)，两者都近似为 \(1\)。

搜索如下形式的素数：

\[
p=k_p s+1,
\]

\[
q=k_q(2t)+1
\]

是启发式 prime search。只要最终找到的 \(p,q\) 确实为素数，上述 order 构造和 gcd 成功性质就是确定成立的。

---

## 4. 三类构造的正确性条件总结

| 类型 | 模数 | ground-truth order | 关键接受条件 |
|---|---:|---:|---|
| Prime-order | \(N=p\) | 素数 \(r\) | \(a=h^k\neq1\) |
| Composite-order | \(N=p\) | 已知分解的 \(r\) | 对所有 \(\ell\mid r\)，\(a^{r/\ell}\neq1\) |
| RSA-style | \(N=pq\) | \(r=2st\) | \(a_p\neq1\)，\(b_q\neq1\)，CRT 合并 |

三类构造都不通过暴力搜索 order。它们都是先构造合适的群结构，再用已知因式分解和局部 order 关系强制得到目标 order。

---

## 5. 建议保留的实现级断言

### Prime-order instance

```python
assert gcd(a, N) == 1
assert pow(a, r, N) == 1
assert a != 1
```

### Composite-order instance

```python
assert gcd(a, N) == 1
assert pow(a, r, N) == 1
for ell in distinct_prime_factors_r:
    assert pow(a, r // ell, N) != 1
```

### RSA-style instance

```python
assert gcd(a, N) == 1
assert pow(a, r, N) == 1
assert pow(a, r // 2, N) not in (1, N - 1)

g_minus = gcd(pow(a, r // 2, N) - 1, N)
g_plus = gcd(pow(a, r // 2, N) + 1, N)

assert g_minus in (p, q)
assert g_plus in (p, q)
assert g_minus * g_plus == N
```

这些断言不是用来暴力发现 order，而是用已知构造参数验证生成结果符合预期。
