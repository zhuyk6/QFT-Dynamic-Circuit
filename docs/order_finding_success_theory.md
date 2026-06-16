# Order-finding benchmark 的成功率理论说明

本文整理 Shor order-finding benchmark 中 **finite-$Q$ ideal**、**arithmetic baseline**、**uniform random baseline** 与 **depolarized 噪声模型**下的 strict order-recovery 成功率。正文只给出核心结论和解释；数学证明放在附录中。

---

## 0. 适用范围与统一符号

本文只分析 **strict order-recovery** 指标，即单样本做 continued fractions，$K$ 个样本做 subset-LCM，再用模幂条件 $a^L \equiv 1 \pmod N$ 验证，最后输出最小通过验证的候选 $L$。若输出等于真实 order $r$，则记为成功。

### 0.1 基本实例

给定 Shor order-finding 实例

$$
(N,a,r,m),
$$

其中

$$
r=\operatorname{ord}_N(a),\qquad Q=2^m.
$$

控制寄存器测量结果记为

$$
y\in\{0,1,\dots,Q-1\}.
$$

continued fractions 后处理记为

$$
\mathrm{CF}_N(y/Q)=\frac{p_Q(y)}{q_Q(y)},
$$

其中分母上限为 $N-1$，并且 $p_Q(y),q_Q(y)$ 互素。

### 0.2 有用分母

定义

$$
D_Q(y)=
\begin{cases}
q_Q(y), & q_Q(y)\mid r,\\
1, & q_Q(y)\nmid r.
\end{cases}
$$

解释：

- 若 $q_Q(y)\mid r$，该样本可能贡献真实 order $r$ 的一部分素因子信息；
- 若 $q_Q(y)\nmid r$，该样本不可能参与构造一个 LCM 正好等于 $r$ 的 subset，因此在分析 **success event** 时可视为无信息样本；
- $D_Q(y)$ 只用于分析“是否成功输出 $r$”。它不会完整描述 wrong/null 失效模式，因为 $q_Q(y)\nmid r$ 仍可能导致 strict 后处理器输出错误的 $r$ 的倍数。

### 0.3 常用数论符号

| 符号 | 含义 |
|---|---|
| $\mathcal P(r)$ | $r$ 的不同素因子集合，即 $\mathcal P(r)=\{p:p\mid r,\ p\text{ prime}\}$ |
| $\operatorname{rad}(r)$ | $r$ 的 square-free kernel，$\operatorname{rad}(r)=\prod_{p\mid r}p$ |
| $\omega(r)$ | $r$ 的不同素因子个数，$\omega(r)=\|\mathcal P(r)\|$ |
| $\tau(r)$ | $r$ 的正因子个数 |
| $\varphi(r)$ | Euler totient function |
| $\mu(n)$ | Möbius function |

---


## 1. 无噪声 arithmetic ideal 的成功率

在 arithmetic ideal 中，每个样本等价于直接得到相位

$$
\frac{s}{r},\qquad s\sim\mathrm{Unif}\{0,1,\dots,r-1\}.
$$

continued fractions 返回的是 $s/r$ 的最简分母：

$$
q_s=\frac{r}{\gcd(s,r)}.
$$

$K$ 个独立样本的 strict order-recovery 成功率为

$$
\boxed{
P_{\mathrm{arith}}^{(K)}
=
\Pr\bigl[\operatorname{lcm}(q_{s_1},\dots,q_{s_K})=r\bigr]
=
\prod_{p\mid r}\left(1-p^{-K}\right).
}
$$

这个公式说明：

1. 成功率只依赖 $r$ 的**不同素因子集合** $\mathcal P(r)$；
2. 不依赖 $r$ 中各素因子的幂指数；
3. 小素因子的影响最大，尤其是 $p=2$；
4. 增大 $K$ 的作用是提升“所有素因子都被至少一个样本覆盖”的概率。

### 1.1 $K=1$ 特例

当 $K=1$ 时，

$$
P_{\mathrm{arith}}^{(1)}
=
\prod_{p\mid r}\left(1-\frac1p\right)
=
\frac{\varphi(r)}{r}.
$$

若 $r$ 是大素数，则

$$
P_{\mathrm{arith}}^{(1)}=1-\frac1r\approx 1.
$$

若

$$
r=2pq
$$

且 $p,q$ 是较大的奇素数，则

$$
P_{\mathrm{arith}}^{(1)}
=
\frac12\left(1-\frac1p\right)\left(1-\frac1q\right)
\approx \frac12.
$$

这解释了 prime order instance 的 $K=1$ 成功率接近 $1$，而 $r=2pq$ instance 的 $K=1$ 成功率接近 $0.5$。

---

## 2. finite-$Q$ ideal 的精确成功率

finite-$Q$ ideal 下，对固定 eigenphase 标签 $s$，测量分布为

$$
P_Q(y\mid s)
=
\frac{1}{Q^2}
\left|
\sum_{x=0}^{Q-1}
\exp\!\left(2\pi i x\left(\frac{s}{r}-\frac{y}{Q}\right)\right)
\right|^2.
$$

令 $s$ 均匀混合后的单样本边缘分布为

$$
\pi_Q(y)=\frac1r\sum_{s=0}^{r-1}P_Q(y\mid s).
$$

对每个 $d\mid r$，定义

$$
\boxed{
B_Q(d)=\Pr_{Y\sim\pi_Q}\bigl[D_Q(Y)\mid d\bigr].
}
$$

则 finite-$Q$ ideal 的 $K$-sample strict 成功率为

$$
\boxed{
P_Q^{(K)}
=
\sum_{d\mid r}
\mu\!\left(\frac{r}{d}\right)
B_Q(d)^K.
}
$$

这里 $B_Q(d)$ 是 divisor lattice 上的累积分布函数。它表示：单个样本给出的“有用分母”是否只含有 $d$ 中已经包含的素因子和幂次。

### 2.1 finite-$Q$ 与 arithmetic 的关系

在 arithmetic limit 中，

$$
B_Q(d)\longrightarrow B_\infty(d)=\frac{d}{r}.
$$

因此

$$
P_Q^{(K)}
\longrightarrow
\sum_{d\mid r}
\mu\!\left(\frac{r}{d}\right)
\left(\frac{d}{r}\right)^K
=
\prod_{p\mid r}\left(1-p^{-K}\right).
$$

所以，当 $Q$ 足够大并且 continued fractions 能稳定恢复理想最简分母时，可以近似认为

$$
\boxed{
P_Q^{(K)}\approx P_{\mathrm{arith}}^{(K)}.
}
$$

当

$$
Q=2^m,\qquad m\ge 2\lceil\log_2 N\rceil,
$$

即 $Q\gtrsim N^2$ 时，通常已经处在 Shor continued-fractions 后处理所需的分辨率尺度。实际是否接近 arithmetic baseline，还取决于 finite-$Q$ Dirichlet kernel 的概率质量是否主要落在正确的 continued-fractions cell 内。

---

## 3. depolarized 噪声下的精确成功率

考虑简单 depolarized 噪声模型

$$
\rho_{\lambda}
=(1-\lambda)\rho+\lambda\frac{I}{Q},
\qquad 0\le\lambda\le1.
$$

测量分布变为

$$
P_{\lambda,Q}(y\mid s)
=(1-\lambda)P_Q(y\mid s)+\lambda\frac1Q.
$$

令 uniform random 分布为

$$
U_Q(y)=\frac1Q.
$$

定义

$$
\boxed{
B_U(d)=\Pr_{Y\sim U_Q}\bigl[D_Q(Y)\mid d\bigr].
}
$$

则 noisy 单样本的累积分布函数为

$$
B_{\lambda,Q}(d)
=(1-\lambda)B_Q(d)+\lambda B_U(d).
$$

因此 depolarized 噪声下的 exact finite-$Q$ strict 成功率为

$$
\boxed{
P_{\lambda,Q}^{(K)}
=
\sum_{d\mid r}
\mu\!\left(\frac{r}{d}\right)
\left[(1-\lambda)B_Q(d)+\lambda B_U(d)\right]^K.
}
$$

### 3.1 曲线形状

当 $K=1$ 时，

$$
P_{\lambda,Q}^{(1)}
=(1-\lambda)P_Q^{(1)}+\lambda P_U^{(1)},
$$

所以曲线必然是 $\lambda$ 的一次函数。

当 $K>1$ 时，

$$
\left[(1-\lambda)B_Q(d)+\lambda B_U(d)\right]^K
$$

是 $\lambda$ 的 $K$ 次多项式。因此 $P_{\lambda,Q}^{(K)}$ 一般是 $K$ 次多项式，而不是直线。

---

## 4. uniform random baseline 的成功率

uniform random baseline 定义为

$$
Y\sim U_Q,
\qquad
U_Q(y)=\frac1Q.
$$

使用同一个 strict 后处理器时，它的 exact 成功率为

$$
\boxed{
P_U^{(K)}
=
\sum_{d\mid r}
\mu\!\left(\frac{r}{d}\right)
B_U(d)^K.
}
$$

令

$$
\epsilon_U
=
\Pr_{Y\sim U_Q}[D_Q(Y)>1].
$$

则有简单上界

$$
\boxed{
P_U^{(K)}
\le 1-(1-\epsilon_U)^K
\le K\epsilon_U.
}
$$

在 big instance 中，uniform random 的 $y/Q$ 几乎不会恰好落入能给出 $r$ 的因子的 continued-fractions cells。通常可以粗略估计为

$$
\epsilon_U
=O\!\left(\frac{\tau(r)\log N}{N}\right),
$$

因此

$$
\boxed{
P_U^{(K)}
=O\!\left(\frac{K\tau(r)\log N}{N}\right)
\approx 0
}
$$

对于固定的小 $K$，例如 $K\in\{1,2,4,8,16\}$，这个概率在大 $N$ 下通常非常小。严格地说，$\lambda=1$ 的端点是 $P_U^{(K)}$，不是数学上必然等于 $0$；但在 big instance 中它通常数值上接近 $0$。

---

## 5. 最关心的近似：$Q$ 足够大且忽略 uniform 成功概率

现在假设：

1. $Q$ 足够大，使 finite-$Q$ ideal 近似 arithmetic baseline：

   $$
   B_Q(d)\approx\frac{d}{r};
   $$

2. uniform random 样本几乎不会偶然给出有用分母，因此在 success 分析中可近似为 blank sample：

   $$
   B_U(d)\approx1.
   $$

代入 depolarized exact formula，得到

$$
\boxed{
P_{\lambda}^{(K)}
\approx
\sum_{d\mid r}
\mu\!\left(\frac{r}{d}\right)
\left[\lambda+(1-\lambda)\frac{d}{r}\right]^K.
}
$$

令

$$
t=\frac{r}{d},
$$

则等价于

$$
\boxed{
P_{\lambda}^{(K)}
\approx
\sum_{t\mid\operatorname{rad}(r)}
\mu(t)
\left[\lambda+\frac{1-\lambda}{t}\right]^K.
}
$$

由于 $\mu(t)=0$ 当 $t$ 含平方因子，所以只需要对 $\operatorname{rad}(r)$ 的因子求和。

### 5.1 binomial mixture 形式

同一个近似还可以写成

$$
\boxed{
P_{\lambda}^{(K)}
\approx
\sum_{j=0}^{K}
\binom{K}{j}
(1-\lambda)^j
\lambda^{K-j}
\prod_{p\mid r}\left(1-p^{-j}\right).
}
$$

约定

$$
\prod_{p\mid r}\left(1-p^{0}\right)=0,
$$

即 $j=0$ 时没有任何 ideal sample，不能恢复 $r>1$。

这个公式的直观含义是：

- $j$ 表示 $K$ 个样本中有多少个样本来自 ideal 分布；
- $\binom{K}{j}(1-\lambda)^j\lambda^{K-j}$ 是出现 $j$ 个 ideal samples 的概率；
- $\prod_{p\mid r}(1-p^{-j})$ 是这 $j$ 个 ideal samples 的 LCM 覆盖所有素因子的概率。

---

## 6. 两个重要实例

### 6.1 prime order：$r=p$

若 $r=p$ 是素数，则

$$
P_\lambda^{(K)}
\approx
1-\left[\lambda+\frac{1-\lambda}{p}\right]^K.
$$

当 $p$ 很大时，

$$
\boxed{
P_\lambda^{(K)}\approx1-\lambda^K.
}
$$

因此：

$$
K=1:\quad P_\lambda^{(1)}\approx1-\lambda,
$$

$$
K=2:\quad P_\lambda^{(2)}\approx1-\lambda^2,
$$

$$
K=4:\quad P_\lambda^{(4)}\approx1-\lambda^4.
$$

这解释了 prime order instance 中：无噪声 $K=1$ 成功率接近 $1$，而 $\lambda=1$ 时成功率接近 uniform baseline，即接近 $0$。

### 6.2 半素数型 order：$r=2pq$

若

$$
r=2pq
$$

且 $p,q$ 是较大的奇素数，则

$$
P_{\mathrm{arith}}^{(K)}
=(1-2^{-K})(1-p^{-K})(1-q^{-K})
\approx1-2^{-K}.
$$

因此

| $K$ | $P_{\mathrm{arith}}^{(K)}\approx1-2^{-K}$ |
|---:|---:|
| 1 | 0.5 |
| 2 | 0.75 |
| 4 | 0.9375 |
| 8 | 0.9961 |
| 16 | 0.9999847 |

在 depolarized 噪声下，若 $p,q$ 很大，则主导项来自素因子 $2$：

$$
\boxed{
P_\lambda^{(K)}
\approx
1-\left(\frac{1+\lambda}{2}\right)^K.
}
$$

端点为

$$
\lambda=0:\quad P_0^{(K)}\approx1-2^{-K},
$$

$$
\lambda=1:\quad P_1^{(K)}\approx0.
$$

---

## 7. 对实验现象的统一解释

### 7.1 prime order，$K=1$

若 $r$ 是大素数，则

$$
P_{\mathrm{arith}}^{(1)}=1-\frac1r\approx1.
$$

finite-$Q$ ideal 在 $Q$ 足够大时近似 arithmetic，因此无噪声成功率接近 $1$。当 $\lambda=1$ 时，样本完全来自 uniform random baseline，其 accidental success 概率在 big instance 中接近 $0$。

### 7.2 $r=2pq$，$K=1$

若 $p,q$ 很大，则

$$
P_{\mathrm{arith}}^{(1)}
=\frac12\left(1-\frac1p\right)\left(1-\frac1q\right)
\approx\frac12.
$$

所以 $K=1$ 成功率约为 $0.5$ 是数论结构导致的，不是 finite-$Q$ 误差导致的。

### 7.3 噪声曲线形状

- $K=1$：

  $$
  P_{\lambda,Q}^{(1)}=(1-\lambda)P_Q^{(1)}+\lambda P_U^{(1)},
  $$

  因此必然近似直线。

- $K>1$：

  $$
  P_{\lambda,Q}^{(K)}
  =
  \sum_{d\mid r}
  \mu\!\left(\frac{r}{d}\right)
  \left[(1-\lambda)B_Q(d)+\lambda B_U(d)\right]^K,
  $$

  因此是 $K$ 次多项式，视觉上会出现类似指数型下降的曲线。

---

# 附录 A：finite-$Q$ ideal 成功率推导

## A.1 finite-$Q$ ideal 测量分布

对固定 $s$，inverse QFT 前的相位态为

$$
|\xi_s\rangle
=
\frac1{\sqrt Q}
\sum_{x=0}^{Q-1}
\exp\!\left(2\pi i\frac{sx}{r}\right)|x\rangle.
$$

理想 inverse QFT 后测得 $y$ 的振幅为

$$
A_s(y)
=
\frac1Q
\sum_{x=0}^{Q-1}
\exp\!\left(2\pi i x\left(\frac{s}{r}-\frac{y}{Q}\right)\right).
$$

因此

$$
P_Q(y\mid s)=|A_s(y)|^2.
$$

令

$$
\Delta=\frac{s}{r}-\frac{y}{Q}.
$$

由等比数列求和，

$$
\sum_{x=0}^{Q-1}e^{2\pi i x\Delta}
=
 e^{\pi i(Q-1)\Delta}
\frac{\sin(\pi Q\Delta)}{\sin(\pi\Delta)}.
$$

所以

$$
P_Q(y\mid s)
=
\frac1{Q^2}
\frac{\sin^2(\pi Q\Delta)}{\sin^2(\pi\Delta)},
$$

当 $\sin(\pi\Delta)=0$ 时取连续极限。

---

## A.2 为什么引入 $D_Q(y)$

单样本 continued fractions 给出 $q_Q(y)$。strict 后处理器成功输出 $r$，等价于某个 subset 的 LCM 正好等于 $r$。

若某个 $q_Q(y)\nmid r$，则它不可能出现在任何 LCM 正好等于 $r$ 的 subset 中。否则该 LCM 会包含 $r$ 没有的素因子，或包含超过 $r$ 的素因子幂次。

因此，在只分析 success event 时，可以把所有 $q_Q(y)\nmid r$ 的样本投影成 $1$：

$$
D_Q(y)=
\begin{cases}
q_Q(y), & q_Q(y)\mid r,\\
1, & q_Q(y)\nmid r.
\end{cases}
$$

这就是“有用分母”的定义。

---

## A.3 strict 成功事件的等价刻画

令

$$
D_i=D_Q(y_i),\qquad i=1,\dots,K.
$$

定义

$$
L_D=\operatorname{lcm}(D_1,\dots,D_K).
$$

则有等价关系

$$
\boxed{
h_K^{\mathrm{strict}}(y_1,\dots,y_K)=r
\quad\Longleftrightarrow\quad
L_D=r.
}
$$

证明如下。

**充分性。** 若 $L_D=r$，则存在一个由真实 denominators 组成的 subset，其 LCM 为 $r$。于是 $r$ 属于 candidate LCM 集合，并且

$$
a^r\equiv1\pmod N.
$$

任何正整数 $L<r$ 都不可能满足 $a^L\equiv1\pmod N$，因为 $r$ 是最小 order。因此 strict 后处理器输出的最小通过验证的候选为 $r$。

**必要性。** 若 strict 后处理器输出 $r$，则 $r$ 必须属于 candidate LCM 集合，即存在某个 subset 的真实 denominators 的 LCM 等于 $r$。该 subset 中的每个 denominator 都必须整除 $r$，因此都会被 $D_Q$ 保留下来。于是 $L_D$ 至少包含 $r$。另一方面，每个 $D_i\mid r$，所以 $L_D\mid r$。因此 $L_D=r$。

---

## A.4 $B_Q(d)$ 的含义

对任意 $d\mid r$，定义

$$
B_Q(d)=\Pr_{Y\sim\pi_Q}[D_Q(Y)\mid d].
$$

这是 divisor lattice 上的累积分布函数。这里的“$D_Q(Y)\mid d$”表示 $D_Q(Y)$ 的所有素因子和幂次都已包含在 $d$ 中。

令

$$
L_D=\operatorname{lcm}(D_1,\dots,D_K).
$$

则

$$
L_D\mid d
\quad\Longleftrightarrow\quad
D_1\mid d,\dots,D_K\mid d.
$$

样本独立，所以

$$
\Pr[L_D\mid d]
=
B_Q(d)^K.
$$

---

## A.5 Möbius 反演

令

$$
G_Q(e)=\Pr[L_D=e],\qquad e\mid r.
$$

由上一节可得

$$
B_Q(d)^K
=
\Pr[L_D\mid d]
=
\sum_{e\mid d}G_Q(e).
$$

这是 divisor lattice 上的求和关系。由 Möbius 反演，

$$
G_Q(r)
=
\sum_{d\mid r}
\mu\!\left(\frac{r}{d}\right)B_Q(d)^K.
$$

又因为 strict 成功等价于 $L_D=r$，所以

$$
P_Q^{(K)}=G_Q(r)
=
\sum_{d\mid r}
\mu\!\left(\frac{r}{d}\right)B_Q(d)^K.
$$

---

# 附录 B：arithmetic baseline 推导

## B.1 单样本返回的分母

在 arithmetic ideal 中，测量相位为

$$
\frac{s}{r},\qquad s\sim\mathrm{Unif}\{0,1,\dots,r-1\}.
$$

设

$$
g=\gcd(s,r).
$$

则

$$
\frac{s}{r}
=
\frac{s/g}{r/g}
$$

是最简分数，所以 continued fractions 返回

$$
q_s=\frac{r}{\gcd(s,r)}.
$$

---

## B.2 计算 $B_\infty(d)$

对任意 $d\mid r$，

$$
B_\infty(d)=\Pr[q_s\mid d].
$$

因为

$$
q_s\mid d
\quad\Longleftrightarrow\quad
\frac{r}{\gcd(s,r)}\mid d.
$$

令

$$
t=\frac{r}{d}.
$$

上式等价于

$$
t\mid s.
$$

在 $s=0,1,\dots,r-1$ 中，满足 $t\mid s$ 的数正好有 $d$ 个。因此

$$
\boxed{
B_\infty(d)=\frac{d}{r}.
}
$$

---

## B.3 arithmetic 成功率闭式公式

代入 finite-$Q$ exact formula 的极限形式：

$$
P_{\mathrm{arith}}^{(K)}
=
\sum_{d\mid r}
\mu\!\left(\frac{r}{d}\right)
\left(\frac{d}{r}\right)^K.
$$

令 $u=r/d$，则

$$
P_{\mathrm{arith}}^{(K)}
=
\sum_{u\mid r}\mu(u)u^{-K}.
$$

由于函数 $u\mapsto \mu(u)u^{-K}$ 是 multiplicative，得到 Euler product：

$$
\boxed{
P_{\mathrm{arith}}^{(K)}
=
\prod_{p\mid r}(1-p^{-K}).
}
$$

---

# 附录 C：depolarized 噪声推导

## C.1 单样本分布的线性混合

depolarized 噪声模型为

$$
\rho_\lambda=(1-\lambda)\rho+\lambda\frac{I}{Q}.
$$

对测量算符

$$
M_y=|y\rangle\langle y|,
$$

有

$$
\begin{aligned}
P_{\lambda,Q}(y\mid s)
&=\operatorname{Tr}\!\left[M_y\rho_\lambda\right]\\
&=(1-\lambda)\operatorname{Tr}[M_y\rho]+\lambda\operatorname{Tr}\!\left[M_y\frac{I}{Q}\right]\\
&=(1-\lambda)P_Q(y\mid s)+\lambda\frac1Q.
\end{aligned}
$$

因此 depolarized 噪声在测量分布层面就是 ideal 分布和 uniform 分布的 convex combination。

---

## C.2 noisy $B$-函数

对 $s$ 做均匀混合后，单样本分布为

$$
\pi_{\lambda,Q}(y)
=(1-\lambda)\pi_Q(y)+\lambda U_Q(y).
$$

因此

$$
\begin{aligned}
B_{\lambda,Q}(d)
&=\Pr_{Y\sim\pi_{\lambda,Q}}[D_Q(Y)\mid d]\\
&=(1-\lambda)B_Q(d)+\lambda B_U(d).
\end{aligned}
$$

---

## C.3 noisy 成功率

与 finite-$Q$ ideal 的推导完全相同，

$$
\Pr[L_D\mid d]
=B_{\lambda,Q}(d)^K.
$$

Möbius 反演给出

$$
\boxed{
P_{\lambda,Q}^{(K)}
=
\sum_{d\mid r}
\mu\!\left(\frac{r}{d}\right)
\left[(1-\lambda)B_Q(d)+\lambda B_U(d)\right]^K.
}
$$

当 $K=1$ 时，该式自动化为线性插值：

$$
P_{\lambda,Q}^{(1)}
=(1-\lambda)P_Q^{(1)}+\lambda P_U^{(1)}.
$$

---

# 附录 D：uniform random baseline 推导与数量级估计

## D.1 exact formula

定义

$$
n_Q(e)=\#\{0\le y<Q:q_Q(y)=e\}.
$$

对 $e\mid r$，$e>1$，uniform random 下

$$
\Pr[D_Q(Y)=e]=\frac{n_Q(e)}{Q}.
$$

而

$$
\Pr[D_Q(Y)=1]
=1-\sum_{\substack{e\mid r\\e>1}}\frac{n_Q(e)}{Q},
$$

因为所有 $q_Q(y)\nmid r$ 的格点也会被投影到 $D_Q(Y)=1$。

于是

$$
B_U(d)
=\Pr[D_Q(Y)\mid d]
=1-\sum_{\substack{e\mid r\\e\nmid d}}\frac{n_Q(e)}{Q}.
$$

代入 Möbius 公式得到

$$
P_U^{(K)}
=
\sum_{d\mid r}
\mu\!\left(\frac{r}{d}\right)B_U(d)^K.
$$

---

## D.2 简单上界

令

$$
\epsilon_U=\Pr[D_Q(Y)>1]
=\sum_{\substack{e\mid r\\e>1}}\frac{n_Q(e)}{Q}.
$$

若 strict 成功恢复 $r>1$，至少需要某个样本满足 $D_Q(Y_i)>1$。因此

$$
P_U^{(K)}
\le
\Pr[\exists i:D_Q(Y_i)>1]
=1-(1-\epsilon_U)^K
\le K\epsilon_U.
$$

---

## D.3 big instance 下为何接近 $0$

uniform random 的 $y/Q$ 均匀铺在 $[0,1)$ 的 $Q$ 个网格点上。continued fractions 返回某个固定分母 $e$，意味着 $y/Q$ 落入某些以分母为 $e$ 的有理数为中心的 best-approximation cells。

当分母上限为 $N-1$ 且 $Q\gtrsim N^2$ 时，每个固定分母 $e$ 的 cell 总长度通常至多为

$$
O\!\left(\frac{\log N}{N}\right).
$$

因此对 $r$ 的所有非平凡因子求和，可得粗略数量级

$$
\epsilon_U
=O\!\left(\frac{\tau(r)\log N}{N}\right).
$$

从而

$$
P_U^{(K)}
\le K\epsilon_U
=O\!\left(\frac{K\tau(r)\log N}{N}\right).
$$

对 big instance 与固定小 $K$，该值通常可以忽略。

---

# 附录 E：忽略 uniform success 后的 $P_\lambda^{(K)}$ 推导

## E.1 从 exact noisy formula 推导

从

$$
P_{\lambda,Q}^{(K)}
=
\sum_{d\mid r}
\mu\!\left(\frac{r}{d}\right)
\left[(1-\lambda)B_Q(d)+\lambda B_U(d)\right]^K
$$

出发。

若 $Q$ 足够大，取

$$
B_Q(d)\approx\frac{d}{r}.
$$

若 uniform success 可忽略，则 uniform sample 在 success 分析中近似为 blank sample，因此

$$
D_Q(Y)\approx1,
\qquad
B_U(d)=\Pr[D_Q(Y)\mid d]\approx1.
$$

于是

$$
P_\lambda^{(K)}
\approx
\sum_{d\mid r}
\mu\!\left(\frac{r}{d}\right)
\left[\lambda+(1-\lambda)\frac{d}{r}\right]^K.
$$

令 $t=r/d$，得到

$$
P_\lambda^{(K)}
\approx
\sum_{t\mid r}
\mu(t)
\left[\lambda+\frac{1-\lambda}{t}\right]^K.
$$

因为 $\mu(t)=0$ 当 $t$ 含平方因子，所以可写为

$$
P_\lambda^{(K)}
\approx
\sum_{t\mid\operatorname{rad}(r)}
\mu(t)
\left[\lambda+\frac{1-\lambda}{t}\right]^K.
$$

---

## E.2 binomial mixture 形式

展开

$$
\left[\lambda+(1-\lambda)\frac{d}{r}\right]^K
=
\sum_{j=0}^{K}
\binom{K}{j}
\lambda^{K-j}(1-\lambda)^j
\left(\frac{d}{r}\right)^j.
$$

代入：

$$
\begin{aligned}
P_\lambda^{(K)}
&\approx
\sum_{j=0}^{K}
\binom{K}{j}
\lambda^{K-j}(1-\lambda)^j
\sum_{d\mid r}
\mu\!\left(\frac{r}{d}\right)
\left(\frac{d}{r}\right)^j.
\end{aligned}
$$

内部求和正是 $j$ 个 arithmetic ideal samples 的成功率：

$$
\sum_{d\mid r}
\mu\!\left(\frac{r}{d}\right)
\left(\frac{d}{r}\right)^j
=
\prod_{p\mid r}(1-p^{-j}).
$$

因此

$$
\boxed{
P_\lambda^{(K)}
\approx
\sum_{j=0}^{K}
\binom{K}{j}
(1-\lambda)^j
\lambda^{K-j}
\prod_{p\mid r}(1-p^{-j}).
}
$$

其中 $j=0$ 项按 $0$ 处理。

---

## E.3 inclusion-exclusion 形式

设

$$
\mathcal P(r)=\{p:p\mid r\}.
$$

对任意素因子子集 $S\subseteq\mathcal P(r)$，定义

$$
t_S=\prod_{p\in S}p,
$$

并约定 $t_\varnothing=1$。

一个样本同时没有覆盖 $S$ 中所有素因子的概率为

$$
\lambda+\frac{1-\lambda}{t_S}.
$$

因此 $K$ 个样本都没有覆盖 $S$ 中所有素因子的概率为

$$
\left(\lambda+\frac{1-\lambda}{t_S}\right)^K.
$$

由 inclusion-exclusion，成功覆盖所有素因子的概率为

$$
\boxed{
P_\lambda^{(K)}
\approx
\sum_{S\subseteq\mathcal P(r)}
(-1)^{|S|}
\left(\lambda+\frac{1-\lambda}{\prod_{p\in S}p}\right)^K.
}
$$

该式与 Möbius 形式完全等价。
