# Order-finding 与 ECDLP 的 QFT 应用 Benchmark：原理、构造与比较

## 0. 文档目的与范围

本文总结两类面向 Shor 算法应用的 QFT benchmark：

1. **Order-finding benchmark**：模拟 Shor 分解整数时的 order-finding 子程序；
2. **DLP/ECDLP benchmark**：模拟 Shor 求解离散对数，特别是椭圆曲线离散对数问题 ECDLP 时的 Fourier-sampling 子程序。

两者不是完全无关的算法：它们都属于 Abelian hidden-subgroup / Fourier-sampling 框架，也都可以从完整 Shor 电路中抽取出 QFT 前的 reduced state，直接制备 phase state，再只执行 QFT+measurement。但是，两者隐藏的数学结构、QFT 输出的数量、经典后处理以及多样本聚合方式均不同，因此应视为两个独立但相关的应用 benchmark。

本文统一采用 **inverse QFT（IQFT）+ measurement** 的约定。若实验执行 forward QFT，只需将输入相位取共轭，或等价地改变相位符号；其余定义不变。

---

## 1. 核心结论

两类 benchmark 的主要差别可以压缩为：

- Order-finding 求解未知 order $r$；单个 QFT 输出一个相位估计 $y/Q\approx s/r$；由于分母 $r$ 未知，需要 continued fractions 恢复候选分母，并通过多样本 LCM 汇集 order 的素因子信息。
- ECDLP 求解未知离散对数 $d$，其中公开点满足 $R=dP$；一个完整 sample 产生两个相关 Fourier 输出 $(y,z)$，分别估计 $s/n$ 和 $(ds\bmod n)/n$；由于群阶 $n$ 已知，后处理主要是 residue rounding / limited search、模逆和公开点验证。
- Order-finding 的 textbook 控制寄存器通常取 $m\approx 2\log_2N$，因为要唯一恢复一个未知分母 $r<N$；ECDLP 的每个 Fourier 寄存器通常只需约 $\log_2n$ 个 qubit，再根据 finite-$Q$ 成功率和经典搜索预算增加少量 padding。
- Order-finding benchmark 只调用一次 QFT，更适合把应用成功率与单个 QFT 子程序直接关联；ECDLP benchmark 调用两个相互独立、但输入标签相关的 QFT，更适合验证 QFT 在 paired-output Shor 应用中的可用性。
- 建议把 **order-finding 作为主 benchmark**，把 **ECDLP 作为 paired-QFT 扩展 benchmark**。

---

## 2. 统一符号与 benchmark 边界

| 符号 | 含义 |
|---|---|
| $m$ | 单个 Fourier 控制寄存器的 qubit 数 |
| $Q$ | Fourier 寄存器维度，$Q=2^m$ |
| $K$ | 一个应用实例允许使用的 algorithmic samples 数，不是实验总 shots 数 |
| $\bot$ | 后处理器拒绝输出 |
| $P_{\mathrm{ideal}}$ | 相同 $Q$、相同后处理器下的无噪声 finite-$Q$ baseline |
| $P_{\mathrm{unif}}$ | 将测量结果替换成均匀随机 bitstring 后，用相同后处理器得到的 baseline |

两类 benchmark 都遵循同一个抽象流程：

$$
\text{完整 Shor 电路}
\longrightarrow
\text{QFT 前 reduced state}
\longrightarrow
\text{直接 phase-state 制备}
\longrightarrow
\text{QFT+M}
\longrightarrow
\text{固定经典后处理器}
\longrightarrow
\text{应用成功率}.
$$

benchmark 的目标不是模拟完整 modular exponentiation 或椭圆曲线点运算，而是隔离并评估：

> 给定具有正确 Shor 应用语义的 QFT 输入态，实验 QFT+M 的输出是否仍足以通过固定后处理恢复应用目标？

需要区分两类信息：

- **公开给后处理器的信息**：真实攻击者可以获得的实例参数及测量结果；
- **benchmark ground truth**：由实例生成器保留，用于直接制备 phase state 和判断成功，但不能交给后处理器。

---

# 第一部分：Order-finding benchmark

## 3. Order-finding 问题

给定整数 $N$ 和满足 $\gcd(a,N)=1$ 的整数 $a$，目标是恢复

$$
r=\operatorname{ord}_N(a),
$$

即满足

$$
a^r\equiv 1\pmod N
$$

的最小正整数 $r$。

### 3.1 已知量与未知量

| 量 | 真实应用中 | benchmark 中的用途 |
|---|---|---|
| $N$ | 已知 | 实例参数、continued-fractions 分母上界、最终 factoring |
| $a$ | 已知 | 定义 order，并用于模幂验证 |
| $r$ | 未知 | benchmark 生成器知道；后处理器不可见；用于制备 phase-state ensemble 和评分 |
| $m,Q$ | 设计参数 | 指定 QFT 精度与电路规模 |
| $y$ | 量子测量结果 | 经典后处理输入 |

benchmark 实例可写为

$$
I_{\mathrm{OF}}=(N,a,r,m),
\qquad
Q=2^m.
$$

严格地说，交给求解器的公开实例是 $(N,a,m)$；$r$ 是隐藏 ground truth。

---

## 4. 从完整 Shor 电路到单个 phase state

令模乘算符为

$$
M_a|z\rangle=|az\bmod N\rangle.
$$

在由 $|1\rangle$ 生成的阶为 $r$ 的循环子空间中，$M_a$ 有本征态 $|u_s\rangle$，满足

$$
M_a|u_s\rangle
=
\exp\!\left(2\pi i\frac{s}{r}\right)|u_s\rangle,
\qquad
s\in\{0,1,\ldots,r-1\}.
$$

完整 Shor/QPE 电路在 IQFT 前可写成

$$
|\Psi_{\mathrm{pre}}\rangle
=
\frac{1}{\sqrt r}
\sum_{s=0}^{r-1}
|\xi_s^{(r,Q)}\rangle\otimes|u_s\rangle,
$$

其中

$$
|\xi_s^{(r,Q)}\rangle
=
\frac{1}{\sqrt Q}
\sum_{x=0}^{Q-1}
\exp\!\left(2\pi i\frac{sx}{r}\right)|x\rangle.
$$

trace out 工作寄存器后，QFT 输入寄存器的 reduced state 为

$$
\rho_{\mathrm{OF}}
=
\frac1r
\sum_{s=0}^{r-1}
|\xi_s^{(r,Q)}\rangle
\langle\xi_s^{(r,Q)}|.
$$

因此 benchmark 不需要实现 modular exponentiation。只需均匀随机选择

$$
s\sim\operatorname{Unif}(\mathbb Z_r),
$$

然后直接制备 $|\xi_s^{(r,Q)}\rangle$。

### 4.1 输入态是单 qubit 乘积态

写

$$
x=\sum_{j=0}^{m-1}x_j2^j,
\qquad x_j\in\{0,1\},
$$

则

$$
|\xi_s^{(r,Q)}\rangle
=
\bigotimes_{j=0}^{m-1}
\frac{|0\rangle+
\exp\!\left(2\pi i\frac{s2^j}{r}\right)|1\rangle}
{\sqrt2}.
$$

因此初态只需 Hadamard 和单 qubit $Z$-axis phase rotations，不需要多 qubit 纠缠态制备。

---

## 5. Order-finding benchmark 的量子流程

每个 algorithmic sample 执行：

1. 均匀随机选择 $s\in\mathbb Z_r$；
2. 制备 $|\xi_s^{(r,Q)}\rangle$；
3. 执行 $\operatorname{IQFT}_Q$；
4. 测量并得到
   $$
   y\in\{0,1,\ldots,Q-1\}.
   $$

对固定 $s$，理想测量分布为

$$
D_{Q,r}(y\mid s)
=
\frac1{Q^2}
\left|
\sum_{x=0}^{Q-1}
\exp\!\left[
2\pi i x\left(\frac{s}{r}-\frac{y}{Q}\right)
\right]
\right|^2.
$$

等价地，令

$$
\Delta=\frac{s}{r}-\frac{y}{Q},
$$

则

$$
D_{Q,r}(y\mid s)
=
\frac1{Q^2}
\frac{\sin^2(\pi Q\Delta)}{\sin^2(\pi\Delta)},
$$

在分母为零时取连续极限。

测量峰位于

$$
\frac yQ\approx\frac sr.
$$

---

## 6. 为什么 textbook order-finding 需要 $m\approx2\log_2N$

continued fractions 唯一恢复 $s/r$ 的经典充分条件是

$$
\left|
\frac yQ-\frac sr
\right|
<
\frac1{2r^2}.
$$

QFT 的相位分辨率尺度约为 $1/Q$，因此需要

$$
Q\gtrsim r^2.
$$

由于只知道

$$
r<N,
$$

textbook worst-case 选择为

$$
Q\gtrsim N^2,
$$

即

$$
m\approx2\lceil\log_2N\rceil.
$$

这不是所有优化 Shor 算法都必须采用的唯一选择。缩短寄存器后，可以通过增加 quantum runs、nearby-frequency search 或 lattice post-processing 做量子—经典 trade-off。但在本 benchmark 的 textbook continued-fractions 定义下，$m\approx2\log_2N$ 是最自然的默认参数。

---

## 7. Order-finding 的经典后处理

### 7.1 单样本 continued fractions

对测量值 $y$，定义

$$
\alpha=\frac yQ.
$$

使用精确有理数而不是浮点数，计算

$$
\operatorname{CF}_N(\alpha)=\frac pq,
$$

其中 $p/q$ 是分母小于 $N$ 的最佳有理逼近之一。

理想极限下，continued fractions 恢复的是 $s/r$ 的最简形式，因此返回分母

$$
q_s=\frac r{\gcd(s,r)}.
$$

因此单次样本不一定直接给出 $r$；它通常只给出 $r$ 的一个因子。

### 7.2 $K$ 个样本的 denominator 聚合

一个 $K$-sample block 为

$$
Y=(y_1,\ldots,y_K).
$$

每个样本产生候选分母 $q_i$。为抵抗 noisy denominators，定义所有可达 subset-LCM：

$$
\mathcal L_0=\{1\},
$$

$$
\mathcal L_i
=
\mathcal L_{i-1}
\cup
\left\{
\operatorname{lcm}(L,q_i):L\in\mathcal L_{i-1}
\right\}.
$$

然后保留通过模幂验证的候选：

$$
\mathcal V
=
\{L\in\mathcal L_K:1<L<N,\ a^L\equiv1\pmod N\}.
$$

严格后处理器定义为

$$
h_{K,\mathrm{OF}}^{\mathrm{strict}}(Y)
=
\begin{cases}
\bot, & \mathcal V=\varnothing,\\
\min\mathcal V, & \mathcal V\neq\varnothing.
\end{cases}
$$

注意：条件 $a^L\equiv1\pmod N$ 只说明 $L$ 是真实 order $r$ 的倍数，不保证 $L=r$。因此还可以定义 order-reduction 和最终 factor-recovery 辅助指标。

### 7.3 Order-finding 也可以做附近搜索

continued fractions 并不排斥 limited search。例如可以：

- 对 $y+\delta$ 的附近频率分别做 continued fractions；
- 搜索候选分母 $q$ 的小倍数并做模幂验证；
- 将多个短寄存器输出放入 lattice reconstruction。

所以 Order-finding 与 ECDLP 的差异不是“一个能搜索、一个不能搜索”，而是：

> Order-finding 的目标分母 $r$ 未知，因此首先需要 rational reconstruction；ECDLP 的群阶 $n$ 已知，因此可以直接把 Fourier 输出缩放到已知 residue 网格。

---

## 8. Order-finding 指标与 baseline

### 8.1 主指标

$$
P_{\mathrm{ord,strict}}^{(K)}
=
\Pr\left[
 h_{K,\mathrm{OF}}^{\mathrm{strict}}(Y)=r
\right].
$$

同时报告

$$
P_{\mathrm{wrong}}^{(K)}
=
\Pr\left[
 h_{K,\mathrm{OF}}^{\mathrm{strict}}(Y)\notin\{r,\bot\}
\right],
$$

$$
P_{\mathrm{null}}^{(K)}
=
\Pr\left[
 h_{K,\mathrm{OF}}^{\mathrm{strict}}(Y)=\bot
\right].
$$

三者满足

$$
P_{\mathrm{ord,strict}}^{(K)}
+
P_{\mathrm{wrong}}^{(K)}
+
P_{\mathrm{null}}^{(K)}
=1.
$$

### 8.2 Arithmetic ideal baseline

在 $Q\to\infty$ 的理想极限中，每个样本返回

$$
q_s=\frac r{\gcd(s,r)}.
$$

$K$ 个样本通过 LCM 恢复完整 order 的概率为

$$
\boxed{
P_{\mathrm{arith,OF}}^{(K)}
=
\prod_{p\mid r}\left(1-p^{-K}\right)
}.
$$

其中乘积遍历 $r$ 的不同素因子。

该公式说明，Order-finding 的理想成功率仍受到 $r$ 的数论结构影响。例如当 $r$ 是大素数时，$K=1$ 成功率接近 $1$；当 $r=2pq$ 且 $p,q$ 为大奇素数时，$K=1$ 成功率约为 $1/2$。

### 8.3 Finite-$Q$ 与 uniform baseline

finite-$Q$ baseline 使用理想分布 $D_{Q,r}(y\mid s)$，但必须经过与实验完全相同的 continued-fractions、subset-LCM 和验证流程。

uniform baseline 使用

$$
y\sim\operatorname{Unif}\{0,\ldots,Q-1\},
$$

并经过相同后处理。它衡量随机 bitstring 经过较强经典后处理后发生 accidental success 的概率。

归一化指标可定义为

$$
\eta_{\mathrm{OF}}^{(K)}
=
\frac{
P_{\mathrm{exp}}^{(K)}-P_{\mathrm{unif}}^{(K)}
}{
P_{\mathrm{ideal}}^{(K)}-P_{\mathrm{unif}}^{(K)}
}.
$$

---

# 第二部分：DLP / ECDLP benchmark

## 9. DLP 与 ECDLP 问题

设 $\mathcal G=\langle P\rangle$ 是一个阶为 $n$ 的循环群，公开点为

$$
R=dP.
$$

DLP 的目标是从公开的 $(P,R,n)$ 恢复

$$
d\in\mathbb Z_n.
$$

ECDLP 是同一问题在椭圆曲线点群上的实例：群运算是椭圆曲线点加法，$P$ 是基点，$R$ 是公钥点。

### 9.1 已知量与未知量

| 量 | 真实应用中 | benchmark 中的用途 |
|---|---|---|
| $\mathcal G$ 或椭圆曲线参数 | 已知 | 定义群运算与验证规则 |
| $P$ | 已知 | 生成元或基点 |
| $n=\operatorname{ord}(P)$ | 已知 | Fourier phase 的分母和模运算模数 |
| $R=dP$ | 已知 | 公开点，用于最终验证 |
| $d$ | 未知 | benchmark 生成器知道；后处理器不可见；用于制备相关 phase states 和评分 |
| $m,Q$ | 设计参数 | 每个 Fourier 寄存器的精度 |
| $(y,z)$ | 量子测量结果 | 经典后处理输入 |

benchmark 实例可写为

$$
I_{\mathrm{DLP}}
=(\mathcal G,P,n,R=dP,m,W),
$$

其中 $W$ 是固定的经典 limited-search 预算。对于 ECDLP，可将 $\mathcal G$ 替换为具体椭圆曲线及有限域参数。

密码学 ECC 通常使用大素数阶子群，因此下文主要假设 $n$ 为素数。

---

## 10. Shor 求解 DLP 的完整量子算法

### 10.1 三个逻辑寄存器

完整算法包含：

- 第一个指数寄存器 $A$，存储 $a$；
- 第二个指数寄存器 $B$，存储 $b$；
- 工作寄存器 $W_G$，存储群元素。

精确数学版本从

$$
|0\rangle_A|0\rangle_B|O\rangle_{W_G}
$$

开始，其中 $O$ 是群单位元。

### 10.2 制备二维均匀叠加

制备

$$
|\Psi_0\rangle
=
\frac1n
\sum_{a,b\in\mathbb Z_n}
|a\rangle|b\rangle|O\rangle.
$$

### 10.3 计算群函数

可逆地计算

$$
f(a,b)=aP+bR.
$$

因为 $R=dP$，有

$$
f(a,b)
=(a+bd)P.
$$

联合态变为

$$
|\Psi_1\rangle
=
\frac1n
\sum_{a,b\in\mathbb Z_n}
|a\rangle|b\rangle|(a+bd)P\rangle.
$$

这一阶段在真实 ECDLP 电路中需要受控椭圆曲线点加法和可逆有限域运算，是完整算法的主要资源开销；但 QFT-only benchmark 会跳过这一部分。

### 10.4 隐藏子群结构

函数满足

$$
f(a-dt,b+t)=f(a,b),
\qquad t\in\mathbb Z_n.
$$

因此隐藏子群为

$$
H
=
\{(-dt,t):t\in\mathbb Z_n\}.
$$

其方向由未知量 $d$ 决定。

### 10.5 测量工作寄存器后的 coset state

为方便推导，假设工作寄存器被测为某个点

$$
C=cP.
$$

则前两个寄存器坍缩为

$$
|\psi_c\rangle
=
\frac1{\sqrt n}
\sum_{b=0}^{n-1}
|c-db\rangle_A|b\rangle_B.
$$

实际电路不必显式测量工作寄存器；将其 trace out 会给出相同的 Fourier 测量统计。

### 10.6 对两个指数寄存器执行 QFT

定义

$$
\operatorname{IQFT}_n|x\rangle
=
\frac1{\sqrt n}
\sum_{u=0}^{n-1}
\exp\!\left(-2\pi i\frac{ux}{n}\right)|u\rangle.
$$

对两个寄存器执行

$$
\operatorname{IQFT}_n\otimes\operatorname{IQFT}_n.
$$

代入 $|\psi_c\rangle$：

$$
\begin{aligned}
&(\operatorname{IQFT}_n\otimes\operatorname{IQFT}_n)|\psi_c\rangle\\
&=
\frac1{n^{3/2}}
\sum_{u,v=0}^{n-1}
\exp\!\left(-2\pi i\frac{uc}{n}\right)
\left[
\sum_{b=0}^{n-1}
\exp\!\left(2\pi i\frac{b(ud-v)}{n}\right)
\right]
|u,v\rangle.
\end{aligned}
$$

括号内的和仅在

$$
v\equiv du\pmod n
$$

时非零，因此输出态为

$$
\boxed{
\frac1{\sqrt n}
\sum_{u=0}^{n-1}
\exp\!\left(-2\pi i\frac{uc}{n}\right)
|u\rangle|du\bmod n\rangle
}.
$$

所以测量得到一对相关结果

$$
(u,v),
$$

满足

$$
\boxed{
v\equiv du\pmod n
}.
$$

若采用 $aP-bR$ 或相反的 QFT 符号约定，关系可能出现负号；这只是 convention，不改变算法。

### 10.7 精确经典后处理

若 $n$ 为素数且 $u\neq0$，则 $u$ 可逆：

$$
\hat d
=
v u^{-1}\pmod n.
$$

最后验证

$$
\hat dP\stackrel{?}=R.
$$

若验证成立，则恢复正确离散对数。

当 $u=0$ 时也有 $v=0$，该 sample 不包含 $d$ 的信息，需要重新运行。理想精确 $n$ 点 QFT 下，单 sample 成功率为

$$
1-\frac1n.
$$

---

## 11. DLP 的 phase-estimation 解释

定义群平移算符

$$
T_P|X\rangle=|X+P\rangle,
$$

以及

$$
T_R|X\rangle=|X+R\rangle.
$$

由于 $R=dP$，有

$$
T_R=T_P^d.
$$

令 $\omega_n=\exp(2\pi i/n)$，定义

$$
|\phi_s\rangle
=
\frac1{\sqrt n}
\sum_{k=0}^{n-1}
\omega_n^{-sk}|kP\rangle.
$$

则

$$
T_P|\phi_s\rangle
=
\omega_n^s|\phi_s\rangle,
$$

$$
T_R|\phi_s\rangle
=
\omega_n^{ds}|\phi_s\rangle.
$$

因此，同一个工作寄存器本征态 $|\phi_s\rangle$ 同时产生两组相关 eigenphases：

$$
\frac sn,
\qquad
\frac{ds\bmod n}{n}.
$$

这解释了为什么 DLP 需要两个 Fourier 输出，也解释了 QFT-only benchmark 应如何直接制备输入态。

---

## 12. 从完整 DLP 算法抽取 QFT-only benchmark

设

$$
Q=2^m.
$$

对任意 $t\in\mathbb Z_n$，定义 $m$-qubit phase state

$$
|\xi_t^{(n,Q)}\rangle
=
\frac1{\sqrt Q}
\sum_{x=0}^{Q-1}
\exp\!\left(2\pi i\frac{tx}{n}\right)|x\rangle.
$$

完整 DLP 电路在 QFT 前、trace out 工作寄存器后，对应的两寄存器 mixed state 为

$$
\boxed{
\rho_{\mathrm{DLP}}
=
\frac1n
\sum_{s=0}^{n-1}
\left(
|\xi_s^{(n,Q)}\rangle
\langle\xi_s^{(n,Q)}|
\right)
\otimes
\left(
|\xi_{ds\bmod n}^{(n,Q)}\rangle
\langle\xi_{ds\bmod n}^{(n,Q)}|
\right)
}.
$$

因此，每个 benchmark sample 应执行：

1. 均匀随机选择
   $$
   s\sim\operatorname{Unif}(\mathbb Z_n);
   $$
2. 制备第一个寄存器
   $$
   |\xi_s^{(n,Q)}\rangle;
   $$
3. 制备第二个寄存器
   $$
   |\xi_{ds\bmod n}^{(n,Q)}\rangle;
   $$
4. 分别执行两次
   $$
   \operatorname{IQFT}_Q;
   $$
5. 测量得到一个相关输出对
   $$
   (y,z).
   $$

关键约束是：两个输入态必须共享同一个随机标签 $s$。不能分别独立选择 $s_1$ 和 $s_2$，否则两次 QFT 输出之间不再包含离散对数关系。

### 12.1 为什么 benchmark 可以在制备时使用 $d$

真实攻击者不知道 $d$，完整 Shor 电路通过公开点 $R=dP$ 和受控群运算自动产生相关相位。

QFT-only benchmark 的生成器人为选择 ground-truth $d$，计算公开点 $R=dP$，再直接制备等效的 reduced QFT input。这里使用 $d$ 只是为了跳过群算术电路；后处理器仍只能看到公开的 $(P,n,R)$ 和测量结果 $(y,z)$，不能访问 $d$。

这与 Order-finding benchmark 中生成器知道真实 $r$，并用其直接制备 $|\xi_s^{(r,Q)}\rangle$，但不把 $r$交给后处理器，是完全相同的 benchmark 方法。

### 12.2 两个输入态仍然只需单 qubit 门

有

$$
|\xi_t^{(n,Q)}\rangle
=
\bigotimes_{j=0}^{m-1}
\frac{|0\rangle+
\exp\!\left(2\pi i\frac{t2^j}{n}\right)|1\rangle}
{\sqrt2}.
$$

所以条件于固定 $s$，两个寄存器之间不需要纠缠；它们的相关性来自共享的经典随机变量 $s$。

---

## 13. DLP 的 finite-$Q$ 理想分布

对一个 phase label $t\in\mathbb Z_n$，定义

$$
D_{Q,n}(y\mid t)
=
\frac1{Q^2}
\left|
\sum_{x=0}^{Q-1}
\exp\!\left[
2\pi i x\left(\frac tn-\frac yQ\right)
\right]
\right|^2.
$$

对固定 $s$ 和 $d$，两个 QFT 条件独立，因此 pair 分布为

$$
\boxed{
P_{Q,n}(y,z\mid s,d)
=
D_{Q,n}(y\mid s)
D_{Q,n}(z\mid ds\bmod n)
}.
$$

对 $s$ 做均匀混合得到 benchmark 的无噪声输出分布：

$$
P_{Q,n}(y,z\mid d)
=
\frac1n
\sum_{s=0}^{n-1}
P_{Q,n}(y,z\mid s,d).
$$

这一定义不需要模拟椭圆曲线 arithmetic，只需经典计算两个 Dirichlet-kernel 分布。

---

## 14. DLP 中的寄存器大小 $m$

令

$$
\ell=\lceil\log_2n\rceil.
$$

### 14.1 精确 $\operatorname{QFT}_n$ 的数学版本

如果能够直接在 $\mathbb Z_n$ 上实现精确 QFT，则每个指数寄存器只需表示 $n$ 个状态，尺度为

$$
m\approx\ell.
$$

### 14.2 标准 qubit QFT 的 power-of-two 版本

实际硬件通常实现

$$
Q=2^m.
$$

因为分母 $n$ 已知，目标只是区分相邻 residue $s$，而不是恢复一个未知分母。相邻 residue 在测量网格上的间距约为

$$
\frac Qn.
$$

因此自然的 benchmark 参数化是

$$
m=\ell+\Delta,
$$

其中 $\Delta$ 是小的 padding length。

建议至少评测

$$
\Delta\in\{0,1,2,4\},
$$

并在固定经典搜索预算 $W$ 下，使用 finite-$Q$ ideal success 决定最小可接受 $m$：

$$
m^*
=
\min\left\{
 m:
 P_{\mathrm{DLP,ideal}}^{(K)}(m,W)
 \ge P_0
\right\}.
$$

增加 $m$ 通常会缩窄相位峰并提高可恢复性，但成功率不必对每个 $m$ 严格单调，因为 $n$ 与 $2^m$ 的数论对齐关系以及后处理搜索区域都会影响结果。因此不能简单地说“$m$ 越大一定越好”；应报告完整 finite-$Q$ baseline。

### 14.3 与 Order-finding 的根本差异

$$
\begin{aligned}
\text{Order-finding:}&\quad
\text{未知分母 }r
\Rightarrow
Q=O(r^2);\\
\text{DLP/ECDLP:}&\quad
\text{已知分母 }n
\Rightarrow
Q=O(n)
\text{ 即可获得常数级恢复概率。}
\end{aligned}
$$

因此对于 RSA-2048 textbook order-finding，单个 QFT 可能使用约 $4096$ 个 phase bits；对于 256-bit prime-order ECDLP，逻辑上是两组各约 $256$ bits 的 Fourier 信息。ECDLP 虽然有两个 QFT stream，但总 phase-bit 数仍可能远小于 RSA-2048 的单个 textbook QFT。

---

## 15. DLP benchmark 的经典后处理

### 15.1 最简单的固定窗口后处理器

对测量结果 $(y,z)$，先计算两个 residue 中心：

$$
s_0
=
\left\lfloor\frac{ny}{Q}\right\rceil\bmod n,
$$

$$
t_0
=
\left\lfloor\frac{nz}{Q}\right\rceil\bmod n.
$$

给定固定搜索半径 $W$，定义候选集合

$$
\mathcal S_W(y)
=
\{s_0+\delta\bmod n:-W\le\delta\le W\},
$$

$$
\mathcal T_W(z)
=
\{t_0+\delta\bmod n:-W\le\delta\le W\}.
$$

遍历

$$
s'\in\mathcal S_W(y),
\qquad
 t'\in\mathcal T_W(z).
$$

若 $s'\neq0$，构造候选

$$
d'=t'(s')^{-1}\pmod n.
$$

然后执行公开验证

$$
d'P\stackrel{?}=R.
$$

定义后处理器

$$
h_{W,\mathrm{DLP}}(y,z)
=
\begin{cases}
d', & d'P=R,\\
\bot, & \text{没有候选通过验证。}
\end{cases}
$$

对于 prime-order subgroup，验证通过的 $d'$ 在 $\mathbb Z_n$ 中唯一，因此在精确经典计算下几乎不存在“输出错误但验证通过”的情况。

### 15.2 该后处理器只是 benchmark 选择，不是唯一 DLP 后处理

更强的实际后处理可以使用：

- 非对称 limited search；
- meet-in-the-middle search；
- lattice reconstruction；
- 增加少量量子群操作与经典搜索的 trade-off。

为了使 benchmark 可复现，必须固定并报告：

$$
(m,W,K,\text{search rule},\text{candidate budget}).
$$

简单二维窗口的候选复杂度约为

$$
O((2W+1)^2)
$$

每个 pair；meet-in-the-middle 或 lattice 方法会有不同的成本模型。

### 15.3 小例子

设

$$
n=13,
\qquad
 d=5.
$$

理想测量得到

$$
u=4,
\qquad
 v=du\bmod13=7.
$$

因为

$$
4^{-1}=10\pmod{13},
$$

所以

$$
\hat d
=7\cdot10\bmod13
=5.
$$

最后验证

$$
5P=R.
$$

---

## 16. DLP 中一个 sample 的定义与 $K$-sample 聚合

DLP benchmark 中，一个完整 algorithmic sample 必须定义为

$$
\boxed{(y,z)}.
$$

它表示同一个随机标签 $s$ 下的两次相关 QFT 输出。不能把 $y$ 和 $z$ 分别计作两个独立 samples。

对 $K$ 个独立 samples：

$$
\bigl((y_1,z_1),\ldots,(y_K,z_K)\bigr),
$$

每个 pair 使用独立随机标签

$$
s_i\sim\operatorname{Unif}(\mathbb Z_n).
$$

在本文建议的 pairwise fixed-window 后处理器中，只要存在一个 pair 恢复并验证出 $d$，整个 block 即成功：

$$
h_{K,\mathrm{DLP}}
=
 d
\quad\Longleftrightarrow\quad
\exists i,
\ h_{W,\mathrm{DLP}}(y_i,z_i)=d.
$$

若单 pair 成功率为 $s_{Q,W}$，不同 pairs 独立，则

$$
\boxed{
P_{\mathrm{DLP}}^{(K)}
=
1-(1-s_{Q,W})^K
}.
$$

此时经典成本随 $K$ 近似线性增长，而不是 subset 数量的指数增长。

但这不是 DLP 的普遍定理：若选择跨多个 runs 的联合 lattice post-processing，则不同 samples 会被联合处理，其经典成本也会随 lattice 维度增长。

此外，虽然 DLP 的 $K$ 可以在实现上取较大值，benchmark 不宜只报告很大的 $K$。因为

$$
1-(1-s)^K
$$

会迅速饱和到 $1$，从而掩盖不同 QFT 实现之间的差异。建议至少报告

$$
K\in\{1,2,4,8\}
$$

的完整曲线，并把 $K=1$ 或 $K=2$ 作为主要对比点。

---

## 17. DLP 指标与 baseline

### 17.1 主指标

$$
P_{\mathrm{key}}^{(K)}
=
\Pr\left[
 h_{K,\mathrm{DLP}}=d
\right].
$$

在 prime-order subgroup 和精确公开验证下，可以定义

$$
P_{\mathrm{wrong}}^{(K)}\approx0,
$$

$$
P_{\mathrm{null}}^{(K)}
=1-P_{\mathrm{key}}^{(K)}.
$$

这里的 null 表示没有任何候选通过 $d'P=R$，而不是输出了另一个有效私钥。

### 17.2 Arithmetic ideal baseline

若使用精确 $\operatorname{IQFT}_n$，单 sample 直接得到

$$
(s,ds\bmod n).
$$

对于 prime $n$，只有 $s=0$ 时无法求逆，因此

$$
s_{\mathrm{arith,DLP}}
=1-\frac1n.
$$

$K$ 个独立 pairs 的成功率为

$$
\boxed{
P_{\mathrm{arith,DLP}}^{(K)}
=1-\frac1{n^K}
}.
$$

更一般地，如果 $n$ 不是素数，并且单 sample 仅在 $s$ 可逆时成功，则

$$
s_{\mathrm{arith,DLP}}
=\frac{\varphi(n)}n.
$$

### 17.3 Finite-$Q$ ideal baseline

定义单 pair 理想成功率

$$
\begin{aligned}
s_{Q,W}
=
\frac1n
\sum_{s=0}^{n-1}
\sum_{y,z=0}^{Q-1}
&D_{Q,n}(y\mid s)
D_{Q,n}(z\mid ds\bmod n)\\
&\cdot
\mathbf1\!\left[
 h_{W,\mathrm{DLP}}(y,z)=d
\right].
\end{aligned}
$$

然后

$$
P_{\mathrm{DLP,ideal}}^{(K)}
=1-(1-s_{Q,W})^K.
$$

### 17.4 Uniform baseline

令

$$
y,z
\sim
\operatorname{Unif}\{0,\ldots,Q-1\},
$$

并使用相同后处理器计算 $s_{U,W}$。

在理想化的 $Q=n$、$W=0$、prime $n$ 情况下，成功 pair 数为 $n-1$，全部 pairs 数为 $n^2$，因此

$$
s_U
=
\frac{n-1}{n^2}
\approx\frac1n.
$$

对于 256-bit prime order，该 accidental success 可以忽略；但当 $W>0$ 或使用更强搜索时，必须用同一个后处理器重新数值计算 uniform baseline。

归一化指标可定义为

$$
\eta_{\mathrm{DLP}}^{(K)}
=
\frac{
P_{\mathrm{exp}}^{(K)}-P_{\mathrm{unif}}^{(K)}
}{
P_{\mathrm{ideal}}^{(K)}-P_{\mathrm{unif}}^{(K)}
}.
$$

---

# 第三部分：两类 benchmark 的关键比较

## 18. 为什么 Order-finding 用 continued fractions，而 DLP 通常不用

### 18.1 Order-finding：未知分母的有理重构

测量只给出

$$
\frac yQ\approx\frac sr,
$$

其中 $s$ 和 $r$ 均未知，只知道 $r<N$。因此后处理必须在所有分母小于 $N$ 的有理数中寻找合适的 $s/r$。continued fractions 正是这个 rational-reconstruction 问题的高效算法。

### 18.2 DLP：分母已知，只需恢复 residue

DLP 输出满足

$$
\frac yQ\approx\frac sn,
$$

$$
\frac zQ\approx\frac{ds\bmod n}{n},
$$

其中 $n$ 已知。因此可以直接计算

$$
\left\lfloor\frac{ny}{Q}\right\rceil,
\qquad
\left\lfloor\frac{nz}{Q}\right\rceil,
$$

再做附近搜索、模逆与公开验证。

对 DLP 输出做 continued fractions 并非数学上禁止，但它主要会重新恢复一个本来已知的分母 $n$，因此通常不是最直接的做法。

### 18.3 两者都可以使用 stronger classical post-processing

| Benchmark | 基础后处理 | 可选增强后处理 |
|---|---|---|
| Order-finding | continued fractions + LCM | nearby frequencies、denominator-multiple search、order reduction、lattice |
| DLP/ECDLP | scaling/rounding + modular division | nearby residues、meet-in-the-middle、lattice、额外量子群操作 |

经典后处理越强，应用成功率通常越高，但 benchmark 对 QFT 本身的区分力可能越弱。因此后处理预算必须被视为 benchmark 定义的一部分，而不能在不同方法之间随意变化。

---

## 19. 多样本聚合的差异

### 19.1 Order-finding

每个理想 sample 只得到

$$
q_i=\frac r{\gcd(s_i,r)},
$$

因此不同 samples 提供的是 order 的互补素因子信息，必须进行 LCM 聚合。

在 noisy benchmark 中，为排除坏分母，strict 后处理可能维护 subset-LCM 候选集合。最坏候选数可能随 $K$ 指数增长，但实际实现可以：

- 合并重复 LCM 状态；
- 设置最大候选预算；
- 使用 beam search；
- 改用 lattice 或固定搜索规则。

所以“Order-finding 的 $K$ 不能大”不是算法定理，而是特定 strict subset-LCM 后处理器的工程约束。

### 19.2 DLP/ECDLP

在 prime-order、pairwise fixed-window 后处理器中，每个有效 pair 已经可以独立恢复完整 $d$。因此不同 pairs 的默认聚合方式只是 logical OR：

$$
\text{至少一个 pair 成功}
\Longrightarrow
\text{整体成功}.
$$

它不需要像 Order-finding 那样汇集不同 samples 的素因子信息。

---

## 20. 总体对比表

| 维度 | Order-finding benchmark | DLP / ECDLP benchmark |
|---|---|---|
| 应用目标 | 恢复未知 order $r$，进一步分解 $N$ | 恢复未知离散对数或私钥 $d$ |
| 公开数学结构 | $a$ 在模 $N$ 乘法群中的作用 | 已知阶循环群 $\langle P\rangle$，公开点 $R=dP$ |
| 已知分母 | 否，$r$ 未知 | 是，群阶 $n$ 已知 |
| 隐藏结构 | 一维周期 | 二维隐藏子群方向 $(-d,1)$ |
| 随机标签 | $s\in\mathbb Z_r$ | $s\in\mathbb Z_n$ |
| QFT 输入 | 一个 phase state $\|\xi_s\rangle$ | 两个相关 phase states $\|\xi_s\rangle\otimes\|\xi_{ds}\rangle$ |
| QFT 调用 | 1 次 | 2 次，即 $\operatorname{QFT}\otimes\operatorname{QFT}$ |
| 一个 sample | 单个整数 $y$ | 一个相关整数对 $(y,z)$ |
| 典型寄存器尺度 | $m\approx2\log_2N$ | 每个寄存器 $m\approx\log_2n+\Delta$ |
| 单 sample 关系 | $y/Q\approx s/r$ | $y/Q\approx s/n$，$z/Q\approx(ds\bmod n)/n$ |
| 基础经典后处理 | continued fractions | known-denominator scaling / limited search |
| 候选计算 | 候选分母 $q$ | $d'=t'(s')^{-1}\bmod n$ |
| 经典验证 | $a^L\equiv1\pmod N$ | $d'P=R$ |
| 多样本聚合 | LCM / subset-LCM | 默认对各 pair 独立处理并取 OR |
| arithmetic 单样本成功率 | $\varphi(r)/r$ | prime $n$ 下 $1-1/n$ |
| arithmetic $K$-sample 成功率 | $\prod_{p\mid r}(1-p^{-K})$ | prime $n$ 下 $1-n^{-K}$ |
| 主要失败模式 | wrong order、null、只恢复部分 order | 主要为 null；精确验证后 wrong 近似为零 |
| 是否易于随 $K$ 饱和 | 取决于 $r$ 的素因子结构 | 很容易快速饱和 |
| 对单个 QFT 的隔离性 | 更强 | 较弱，测量两次 QFT 的组合效果 |
| 应用后处理丰富度 | 高 | 相对简单 |

---

# 第四部分：简单 depolarizing-noise 扩展

## 21. Order-finding 的输出级 depolarizing 模型

考虑

$$
\rho_\lambda
=(1-\lambda)\rho
+\lambda\frac{I}{Q}.
$$

测量分布为

$$
P_{\lambda,Q}(y\mid s)
=(1-\lambda)D_{Q,r}(y\mid s)
+\lambda\frac1Q.
$$

使用与无噪声和 uniform baseline 相同的后处理器，即可得到 $P_{\mathrm{OF}}^{(K)}(\lambda)$。对于 strict order-recovery，附件中的成功率理论可通过 divisor lattice 上的 Möbius 反演给出精确 finite-$Q$ 公式。

在 $Q$ 足够大且 uniform accidental success 可忽略时：

- 若 $r=p$ 是大素数，
  $$
  P_{\mathrm{OF}}^{(K)}(\lambda)
  \approx1-\lambda^K;
  $$
- 若 $r=2pq$ 且 $p,q$ 是大奇素数，
  $$
  P_{\mathrm{OF}}^{(K)}(\lambda)
  \approx
  1-\left(\frac{1+\lambda}{2}\right)^K.
  $$

这说明 Order-finding 的噪声曲线同时由 QFT 输出噪声和 $r$ 的素因子结构决定。

---

## 22. DLP 的输出级 depolarizing 模型

### 22.1 对整个 pair 施加 global depolarizing

若定义

$$
P_\lambda(y,z)
=(1-\lambda)P_{Q,n}(y,z)
+\lambda\frac1{Q^2},
$$

单 pair 成功率为

$$
s_\lambda
=(1-\lambda)s_{Q,W}
+\lambda s_{U,W}.
$$

$K$ 个独立 pairs 的成功率为

$$
P_{\mathrm{DLP}}^{(K)}(\lambda)
=
1-(1-s_\lambda)^K.
$$

### 22.2 两个 QFT 分别受到独立 depolarizing

定义单个 QFT 输出分布

$$
D_{\lambda,Q,n}(y\mid t)
=(1-\lambda)D_{Q,n}(y\mid t)
+\lambda\frac1Q.
$$

则单 pair 成功率为

$$
\begin{aligned}
s_{\lambda,Q,W}
=
\frac1n
\sum_s\sum_{y,z}
&D_{\lambda,Q,n}(y\mid s)
D_{\lambda,Q,n}(z\mid ds)\\
&\cdot
\mathbf1[h_{W,\mathrm{DLP}}(y,z)=d].
\end{aligned}
$$

在理想化的精确 residue、$W=0$、prime $n$ 情况下，可以得到

$$
\boxed{
s_\lambda
=
\left(1-\frac1n\right)
\left[
(1-\lambda)^2
+
\frac{2\lambda-\lambda^2}{n}
\right]
}.
$$

对 256-bit prime order，$1/n$ 项可以忽略，因此

$$
s_\lambda\approx(1-\lambda)^2.
$$

这与 pair-level global depolarizing 下的近似 $s_\lambda\approx1-\lambda$ 不同。因此论文必须明确 $\lambda$ 是：

- per-QFT noise；还是
- per-pair global noise。

---

# 第五部分：benchmark 选择建议

## 23. 哪一个更适合作为 QFT 面向应用 benchmark

### 23.1 Order-finding 更适合作为主 benchmark

原因包括：

1. **只调用一个 QFT**，应用成功率更容易归因到单个 QFT+M 子程序；
2. phase-state ensemble、finite-$Q$ baseline、uniform baseline 和 depolarizing 理论已经完整；
3. 不同 $r$ 的素因子结构提供自然的实例难度层次；
4. strict/wrong/null 可以区分“误导性输出”和“信息不足”；
5. 成功率不容易在很小的 $K$ 下统一饱和，更有利于区分不同 QFT 实现。

Order-finding 后处理较复杂不是缺点。应用 benchmark 本来就需要固定真实应用中的经典语义；关键是把 continued fractions、LCM、验证和搜索预算完整公开并保持不变。

### 23.2 ECDLP 更适合作为 paired-QFT 扩展 benchmark

ECDLP benchmark 的价值在于：

1. 验证同一个 QFT 子程序被调用两次后，两个相关输出是否仍足以恢复应用目标；
2. 后处理简单且具有确定性的公开点验证；
3. 与 ECC-256 的实际量子威胁模型直接对应；
4. 可用于研究两个 QFT streams 的相关失败、非对称噪声和 sequential / semiclassical 实现。

但它作为唯一主 benchmark 有三个问题：

- 一个 sample 同时包含两次 QFT，难以把失败归因到单次 QFT；
- prime-order DLP 的 arithmetic one-shot success 接近 $1$，容易快速饱和；
- 强公开验证使 wrong output 几乎消失，失效模式不如 Order-finding 丰富。

### 23.3 推荐组合

$$
\boxed{
\text{Primary benchmark: Order-finding}
}
$$

$$
\boxed{
\text{Secondary benchmark: ECDLP paired-QFT}
}
$$

主文可以用 Order-finding 证明 batched dynamic QFT 在单次 QFT 应用中的价值；补充实验再用 ECDLP 说明该方法能够推广到需要两组相关 Fourier outputs 的 Shor 应用。

---

## 24. 与后续 QEC 资源分析的接口

两类 benchmark 都可以提供一个应用级函数：

$$
P_{\mathrm{app}}
=
F(m,K,\lambda,\text{post-processing budget},\text{instance}).
$$

它可以用于反推出允许的 effective application-level error budget。但它不能单独给出物理 qubit 数，因为还需要完成映射：

$$
p_{\mathrm{phys}}
\longrightarrow
p_L(\text{code},d_c)
\longrightarrow
\lambda_{\mathrm{block}}
\longrightarrow
P_{\mathrm{app}}.
$$

其中必须加入：

- 完整 modular exponentiation 或椭圆曲线 arithmetic 的逻辑 gate count / depth；
- QEC code、distance 和 decoder；
- magic-state factories；
- routing、memory 和 feed-forward；
- 目标运行时间与允许重复次数 $K$。

因此，QFT benchmark 最适合作为完整资源模型中的一个 **post-processing-aware application success layer**，而不是替代完整 fault-tolerant resource estimation。

---

# 第六部分：可直接引用的正式定义

## 25. Order-finding benchmark 定义

> **Shor post-processed strict order-recovery benchmark**
>
> 给定实例 $(N,a,r,m)$，其中 $r=\operatorname{ord}_N(a)$、$Q=2^m$。对每个 quantum sample，均匀随机选择 $s\in\mathbb Z_r$，制备
> $$ |\xi_s^{(r,Q)}\rangle = \frac1{\sqrt Q} \sum_{x=0}^{Q-1} \exp\!\left(2\pi i\frac{sx}{r}\right)|x\rangle,$$
> 执行 $\operatorname{IQFT}_Q$ 并测量得到 $y$。将 $K$ 个独立测量结果输入固定的 continued-fractions、subset-LCM 和 modular-validation 后处理器。指标定义为后处理器直接输出真实 order $r$ 的概率。

主指标为

$$
P_{\mathrm{ord,strict}}^{(K)}
=
\Pr[h_{K,\mathrm{OF}}^{\mathrm{strict}}=r].
$$

同时报告

$$
P_{\mathrm{wrong}}^{(K)},
\qquad
P_{\mathrm{null}}^{(K)},
\qquad
\eta_{\mathrm{OF}}^{(K)}.
$$

---

## 26. ECDLP benchmark 定义

> **ECDLP post-processed private-key recovery benchmark**
>
> 给定 prime-order 椭圆曲线子群实例 $(\mathcal G,P,n,R=dP,m,W)$，其中 $Q=2^m$。对每个 quantum sample，均匀随机选择 $s\in\mathbb Z_n$，制备相关 phase-state pair
> $$
> |\xi_s^{(n,Q)}\rangle
> \otimes
> |\xi_{ds\bmod n}^{(n,Q)}\rangle,
> $$
> 对两个寄存器分别执行 $\operatorname{IQFT}_Q$ 并测量得到 $(y,z)$。将该 pair 输入固定预算的 residue search 后处理器，计算候选
> $$
> d'=t'(s')^{-1}\pmod n,
> $$
> 并通过
> $$
> d'P\stackrel{?}=R
> $$
> 验证。一个 algorithmic sample 是完整的相关输出对 $(y,z)$；指标定义为 $K$ 个独立 pairs 中至少有一个恢复真实 $d$ 的概率。

主指标为

$$
P_{\mathrm{key}}^{(K)}
=
\Pr[h_{K,\mathrm{DLP}}=d].
$$

同时报告

$$
P_{\mathrm{null}}^{(K)},
\qquad
\eta_{\mathrm{DLP}}^{(K)},
$$

并固定公开

$$
(m,W,K,\text{search rule},\text{candidate budget}).
$$

---

## 27. 最终建议的实验汇报矩阵

| 类别 | 建议参数或指标 |
|---|---|
| Order-finding 实例 | 对齐 / 非对齐 $r$，prime order，$r=2pq$，多个 $N$ 和 $m$ |
| Order-finding $K$ | $K\in\{1,2,4,8,16\}$ |
| Order-finding 指标 | strict、wrong、null、factor success、normalized success |
| ECDLP 实例 | 小规模 prime-order toy curves；资源允许时扩展到合成 256-bit phase states |
| ECDLP $m$ | $m=\lceil\log_2n\rceil+\Delta$，$\Delta\in\{0,1,2,4\}$ |
| ECDLP $K$ | $K\in\{1,2,4,8\}$，重点报告 $K=1,2$ |
| ECDLP 后处理 | 固定 $W$ 或固定候选预算；不可在不同 QFT 方法之间改变 |
| ECDLP 指标 | key recovery、null、normalized success |
| 共同 baseline | finite-$Q$ ideal、arithmetic ideal、uniform random |
| 共同补充指标 | process fidelity、TVD、执行时间、live-qubit exposure |

---

## 28. 来源说明

DLP/ECDLP 的完整算法、两指数寄存器结构和椭圆曲线实现背景，参考：

- Peter W. Shor, *Algorithms for quantum computation: discrete logarithms and factoring*, 1994；
- John Proos and Christof Zalka, *Shor's discrete logarithm quantum algorithm for elliptic curves*, arXiv:quant-ph/0301141；
- Martin Ekerå, *Revisiting Shor's quantum algorithm for computing general discrete logarithms*, arXiv:1905.09084；
- Martin Roetteler, Michael Naehrig, Krysta M. Svore, and Kristin Lauter, *Quantum resource estimates for computing elliptic curve discrete logarithms*, arXiv:1706.06752。

本文中的 ECDLP QFT-only phase-state benchmark 是将完整 DLP 电路在 QFT 前对工作寄存器取 partial trace 后得到的直接构造；固定窗口 residue search 是为了 benchmark 可复现而选择的简化后处理器，不声称是最优 DLP 后处理算法。
