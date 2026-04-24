# 针对 Shor 算法应用的 QFT Benchmark 实验方案（草案）

## 1. 方案定位

现有 QFT+Measurement（QFT+M）实验通常更强调**量子过程本身**的正确性，例如 process fidelity；动态 QFT 的 PRL 也是以 certified process fidelity 为主，并额外用一个简单的 periodic state 展示四峰分布。该文同时明确指出，QFT+M 是 Shor 算法和量子相位估计（QPE）的核心子程序。另一方面，IonQ 的 application-oriented benchmark 使用 plurality voting 聚合不同编译 variant 的直方图，并且作者明确说明这种方法在理想输出主要集中在一个或少数几个 bitstring 时尤其有效。基于这两点，本方案的目标不是再定义一个“量子 fidelity”，而是引入一个**Shor-specific、post-processing-aware 的应用型 benchmark**：直接问“这些 QFT+M 输出，经过 Shor 的经典后处理之后，能否恢复正确 order 或因子”。 ([arXiv][1])

本方案的核心思想是：不要求实验输出分布逐点逼近理想分布，也不要求理想结果是单一 bitstring；而是把每个测量 bitstring 当成 Shor order-finding 子程序的输出，送入 continued fractions、LCM、模幂验证和 gcd 这条经典后处理链，统计最终恢复正确 order 或因子的概率。IBM Quantum Learning 和 Google Cirq 的教程都采用了这种后处理框架；IBM 还明确指出，多次样本的分母取 least common multiple（LCM）可以高概率恢复真正的 order。Ekerå 则进一步证明，若在经典后处理中加入固定预算的有限搜索，单次 quantum run 的 order-recovery 成功率可以显著提高。 ([IBM Quantum][2])

---

## 2. 适用范围与默认约定

本方案适用于如下实验设置：

1. 只运行 **最后的 QFT / inverse QFT + measurement**；
2. 不完整实现 modular exponentiation；
3. 允许事先选择某个 Shor 实例 $(N,a)$，其中 $a$ 与 $N$ 互素；
4. 允许制备一个尽量简单的初态，然后执行 QFT+M；
5. 最终从采样结果中计算应用指标。

**默认约定：** 以下正文以 **inverse QFT + measurement**（即 Shor/QPE 里的标准形式）叙述。若硬件上实际运行的是 **QFT + measurement**，只需把输入态中的相位取共轭，或等价地把相位符号取反；后处理流程与指标定义不变。

---

## 3. 符号表

| 符号                               | 含义                                                            |
| -------------------------------- | ------------------------------------------------------------- |
| $N$                              | 待分解的奇合数，或更一般地，Shor order-finding 的模数                          |
| $a$                              | 满足 $\gcd(a,N)=1$ 的整数，且 $1<a<N$                                |
| $r$                              | $a$ 在模 $N$ 下的 order，即最小正整数，使 $a^r \equiv 1 \pmod N$           |
| $m$                              | 控制寄存器（即 QFT 输入寄存器）的 qubit 数                                   |
| $Q$                              | $Q =2^m$                                                     |
| $s$                              | Shor/QPE 中的 eigenphase 标签，$s \in \{0,1,\dots,r-1\}$             |
| $x$                              | 计算基索引，$x \in \{0,1,\dots,Q-1\}$                                 |
| $y$                              | 测量得到的整数输出，$y \in \{0,1,\dots,Q-1\}$                             |
| $\alpha$                         | 单次测量对应的相位估计值，$\alpha = y/Q$                                   |
| $K$                              | 一个算法实例允许使用的 QFT+M quantum samples 数；即采样成本                     |
| $C_s(y)$                         | 对固定 $s$ 的实验中，测得整数 $y$ 的计数                                     |
| $T_s$                            | 固定 $s$ 时总 shot 数，$T_s = \sum_y C_s(y)$                        |
| $\hat P_{\mathrm{exp}}(y\mid s)$ | 实验条件分布，$\hat P_{\mathrm{exp}}(y\mid s)=C_s(y)/T_s$            |
| $\mathrm{CF}_N(\alpha)$          | 对 $\alpha$ 做 continued fractions，在分母上限 $N-1$ 下得到的最佳有理逼近 $p/q$ |
| $q$                              | 单次样本经 continued fractions 得到的候选分母                             |
| $\mathcal L$                     | 一个 $K$-sample block 上由各个 $q_i$ 生成的 candidate LCM 集合           |
| $\mathcal V$                     | 通过模幂验证 $a^L \equiv 1 \pmod N$ 的候选集合                           |
| $\bot$                           | 后处理器拒绝输出，即“没有可信 order / 因子”                                   |
| $h_K^{\rm strict}$               | 严格 order-recovery 后处理器                                        |
| $h_K^{\rm red}$                  | 带 order reduction 的后处理器                                       |
| $h_K^{\rm fac}$                  | 因子恢复后处理器                                                      |
| $P_{\rm ord,strict}^{(K)}$       | $K$-sample 严格 order 恢复成功率                                     |
| $P_{\rm ord,red}^{(K)}$          | $K$-sample 约化后 order 恢复成功率                                    |
| $P_{\rm fac}^{(K)}$              | $K$-sample 因子恢复成功率                                            |
| $P_{\rm wrong}^{(K)}$            | 严格后处理输出了错误 order 的概率                                          |
| $P_{\rm null}^{(K)}$             | 严格后处理拒绝输出的概率                                                  |

---

## 4. Benchmark 的核心定义

### 4.1 Benchmark 实例

选取一个实例
$$
I=(N,a,r,m),
$$
其中
$$
r = \operatorname{ord}_N(a), \qquad Q=2^m.
$$

建议实例集 $\mathcal I$ 同时覆盖：

* **对齐型**：$r \mid Q$，理想峰恰好落在整数网格上；
* **非对齐型**：$r \nmid Q$，理想峰会展宽到邻近整数；
* **偶数 order**：便于定义 factor success；
* **奇数 order**：只用于 order benchmark，不用于 factor benchmark。

建议不要只用 $r=4$ 这种过于友好的案例。动态 QFT 论文中的 periodic-state 可视化就是 4 峰的简单 sanity check，但它不能充分代表一般 Shor 实例。 ([arXiv][1])

一个可行的小规模实例集示例：

| 实例 | $N$ | $a$ | $r$ | 用途                  |
| -- | --: | --: | --: | ------------------- |
| I1 |  15 |   2 |   4 | sanity check        |
| I2 |  21 |   4 |   3 | 非因式恢复，仅 order       |
| I3 |  21 |   2 |   6 | order + factor      |
| I4 |  35 |   2 |  12 | 非对齐型、order + factor |
| I5 |  55 |   2 |  20 | 更复杂的 even order     |

若目标是尽量贴近 textbook Shor，控制寄存器位数通常应满足 $m \approx 2\lceil \log_2 N\rceil$；IBM Learning 也指出，取 $m = 2\log N + 1$ 足以保证 phase estimation 的精度要求。若实际硬件受限只能取更小的 $m$，则必须同时报告 finite-$Q$ ideal baseline，而不能只与 $Q\to\infty$ 的理想极限比较。 ([IBM Quantum][2])

---

## 5. 实验流程

## 5.1 初态制备：Shor-like phase-state ensemble

### 5.1.1 目标输入态

对每个
$$
s \in \{0,1,\dots,r-1\},
$$
定义输入态
$$
|\xi_s\rangle =
\frac{1}{\sqrt Q}
\sum_{x=0}^{Q-1}
e^{2\pi i sx/r}|x\rangle ,
$$
这里采用的是 **inverse QFT convention**。若实际执行的是 forward QFT，则改用
$$
|\tilde{\xi}_s\rangle =
\frac{1}{\sqrt Q}
\sum_{x=0}^{Q-1}
e^{-2\pi i sx/r}|x\rangle .
$$

### 5.1.2 为什么要对 $s$ 做均匀随机

在完整 Shor 算法中，inverse QFT 之前，控制寄存器与工作寄存器的联合态可写成
$$
\frac{1}{\sqrt r}\sum_{s=0}^{r-1} |\xi_s\rangle \otimes |u_s\rangle,
$$
其中 $|u_s\rangle$ 是对应本征值 $e^{2\pi i s/r}$ 的本征态。IBM Learning 明确指出，对 $|1\rangle$ 运行 phase estimation，等效于均匀随机地选到一个 $k \in \{0,\dots,r-1\}$ 的 eigenphase 标签。也就是说，真实 Shor 并不是固定一个 $s$，而是隐式地对 $s$ 做均匀混合。 ([IBM Quantum][2])

因此，主 benchmark 必须使用
$$
s \sim \mathrm{Unif} \{0,1,\dots,r-1\}
$$
的 **全 $s$ ensemble**，而不是固定 $s=1$。固定 $s=1$ 可以作为附加诊断实验，但不能作为主 application benchmark。

### 5.1.3 物理可实现性：产品态分解

令
$$
x = \sum_{j=0}^{m-1} x_j 2^j,\qquad x_j\in\{0,1\},
$$
则
$$
|\xi_s\rangle = 
\bigotimes_{j=0}^{m-1}
\frac{|0\rangle + e^{2\pi i s 2^j/r}|1\rangle}{\sqrt 2}.
$$

因此，该输入态只需要：

* 每个 qubit 一个 Hadamard；
* 每个 qubit 一个 $Z$-axis 相位旋转。

**不需要多比特纠缠态制备。**
这正符合你们“初态制备不能太复杂”的实验约束。

---

## 5.2 数据采集

对每个 benchmark 实例 $I=(N,a,r,m)$：

### 方案 A：在线随机化 $s$

每个 shot 前随机选择
$$
s \sim \mathrm{Unif}\{0,1,\dots,r-1\},
$$
制备 $|\xi_s\rangle$，执行 QFT/IQFT+M，并记录测量整数 $y$。

### 方案 B：离线等权混合（更推荐）

对每个 $s$ 单独运行实验，收集
$$
C_s(y), \qquad T_s=\sum_y C_s(y),
$$
得到条件分布
$$
\hat P_{\mathrm{exp}}(y\mid s) = \frac{C_s(y)}{T_s}.
$$

后续在经典后处理中按均匀权重混合 $s$。
这种做法有两个优点：

1. 便于调试每个 $s$ 的失效模式；
2. 不改变主 benchmark 的 Shor-like 语义。

### 输出整数的统一定义

若电路末尾省略了 QFT 的终端 SWAP，则必须先做**固定的 bit-reversal**，再把 bitstring 解码为整数
$$
y\in\{0,\dots,Q-1\}.
$$
整个实验中必须始终使用同一 endianness 约定。

---

## 5.3 单次样本的 continued-fractions 处理

对任意一次测量结果 $y$，定义
$$
\alpha = \frac{y}{Q}.
$$

令
$$
\mathrm{CF}_N(\alpha)=\frac{p}{q}
$$
表示：对 $\alpha$ 做 continued fractions，并在分母上限 $N-1$ 下取得到的最佳有理逼近，其中 $p,q$ 互素且 $1\le q < N$。

**实现建议：** 不要把 $\alpha$ 转成浮点数；直接把 $\alpha$ 当作精确有理数 $y/Q$ 处理。

若 $p=0$，则视该样本为**无信息样本**，等价于 $q=1$。

Cirq 的 Shor 教程给出的经典后处理正是：把 exponent register 的测量值转成 $\alpha=y/2^m$，使用 continued fractions 在分母上界 $n$ 内求有理逼近，再检查 $x^r\bmod n =1$ 是否成立；IBM Learning 也给出了相同思路，并明确指出当得到的是 $k/r$ 的最简形式时，输出分母只一定整除真正的 $r$。 ([Google Quantum AI][3])

---

## 5.4 $K$-sample block 的 Shor 后处理器

设一个算法实例允许使用 $K$ 个独立的 quantum samples。
记这一批样本为
$$
Y=(y_1,y_2,\dots,y_K).
$$

对每个 $y_i$，计算
$$
\mathrm{CF}_N(y_i/Q)=p_i/q_i.
$$

保留所有 $q_i>1$ 的 informative denominators。

### 5.4.1 候选 LCM 集合

定义递推集合
$$
\mathcal L_0={1},
$$
$$
\mathcal L_i =
\mathcal L_{i-1}
\cup
\left\{
\mathrm{lcm} (L,q_i):L\in \mathcal L_{i-1}
\right\},
\qquad i=1,\dots,K.
$$

最后定义
$$
\mathcal L =
\{L\in\mathcal L_K:\ 1<L<N\}.
$$

这相当于对所有子集做 subset-LCM，但实现上不需要显式枚举 $2^K$ 个子集；只需动态维护“可达 LCM 集合”。对 $K\in\{1,2,4,8,16\}$ 这类应用规模，代价很低。

### 5.4.2 模幂验证

定义
$$
\mathcal V =
\{L\in\mathcal L:\ a^L \equiv 1 \pmod N\}.
$$

注意：
$$
a^L \equiv 1 \pmod N
$$
只能说明 $L$ 是真正 order $r$ 的倍数；不能单独说明 $L=r$。

---

## 5.5 三个后处理器

### 5.5.1 严格 order-recovery 后处理器

定义
$$
h_K^{\rm strict}(Y) =
\begin{cases}
\bot, & \mathcal V=\varnothing, \\
\min \mathcal V, & \mathcal V\neq\varnothing.
\end{cases}
$$

解释：

* 若没有任何 candidate 通过模幂验证，则拒绝输出；
* 否则输出最小的通过验证的候选。

这是一种**保守、可复现、与 QFT 子程序本身耦合较强**的后处理器。

### 5.5.2 带 order reduction 的后处理器

对每个 $L\in\mathcal V$，定义
$$
\operatorname{red}(L) =
\min\{d:\ d\mid L,\ a^d \equiv 1 \pmod N\}.
$$

然后定义
$$
h_K^{\rm red}(Y) =
\begin{cases}
\bot, & \mathcal V=\varnothing, \\
\min_{L\in\mathcal V}\operatorname{red}(L), & \mathcal V\neq\varnothing.
\end{cases}
$$

解释：

* 若某个 candidate $L$ 只是 $r$ 的倍数，如 $L=2r$ 或 $3r$，这里允许把它**约化回真正的最小 order**；
* 这更贴近“应用尽量利用所有经典信息”的视角；
* 但它比 strict 版本更依赖经典后处理，因此建议作为**辅助指标**而非唯一主指标。

Ekerå 的工作支持这一点：有限预算的经典搜索可以显著提高 order-recovery 的成功率，因此经典搜索预算本身应被视为 benchmark 定义的一部分。 ([arXiv][4])

### 5.5.3 因子恢复后处理器

定义
$$
h_K^{\rm fac}(Y)
$$
为如下过程：

1. 遍历 $\mathcal V$ 中所有偶数 $L$；
2. 计算
   $$
   d_-(L)=\gcd(a^{L/2}-1,N),\qquad
   d_+(L)=\gcd(a^{L/2}+1,N);
   $$
3. 若存在
   $$
   1<d_\pm(L)<N,
   $$
   则输出任意一个这样的非平凡因子；
4. 若所有候选都失败，则输出 $\bot$。

这个指标最接近“Shor 最终是否成功 factoring”，但它也最宽松，因为某些 order 的倍数有时仍然能给出正确因子。因此建议把它作为**最应用化的辅助指标**，而不是唯一主指标。

---

## 6. 指标定义

## 6.1 主指标：严格 order 恢复成功率

定义
$$
P_{\rm ord,strict}^{(K)} =
\Pr\left[h_K^{\rm strict}(Y)=r\right].
$$

这是本方案建议的**主 benchmark 指标**。

含义：给定 $K$ 个独立 QFT+M samples，Shor 后处理器是否能**直接输出正确 order**。

---

## 6.2 辅助指标 1：约化后 order 恢复成功率

定义
$$
P_{\rm ord,red}^{(K)} =
\Pr\left[h_K^{\rm red}(Y)=r\right].
$$

含义：允许有限的经典 order reduction 之后，能否恢复正确 $r$。

---

## 6.3 辅助指标 2：因子恢复成功率

定义
$$
P_{\rm fac}^{(K)} =
\Pr\left[h_K^{\rm fac}(Y)\neq\bot\right].
$$

含义：是否最终得到 $N$ 的非平凡因子。

---

## 6.4 失效模式分解

对 strict 后处理器，再定义

$$
P_{\rm wrong}^{(K)} =
\Pr\left[h_K^{\rm strict}(Y)\neq r,\ h_K^{\rm strict}(Y)\neq\bot\right],
$$

$$
P_{\rm null}^{(K)} =
\Pr\left[h_K^{\rm strict}(Y)=\bot\right].
$$

三者满足
$$
P_{\rm ord,strict}^{(K)}
+
P_{\rm wrong}^{(K)}
+
P_{\rm null}^{(K)}
= 1.
$$

这三个量能区分三种失败模式：

* **succ 高**：QFT+M 已足以恢复真正 order；
* **wrong 高**：后处理被 noisy denominators 误导，输出了错误答案；
* **null 高**：数据不足以支撑任何可信输出。

---

## 7. 如何从实验数据估计这些指标

设对每个 $s$ 都已收集到条件分布 $\hat P_{\rm exp}(y\mid s)$。

对固定的 $K$，进行 $M_{\rm MC}$ 次 classical Monte Carlo 重采样：

1. 对每个 trial $t$，独立采样
   $$
   s_1^{(t)},\dots,s_K^{(t)}
   \sim
   \mathrm{Unif}\{0,1,\dots,r-1\};
   $$
2. 对每个 $i$，从经验分布
   $$
   y_i^{(t)} \sim \hat P_{\rm exp}(\cdot \mid s_i^{(t)})
   $$
   抽取一个测量结果；
3. 令
   $$
   Y^{(t)}=(y_1^{(t)},\dots,y_K^{(t)});
   $$
4. 计算
   $$
   h_K^{\rm strict}(Y^{(t)}),\quad
   h_K^{\rm red}(Y^{(t)}),\quad
   h_K^{\rm fac}(Y^{(t)});
   $$
5. 取 Monte Carlo 平均。

例如：
$$
\widehat{P}_{\rm ord,strict}^{(K)} =
\frac{1}{M_{\rm MC}}
\sum_{t=1}^{M_{\rm MC}}
\mathbf{1}\left[h_K^{\rm strict}(Y^{(t)})=r\right].
$$

其他指标同理。

### 置信区间

建议对原始 counts 做 bootstrap，再在每个 bootstrap 副本上重复上述 Monte Carlo 过程，从而给出 $95%$ 置信区间。

### $K$ 的解释

这里的 $K$ 是**算法采样成本**，不是实验总 shot 数。

* $K=1$：单次 QFT+M 调用后就做后处理；
* $K=8$：允许 8 次独立 quantum samples 后再做后处理。

实验上可以收集成千上万 shots，但那些 shots 只是用来估计
$$
P^{(K)}
$$
这条曲线，不意味着真实算法一次实例会把所有 shots 都拿去后处理。

---

## 8. Baseline 与归一化

## 8.1 finite-$Q$ ideal baseline

对每个 $s$，理想 inverse QFT 的测量分布为
$$
P_{\rm ideal}(y\mid s) =
\frac{1}{Q^2}
\left|
\sum_{x=0}^{Q-1}
e^{2\pi i x\left(s/r-y/Q\right)}
\right|^2.
$$

等价地，可写为
$$
P_{\rm ideal}(y\mid s) =
\frac{1}{Q^2}
\frac{
\sin^2\left(\pi Q\left(s/r-y/Q\right)\right)
}{
\sin^2\left(\pi \left(s/r-y/Q\right)\right)
},
$$
当分母为 0 时取其连续极限。

然后用**和实验完全相同**的后处理器、完全相同的 $K$、完全相同的 Monte Carlo 过程，定义：

$$
P_{{\rm ord,strict,ideal}}^{(K)},\qquad
P_{{\rm ord,red,ideal}}^{(K)},\qquad
P_{{\rm fac,ideal}}^{(K)}.
$$

这一步不需要运行任何无噪声量子模拟器；只需要经典计算理想分布，再喂给同一个后处理器。

---

## 8.2 arithmetic ideal baseline

定义 **arithmetic ideal** 为 $Q\to\infty$ 极限下的理想参考：每个样本不再经历有限精度的 QFT 展宽，而是直接返回
$$
q_s = \frac{r}{\gcd(s,r)},
\qquad
s\sim\mathrm{Unif}\{0,\dots,r-1\}.
$$

如果只使用“单样本 continued fractions + 多样本 LCM”的后处理，那么
$$
P_{\rm arith}^{(K)} =
\Pr\left[
\operatorname{lcm}(q_{s_1},\dots,q_{s_K})=r
\right].
$$

这个概率有闭式公式
$$
P_{\rm arith}^{(K)} =
\prod_{p\mid r}\left(1-p^{-K}\right),
$$
其中乘积遍历 $r$ 的**不同素因子** $p$。

这个量可以理解为：

* **数论随机性**带来的理想极限成功率；
* **lcm-only 后处理**下的 $Q\to\infty$ 参考值。

它不是所有可能后处理的 universal upper bound，因为一旦允许更强的 classical search，理想成功率还可以继续提高。Ekerå 的结果就体现了这一点。 ([arXiv][4])

---

## 8.3 uniform random baseline

定义 random baseline 为：
$$
y \sim \mathrm{Unif}\{0,1,\dots,Q-1\},
$$
然后用**同一个**后处理器计算成功率：

$$
P_{{\rm ord,strict,unif}}^{(K)},\quad
P_{{\rm ord,red,unif}}^{(K)},\quad
P_{{\rm fac,unif}}^{(K)}.
$$

它的作用是扣除“偶然成功”：

* 随机 bitstring 也可能偶然拼出正确 $r$；
* 更宽松的后处理（例如允许 reduction、factor test）更容易出现这种 accidental success。

---

## 8.4 归一化指标

建议主文报告以下归一化 strict 指标：

$$
\eta_{\rm ord,strict}^{(K)} =
\frac{
P_{{\rm ord,strict,exp}}^{(K)} -
P_{{\rm ord,strict,unif}}^{(K)}
}{
P_{{\rm ord,strict,ideal}}^{(K)} -
P_{{\rm ord,strict,unif}}^{(K)}
}.
$$

同理可以定义
$$
\eta_{\rm ord,red}^{(K)},\qquad
\eta_{\rm fac}^{(K)}.
$$

其中：

* 分子表示实验相对随机输出提升了多少；
* 分母表示在相同 $Q$、相同后处理器下，理想 QFT+M 相对随机输出最多能提升多少。

---

## 9. 建议的结果汇报方式

## 9.1 主图

对每个实例 $I$，画出随 $K$ 变化的曲线：

* $P_{\rm ord,strict}^{(K)}$
* $P_{\rm wrong}^{(K)}$
* $P_{\rm null}^{(K)}$
* $\eta_{\rm ord,strict}^{(K)}$

建议 $K$ 取
$$
K\in\{1,2,4,8,16\}.
$$

## 9.2 辅图

同样画出

* $P_{\rm ord,red}^{(K)}$
* $P_{\rm fac}^{(K)}$

这两条曲线有助于回答：

* “如果允许有限的经典修正，实验输出到底有多接近可用？”
* “如果从最终 factoring 的角度看，这个 QFT+M 子程序到底有多有用？”

## 9.3 补充 benchmark

你们现在已经在做的两个量建议保留：

1. **process fidelity**：衡量量子过程误差；
2. **periodic-state TVD**：衡量完整输出分布与理想分布的偏差。

但建议把它们放在“补充基准”位置，而把
$$
P_{\rm ord,strict}^{(K)}
$$
放在主 benchmark 位置。这样三者形成互补关系：

| 指标                          | 回答的问题                                    |
| --------------------------- | ---------------------------------------- |
| process fidelity            | “这个 QFT+M 过程离理想量子过程有多近？”                 |
| TVD                         | “输出概率分布整体离理想分布有多近？”                      |
| Shor post-processed success | “从 Shor 应用角度，这些输出是否足以恢复 order / factor？” |

---

## 10. 建议的论文中主结论表述

建议正文把主 benchmark 定义为：

> **Shor post-processed strict order-recovery success probability**
>
> 对一个给定的 Shor 实例 $(N,a,r,m)$，在均匀随机的 eigenphase 标签 $s\in\{0,\dots,r-1\}$ 上制备对应 phase state，执行 QFT+measurement，并将所得的 $K$ 个独立测量结果输入到固定的经典后处理器（continued fractions + subset-LCM + modular validation）。指标定义为该后处理器直接输出正确 order $r$ 的概率。

补充定义：

> **Reduced order-recovery success probability**
>
> 在上述基础上，允许固定预算的 classical order reduction。

> **Factor-recovery success probability**
>
> 在上述基础上，进一步执行 Shor 的 gcd 步骤，统计最终恢复非平凡因子的概率。

---

# 附录 A：数学原理

## A.1 从完整 Shor 状态到本方案的输入态

令 $M_a$ 表示模 $N$ 的乘法算符，其本征态记为
$$
|u_s\rangle,\qquad s=0,1,\dots,r-1,
$$
对应本征值
$$
e^{2\pi i s/r}.
$$

在 Shor/QPE 图像中，inverse QFT 之前的联合态可写为
$$
|\Psi_{\rm pre}\rangle =
\frac{1}{\sqrt r}
\sum_{s=0}^{r-1}
|\xi_s\rangle\otimes |u_s\rangle,
$$
其中
$$
|\xi_s\rangle =
\frac{1}{\sqrt Q}
\sum_{x=0}^{Q-1}
e^{2\pi i sx/r}|x\rangle.
$$

由于 $|u_s\rangle$ 两两正交，trace out 工作寄存器后，控制寄存器的 reduced state 为
$$
\rho_{\rm in} =
\operatorname{Tr}_{\rm work}\bigl(|\Psi_{\rm pre}\rangle\langle\Psi_{\rm pre}|\bigr) = 
\frac{1}{r}
\sum_{s=0}^{r-1}
|\xi_s\rangle\langle\xi_s|.
$$

因此，下面两种做法是等价的：

1. 每个 shot 随机选 $s$，制备 $|\xi_s\rangle$；
2. 对每个 $s$ 分别采集数据，再在后处理中按均匀权重混合。

IBM Learning 也是用“对 $|1\rangle$ 运行 phase estimation 等效于均匀随机选择 $k$”的方式解释这一点。 ([IBM Quantum][2])

---

## A.2 为什么 $|\xi_s\rangle$ 是乘积态

写
$$
x=\sum_{j=0}^{m-1} x_j 2^j,\qquad x_j\in{0,1}.
$$
则
$$
e^{2\pi i sx/r} =
e^{2\pi i s(\sum_j x_j2^j)/r} =
\prod_{j=0}^{m-1}
e^{2\pi i s x_j 2^j/r}.
$$

因此
$$
|\xi_s\rangle =
\frac{1}{\sqrt Q}
\sum_{x_0,\dots,x_{m-1}\in{0,1}}
\prod_{j=0}^{m-1}
e^{2\pi i s x_j 2^j/r}
|x_0,\dots,x_{m-1}\rangle
$$
可因式分解为
$$
|\xi_s\rangle =
\bigotimes_{j=0}^{m-1}
\frac{|0\rangle + e^{2\pi i s2^j/r}|1\rangle}{\sqrt 2}.
$$

所以制备该态只需单比特门。

---

## A.3 理想 inverse QFT 后的测量分布

理想 inverse QFT 的定义为
$$
\mathrm{IQFT}|x\rangle =
\frac{1}{\sqrt Q}
\sum_{y=0}^{Q-1}
e^{-2\pi i xy/Q}|y\rangle.
$$

于是
$$
\mathrm{IQFT}|\xi_s\rangle =
\frac{1}{Q}
\sum_{y=0}^{Q-1}
\left(
\sum_{x=0}^{Q-1}
e^{2\pi i x(s/r-y/Q)}
\right)|y\rangle .
$$

记
$$
A_s(y) =
\frac{1}{Q}
\sum_{x=0}^{Q-1}
e^{2\pi i x(s/r-y/Q)},
$$
则理想测量概率为
$$
P_{\rm ideal}(y\mid s)=|A_s(y)|^2.
$$

利用等比数列求和，
$$
\sum_{x=0}^{Q-1} e^{2\pi i x\Delta} =
e^{\pi i (Q-1)\Delta}
\frac{\sin(\pi Q\Delta)}{\sin(\pi \Delta)},
\qquad
\Delta=s/r-y/Q,
$$
因此
$$
P_{\rm ideal}(y\mid s) =
\frac{1}{Q^2}
\frac{\sin^2(\pi Q\Delta)}{\sin^2(\pi \Delta)}.
$$

这说明：

* 当 $r \mid Q$ 时，理想分布会在整数网格上形成尖峰；
* 当 $r \nmid Q$ 时，理想分布会展宽到邻近整数。

---

## A.4 continued fractions 为何只返回 $r$ 的因子

若理想测量足够精确，则 continued fractions 会把
$$
\alpha=\frac{y}{Q}
$$
识别为
$$
\frac{s}{r}
$$
的最简形式。

设
$$
g = \gcd(s,r),
\qquad
s=gs',\quad r=gr',
\qquad
\gcd(s',r')=1.
$$
则
$$
\frac{s}{r}=\frac{s'}{r'}
$$
已约分，因此 continued fractions 返回的分母是
$$
q = r' = \frac{r}{\gcd(s,r)}.
$$

所以：

* 当 $\gcd(s,r)=1$ 时，单次样本直接给出 $q=r$；
* 当 $\gcd(s,r)>1$ 时，只得到 $r$ 的一个因子。

IBM Learning 明确指出，continued fractions 得到的是 $k/r$ 的 lowest terms，因此只能保证分母 $v$ 整除真正的 $r$，而多次采样的 denominator 取 LCM 后会高概率恢复 $r$。 ([IBM Quantum][2])

---

## A.5 为什么 LCM 能恢复 $r$

设一批理想样本对应的标签为
$$
s_1,\dots,s_K,
$$
每个样本返回
$$
q_i = \frac{r}{\gcd(s_i,r)}.
$$

若对每个素因子 $p\mid r$，都至少存在一个 $i$ 使
$$
p \nmid s_i,
$$
则 $q_i$ 中会保留 $p$ 的全部指数，因此
$$
\operatorname{lcm}(q_1,\dots,q_K)=r.
$$

反之，若某个素因子 $p\mid r$ 对所有 $i$ 都满足 $p\mid s_i$，则所有 $q_i$ 都缺少该素因子的一部分，LCM 也达不到 $r$。

对单个随机 $s$，事件“$p\nmid s$”的概率为
$$
1-\frac{1}{p}.
$$
对 $K$ 个独立样本，事件“至少有一个样本满足 $p\nmid s_i$”的概率为
$$
1-\frac{1}{p^K}.
$$

不同素因子通过中国剩余结构可视为独立约束，因此 arithmetic ideal 成功率为
$$
P_{\rm arith}^{(K)} =
\prod_{p\mid r}\left(1-p^{-K}\right).
$$

这个结果表明，在 lcm-only 理想极限下，成功率只依赖于 $r$ 的**不同素因子集合**，而不依赖其幂指数。

---

## A.6 为什么 noisy 样本会产生 $r$ 的倍数

在理想情况下，continued fractions 返回的分母必为 $r$ 的因子。
但在有噪声时，$y/Q$ 可能更接近一个完全不同的有理数，例如
$$
\frac{5}{12},
$$
于是得到
$$
q=12=2r
$$
这样的 order 倍数。

若多个样本给出
$$
q_1=3,\qquad q_2=4,
$$
则
$$
\operatorname{lcm}(3,4)=12,
$$
也可能是 $r=6$ 的倍数。

此时模幂验证
$$
a^L\equiv1\pmod N
$$
只能说明 $L$ 是真正 order 的倍数，而不能单独证明 $L=r$。这正是为何本文区分：

* strict order-recovery；
* reduced order-recovery；
* factor recovery。

---

## A.7 strict / wrong / null 的含义

对 strict 后处理器
$$
h_K^{\rm strict}(Y)\in \{2,\dots,N-1\}\cup\{\bot\},
$$
定义：
- 成功
   $$
   h_K^{\rm strict}(Y)=r.
   $$
- 错误
   $$
   h_K^{\rm strict}(Y)\neq r,\qquad h_K^{\rm strict}(Y)\neq\bot.
   $$
   典型情形是：
   * noisy denominator 让最小有效候选变成 (2r,3r,\dots)；
   * 或者返回了完全错误的候选。
- 空输出
   $$
   h_K^{\rm strict}(Y)=\bot.
   $$
   说明当前这 $K$ 个样本无法支持任何可信 candidate。

---

## A.8 arithmetic ideal 不是 universal upper bound

若后处理严格限定为：

1. 单样本只做 continued fractions；
2. 多样本只做 LCM；
3. 不做额外 multiple search；
4. 不做 order reduction；

则 $P_{\rm arith}^{(K)}$ 就是 $Q\to\infty$ 理想极限下的成功率。

但它**不是所有可能后处理的 universal upper bound**。原因是：

* 若允许 classical search 或 order reduction，成功率可以更高；
* Ekerå 已证明有限预算的 classical search 可以显著提高单次 quantum run 的恢复概率。 ([arXiv][4])

因此，实验归一化时更推荐使用 **finite-$Q$ ideal baseline**，而把 arithmetic ideal 作为“数论极限参考”。

---

## A.9 与现有 benchmark 的关系

本方案与现有基准的关系可概括为：

* **process fidelity**：量子过程基准；
* **periodic-state / TVD**：输出分布基准；
* **Shor post-processed success**：应用后处理基准。

动态 QFT PRL 提供了过程基准与简单周期态可视化；IonQ AQ benchmark 提供了面向应用的 end-to-end 思路，但其 plurality voting 明显更适合输出集中在一个或若干 bitstring 的应用。Shor/QFT 的理想输出天然是围绕 $s/r$ 的多峰结构，因此更合理的应用型指标是“continued fractions + LCM/gcd”能否恢复 order / factor，而不是单峰命中率。 ([arXiv][1])

---

# 建议的最终采用版本

若只保留一个主指标，建议采用

$$
\boxed{
P_{\rm ord,strict}^{(K)} =
\Pr\left[h_K^{\rm strict}(Y)=r\right]
}
$$

并同时报告

$$
P_{\rm wrong}^{(K)},\qquad
P_{\rm null}^{(K)},\qquad
\eta_{\rm ord,strict}^{(K)}.
$$

若正文允许两个应用指标，则再加上

$$
\boxed{
P_{\rm fac}^{(K)} =
\Pr\left[h_K^{\rm fac}(Y)\neq\bot\right]
}
$$

这样就能同时回答：

1. **QFT+M 本身是否足以恢复正确 order？**
2. **从最终 Shor factoring 的角度看，这些输出是否已经有应用价值？**

[1]: https://arxiv.org/html/2403.09514v2 "https://arxiv.org/html/2403.09514v2"
[2]: https://quantum.cloud.ibm.com/learning/courses/fundamentals-of-quantum-algorithms/phase-estimation-and-factoring/shor-algorithm "https://quantum.cloud.ibm.com/learning/courses/fundamentals-of-quantum-algorithms/phase-estimation-and-factoring/shor-algorithm"
[3]: https://quantumai.google/cirq/experiments/shor "https://quantumai.google/cirq/experiments/shor"
[4]: https://arxiv.org/abs/2201.07791 "https://arxiv.org/abs/2201.07791"
