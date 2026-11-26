# IDEA 03 · RRBU — 重学习鲁棒双层遗忘
## Relearning-Robust Bilevel Unlearning against Recovery Attacks

| 项 | 值 |
|---|---|
| 排名 | **#3** |
| 五维 (H/F/S/C/B) | 8 / 6 / 9 / 7 / 8 |
| 均分 | **7.6** |
| 裁决 | **Strong Accept**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU forget10（含保留子集用于模拟重学习攻击） |
| 基线 | R2MU、SimNPO/TPO、PARS（表征擦除 SOTA 对照） |

---

## 1. Problem Anchor（一句话故事）

> UnPact、PARS、UNDO 都证明「遗忘能被几步微调／关键词强调复活」；现有方法却只优化**当前**模型的零泄漏。RRBU 把 K 步重学习攻击放进双层优化的内层，让外层直接优化「攻击之后仍不泄漏」的平坦最优点。

锚定问题：**可靠的遗忘应对「恢复攻击」鲁棒，而不是对当前权重的一次性抑制**。这是「输出抑制 ≠ 知识擦除」这一诊断结论的训练闭环化。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| UnPact (arXiv:2509.24675) | 发现知识可经关键词强调 / 轻微调恢复；Recovery Rate | **诊断**，不把攻击纳入训练 |
| PARS (arXiv:2609.36612) | probe-adversarial 目标，抗 probing / relearning 更强 | 对抗的是**探针可提取性**，非显式的 K 步梯度恢复轨迹 |
| UNDO / Distillation Robustifies Unlearning (arXiv:2506.06278) | 指出行为对齐≠能力删除 | 观察 / 鲁棒性讨论 |
| Margin Cliff (arXiv:2607.27836) | 14 方法收敛到 margin cliff、梯度消失 | 分析，无闭环训练 |

**Delta**：把「恢复攻击」写成可微的 **K 步 unrolled 内层**，外层最小化攻击后的 forget loss；并给出 $\mathrm{RR}(K)$–Utility 的 Pareto。据现有检索，未见把「K 步 relearning attack」作为内层算子的 LRM 遗忘方法。

---

## 3. 方法（LaTeX）

**3.1 内层：模拟恢复攻击**
$$
\theta^{(0)}=\theta,\qquad
\theta^{(k+1)}=\theta^{(k)}-\eta_{\text{atk}}\,\nabla_{\theta}\mathcal{L}_{\text{rec}}\big(\theta^{(k)};\,D_f^{\text{sub}}\big),
\quad k=0,\dots,K-1
$$
其中 $\mathcal{L}_{\text{rec}}=-\mathbb{E}_{(x,a)\in D_f^{\text{sub}}}\log p_\theta(a\mid x)$（攻击者在小的 forget 子集上做梯度上升式复活）。

**3.2 外层：攻击后仍遗忘的双层目标**
$$
\boxed{\;
\min_{\theta}\ \ \underbrace{\mathcal{L}_{f}\big(\theta^{(K)}(\theta)\big)}_{\text{post-attack forget loss}}
\;+\;\lambda\,\mathcal{L}_{r}(\theta)
\;+\;\rho\,\big\|\theta-\theta_0\big\|_2^2
\;}
$$
$$
\text{s.t.}\quad \theta^{(K)}(\theta)=\mathrm{GD}^{(K)}_{\text{atk}}(\theta)
$$
超梯度按 unrolled 反向传播：
$$
\nabla_\theta \mathcal{L}_f(\theta^{(K)})
=
\Big[\prod_{k=0}^{K-1}\big(I-\eta_{\text{atk}}\nabla^2\mathcal{L}_{\text{rec}}(\theta^{(k)})\big)\Big]\,
\nabla_{\theta^{(K)}}\mathcal{L}_f
$$
实践中 $K\le 5$，用一阶近似（忽略二阶项）降成本。

**3.3 平坦最小点（SAM 风格等价正则）**
$$
\min_\theta\max_{\|\epsilon\|_2\le\rho_{\text{sam}}}
\Big[\mathcal{L}_f(\theta+\epsilon)+\lambda\mathcal{L}_r(\theta+\epsilon)\Big]
$$
直觉：让「被攻击方向」上的邻域也不泄漏，等价于对恢复攻击鲁棒。

**3.4 评估量**
$$
\mathrm{RR}(K)=\mathbb{P}_{x\in D_f}\Big[\ \mathcal{A}_K(\theta)\ \text{recovers}\ e_f\ \Big],
\qquad
\Delta_{\text{Utility}}=\mathrm{Utility}(\theta_0)-\mathrm{Utility}(\theta)
$$

---

## 4. R-TOFU 接口

- 攻击集 $D_f^{\text{sub}}\subset$ forget10（如 40 条），eval 在**未参与攻击**的 held-out forget 上做，避免记忆化。
- 指标：标准 AFE/CFE/MU/LEAK/Retain ROUGE-L + 核心 $\mathrm{RR}(K)$ 曲线（$K=0,1,3,5,10,75$，对标 PARS Fig.1 的 Relearn75）+ collapse。
- 对照：R2MU、SimNPO、TPO、PARS 在**同一攻击协议**下的 $\mathrm{RR}(K)$。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | R2MU / PARS（无攻击内层） | $\mathrm{RR}(5)$ 显著 > 0（复现「会被复活」） |
| T1 | RRBU（$K=3$ unrolled） | $\mathrm{RR}(5)$ 相对 T0 降 ≥ 0.3，且 $\Delta_{\text{Utility}}\le 0.05$ |
| T2 | 消融：$K=0$（退化为普通遗忘） | $\mathrm{RR}(5)$ 回弹 → 证明攻击内层必要 |
| T3 | 消融：去掉 SAM 平坦项 | 对未见 $K$（如 $K=10$）的鲁棒性下降 |

**决定性实验**：T1 vs T2（隔离「内层攻击」贡献）。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 8（机制分） | 直接优化「攻击后泄漏」这一更强指标；待 T1 |
| Faster | 6（机制分） | 双层 + unrolling 使单步变贵（故低于其他 idea），但相对 exact-retrain 仍快；$K$ 小、一阶近似 |
| Stronger | 9（机制分） | **本方法的主轴**：对已知最强恢复攻击族鲁棒，含未见 $K$ 外推；待 T3 |
| Cheaper | 7（机制分） | 免全量重训、复用 forget 子集、LoRA；但需额外前向/反向 |
| Broader | 8（机制分） | 「把攻击纳入内层」是可套到任意遗忘损失的通用协议 |

**裁决：Strong Accept**（S=9、B=8，零 CRITICAL）。机制分 → *pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. unrolled 超梯度方差大、$K$ 大时不稳定 → 限 $K\le5$ + 一阶近似 + 梯度裁剪。
2. 攻击者若换成**非梯度**恢复（关键词强调、激活编辑，UnPact / UDS 类），$\mathcal{L}_{\text{rec}}$ 未覆盖 → 需在攻击集 $\mathcal{A}$ 里含 prompt 强调与 patching 两类（与 CSDA 的 patching 资产复用）。
3. 「抗复活」可能与「保留邻近知识」冲突 → 用 IDEA 09 `RIPPLE` 的一跳邻居正则配套。
4. 本机无 GPU、无跨家族评审；分数为机制推断，`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/rrbu/bilevel.py`（unrolled 内层 + 一阶超梯度）；攻击协议脚本 `scripts/rrbu/relearn_attack.py`。训练基于 `scripts/upbd/train_upbd.py` 的 LoRA 骨架。冻结：fp32 / seeds 1001, 2002 / held-out 20 行。
