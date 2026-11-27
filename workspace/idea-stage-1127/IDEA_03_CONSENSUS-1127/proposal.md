# IDEA 03 · CONSENSUS — 多采样自一致性共识恢复攻击 + 共识不变目标
## Consensus-Invariant Unlearning against Majority-Vote Recovery in LRMs

| 项 | 值 |
|---|---|
| 排名 | **#3（第二轮 1127）** |
| 五维 (H/F/S/C/B) | 8 / 6 / 9 / 7 / 8 |
| 均分 | **7.60** |
| 裁决 | **Strong Accept**（机制分，worth pursuing, pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU `forget01/05/10`，DeepSeek-R1-Distill-Llama-8B |
| 基线 | R2MU、RRBU（首轮）、STaR-MCS、TTRL / Self-Consistency |

---

## 1. Problem Anchor（一句话故事）

> LRM 的强项就是「多采样 + 多数投票（self-consistency）」把答案稳住。可这恰恰是遗忘的漏洞：单次生成被压住了，采样 16 次一投票，forget 实体又成了共识答案。现有恢复攻击都是 relearning 权重或 steering 表征，没人把「模型自己的多数共识」本身当攻击面。CONSENSUS 把 majority-vote 写成可微的恢复攻击者，再让遗忘对这个共识攻击者不变。

锚定问题：**LRM 遗忘必须在「多采样共识」这一推理时缩放手段下仍成立，而不是只在 greedy 单样本下成立**。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| STaR (AAAI 2026) | MCS 多解码一致性**评分** | 评测指标；非恢复攻击、非训练目标 |
| TTRL / Self-Consistency (2025–2026) | majority vote 作伪标签做 RL | 提升推理能力；不用于遗忘，不作攻击 |
| RRBU（首轮 #3） | 参数级 K 步 relearning 攻击 + 双层 | 攻击在**权重空间**；非采样共识空间 |
| CiPO (ACL 2026) | 反事实轨迹 DPO | 自承认固定模板成侧信道；未处理多采样共识 |
| MCS (STaR) | 一致性打分 | 无 min-max 训练闭环 |

**Delta**：把 LRM 的「多次采样 → 多数答案」形式化为恢复攻击者 $\mathcal A_{\text{vote}}$，写 $\min_\theta\max_{N,\tau}$ 的共识不变目标，并给出「共识泄漏率」$\mathrm{CLR}$ 主判据。据现有检索，未见把 majority-vote 自一致性当 LRM 遗忘恢复攻击与训练目标的工作。

**与首轮 10 个的划界**：最像 `RRBU`（恢复攻击双层）。不同轴：RRBU 的攻击是**参数级 relearning / 梯度 unroll**（作用在 $\theta$）；CONSENSUS 的攻击是**推理时采样共识**（作用在输出分布 $\pi_\theta(\cdot\mid x)$ 的 $N$ 次采样），不改权重。二者正交可叠加。

---

## 3. 方法（LaTeX）

对问题 $x$，从 $\pi_\theta$ 采样 $N$ 条轨迹 $\{y^{(i)}\}_{i=1}^N$，答案提取器 $\mathrm{ans}(\cdot)$，多数投票伪标签 $\hat a_N=\mathrm{MajVote}\{\mathrm{ans}(y^{(i)})\}$，$e_f$ 为 forget 实体。

**3.1 共识恢复攻击（内层 $\max$）**
$$
\mathcal A_{\text{vote}}(x;N,\tau):\quad
\hat a_N=\arg\max_{a}\ \frac{1}{N}\sum_{i=1}^{N}\mathbb 1\big[\mathrm{ans}(y^{(i)})=a\big],
\qquad y^{(i)}\sim\pi_\theta(\cdot\mid x;\tau),
$$
$$
\mathrm{CLR}_\theta(N,\tau)
=\mathbb E_{x\in D_f}\Big[\mathbb 1\big[\hat a_N \equiv e_f\big]\Big].
$$
恢复成功 = 单次不漏但投票漏。注意 $\mathrm{CLR}$ 关于 $N$ 通常非降（更多采样放大残留偏好）。

**3.2 共识不变 min-max 目标（核心）**
$$
\min_\theta\ \max_{N\in\{1,\dots,N_{\max}\},\,\tau\in\mathcal T}\ 
\Big[\,\mathrm{CLR}_\theta(N,\tau)-\rho_\theta^{\text{greedy}}(x)\,\Big]_+
\;+\;\lambda\,\mathrm{KL}\big(p_\theta\|p_{\theta_0}\big)_{D_r}
\;+\;\mu\,\mathcal L_{\text{step}},
$$
其中 $\mathcal L_{\text{step}}$ 约束步级隐藏表示（避免过度扰动致 collapse）。内层对采样数 $N$ 与温度 $\tau$ 做最坏选择；外层用 Straight-Through / REINFORCE 估计投票梯度的可微松弛：
$$
\widehat{\mathrm{CLR}}_\theta
\approx\mathbb E\Big[\sigma\Big(\tfrac{1}{N}\sum_i s_\theta(y^{(i)},e_f)-\gamma\Big)\Big],
\quad s_\theta=\log p_\theta(e_f\mid \text{path}).
$$

**3.3 共识 margin 证书（机制推断，待验证）**
$$
\mathrm{CM}_\theta
=\max_{N,\tau}\ \Big[\mathrm{CLR}_\theta(N,\tau)-\mathrm{CLR}_\theta(1,\tau)\Big]
\xrightarrow[]{\mathcal L\to 0}\ 0,
$$
即「采样越多越不该多漏」。报告 $\mathrm{CM}$ 作为遗忘对 test-time-scaling 的鲁棒证书。

参数更新走 LoRA，base 冻结。

---

## 4. R-TOFU 接口

- 数据：`forget10/task_id=1` 400 forget + retain；生成侧复用 `redact/inference.py` 的多采样与 `</think>` 切分。
- 采样网格：$N\in\{1,4,8,16\}$，$\tau\in\{0.6,1.0\}$；答案抽取用 R-TOFU gold answer 匹配 + Hy4 判分（Hy4 是 metric judge，非 reviewer）。
- 指标：核心 $\mathrm{CM}$、$\mathrm{CLR}(16,\cdot)$、R-TOFU step-wise、LEAK、collapse、Retain ROUGE-L。
- 开销：每评估点 $N$ 次生成，训练用可微松弛降本。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | R2MU / CoT-only GA | $\mathrm{CLR}(16)-\mathrm{CLR}(1)$ 显著 $>0$（共识复活存在） |
| T1 | CONSENSUS（min-max 开） | $\mathrm{CM}\le 0.05$ 且 $\mathrm{CLR}(16)\le 0.05$ 且 collapse 增幅 $<0.20$ |
| T2 | 消融：内层只取 $N=1$（退化为 greedy 压泄漏） | $N=16$ 泄漏回弹 → 证明「共识攻击者」必要 |
| T3 | 消融：换成 RRBU 的参数级 relearning 攻击 | $\mathrm{CM}$ 不降（攻击空间错配）→ 证明采样轴独立 |

**决定性实验**：T1 vs T2（隔离共识轴）。若无差异，核心机制被驳倒 → 转向。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 8（机制分） | 直接压最坏采样数下的共识泄漏；待 T1 验证 |
| Faster | 6（机制分） | 每步多采样 + 可微松弛，开销高于单样本；相对 exact-retrain 仍快 |
| Stronger | 9（机制分） | **主轴**：对 test-time-scaling / 自一致性解码鲁棒，直击 LRM 特有复活；待跨 split |
| Cheaper | 7（机制分） | 无需新标注，复用既有生成与判分；采样算力是主要成本 |
| Broader | 8（机制分） | 「共识不变」可迁移到任意带 self-consistency / best-of-N 的推理模型 |

**裁决：Strong Accept**（两维 ≥8、零 CRITICAL）。高分全为机制分 → *worth pursuing, pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. 可微松弛与真实投票有 gap → 必须同时报真实 $\mathrm{MajVote}$ 的 $\mathrm{CLR}$，不以松弛值充当结论。
2. 强压共识可能伤及正常 self-consistency 推理增益 → 用 Retain ROUGE-L + 数学 CoT 对照兜底。
3. $N_{\max}$ 取太小则证书无意义 → 报告 $\mathrm{CLR}(N)$ 曲线而非单点。
4. 本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。Hy4 为 metric judge，非独立 reviewer。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/consensus/vote_attack.py`（多采样 + 投票 + 可微松弛）与 `scripts/consensus/consensus_loss.py`；训练复用 hook 骨架。冻结：fp32 / 1024 max new tokens / seeds 1001, 2002。
