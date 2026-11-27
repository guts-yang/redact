# IDEA 05 · REVERSE — 推理时可逆半径约束（单向量恢复阻断）
## Bounded Reversibility Radius against Single-Direction Steering Recovery

| 项 | 值 |
|---|---|
| 排名 | **#5（第二轮 1127）** |
| 五维 (H/F/S/C/B) | 8 / 6 / 9 / 7 / 7 |
| 均分 | **7.40** |
| 裁决 | **Strong Accept**（机制分，worth pursuing, pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU `forget01/05/10` + 5 语言子集（借 ICML 2026 方法），DeepSeek-R1-Distill-Llama-8B |
| 基线 | Multilingual Unlearning steering (ICML 2026)、首轮 RRBU / REVERSE 无 |

---

## 1. Problem Anchor（一句话故事）

> ICML 2026 的 Multilingual Unlearning 给出一记重锤：一个**推理时 steering 向量**（不用重训、不用答案前缀）就能跨语言恢复 50%–90% 的已忘知识。这说明现有遗忘只是「表面抑制」，且抑制被压缩在单一方向上。REVERSE 不否认可逆性，而是把它变成**可优化的约束**：要求不存在任何低维方向 $v$ 能把泄漏拉回来，即把「可逆半径」压到零。

锚定问题：**真正的遗忘应对「单向量、推理时、免重训」这一最轻量的恢复攻击鲁棒，而不是只对重训/越狱鲁棒**。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| Multilingual Unlearning (ICML 2026, arXiv:2606.03291) | 用 steering 向量**证明**可逆（50%–90% 恢复） | 诊断/攻击；不提防御、不把可逆半径写成训练约束 |
| RRBU（首轮 #3） | 抗参数级 relearning 攻击 | 攻击在权重空间；非推理时单向量 |
| PARS (arXiv:2609.36612) | 证明输出抑制与表征擦除可分离 | 诊断；未给「单方向不可恢复」训练目标 |
| CRISP (ACL 2026) | SAE 特征焊死，抗参数访问绕过 | 针对 hook 绕过；非推理时 steering 方向 |

**Delta**：定义「可逆半径」 $\mathcal R_\theta(\varepsilon)=\sup\{\|v\|:\ \mathrm{Leak}(\theta;+v)\ge\varepsilon\}$，把它作为训练期可优化约束（对最坏低秩方向的 min-max），并给出「单向量恢复阻断」判据。据现有检索，未见把 steering 可逆性从诊断转成训练期半径约束的工作。

**与首轮 10 个的划界**：最像 `RRBU`（恢复攻击鲁棒）与 `ORTHO`（子空间）。不同轴：RRBU 抗**权重 relearning**；REVERSE 抗**推理时激活空间低秩方向**。ORTHO 是训练期梯度投影保推理能力；REVERSE 是约束「任意单方向都不能恢复泄漏」。

---

## 3. 方法（LaTeX）

设前向在第 $\ell$ 层注入扰动 $h\leftarrow h+v$（$v$ 为常向量，即 steering），$e_f$ 为 forget 实体，$\mathrm{Leak}(\theta;v)$ 为注入 $v$ 后 forget 集泄漏率。

**3.1 可逆半径（核心量）**
$$
\mathcal R_\theta(\varepsilon)
=\sup_{v\in\mathcal V,\ \|v\|\le r_{\max}}
\Big\{\ \|v\|\ :\ \mathrm{Leak}(\theta;v)\ \ge\ \varepsilon\ \Big\},
\qquad \mathcal V=\text{低秩/单方向子空间}.
$$
ICML 2026 的现象即 $\mathcal R_{\theta_{\text{unl}}}$ 很大（存在一个方向能恢复）。目标 $\mathcal R_\theta\to 0$。

**3.2 最坏方向 min-max 训练**
$$
\min_\theta\ \max_{v:\|v\|\le r_{\max}}\ 
\Big[\,\mathrm{Leak}(\theta;v)-\mathrm{Leak}(\theta;0)\,\Big]_+
\;+\;\lambda\,\mathrm{KL}\big(p_\theta\|p_{\theta_0}\big)_{D_r}
\;+\;\mu\,\underbrace{\big\|\,\phi_\theta^{(\ell)}(D_f)-\phi_{\theta_0}^{(\ell)}(D_f)\,\big\|_{\text{proj}}}_{\text{抑制方向的秩/幅度约束}} .
$$
内层 $\max_v$ 用幂迭代/一步梯度上升近似最坏方向 $v^\*$：
$$
v^\*\approx r_{\max}\cdot
\frac{\nabla_v\,\mathrm{Leak}(\theta;v)\big|_{v=0}}
{\big\|\nabla_v\,\mathrm{Leak}(\theta;v)\big|_{v=0}\big\|_2}.
$$

**3.3 抑制方向去低秩化（机制推断）**
$$
\mathrm{Leak}(\theta;v)\le\mathrm{Leak}(\theta;0)
+\big\|\nabla_h\ell_{\text{leak}}\big\|\cdot\|v\|
+\mathcal O(\|v\|^2).
$$
若把 forget 抑制分散到**多个正交方向**（而非单一 $v_f$），则单方向 $v$ 的一阶增益被摊薄，$\mathcal R_\theta$ 收缩。训练用「抑制矩阵秩」正则 $\|\Sigma_f\|_*$ 鼓励高秩/分散抑制。报告 $\mathcal R_\theta$ 与 $\mathrm{rank}(\Delta_\theta)$ 的相关系数。

参数更新走 LoRA，base 冻结。

---

## 4. R-TOFU 接口

- 数据：`forget10/task_id=1` 400 forget + retain；steering 评估借 ICML 2026 的「遗忘前后表示差 → 反向注入」协议（不引用其数字，本地重测）。
- 方向注入：在 `</think>` 前的 CoT 步隐藏表示上加 $v$，复用 `inference_regen.py` 的前向钩子。
- 指标：核心 $\mathcal R_\theta(0.05)$、单向量恢复率（目标 $\le 0.10$）、R-TOFU step-wise、LEAK、collapse、Retain ROUGE-L。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | R2MU / CoT-only GA / 首轮方法 | 单向量 steering 恢复率显著 $>0.3$（复现可逆性） |
| T1 | REVERSE（去低秩 + min-max） | $\mathcal R_\theta(0.05)\le 0.1\,r_{\max}$ 且单向量恢复率 $\le 0.10$ |
| T2 | 消融：去掉去低秩正则，只做 min-max | 存在单一 $v^\*$ 仍可恢复 → 证明「分散抑制」必要 |
| T3 | 消融：换成 RRBU 的参数级 relearning 防御 | 单向量恢复率不降 → 证明推理时方向轴独立 |

**决定性实验**：T1 vs T2（隔离去低秩机制）。若无差异，核心机制被驳倒 → 转向。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 8（机制分） | 把最轻量恢复攻击纳入优化，机制上逼近「真擦除」；待 T1 |
| Faster | 6（机制分） | 每步需内层最坏方向搜索；相对 exact-retrain 仍快 |
| Stronger | 9（机制分） | **主轴**：直接对 ICML 2026 的 steering 可逆性证据做训练闭环；待跨 split |
| Cheaper | 7（机制分） | 无需新标注，复用前向钩子；方向搜索是主要算力 |
| Broader | 7（机制分） | 「可逆半径」可迁移到任意激活可被 steering 的模型；未跨域验证 |

**裁决：Strong Accept**（两维 ≥8、零 CRITICAL）。高分全为机制分 → *worth pursuing, pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. 高秩/多方向攻击（不止单向量）可绕过 $\mathcal R$ 定义 → 报告 $k$-方向恢复率并声明威胁模型边界（不夸成「全攻击鲁棒」）。
2. 去低秩正则可能伤及正常表示的低秩结构 → 用 Retain ROUGE-L 与 $\lambda$ 锚兜底。
3. $v^\*$ 的一步近似可能低估最坏方向 → 用幂迭代多步 + 随机方向网格复核。
4. 本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/reverse/steering_eval.py`（单/多方向注入 + 恢复率）与 `scripts/reverse/radius_loss.py`（去低秩 + min-max）；训练复用 hook 骨架。冻结：fp32 / seeds 1001, 2002。
