# IDEA 05 · ORTHO — 推理子空间正交保护
## Reasoning-Subspace Orthogonal Protection against Collapse

| 项 | 值 |
|---|---|
| 排名 | **#5** |
| 五维 (H/F/S/C/B) | 8 / 7 / 8 / 7 / 8 |
| 均分 | **7.6** |
| 裁决 | **Strong Accept**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU Retain + 数学 CoT 语料（估计推理子空间） |
| 基线 | R2MU（collapse 0.95+）、REDaCT（推理时零 collapse 对照） |

---

## 1. Problem Anchor（一句话故事）

> R2MU 之所以把推理能力一起压垮，是因为遗忘梯度沿着「通用推理方向」也在推。ORTHO 先用 retain / 数学 CoT 语料估出「推理能力子空间」，把遗忘梯度投影到它的正交补，并给出一个可验证的 collapse 上界。

锚定问题：**遗忘与推理能力的冲突可以几何化，而不是靠调权重硬凑**。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| R-MUSE (CVPR 2026) | **多模态**、推理时子空间投影（遗忘子空间 / 推理保留子空间） | 多模态 + 免训练推理时，非纯文本 LRM 训练期正交约束 |
| Geometric Unlearning (arXiv:2605.01735) | prompt-conditioned hidden state 对齐到 safe-behavior 子空间 | 对齐到「安全子空间」，未显式建模「推理能力方向」并给 collapse 上界 |
| R2MU (2025) | 数学语料 CoT 监督正则 | 软正则，无显式子空间投影、无上界 |
| UNLEARN / subspace discrimination（综述二手） | 权重子空间 SVD 去交集 | 权重空间、LLM 通用，非 LRM 激活层推理方向 |

**Delta**：在纯文本 LRM 的**激活层**估计推理能力子空间 $S_r$，训练期做**梯度正交投影** $g^\perp=(I-UU^\top)g_f$，并把「投影残差」与 collapse 绑定为上界。据现有检索，未见「推理能力子空间 + collapse 形式化上界 + R-TOFU step-wise 验证」三件套。

---

## 3. 方法（LaTeX）

**3.1 推理能力子空间估计**
在 retain / 数学 CoT 语料上收集第 $\ell$ 层步级激活 $h_t$，协方差
$$
C_r=\frac{1}{N}\sum_{t=1}^{N}(h_t-\bar h)(h_t-\bar h)^\top,\qquad
U=[u_1,\dots,u_k]=\mathrm{eigvec}_{\text{top-}k}(C_r)
$$
$$
S_r=\mathrm{span}(U),\qquad P_\perp=I-UU^\top
$$

**3.2 遗忘梯度正交投影**
$$
g_f=\nabla_\theta\mathcal{L}_{\text{forget}},\qquad
\boxed{\;g_f^\perp=P_\perp\,g_f=(I-UU^\top)\nabla_\theta\mathcal{L}_{\text{forget}}\;}
$$
$$
\theta\leftarrow\theta-\eta\big(g_f^\perp+\lambda\, g_r\big),
\qquad g_r=\nabla_\theta\mathcal{L}_{\text{retain}}
$$

**3.3 表示级目标**
$$
\mathcal{L}_{\text{ORTHO}}
=
\mathcal{L}_{\text{forget}}
+
\mu\underbrace{\big\|U^\top\big(\phi_\theta-\phi_{\theta_0}\big)\big\|_F^2}_{\text{推理方向上的偏移惩罚}}
+
\lambda\,\mathrm{KL}\big(p_\theta(\cdot\mid x_r)\,\|\,p_{\theta_0}(\cdot\mid x_r)\big)
$$

**3.4 Collapse 上界（机制推断，待验证）**
设推理链完整度由 $U$ 方向的表示漂移决定，则
$$
\mathrm{Collapse}\;\le\;\kappa\,\big\|U^\top\Delta\phi\big\|_F\;+\;\mathcal{O}\big(\|\Delta\phi\|^2\big),
\qquad \Delta\phi=\phi_\theta-\phi_{\theta_0}
$$
ORTHO 把右端第一项压到 $\approx0$，故 collapse 受控。**该上界是本 idea 的可证伪核心**：若实测不成立，机制被驳倒。

**3.5 与 REDaCT 的关系**：REDaCT 在推理时检测泄漏句并注入 corrective embedding；ORTHO 在训练时约束梯度方向，二者可叠加（ORTHO 保能力，REDaCT 保无泄漏）。

---

## 4. R-TOFU 接口

- 子空间估计用 R-TOFU retain split + 仓库既有 `data/reasoned_idk/` 之外的数学 CoT（或 `config/training_config/` 指向的通用推理语料）；需新增一次协方差统计。
- 指标：R-TOFU step-wise（验证推理链保真）、collapse、Retain ROUGE-L、LEAK/MU，加核心 $\big\|U^\top\Delta\phi\big\|_F$ 与实测 collapse 的相关系数（验证 §3.4 上界）。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | R2MU | 复现 collapse ≥ 0.90 |
| T1 | ORTHO（$g_f^\perp$） | collapse 增幅 < 0.20，LEAK ≤ 0.05，step-wise 保真度 ≥ T0 + 0.10 |
| T2 | 消融：随机正交基 $U_{\text{rand}}$（同维度） | collapse 回弹 → 证明「推理方向」而非「任意低维投影」 |
| T3 | 消融：去掉 $\mu$ 项只留梯度投影 | $\big\|U^\top\Delta\phi\big\|_F$ 与 collapse 相关性消失 / 上界失效 |

**决定性实验**：T1 vs T2（隔离推理子空间的特异性）+ 上界相关性检验。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 8（机制分） | 几何约束在机制上同时保遗忘与保推理；待 T1 |
| Faster | 7（机制分） | 协方差一次性估计 + LoRA；相对 exact-retrain 快 |
| Stronger | 8（机制分） | 直接治理 collapse 这一已知 brittleness，含可验证上界；待 T2 |
| Cheaper | 7（机制分） | 复用 retain 语料与既有 CSV，无新标注 |
| Broader | 8（机制分） | 「能力子空间正交保护」可迁移到任意能力（代码/数学/工具），通用 protocol |

**裁决：Strong Accept**（H/B=8、零 CRITICAL）。机制分 → *pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. 推理能力未必低维线性可分；$U$ 可能混入 forget 相关方向 → 用「投影后 forget 梯度是否仍有效」做健康检验，必要时做 subspace discrimination（去 $S_r\cap S_f$）。
2. 与 R-MUSE / Geometric Unlearning 近邻较强，**F1 = MAJOR**（可划界：纯文本 LRM / 训练期 / 梯度投影 / 上界），故裁决虽为 Strong Accept，论文必须在 related work 主动划界，否则可能被判「移植」。
3. 上界 §3.4 若为假 → 整个机制主张需退回为纯经验方法（诚实性闸门要求预先声明）。
4. 本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/ortho/reasoning_subspace.py`（协方差 + top-k 特征）；训练在 `scripts/upbd/train_upbd.py` 的 `loss.backward()` 前插入梯度投影 hook。冻结：fp32 / seeds 1001, 2002 / LoRA r=8。
