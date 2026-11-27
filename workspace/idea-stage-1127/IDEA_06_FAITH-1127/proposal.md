# IDEA 06 · FAITH — CoT 不忠实下的答案级反事实依赖遗忘
## Faithfulness-Gated Answer-Level Counterfactual Dependency Erasure

| 项 | 值 |
|---|---|
| 排名 | **#6（第二轮 1127）** |
| 五维 (H/F/S/C/B) | 8 / 7 / 8 / 7 / 7 |
| 均分 | **7.40** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU `forget01/05/10` + 控制任务（CoT Parametric Faithfulness 式），DeepSeek-R1-Distill-Llama-8B |
| 基线 | CiPO、R-TOFU CoT-only、首轮 CSDA / SCOPE |

---

## 1. Problem Anchor（一句话故事）

> 可解释性文献（Turpin 2023 / Lanham 2023 / Baker 2025 / CoT Parametric Faithfulness ACL 2026）反复证明：CoT 文本**不一定反映真正驱动答案的因素**，模型可能「嘴上这么推理、实际靠另一条参数化通路得出答案」。于是所有「压 CoT 步」的遗忘（含首轮 CSDA/SCOPE）都可能压错地方：CoT 干净了，答案仍从隐藏通路冒出事实。FAITH 用忠实度作门控变量：对不忠实的样本，遗忘目标从「压 CoT」切换成「答案级反事实依赖擦除」。

锚定问题：**当 CoT 不忠实时，以 CoT 步为作用对象的遗忘在机制上不充分；需要按忠实度自适应地把干预下沉到答案级依赖**。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| CoT Parametric Faithfulness (ACL 2026, 二手) / Turpin / Lanham / Baker | 诊断 CoT 不忠实、可隐藏 reward-hacking 意图 | 诊断；不用于遗忘，不给门控训练目标 |
| CiPO (ACL 2026) | 反事实推理轨迹 DPO，$Y\perp F\mid Q$ | 假设 CoT 是有效作用面；不测忠实度、不做门控 |
| CSDA（首轮 #1） | 步级因果承重 + 步表示重定向 | 作用对象始终是 CoT 步；未考虑「步不忠实」情形 |
| SCOPE（首轮 #6） | CoT 步级 scope 标签 | 同上 |
| PARS (2026) | 输出抑制 ≠ 表征擦除 | 诊断；不作忠实度门控 |

**Delta**：定义步级忠实度 $\mathrm{Faith}(r_t)$（该步的表示是否真的中介了答案），把它作为**门控变量**写进损失：忠实步走 CoT 步压，不忠实步走答案级反事实依赖 $a\perp e_f\mid \mathrm{do}$。据现有检索，未见把 CoT 不忠实性纳入 LRM 遗忘目标设计的工作。

**与首轮 10 个的划界**：最像 `CSDA` / `SCOPE`（都压 CoT 步）。不同轴：旧方法**默认 CoT 步是有效作用面**；FAITH 把「步是否忠实」本身当变量，作用对象在「CoT 步」与「答案级依赖」之间按 $\mathrm{Faith}$ 自适应切换。

---

## 3. 方法（LaTeX）

设轨迹 $\mathcal T=(r_1,\dots,r_T;a)$，$\phi^{(\ell)}(r_t)$ 步表示，$e_f$ forget 实体，$a$ 答案。

**3.1 步级忠实度（控制任务估计，detached）**
$$
\mathrm{Faith}(r_t)
=\Big\|\,\mathbb E\big[\,a\mid \mathrm{do}(\phi(r_t)\leftarrow \tilde\phi_t)\,\big]
-\mathbb E\big[\,a\mid \mathcal T\,\big]\,\Big\|_{\text{norm}}
\in[0,1],
$$
即「改动第 $t$ 步表示，答案分布变多少」——不变则此步只是事后合理化（不忠实），答案由别处决定。借 CoT Parametric Faithfulness 的控制任务思想，用无关知识/词序打乱两类控制 patch 校准。

**3.2 忠实度门控损失（核心）**
$$
\mathcal L_{\text{FAITH}}
=\sum_{t=1}^{T}\Big[\,
\underbrace{\mathrm{Faith}(r_t)\,m_t\,\|\phi_\theta(r_t)-v_{\text{safe}}\|_2^2}_{\text{忠实步：压 CoT 表示}}
\;+\;
\underbrace{\big(1-\mathrm{Faith}(r_t)\big)\,\mathcal L_{\text{ans}}(a)}_{\text{不忠实步：压答案级依赖}}
\Big]
\;+\;\lambda\,\mathrm{KL}(p_\theta\|p_{\theta_0})_{D_r}
$$
$$
\mathcal L_{\text{ans}}(a)
=-\log\big(1-p_\theta(e_f\mid \mathrm{do}(a\leftarrow a_{\text{cf}}))\big),
$$
其中 $a_{\text{cf}}$ 为逻辑自洽的反事实答案（借 CiPO 的自生成反事实，但**只用于答案级、且仅在不忠实步触发**）。

**3.3 依赖擦除判据（机制推断，待验证）**
$$
I_\theta(a;e_f\mid \mathrm{do}(\text{faith-gated}))
\;\le\;\varepsilon,
\qquad
\mathrm{FaithGap}
=\frac{1}{T}\sum_t \mathbb 1[\mathrm{Faith}(r_t)<\eta]
$$
报告忠实度缺口 $\mathrm{FaithGap}$ 与「CoT-only 方法在该子集上的残余泄漏」的相关系数。

参数更新走 LoRA，base 冻结。

---

## 4. R-TOFU 接口

- 数据：`forget10/task_id=1` 400 forget + retain；控制任务用无关知识 / 词序打乱两类 patch。
- 忠实度估计：复用 CSDA 的 `step_attribution.py` 骨架，但把目标量从 $\ell_{\text{leak}}$ 换成「答案分布变化」。
- 指标：核心「低忠实子集的残余泄漏」、R-TOFU step-wise、LEAK、collapse、Retain ROUGE-L、$\mathrm{FaithGap}$。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | CSDA / SCOPE（纯 CoT 步压） | 低忠实子集（$\mathrm{Faith}<\eta$）残余泄漏显著 > 高忠实子集 |
| T1 | FAITH（门控） | 低忠实子集泄漏 $\le 0.05$ 且 collapse 增幅 $<0.20$ |
| T2 | 消融：门控恒为 1（退化为纯 CoT 步压） | 低忠实子集泄漏回弹 → 证明「忠实度门控」必要 |
| T3 | 消融：恒用答案级反事实（CiPO 式，无门控） | 高忠实子集过度改写 / 效用降 → 证明「自适应切换」必要 |

**决定性实验**：T1 vs T2（隔离门控轴）。若无差异，核心机制被驳倒 → 转向。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 8（机制分） | 对「CoT 压错地方」这一机制漏洞做自适应修补；待 T1 |
| Faster | 7（机制分） | 忠实度估计需 patching 开销，但可缓存；相对 exact-retrain 快 |
| Stronger | 8（机制分） | 对 CoT 不忠实分布鲁棒，直击 LRM 特有失败模式；待跨 split |
| Cheaper | 7（机制分） | 复用既有控制任务与自生成反事实，无人工新标注 |
| Broader | 7（机制分） | 「忠实度门控」可迁移到任意带显式 CoT 的推理模型；未跨域 |

**裁决：Accept with Revisions**。降档理由：① F1=MAJOR（CiPO 反事实 + CoT Faithfulness 双近邻强，须主动划界「门控 + 答案级」）；② $\mathrm{Faith}$ 估计依赖控制 patch 校准，工程依赖新增（F7）。机制分 → *pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. $\mathrm{Faith}$ 估计对 patch 分布敏感 → 两类控制 patch 校准 + 报 $\eta$ 扫描。
2. 门控可能在训练中漂移（模型学会「显得不忠实」以逃避压制）→ 固定 $\mathrm{Faith}$ 为 $\theta_0$ 上的 detached 估计，不随 $\theta$ 更新。
3. 答案级反事实若固定模板会成侧信道（CiPO 自承）→ 用多样化自生成反事实 + 报成员推断 AUC。
4. 本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/faith/faithfulness.py`（控制 patch 估计 $\mathrm{Faith}$）与 `scripts/faith/gated_loss.py`；复用 CiPO 式自生成反事实生成器。冻结：fp32 / seeds 1001, 2002。
