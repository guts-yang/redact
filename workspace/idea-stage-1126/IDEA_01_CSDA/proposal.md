# IDEA 01 · CSDA — 因果步级依赖消融遗忘
## Causal Step-wise Dependency Ablation for CoT Unlearning

| 项 | 值 |
|---|---|
| 排名 | **#1** |
| 五维 (H/F/S/C/B) | 9 / 7 / 8 / 8 / 7 |
| 均分 | **7.8** |
| 裁决 | **Strong Accept**（机制分，worth pursuing, pending validation） |
| 状态 | provisional · 无跨家族评审 · evidence_gate = BLOCKED |
| 数据集 | R-TOFU `forget01/05/10`，DeepSeek-R1-Distill-Llama-8B |
| 基线 | R2MU、Reasoned IDK、CoT-only GA（R-TOFU 原生）、REDaCT |

---

## 1. Problem Anchor（一句话故事）

> R2MU 把「forget 相关推理步」无差别地全部误导到随机向量，于是推理链一起垮掉；CSDA 先用因果归因找出 CoT 里真正把模型拽向 forget answer 的**承重步**，只扰动这一最小集合，逻辑链自然断掉而通用推理结构保留。

锚定问题：**在 LRM 中，遗忘一个事实的最小充分干预是「步」而不是「token」或「整条轨迹」**。R-TOFU 已经证明 CoT-only 干预比 answer-only 更关键，但没有回答「CoT 的哪一步该被动」。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| R-TOFU (EMNLP 2025) | step-wise **评测**指标、CoT-only 整体干预 | 不区分步的因果贡献，不做步级选择 |
| R2MU (2025) | 对全部 forget 相关步做表示扰动 | 无差别 → reasoning collapse 0.95+ |
| PARS (arXiv:2609.36612) / UDS | 表征层 / 激活层的**审计**与「遗忘深度」 | 审计结论，不产出「选哪一步做训练干预」的方法 |
| REDaCT (EMNLP 2026) | 推理时**句级检测** + corrective embedding | 推理时后处理，非训练期因果最小干预 |

**Delta（一句话可验证）**：把 activation-patching 的**步级因果贡献**变成训练期的**离散干预掩码**，并用 R-TOFU 的 step-wise 指标验证「只断承重步」。据现有检索资料，未见把「CoT 步因果承重 = 遗忘干预目标」形式化为训练方法的工作。

---

## 3. 方法（LaTeX）

设一条生成轨迹 $\mathcal{T}=(r_1,\dots,r_T; a)$，其中 $r_t$ 为第 $t$ 个推理步，$a$ 为最终答案；$\phi_\theta^{(\ell)}(r_t)$ 为第 $\ell$ 层在步 $t$ 的隐藏表示，$e_f$ 为 forget 实体。

**3.1 步级因果贡献（activation patching）**
$$
c_t \;=\; \mathbb{E}_{x\in D_f}\Big[\, \ell_{\text{leak}}\big(a \mid \mathrm{do}(\phi(r_t)\leftarrow \tilde\phi_t)\big) \;-\; \ell_{\text{leak}}\big(a \mid \mathcal{T}\big) \,\Big],
\qquad \tilde\phi_t \sim \mathcal{N}(\mu_{\text{retain}},\Sigma_{\text{retain}})
$$
其中 $\ell_{\text{leak}} = \log p_\theta(e_f \mid r_{\le t})$，$c_t$ 越大表示该步对「导出 forget answer」越承重。

**3.2 最小承重步集**
$$
\min_{\mathbf{m}\in\{0,1\}^{T}}\ \|\mathbf{m}\|_0
\quad \text{s.t.}\quad
\mathrm{Leak}\big(\mathcal{T}\mid \mathbf{m}\big)\;\le\;\varepsilon,
\qquad
m_t=\mathbb{1}\!\left[c_t \ge \tau\right]
$$
实践中用连续松弛 $\mathbf{m}=\sigma\big((c_t-\tau)/\kappa\big)$ 以便梯度回传。

**3.3 训练目标**
$$
\mathcal{L}_{\text{CSDA}}
=
\underbrace{\sum_{t=1}^{T} m_t\,\big\|\phi_\theta^{(\ell)}(r_t)-v_{\text{safe}}\big\|_2^2}_{\text{步级表示重定向}}
\;+\;
\lambda\,\underbrace{\mathrm{KL}\big(p_\theta(\cdot\mid x_{\text{ret}})\,\|\,p_{\theta_0}(\cdot\mid x_{\text{ret}})\big)}_{\text{retain 锚}}
\;+\;
\mu\,\underbrace{\sum_{t:\,m_t=0}\big\|\phi_\theta(r_t)-\phi_{\theta_0}(r_t)\big\|_2^2}_{\text{非承重步结构保持}}
$$
$v_{\text{safe}}$ 为 retain CoT 语料上的均值方向（不是随机向量，区别于 R2MU 的随机 misdirection）。参数更新走 LoRA（$r=8$），base 冻结。

**3.4 塌陷控制量（机制推断，待验证）**
$$
\mathrm{Collapse} \;\propto\; \sum_{t} m_t \cdot c_t^{-1}\;\xrightarrow[\text{只选高 }c_t]{}\; \text{小}
$$
即承重步越少、越集中，非承重推理步被扰动的概率越低。

---

## 4. R-TOFU 接口

- 数据：`forget10/task_id=1` 的 400 forget + 400 protocol-retain + 100 real_authors + 117 world_facts；held-out 用 `--holdout` 前 20 行。
- 指标：R-TOFU step-wise ROUGE-L / Cosine / LLM-as-Judge（**核心**），加 AFE/CFE_judge、LEAK、collapse、Retain ROUGE-L，以及本方法新增 $\|\mathbf{m}\|_0 / T$（干预稀疏度）。
- 归因开销：$\mathcal{O}(T)$ 次 patching/样本，可用缓存激活复用。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | R2MU / CoT-only GA（全步扰动） | 复现 collapse ≥ 0.90 |
| T1 | CSDA（$m_t$ 由 $c_t$ 选） | clean-cell LEAK ≤ 0.05 且 collapse 增幅 < 0.20，Retain ROUGE-L ≥ T0 − 0.02 |
| T2 | 消融：随机选等量步（$\mathbf{m}$ 随机） | collapse 或 leakage 显著劣于 T1 → 证明「因果承重」而非「步数少」是机制 |
| T3 | 消融：去掉结构保持项（$\mu=0$） | Retain ROUGE-L 掉 > 0.1 → 证明非承重步保护必要 |

**决定性实验**：T1 vs T2。若二者无差异，核心机制（步级因果承重）被驳倒 → 转向。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 9（机制分） | 最小充分干预在机制上同时压 leakage 与 collapse；待 T1 验证 |
| Faster | 7（机制分） | 相对 exact-retrain 少若干数量级梯度步；LoRA r=8；但需 patching 归因开销 |
| Stronger | 8（机制分） | 直接针对 CoT 结构鲁棒性 + 步级噪声容差；待跨 split 验证 |
| Cheaper | 8（机制分） | 复用冻结 target 与既有 CSV，无新标注、无全量重训 |
| Broader | 7（机制分） | 「步级因果承重」可迁移到任意带显式 CoT 的推理模型；未跨域验证 |

**裁决：Strong Accept**（两维 ≥8、零 CRITICAL）。高分全为机制分 → *worth pursuing, pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. $c_t$ 估计对 patching 分布 $\tilde\phi_t$ 敏感；若 retain 分布与 forget 步重叠，承重步可能被误判 → 用两类控制 patch（无关知识 / 词序打乱）做对照（借鉴 CoT Parametric Faithfulness 的控制任务思想）。
2. 「只断承重步」可能让模型用**新的推理路径**绕回 forget answer（R-TOFU 的 ZeroThink 现象）→ 必须联合 IDEA 04 `DISCO` 的解码不变约束。
3. 本机无 GPU、checkpoint 在服务器；全部结论待服务器上复现。评审为执行者本地初筛（hy4 为项目 override，provisional），`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

归因脚本需新增 `scripts/csda/step_attribution.py`（读 `redact/inference_regen.py` 的 CoT 切分）。训练复用 `scripts/upbd/train_upbd.py` 的 LoRA 骨架，替换损失为 $\mathcal{L}_{\text{CSDA}}$。协议冻结 fp32 / 1024 max new tokens / seeds 1001, 2002。
