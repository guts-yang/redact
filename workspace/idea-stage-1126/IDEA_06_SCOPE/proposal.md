# IDEA 06 · SCOPE — LRM CoT 等价类边界遗忘
## Step-wise Scope-Consistent Equivalence-Class Unlearning

| 项 | 值 |
|---|---|
| 排名 | **#6** |
| 五维 (H/F/S/C/B) | 7 / 6 / 9 / 7 / 8 |
| 均分 | **7.4** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU + 仓库 `paraphrases/forget10*.jsonl`（改写/间接问句） |
| 基线 | R-TOFU Answer IDK / Direct IDK、TRU（LLM 等价类，二手来源） |

---

## 1. Problem Anchor（一句话故事）

> 「遗忘一个作者」不该只忘 400 道题，而该忘「这个作者的所有等价问法、翻译、改写、间接指代」；TRU 把 scope 定义成等价类，但在普通 LLM 上做。SCOPE 把等价类下沉到 **LRM 的每一个 CoT 步**，要求 in-scope 的每一步推理都被压、out-of-scope 的每一步都保。

锚定问题：**LRM 的遗忘范围必须是 step-wise 的，否则改写问句可以从中间推理步绕回事实**。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| TRU (ICLR 2026, 二手来源) | 形式化 scope unlearning 等价类 $[x]_T$，用推理链 + 可解释拒答 | 普通 LLM / 答案级 scope，无 step-wise，无 R-TOFU CoT |
| R-TOFU (EMNLP 2025) | step-wise **评测**指标 | 不做等价类边界训练 |
| 仓库 `paraphrases/` | 已有改写库（但 METHOD_STATUS 记为「拒答改写，不可复现」） | 资产待修复，非方法 |
| ShadowBench / N05 span | 「删名字≠删知识」、span 级 | 无等价类、无 step-wise 一致性目标 |

**Delta**：把等价类条件从「样本级」提升到「CoT 步级」，定义 $\mathrm{ScopeStep}(r_t)$，并给出 step-wise 一致性约束。据现有检索，未见在 LRM 上做等价类边界 + step-wise 联合的工作。

---

## 3. 方法（LaTeX）

**3.1 等价类与步级标签**
$$
[x]_T=\{\tilde x:\tilde x\sim_T x\},\qquad
g_t=\mathbb{1}\!\big[\,r_t\ \text{belongs to in-scope equivalence class}\,\big]
$$
in-scope（含改写、翻译、间接问句）的每一步 CoT 都标 $g_t=1$；out-of-scope（retain / aux）标 $0$。

**3.2 Step-wise scope 损失**
$$
\boxed{\;
\mathcal{L}_{\text{SCOPE}}
=
\sum_{x\in D_f}\sum_{t=1}^{T}
\Big[\,g_t\,\mathcal{L}_{\text{sup}}(r_t)
\;+\;(1-g_t)\,\mathcal{L}_{\text{keep}}(r_t)\,\Big]
\;+\;\lambda\,\mathcal{L}_{r}
\;}
$$
$$
\mathcal{L}_{\text{sup}}(r_t)=-\log\big(1-p_\theta(e_f\mid r_{\le t})\big),\qquad
\mathcal{L}_{\text{keep}}(r_t)=\mathrm{KL}\big(p_\theta(\cdot\mid r_{\le t})\,\|\,p_{\theta_0}(\cdot\mid r_{\le t})\big)
$$

**3.3 一致性判据**
$$
\mathrm{SCS}=\frac{1}{|[x]_T|}\sum_{\tilde x\in[x]_T}
\min_{t}\, \mathbb{1}\big[\mathrm{Leak}(r_t(\tilde x))\le\varepsilon\big]
$$
即「每个等价问法的每一步都不泄漏」的比例，目标 $\mathrm{SCS}\to 1$。

**3.4 与 DISCO 的区别**：DISCO 管解码策略前缀，SCOPE 管**语义等价问法**；二者正交，可叠加。

---

## 4. R-TOFU 接口

- 等价类构造：用 `paraphrases/forget10.jsonl` / `forget10.deepseek.jsonl`，但必须先按 `METHOD_STATUS.md` 的教训**修复为「问题改写」而非「拒答改写」**；新增翻译 / 间接指代问句模板。
- 指标：R-TOFU step-wise + 核心 $\mathrm{SCS}$（等价类内最坏步泄漏率）+ Retain ROUGE-L + collapse。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | Answer IDK / Direct IDK | 等价问法上 step-wise 泄漏显著 > 原题 |
| T1 | SCOPE | $\mathrm{SCS}\ge 0.90$ 且 Retain ROUGE-L 不掉 > 0.02 |
| T2 | 消融：$g_t\equiv1$（整条 CoT 压，退化为样本级 scope） | 改写问法上泄漏回弹 → 证明 step-wise 必要 |
| T3 | 消融：只在答案步加 $\mathcal{L}_{\text{sup}}$ | 中间步残留 → 证明 LRM 特有必要 |

**决定性实验**：T1 vs T2（隔离步级 vs 样本级 scope）。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 7（机制分） | 等价类内最坏步泄漏下降；上限受改写库质量约束，故不给 8 |
| Faster | 6（机制分） | 等价类扩充使训练样本数上升；相对 exact-retrain 仍快 |
| Stronger | 9（机制分） | **主轴**：对改写/翻译/间接问句 OOD 鲁棒，直击 LRM 特有绕过 |
| Cheaper | 7（机制分） | 复用仓库 paraphrase 资产（需修复），无需新人工标注 |
| Broader | 8（机制分） | 等价类框架可迁移到多语言 / 多模态推理 |

**裁决：Accept with Revisions**。降档理由：① F1=MAJOR（TRU 等价类近邻强，需主动划界）；② 数据资产 `paraphrases/` 被记为「拒答改写，不可复现」，**修订前置**：必须先修复等价问句库再跑 T1/T2，否则 F7 数据风险。机制分 → *pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. 等价类覆盖不全 → 攻击者用未覆盖的问法仍可绕；需报告覆盖率 $\mathrm{cov}([x]_T)$。
2. 「改写库是拒答改写」的历史坑必须先在 §4 解决，否则整个 T0/T1 不可信。
3. 与 `N09 通道分解`（已被 K-Bench 抢先）不同：SCOPE 不做 CoT-vs-answer 通道，做的是语义边界。
4. 本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

先新增 `scripts/scope/rebuild_equivalence.py` 修复 paraphrase 库；再在 `scripts/upbd/train_upbd.py` 骨架上把 step-wise $g_t$ 标签接入。冻结：fp32 / seeds 1001, 2002 / 400 forget + 217 aux。
