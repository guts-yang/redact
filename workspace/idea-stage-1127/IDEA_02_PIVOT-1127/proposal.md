# IDEA 02 · PIVOT — 跨语言 CoT 枢轴下的遗忘转移与去枢轴化
## De-Pivoting Cross-Lingual CoT Transfer in LRM Unlearning

| 项 | 值 |
|---|---|
| 排名 | **#2（第二轮 1127）** |
| 五维 (H/F/S/C/B) | 7 / 6 / 9 / 7 / 9 |
| 均分 | **7.60** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU（翻译/改写为多语言遗忘集）+ 仓库 `paraphrases/` 资产（需修复） |
| 基线 | Multilingual Unlearning (ICML 2026)、R-TOFU Answer IDK、首轮 SCOPE |

---

## 1. Problem Anchor（一句话故事）

> LRM 常用「非英文提问、英文 CoT 思考、再译回作答」的枢轴路径。Multilingual Unlearning (ICML 2026) 证明普通 LLM 的遗忘只动后段语言特化层、跨语言易被 steering 恢复；但没人处理 LRM 里这条「输入语言 ≠ 推理语言」的 CoT 枢轴：只忘英文轨迹，母语提问经英文枢轴仍能把事实绕回来。PIVOT 把枢轴本身当成遗忘的作用对象。

锚定问题：**LRM 的遗忘范围必须覆盖「输入语言 → 枢轴语言 CoT → 输出语言」整条链，否则跨语言提问可从英文枢轴步绕回事实**。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| Multilingual Unlearning (ICML 2026, arXiv:2606.03291) | 5 语言 TOFU 转移矩阵 + steering 可逆性诊断 | 普通 LLM，非 LRM；无 CoT 枢轴、无训练方法 |
| Long CoT Across Languages (ICLR 2026) | 诊断 En-CoT / Target-CoT 推理迁移 | 推理能力迁移，不涉及遗忘 |
| SCOPE（首轮 #6） | 同语言内改写等价类的 CoT 步级 scope | 单一语言内的语义等价；不跨语言、不处理枢轴 |
| TRU / MAGE | 普通 LLM 等价类 / 记忆图 | 答案级 / 实体级，无 LRM CoT |

**Delta**：把「跨语言」从查询语言维度提升到 **LRM 的 CoT 枢轴路径维度**，定义去枢轴一致性 $\mathrm{DPC}$，并给出跨语言步级监督目标。据现有检索，未见在 LRM 上做「语言枢轴 CoT」遗忘转移与去枢轴化训练的工作。

**与首轮 10 个的划界**：最像 `SCOPE`（等价类）。不同轴：SCOPE 是**同语言内**的语义改写等价类（作用对象=问句改写）；PIVOT 是**跨语言 + 语言内枢轴**（输入语言 $L_q$ ≠ 推理语言 $L_{\text{cot}}$），机制是去枢轴化，不是等价类扩张。

---

## 3. 方法（LaTeX）

设语言集合 $\mathcal L$，查询 $x^{L_q}$ 用语言 $L_q$ 提问，模型以枢轴语言 $L_p$（通常英文）生成 CoT $\mathcal T^{L_p}=(r_1^{L_p},\dots,r_T^{L_p})$，再输出 $a^{L_o}$。$e_f$ 为 forget 实体，$\phi_\theta^{(\ell)}(r_t)$ 为步级隐藏表示。

**3.1 枢轴转移泄漏**
$$
\rho_\theta(L_q,L_p,L_o)
=\mathbb E_{x\in D_f^{L_q}}\Big[\mathbb 1\big[e_f\in \mathrm{Gen}_\theta(x^{L_q}\!\to\!\mathcal T^{L_p}\!\to\! a^{L_o})\big]\Big].
$$
ICML 2026 的诊断结论对应「$L_p$ 固定、$L_q$ 变化时 $\rho$ 回升」。本方法 additionally 约束 $L_p$ 这一 LRM 特有维度。

**3.2 去枢轴一致性（核心）**
$$
\mathrm{DPC}_\theta
=\frac{1}{|\mathcal L_q||\mathcal L_p|}
\sum_{L_q,L_p}\ \max_{t}\ \mathbb 1\big[\mathrm{Leak}(r_t^{L_p}(x^{L_q}))>\varepsilon\big],
\qquad \text{目标 } \mathrm{DPC}\to 0 .
$$
即「任意查询语言 × 任意枢轴语言组合下，每一步 CoT 都不泄漏」。

**3.3 跨语言步级目标**
$$
\mathcal L_{\text{PIVOT}}
=\sum_{L_q,L_p}\sum_{t=1}^{T}
\Big[\,g_t^{L_q,L_p}\,\mathcal L_{\text{sup}}(r_t^{L_p})
\;+\;(1-g_t^{L_q,L_p})\,\mathcal L_{\text{keep}}(r_t^{L_p})\Big]
\;+\;\lambda\,\mathcal L_r
$$
$$
\mathcal L_{\text{sup}}(r_t)=-\log\big(1-p_\theta(e_f\mid r_{\le t})\big),\qquad
\mathcal L_{\text{keep}}(r_t)=\mathrm{KL}\big(p_\theta(\cdot\mid r_{\le t})\,\|\,p_{\theta_0}(\cdot\mid r_{\le t})\big),
$$
其中 $g_t^{L_q,L_p}=\mathbb 1[r_t$ belongs to forget in-scope$]$ 为跨语言步级标签。

**3.4 共享语义空间约束（借 ICML 2026 的机制发现）**
$$
\mathcal L_{\text{shared}}
=\big\|\,\phi_\theta^{(\ell_{\text{mid}})}(x^{L_q})-\phi_{\theta_0}^{(\ell_{\text{mid}})}(x^{L_q})\,\big\|_2^2,\quad
\ell_{\text{mid}}\in\text{前中段共享层},
$$
强制遗忘**不要只发生在后段语言特化层**（那正是 steering 可逆的根源），而在跨语言共享语义空间里一致。

参数更新走 LoRA，base 冻结。

---

## 4. R-TOFU 接口

- 多语言遗忘集：把 `forget10` 的 question 用受控模板译为 $L_q\in\{$EN, ZH, DE, RU, TR$\}$（借 ICML 2026 的五语言设定），CoT 枢轴 $L_p\in\{$EN, $L_q\}$（英文枢轴 / 母语 CoT 两档）。
- **修订前置**：`paraphrases/` 被记「拒答改写，不可复现」，须先按 METHOD_STATUS 教训修复为「问题改写」再跑 T1/T2，否则 F7 数据风险。
- 指标：核心 $\mathrm{DPC}$、跨语言 step-wise 泄漏、steering 恢复率（借 ICML 2026 方法）、Retain ROUGE-L、collapse。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | 英文单语遗忘（R-TOFU Answer IDK） | 母语提问 + 英文枢轴下 step-wise 泄漏显著 > 原题 |
| T1 | PIVOT（跨语言步级 + $\mathcal L_{\text{shared}}$） | $\mathrm{DPC}\ge 0.90$ 的补（$\le 0.10$ 泄漏）且 steering 恢复率降 $\ge 0.3$ |
| T2 | 消融：去掉 $\mathcal L_{\text{shared}}$（等同只压后段） | steering 恢复率回弹 → 证明「共享层一致」必要 |
| T3 | 消融：只做查询语言覆盖、固定英文枢轴 | 母语 CoT 档泄漏回弹 → 证明「去枢轴」而非「多语覆盖」是机制 |

**决定性实验**：T1 vs T3（隔离枢轴轴）。若二者无差异，核心机制被驳倒 → 转向。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 7（机制分） | 跨语言最坏步泄漏下降；上限受翻译/改写库质量约束，故不给 8 |
| Faster | 6（机制分） | 语言×枢轴组合使训练样本数上升；相对 exact-retrain 仍快 |
| Stronger | 9（机制分） | **主轴**：对跨语言 + 枢轴绕过 OOD 鲁棒，直击 LRM 特有绕过；待跨 split |
| Cheaper | 7（机制分） | 复用 paraphrase 资产（需修复）与机器翻译模板，无需人工新标注 |
| Broader | 9（机制分） | 去枢轴框架可迁移到任意多语 / 多模态推理模型 |

**裁决：Accept with Revisions**。降档理由：① F1=MAJOR（Multilingual Unlearning 近邻强，需主动划界「LRM CoT 枢轴 vs 普通 LLM 答案级」）；② 数据资产 `paraphrases/` 需先修复。机制分 → *pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. 机器翻译/枢轴模板覆盖不全 → 攻击者用未见语言绕；须报覆盖率 $\mathrm{cov}(\mathcal L_q\times\mathcal L_p)$。
2. 共享层约束可能伤及多语通用能力 → 用 Retain ROUGE-L 与 $\mathcal L_r$ 兜底。
3. steering 恢复率评估借自外部诊断，需在同一 checkpoint 上重测，不直接引用其数字。
4. 本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/pivot/build_multilingual_forget.py`（修复等价问句库 + 枢轴模板）与 `scripts/pivot/shared_layer_loss.py`；训练复用 hook 骨架。冻结：fp32 / seeds 1001, 2002 / 400 forget + 217 aux。
