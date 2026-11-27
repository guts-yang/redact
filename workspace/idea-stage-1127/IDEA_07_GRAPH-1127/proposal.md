# IDEA 07 · GRAPH — 最小锚点驱动的自生成 CoT 步级遗忘监督
## Anchor-Driven Self-Generated Step-Level Supervision for Corpus-Free LRM Unlearning

| 项 | 值 |
|---|---|
| 排名 | **#7（第二轮 1127）** |
| 五维 (H/F/S/C/B) | 7 / 6 / 8 / 9 / 7 |
| 均分 | **7.40** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU `forget01/05/10` + 最轻量锚点（实体名），DeepSeek-R1-Distill-Llama-8B |
| 基线 | MAGE (arXiv:2604.13777)、SHRED、首轮 SCOPE |

---

## 1. Problem Anchor（一句话故事）

> 真实「被遗忘权」请求里，用户往往只给得出一个名字，交不出一整份 forget 语料库；而现有 LRM 遗忘全靠外部提供的完整 forget set（含 CoT）。MAGE 证明普通 LLM 能从最小锚点自生成实体级监督，但它是答案/实体级、不产 CoT 步级监督。GRAPH 把这条 corpus-free 范式下沉到 LRM：从一个锚点探测出模型自己的记忆图，再自生成**步级**的 forget / neighbor 监督，让 LRM 遗忘不再依赖外部 CoT 库。

锚定问题：**LRM 遗忘应能从最轻量、可审计的锚点出发，自生成 CoT 步级监督，而不是要求用户提交完整推理轨迹语料**。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| MAGE (arXiv:2604.13777, 2026) | 最小锚点 → 记忆图 → 实体级自生成监督（TOFU/RWKU） | 普通 LLM，答案/实体级；不产 CoT 步级 forget 监督 |
| SHRED (NeurIPS 2026) | 无保留集，但仍需 forget set 样本 | 仍依赖完整 forget 集；非锚点驱动 |
| SCOPE（首轮 #6） | 等价类 + 步级标签 | 依赖人工 paraphrase 库（且记为「拒答改写，不可复现」） |
| FRUL / Selective Forgetting | RAG 识别敏感 CoT 片段 | 需外部检索语料；非自生成 |

**Delta**：把 corpus-free 的「记忆图 → 监督」从实体 QA 对扩展到 **CoT 步序列**：每个记忆图路径生成一条带步级标签的推理轨迹 $(\mathcal T, g_t)$，再做步级 scope 损失。据现有检索，未见在 LRM 上做锚点驱动、自生成 CoT 步级遗忘监督的工作。

**与首轮 10 个的划界**：最像 `SCOPE`（步级 scope）。不同轴：SCOPE 的步级标签来自**外部 paraphrase 库**（且该资产被记为不可复现）；GRAPH 的步级标签来自**模型自身记忆图的自生成路径**，无外部库、可审计、成本极低（故 Cheaper 给 9）。

---

## 3. 方法（LaTeX）

设用户只提供锚点 $z$（如实体名 $e_f$）。$\theta_0$ 记忆探测：迭代从 $z$ 扩展出原子事实，构成加权局部记忆图 $G=(V,E,w)$，边权 $w$ 由跨多次生成的频次/记忆强度估计。

**3.1 记忆图与强度加权路径采样**
$$
G_z=(V_z,E_z,w),\qquad
w(e)=\frac{\#\text{生成中该事实稳定复现的次数}}{\#\text{探测查询}},
$$
$$
\mathcal P_z=\{\,p:\ \text{强度加权随机游走路径, 带覆盖率约束}\,\}.
$$
强权重路径 = 模型确实记住的内容（该忘）；弱权重/邻接路径 = 邻近良性知识（该保）。

**3.2 步级监督自生成（核心）**
每条路径 $p$ 由模型自生成一条 CoT 轨迹 $\mathcal T_p=(r_1,\dots,r_{T_p};a_p)$，并给每一步打标签：
$$
g_t=\mathbb 1\big[\,r_t\ \text{属于 forget 作用域（强权重路径）}\,\big],
\qquad
\mathcal D_f^{\text{self}}=\{(\mathcal T_p,\{g_t\})\}_{p\in\mathcal P_z^{\text{forget}}},
\quad
\mathcal D_r^{\text{self}}=\{\mathcal T_p\}_{p\in\mathcal P_z^{\text{neighbor}}}.
$$
与 MAGE 的区别：生成的是**带步级标签的推理轨迹**，不是单条 QA 对。

**3.3 步级 scope 损失（复用 SCOPE 形式，但监督自生成）**
$$
\mathcal L_{\text{GRAPH}}
=\sum_{(\mathcal T,\{g_t\})\in\mathcal D_f^{\text{self}}}\sum_{t=1}^{T}
\Big[\,g_t\,\mathcal L_{\text{sup}}(r_t)
+(1-g_t)\,\mathcal L_{\text{keep}}(r_t)\,\Big]
+\lambda\,\mathcal L_r(\mathcal D_r^{\text{self}}),
$$
$$
\mathcal L_{\text{sup}}(r_t)=-\log\big(1-p_\theta(e_f\mid r_{\le t})\big),\qquad
\mathcal L_{\text{keep}}(r_t)=\mathrm{KL}\big(p_\theta(\cdot\mid r_{\le t})\|p_{\theta_0}(\cdot\mid r_{\le t})\big).
$$

**3.4 覆盖率与审计判据（机制推断，待验证）**
$$
\mathrm{cov}(G_z)=\frac{|\{e_f\text{ 相关事实被探测到}\}|}{|\text{forget set 原子事实}|},
\qquad
\mathrm{SCS}^{\text{self}}=\frac{1}{|\mathcal D_f^{\text{self}}|}
\sum_{\mathcal T}\min_t \mathbb 1[\mathrm{Leak}(r_t)\le\varepsilon].
$$
报告自生成监督覆盖率 $\mathrm{cov}$ 与「自生成 vs 外部 gold forget set」的遗忘效果差距 $\Delta_{\text{MU}}$。

参数更新走 LoRA，base 冻结。

---

## 4. R-TOFU 接口

- 锚点：每个 forget 作者只给名字 $z=e_f$（最轻量）；gold `forget10` 仅用于**评测对照**，不进训练。
- 记忆图：复用 RIPPLE 的 `KnowledgeGraph.from_claims` 思路，但改为「从锚点探测」而非「从 claims.json 建图」（框架一致、数据来源不同）。
- 指标：核心 $\Delta_{\text{MU}}$、$\mathrm{cov}(G_z)$、$\mathrm{SCS}^{\text{self}}$、R-TOFU step-wise、Retain ROUGE-L、collapse。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | 外部 gold forget set 的 SCOPE（上界参考） | 记录 $\mathrm{MU}_{\text{gold}}$、$\mathrm{SCS}_{\text{gold}}$ |
| T1 | GRAPH（纯锚点自生成步级监督） | $\Delta_{\text{MU}}=\mathrm{MU}_{\text{gold}}-\mathrm{MU}_{\text{self}}\le 0.05$ 且 $\mathrm{SCS}^{\text{self}}\ge 0.90$ |
| T2 | 消融：自生成但只给答案级监督（等价 MAGE） | $\mathrm{SCS}^{\text{self}}$ 显著降 → 证明「步级监督」必要 |
| T3 | 消融：不用记忆图、直接自提示生成 | $\mathrm{cov}(G_z)$ 降、中间步残留 → 证明「记忆图覆盖」必要 |

**决定性实验**：T1 vs T2（隔离步级监督轴）与 T1 vs T3（隔离记忆图轴）。若 T1 与 T0 差距过大或两消融无差异，核心机制被驳倒 → 转向。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 7（机制分） | 自生成监督逼近外部 gold 的效果；上限受探测覆盖率约束，故不给 8 |
| Faster | 6（机制分） | 探测 + 自生成使前置阶段变长；相对 exact-retrain 仍快 |
| Stronger | 8（机制分） | 对「锚点不全 / 问法未见」鲁棒；待跨 split |
| Cheaper | 9（机制分） | **主轴**：无需外部 forget set / 人工 paraphrase 库 / 新标注，只给一个名字；最贴合真实被遗忘权 |
| Broader | 7（机制分） | corpus-free 锚点范式可迁移到多语 / 多模态；未跨域 |

**裁决：Accept with Revisions**。降档理由：① F1=MAJOR（MAGE 近邻强，须主动划界「LRM CoT 步级 vs 普通 LLM 实体级」）；② 探测阶段依赖模型自我复述 fidelity，覆盖率是前置风险（F7）。机制分 → *pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. 模型探测不出自己记住的内容（稀疏记忆如 TOFU 虚构作者）→ $\mathrm{cov}$ 低，T1 直接失败；这是可证伪点，不掩饰。
2. 自生成轨迹可能含幻觉「伪知识」，压了不存在的东西 → 用 $w(e)$ 强度阈值过滤并报噪声率。
3. 锚点本身可能触发成员推断（用户只给名字）→ 报审计日志与最小披露约束，声明威胁模型。
4. 本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/graph/anchor_probe.py`（锚点 → 记忆图）与 `scripts/graph/self_supervision.py`（路径 → 步级标签轨迹）；复用 SCOPE 的 `scope_loss`。冻结：fp32 / seeds 1001, 2002 / 400 forget 仅作评测。
