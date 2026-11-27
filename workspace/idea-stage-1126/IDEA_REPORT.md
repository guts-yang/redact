# LRM 机器遗忘 · Idea Discovery + Idea Evaluator 报告

**方向**: Large Reasoning Model (LRM) 机器遗忘创新 idea 调研
**数据集 / 基线**: R-TOFU（EMNLP 2025, arXiv:2505.15214）作为主数据集之一与基线协议；对照含 R2MU、Reasoned IDK、REDaCT、CiPO、STaR
**日期**: 2026-11-26
**流程**: `/idea-discovery`（Phase 1 文献调研 + Phase 2 idea 生成）→ `/idea-evaluator`（逐条八步评估）→ 双门槛筛选 → 排名 → 10 份 proposal
**输出根目录**: `/Users/chenyliao/Repositories/Machine_Unlearning/redact/workspace/`

---

## 0. 执行边界与诚实性声明（务必先读）

本次按用户显式指令对标准 `/idea-discovery` 流水线做了裁剪，以下依赖**确实不可用**，已如实标注，未假装执行：

| 标准环节 | 状态 | 实际做法 |
|---|---|---|
| Gemini 增强的 `/research-lit` | 不可用（gemini-cli / Codex MCP 未挂载） | 用真实 `web_search` 多轮检索 grounding，来源标注见 §1 |
| `/novelty-check` Phase C 跨模型评审 | 不可用（codex/gemini/claude 后端均为 `null`） | 新颖性判断由执行者基于检索到的近邻做 metadata 级划界，标为 **provisional** |
| `/research-review` 外部评审 | 不可用（无跨家族评审线程） | 由用户指定的 `/idea-evaluator` 承担评估，裁决为**执行者本地初筛** |
| GPU pilot（MAX_PILOT_IDEAS=3） | 不可用（本机 `torch.cuda.is_available()=False`，checkpoint 在服务器 `/usr/local/...`） | 全部 idea 为**机制分 + 命名的决定性验证实验**，无实测数字（遵守「不编造」纪律） |
| `run_state.py` / `idea_discovery_gate.py` per-stage evidence gate | `BLOCKED`（`gate_status.json`：helper 在所有 canonical 路径均缺失） | 不伪造 receipt；本报告 `evidence_gate = BLOCKED`，所有裁决 provisional |
| 单一 `FINAL_PROPOSAL.md` + pilot 结果 | 被用户显式覆盖 | 用户要求 10 份并列 proposal + 排名，故以本文为总览、`workspace/idea-stage-1126/IDEA_XX_*/proposal.md` 为 Individual 工件 |

**结论**：本轮是「候选 idea 生成与初筛」，不是投稿级验证。`assurance = draft`，`evidence_gate = BLOCKED`，无跨家族接受。任何 idea 进入实验前需先跑其 proposal 中的决定性验证实验，并取得真实跨模型评审。

---

## 1. 文献 Landscape（真实检索 grounding）

检索来源：`web_search`（多轮，2026-11-26）。下列条目为检索命中，按标题/摘要做 **metadata 级**判断；凡来自二手论文笔记站点（papernotes / graphastra / lacuna / aclanthology 镜像）者已标注，引用时仅作 landscape 描述，不以其中的数字作为打分证据。

**基准与数据集**
- **R-TOFU** — Yoon, Jeung, No, *R-TOFU: Unlearning in Large Reasoning Models*, EMNLP 2025, arXiv:2505.15214 / ACL 2025.emnlp-main.265。首个 LRM 遗忘基准；在 TOFU 虚构作者档案上用 GPT-4o 生成 model-aligned CoT；提出 step-wise 指标（ROUGE-L / Cosine / LLM-as-Judge）；split `forget01/05/10`；目标模型 DeepSeek-R1-Distill-Llama-8B。**关键发现**：answer-only 遗忘不足（CoT 残留 forget traces）、CoT-only 最有效、Reasoned IDK 优于直接拒答、**ZeroThink/LessThink 解码会复活已遗忘内容**。
- TOFU — Maini et al., 2024（R-TOFU 的上游虚构作者数据集）。

**方法（LRM 专属）**
- **R2MU / R²MU** — Wang et al., 2025, arXiv:2506.12963。`unthinking` loss 把 forget 相关推理步表示误导到随机向量 + 数学语料 CoT 监督；副作用是 **reasoning collapse**（REDaCT 报告其 collapse 达 0.95–1.0）。
- **REDaCT** — Maccoby & Wang, Findings EMNLP 2026, arXiv:2610.12769, ACL 2026.findings-emnlp.490。推理时句级 leak detector + 离线 corrective embedding，零 collapse；**即本 `redact` 仓库复现的他人方法（参考 arm）**。
- **CiPO** — *Counterfactual Unlearning for LRMs through Iterative Preference Optimization*, ACL 2026, arXiv:2604.15847（二手笔记站点，ID 待核验）。反事实推理轨迹作 DPO chosen，迭代偏好优化；自承「固定反事实/拒答模式可成为信息泄露通道」。
- **STaR** — AAAI 2026, arXiv:2601.9281（历史记忆）。推理时敏感轨迹调控 + **MCS 多解码一致性评分**（评估指标）。
- **R-MUSE** — CVPR 2026（二手笔记站点）。多模态推理保留遗忘，遗忘子空间 / 推理保留子空间投影，免训练推理时。
- **GUARD** — 2024。推理时 prompt 分类器 + token/semantic matching 生成门控。
- **TRU / Explainable LLM Unlearning through Reasoning** — ICLR 2026（二手笔记站点）。推理链 + 可解释拒答作 unlearning target，形式化 **scope unlearning**（等价类）。
- **Selective Forgetting / FRUL** — PAKDD 2026（综述二手来源）。CoT 中间步遗忘。
- **ASU** — ICLR 2026, arXiv:2603.01285（本地记忆）。注意力平滑 τ>1 + 自蒸馏。

**损失函数红海（避让）**：GA → GradDiff → NPO → SimNPO → AltPO → TPO → CE-U → Unilogit；权重定位/显著性、蒸馏教师构造、新基准构建、单点攻击评估。历史记忆明确标注这些为高度拥挤区。

**审计 / 诊断 / 评估（重要近邻，决定 F1 判断）**
- **UnPact** — Zhang, Qian, Huang, Hong, Huang, Xu, Zhang, Qiu（清华+蚂蚁）, arXiv:2509.24675, 2025。prompt attribution 揭示「遗忘=破坏关键词关注」，知识可经关键词强调恢复；提出 Recovery Rate / Destructive Rate 两难。
- **可逆性遗忘表示诊断** — 港理工/CMU/UCSC, arXiv:2505.16831, 2025。PCA/CKA/FIM 工具箱；「真正的遗忘是结构抹除而非行为抑制」。
- **PARS / Hidden-State Leakage** — Reisizadeh, Ruan, Liu, Hong, arXiv:2609.36612, 2026。理论证明输出抑制与表征擦除可分离；probe-adversarial 目标。
- **PID 审计** — Goel, Ritter, Gurevych, arXiv:2601.15111, EACL 2026。Partial Information Decomposition 分解 unlearned / residual knowledge。
- **Honesty in LLM Unlearning** — Gu et al., 2026（综述二手来源）。自我知识 vs 自我表达，检测「知道但拒答」。
- **UDS / Activation Patching for Depth** — Lee et al., 2026（综述二手来源）。白盒激活修补测遗忘深度。
- **RippleBench** — Rinberg et al., 2025（综述二手来源）。语义邻近概念的涟漪效应 MCQ 评估。
- **Margin Cliff** — arXiv:2607.27836（历史记忆）。14 种方法收敛到 margin cliff、token 饱和后梯度消失。
- **PMC / Partial Model Collapse** — ICLR 2026, arXiv:2507.04219。把模型坍缩反过来当遗忘工具。
- **LLaMA CoT collapse 文献** — emergentmind 综述（二手）：coupling tax、commitment boundaries、post-hoc rationalization。
- **CoT Parametric Faithfulness** — ACL 2026（二手）：用控制任务解耦遗忘副作用。

**结构性空白（本轮 idea 的立论依据）**：
1. 「会被复活 / 会坍塌 / 输出≠表征」已被**诊断**，但把攻击、坍塌、复活纳入**训练闭环**的工作少；
2. 审计多在**表征/token 层**，「CoT 步级因果承重」这一 LRM 特有粒度少有训练方法；
3. R-TOFU 的 **ZeroThink 解码复活**只作为发现 / MCS 只作为评估，缺 min-max 训练目标；
4. 难度/纠缠度只用于**预测**，缺「预测→干预强度调度」闭环；
5. 拒答/反事实模板的**侧信道**（成员推断）刚被 CiPO 自承、未被攻防化；
6. 序列遗忘、多跳涟漪、承诺边界早停、等价类 step-wise 一致性在 LRM 上均属空白。

---

## 2. 评估口径（用户双门槛）

- 框架：`/idea-evaluator`（八步：定位 → F1–F10 致命缺陷审计 → 生命周期 → 五维 → 范式四问 → 可行性 → 诚实性闸门 → 三档裁决）。
- **硬门槛 A**：五维（Higher / Faster / Stronger / Cheaper / Broader）**算术均分 ≥ 7.0**。
- **硬门槛 B**：裁决 ∈ {**Strong Accept**, **Accept with Revisions**}（即 ≥ Accept with Revisions）。
- 评分纪律：每维起评 5；机制成立的维度标「机制分，待验证实验」并命名决定性实验；无依据者留 5 并注 `no grounds given`。
- **Faster / Cheaper 参照系声明**：以 *exact unlearning（排除 forget set 全量重训）* 为参照，近似遗忘方法在机制上均更快更省，故二者可给机制分；若连相对该参照也无优势则留 5。此口径对 13 个 idea 一致适用。
- 淘汰逻辑：任一 F1 命中红海（判 Reject and Pivot）、或出现 CRITICAL（如数据已驳倒核心机制）、或均分 < 7.0 → 出局。

---

## 3. 排名（10 个达标候选，按均分降序；同分依次按 Stronger → Broader → Higher 决胜）

| 排名 | ID | 名称 | H | F | S | C | B | 均分 | 裁决 | 目录 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | CSDA | 因果步级依赖消融遗忘 | 9 | 7 | 8 | 8 | 7 | **7.8** | Strong Accept | `workspace/idea-stage-1126/IDEA_01_CSDA/` |
| 2 | EAS | 纠缠度自适应干预调度 | 8 | 8 | 8 | 8 | 7 | **7.8** | Strong Accept | `workspace/idea-stage-1126/IDEA_02_EAS/` |
| 3 | RRBU | 重学习鲁棒双层遗忘 | 8 | 6 | 9 | 7 | 8 | **7.6** | Strong Accept | `workspace/idea-stage-1126/IDEA_03_RRBU/` |
| 4 | DISCO | 解码不变的参数级抑制 | 8 | 7 | 9 | 7 | 7 | **7.6** | Strong Accept | `workspace/idea-stage-1126/IDEA_04_DISCO/` |
| 5 | ORTHO | 推理子空间正交保护 | 8 | 7 | 8 | 7 | 8 | **7.6** | Strong Accept | `workspace/idea-stage-1126/IDEA_05_ORTHO/` |
| 6 | SCOPE | LRM CoT 等价类边界遗忘 | 7 | 6 | 9 | 7 | 8 | **7.4** | Accept with Revisions | `workspace/idea-stage-1126/IDEA_06_SCOPE/` |
| 7 | SEQU | 序列遗忘的 CoT 漂移校正 | 8 | 6 | 8 | 7 | 8 | **7.4** | Accept with Revisions | `workspace/idea-stage-1126/IDEA_07_SEQU/` |
| 8 | CFA | 反事实侧信道审计与去关联 | 7 | 7 | 8 | 7 | 8 | **7.4** | Accept with Revisions | `workspace/idea-stage-1126/IDEA_08_CFA/` |
| 9 | RIPPLE | 多跳涟漪局部化 | 8 | 7 | 8 | 7 | 7 | **7.4** | Accept with Revisions | `workspace/idea-stage-1126/IDEA_09_RIPPLE/` |
| 10 | BOUND | 承诺边界可控早停 + 删失评测 | 8 | 8 | 7 | 8 | 6 | **7.4** | Accept with Revisions | `workspace/idea-stage-1126/IDEA_10_BOUND/` |

每份 `proposal.md` 均含：Problem Anchor、与近邻的 delta、LaTeX 形式化方法、R-TOFU 接口、决定性验证实验（T0/T1/T2 + 预注册判据）、五维评分表与裁决、失败模式与局限。

---

## 4. 淘汰记录（3 个未达标 idea，保留为反重复记忆）

| ID | 方向 | 均分 | 裁决 | 淘汰理由 |
|---|---|---:|---|---|
| X1 | 「R-TOFU++ 更细粒度 step-wise 新基准」 | 5.0 | Reject and Pivot | F1 命中红海：R-TOFU 已带 step-wise 指标，RippleBench / RMLLMU-Bench 已覆盖评测；「新基准构建」为历史记忆明确的高拥挤区 |
| X2 | 「新 token 级 unlikelihood / margin 损失变体」 | 4.6 | Reject and Pivot | F1 红海（GA→…→Unilogit 谱系）且**数据已驳倒核心机制**：本仓库 `docs/LCES_FULL_RESULTS.md` 实测 LCES 5-epoch 把泄漏转成 0.485 幻觉 / 0.240 拒答、MU 0.4170 < R2MU 0.5269 → 自动 CRITICAL 锁死 |
| X3 | 「全 CoT 序列整体 KL 到随机教师」 | 4.6 | Reject and Pivot | 实质是 R2MU `unthinking` 的翻版（dominated by prior work），且已知导致 0.95+ collapse；无新机制 |

---

## 5. 与仓库既有 idea 的去重说明

- 当前 `fuxian` 分支的 `docs/METHOD_STATUS.md` 中 **UPBD**（active，边界继承 + 行为修复）、**CCPD**、**causal concept-direction ablation**（backup）均未纳入本轮，避免重复。
- `origin/Auto` 分支 `idea-stage/candidates/N02–N12` 与 `ROUND1/2_ELIMINATED.md` 已读取；其中 **N02 删失区间**、**N05 span 级**、**N06 状态不变**、**N07 效用 Pareto 门**、**N10 状态位移**、**N12 know-check 门** 与 BOUND / DISCO / SEQU 部分相邻，已在对应 proposal 中显式划界（BOUND = 方法而非仅评测；DISCO = 解码 min-max 而非生成状态同态；SEQU = 多轮漂移而非单轮加权）。
- **N01/N04/N09/N11** 已被 STaR-MCS / Coverage-Aware / K-Bench / REDaCT 抢先，本轮不再复活。

---

## 6. 下一步

- [ ] 对 Top-1 `CSDA` 跑其 `proposal.md §5` 的 T0/T1/T2（需服务器 GPU + R-TOFU forget10 + activation patching 脚本）。
- [ ] 取得真实跨家族评审线程（当前仅 hy4 = 用户指定 override，provisional），把 `evidence_gate` 从 `BLOCKED` 解锁。
- [ ] 对通过的 idea 调 `/experiment-plan` → `/experiment-bridge`；对未过的按 proposal §7 失败模式转向。

> 本报告为候选初筛，不是投稿级结论。所有分数为机制分，标「待验证实验」，无实测数字。
