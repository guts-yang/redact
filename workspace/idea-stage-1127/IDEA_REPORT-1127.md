# LRM 机器遗忘 · Idea Discovery + Idea Evaluator 报告（第二轮 1127）

**方向**: Large Reasoning Model (LRM) 机器遗忘创新 idea 调研（第二轮，与 2026-11-26 首轮 10 个 ideas 显式区分）
**数据集 / 基线**: R-TOFU（EMNLP 2025, arXiv:2505.15214）作为主数据集之一与基线协议；对照含 R2MU、Reasoned IDK、REDaCT、CiPO、STaR、GUARD、CRISP、SAGO、SHRED、CATA/MCU、Multilingual Unlearning (ICML 2026)
**日期**: 2026-11-27
**流程**: `/idea-discovery`（Phase 1 文献调研 + Phase 2 idea 生成）→ `/idea-evaluator`（逐条八步评估）→ 双门槛筛选 → 排名 → 10 份 proposal
**输出根目录**: `/Users/chenyliao/Repositories/Machine_Unlearning/redact/workspace/`
**本轮目录后缀**: 全部 10 个新目录带 `-1127` 时间戳，与旧 `IDEA_01_CSDA`…`IDEA_10_BOUND` 物理区分

---

## 0. 执行边界与诚实性声明（务必先读）

本轮按用户显式指令对标准 `/idea-discovery` 流水线做了裁剪，以下依赖**确实不可用**，已如实标注，未假装执行：

| 标准环节 | 状态 | 实际做法 |
|---|---|---|
| Gemini 增强 `/research-lit` | 不可用（gemini-cli / Codex MCP 未挂载） | 用真实 `web_search` 多轮检索 grounding，来源标注见 §1 |
| `/novelty-check` Phase C 跨模型评审 | 不可用（codex/gemini/claude 后端 `null`，`reviewer.json` 缺失） | 新颖性判断由执行者基于检索到的近邻做 metadata 级划界，标为 **provisional** |
| `/research-review` 外部评审 | 不可用（无跨家族评审线程） | 由用户指定的 `/idea-evaluator` 承担评估，裁决为**执行者本地初筛** |
| GPU pilot（MAX_PILOT_IDEAS=3） | 不可用（本机 `torch.cuda.is_available()=False`，checkpoint 在服务器 `/usr/local/...`） | 全部 idea 为**机制分 + 命名的决定性验证实验**，无实测数字（遵守「不编造」纪律） |
| `run_state.py` / `idea_discovery_gate.py` per-stage evidence gate | `BLOCKED`（helper 在所有 canonical 路径均缺失） | 不伪造 receipt；本报告 `evidence_gate = BLOCKED`，所有裁决 provisional |
| 单一 `FINAL_PROPOSAL.md` + pilot | 被用户显式覆盖 | 用户要求 10 份并列 proposal + 排名，故以本文为总览、`workspace/idea-stage-1127/IDEA_RANKING-1127.json` 为机器可读排名、`workspace/idea-stage-1127/IDEA_XX_*-1127/proposal.md` 为 Individual 工件 |

**结论**：本轮是「候选 idea 生成与初筛」，不是投稿级验证。`assurance = draft`，`evidence_gate = BLOCKED`，无跨家族接受。任何 idea 进入实验前需先跑其 proposal 中的决定性验证实验，并取得真实跨模型评审。

---

## 1. 文献 Landscape（真实检索 grounding，2026-11-27）

检索来源：`web_search`（多轮）。下列条目为检索命中，按标题/摘要做 **metadata 级**判断；凡来自二手论文笔记站点（papernotes / graphastra / alphaxiv / lacuna / ACL Anthology 镜像）者已标注，引用时仅作 landscape 描述，不以其中数字作为打分证据。

**基准与数据集**
- **R-TOFU** — Yoon, Jeung, No, EMNLP 2025, arXiv:2505.15214 / ACL 2025.emnlp-main.265。首个 LRM 遗忘基准；GPT-4o 生成 model-aligned CoT；step-wise 指标（ROUGE-L / Cosine / LLM-as-Judge）；`forget01/05/10`；目标模型 DeepSeek-R1-Distill-Llama-8B。**关键发现**：answer-only 不足（CoT 残留）、CoT-only 最有效、Reasoned IDK 优于直接拒答、**ZeroThink/LessThink 解码复活已忘内容**。
- TOFU — Maini et al., 2024（上游虚构作者数据集）。

**LRM 专属方法（2025–2026，已成型，构成 F1 近邻）**
- **R2MU** — 2025, arXiv:2506.12963。`unthinking` loss 把 forget 推理步表示误导到随机向量；副作用 reasoning collapse 0.95–1.0。
- **REDaCT** — Findings EMNLP 2026, arXiv:2610.12769。推理时句级 leak detector + corrective embedding，零 collapse；本 `redact` 仓库复现的参考 arm。
- **CiPO** — ACL 2026, arXiv:2604.15847。反事实推理轨迹作 DPO chosen，迭代偏好优化；形式化 $Y\perp F\mid Q$；自承「固定反事实/拒答模式可成为信息泄露通道」。
- **STaR** — AAAI 2026, arXiv:2601.9281。无参数推理时敏感轨迹调控 + MCS 多解码一致性评分 + 多粒度 MIA。
- **GUARD** — 2026, arXiv:2609.21677。引导答案-推理蒸馏学习「自然遗忘轨迹」，提出 NFRS 评分（结构稳定性/流畅度/无幻觉替代）。
- **FRUL / Selective Forgetting** — PAKDD 2026。RAG + 特征替换，用中性占位符替换 CoT 敏感片段。

**2026 新涌现、直接决定本轮 F1 划界的近邻（首轮报告未覆盖）**
- **CRISP** — Ashuach, Arad, Mueller, Tutek, Belinkov, ACL 2026, arXiv:2508.13650 / 2026.acl-long.82。SAE 对比特征选择 + LoRA 把显著特征激活「焊死」为零，WMDP 上持久概念遗忘。**「SAE 特征消融」已被占据**。
- **SAE Feature Unlearning is Shallow** — Field & Yampolskiy, ICML 2026 MechInterp Workshop。消融单语特征只压制「说」不压制「懂」，证明 SAE 干预是浅层的。
- **SAGO / Asymmetric Two-Task** — ACL 2026, arXiv:2604.14808。把遗忘建模成「retain 主、forget 辅」的非对称两任务，元素级符号门控/PCGrad，WMDP Bio MMLU 恢复 44.6%→96.0%。**「梯度几何/符号门控」已成片**（另有 GRU、OrthoGrad、LUR、OFMU、VLA-Forget）。
- **SHRED** — Hu et al., NeurIPS 2026, arXiv:2605.07482。无保留集遗忘：高惊异度 token logit 降级 + Top-K 自蒸馏，TOFU/MUSE/RWKU/Hubble 新 Pareto。**「无保留集 token 级自蒸馏」已被占据**。
- **MAGE** — arXiv:2604.13777。最小锚点 + 记忆图自生成 corpus-free 监督（TOFU/RWKU）。
- **CATA / MCU** — arXiv:2605.18610 / 2608.04548。持续遗忘的冲突规避 task arithmetic / 一次性适配器合并。**「持续遗忘合并」已被占据**（且旧 SEQU 已覆盖序列漂移）。
- **Multilingual Unlearning in LLMs** — Xiang et al., ICML 2026, arXiv:2606.03291。5 语言 TOFU 转移矩阵；发现遗忘只动后段语言特化层、共享语义空间基本不动；单一推理时 steering 向量可跨语言恢复 50%(Qwen)/90%(Gemma)。**「跨语言转移」已被诊断，但 LRM CoT 枢轴场景未被处理**。
- **Recoverable Drift** — arXiv:2610.11915（PTP-U，2026-10）。遗忘后有大量「可恢复漂移」，把内部表示回拉可恢复效用而几乎不改遗忘程度。**「整体回拉」已被占，但步级 CoT 分解未被做**。
- **UnPact** — arXiv:2509.24675, 2025。prompt attribution：遗忘=破坏关键词关注；知识可经关键词强调恢复；Recovery Rate / Destructive Rate 两难。
- **PARS / Hidden-State Leakage** — arXiv:2609.36612, 2026。输出抑制与表征擦除可分离；probe-adversarial 目标。
- **CoT Parametric Faithfulness** — ACL 2026（二手）。CoT 可不忠实（不反映真实决策因素）；控制任务解耦。
- **综述级开放问题** — arXiv:2601.13264（Unlearning in LLMs: Methods, Evaluation, Open Challenges）、HAL hal-05653389（When Forgetting Fails，50 方法三阶段分类 + 分层认证框架）、阿里云综述（2026-11-25）：均指向「认证保证、可逆性/对抗再学习、Compound AI 系统、跨语言/多模态、机制级深层擦除与独立审计」为 2026 未解前沿。

**结构性空白（本轮 idea 的立论依据，与首轮 10 个不重叠）**
1. R-TOFU 的 ZeroThink/LessThink 复活只是「发现」，STaR 的 MCS 只是评测；把**连续推理预算**当攻击维度做 min-max 训练、要求预算单调安全，未见。
2. 多语言遗忘已在普通 LLM 上诊断，但 LRM 的「非英文提问 → 英文 CoT 枢轴思考」这一特有路径下的遗忘转移与训练方法，未见。
3. 恢复攻击多为 relearning / steering；把 LRM 的**多采样自一致性（majority vote）本身**当恢复攻击与训练目标，未见。
4. 可恢复漂移已出现，但把漂移**分解到 CoT 步级**、只回拉非承重步，未见。
5. 可逆性已被 steering 证明，但把「单一推理方向恢复半径」写成训练期可优化约束，未见。
6. CoT 不忠实（推理文本≠真实驱动）已诊断，但据此把遗忘目标从「压 CoT」改成「答案级依赖擦除 + 忠实度门控」，未见。
7. MAGE 的 corpus-free 是普通 LLM 的实体级；LRM 上「最小锚点 → 自生成 CoT 步级监督」未见。
8. UnPact 的关键词归因是审计；把步级关键词归因做成「恢复-破坏两难」的训练闭环，未见。
9. 跨解码策略（DISCO 旧）讲的是输出不变；跨策略一致的**擦除深度定位与证书**（结构层而非行为层）未见。
10. PARS 指出输出/表征可分离；在 **CoT 步级**做 probe-adversarial 的「不可解码擦除」未见。

**红海避让区（本轮 F1 直接淘汰，见 §4）**：token 级 unlikelihood/margin 损失（GA→…→Unilogit）、新 step-wise 基准、梯度符号门控/PCGrad、无保留集 token 自蒸馏（SHRED）、SAE 特征消融（CRISP）、模型合并持续遗忘（CATA/MCU）、单纯量化链路、推理时轨迹调控（STaR/GUARD/TTRL/ICUL）。

---

## 2. 评估口径（用户双门槛）

- 框架：`/idea-evaluator`（八步：定位 → F1–F10 致命缺陷审计 → 生命周期 → 五维 → 范式四问 → 可行性 → 诚实性闸门 → 三档裁决）。
- **硬门槛 A**：五维（Higher / Faster / Stronger / Cheaper / Broader）**算术均分 ≥ 7.0**。
- **硬门槛 B**：裁决 ∈ {**Strong Accept**, **Accept with Revisions**}（即 ≥ Accept with Revisions）。
- 评分纪律：每维起评 5；机制成立的维度标「机制分，待验证实验」并命名决定性实验；无依据者留 5 注 `no grounds given`。
- **Faster / Cheaper 参照系**：以 *exact unlearning（排除 forget set 全量重训）* 为参照，近似遗忘方法在机制上均更快更省，故二者可给机制分；若连相对该参照也无优势则留 5。此口径对全部候选一致适用。
- 淘汰逻辑：任一 F1 命中红海（判 Reject and Pivot）、或出现 CRITICAL（数据已驳倒核心机制）、或均分 < 7.0 → 出局。
- **去重纪律**：每个候选在 proposal §2 显式列出「与旧 10 个 ideas 的划界」与「与 2026 新近邻的划界」，必须给出至少一个不同轴（作用对象 / 机制 / 输入粒度 / 问题设定）。

---

## 3. 排名（10 个达标候选，按均分降序；同分依次按 Stronger → Broader → Higher 决胜）

| 排名 | ID | 名称 | H | F | S | C | B | 均分 | 裁决 | 目录 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | BUDGET | 推理预算单调安全的连续预算 min-max 遗忘 | 8 | 7 | 9 | 7 | 8 | **7.80** | Strong Accept | `workspace/idea-stage-1127/IDEA_01_BUDGET-1127/` |
| 2 | PIVOT | 跨语言 CoT 枢轴下的遗忘转移与去枢轴化 | 7 | 6 | 9 | 7 | 9 | **7.60** | Accept with Revisions | `workspace/idea-stage-1127/IDEA_02_PIVOT-1127/` |
| 3 | CONSENSUS | 多采样自一致性共识恢复攻击 + 共识不变目标 | 8 | 6 | 9 | 7 | 8 | **7.60** | Strong Accept | `workspace/idea-stage-1127/IDEA_03_CONSENSUS-1127/` |
| 4 | DRIFT | 可恢复漂移的 CoT 步级分解与效用回补 | 8 | 7 | 7 | 8 | 8 | **7.60** | Accept with Revisions | `workspace/idea-stage-1127/IDEA_04_DRIFT-1127/` |
| 5 | REVERSE | 推理时可逆半径约束（单向量恢复阻断） | 8 | 6 | 9 | 7 | 7 | **7.40** | Strong Accept | `workspace/idea-stage-1127/IDEA_05_REVERSE-1127/` |
| 6 | FAITH | CoT 不忠实下的答案级反事实依赖遗忘 | 8 | 7 | 8 | 7 | 7 | **7.40** | Accept with Revisions | `workspace/idea-stage-1127/IDEA_06_FAITH-1127/` |
| 7 | GRAPH | 最小锚点驱动的自生成 CoT 步级遗忘监督 | 7 | 6 | 8 | 9 | 7 | **7.40** | Accept with Revisions | `workspace/idea-stage-1127/IDEA_07_GRAPH-1127/` |
| 8 | ATTRIB | 步级关键词归因与恢复-破坏两难阻断 | 8 | 7 | 8 | 8 | 6 | **7.40** | Accept with Revisions | `workspace/idea-stage-1127/IDEA_08_ATTRIB-1127/` |
| 9 | LAYER | 跨解码策略一致的擦除深度定位与证书 | 7 | 6 | 8 | 7 | 8 | **7.20** | Accept with Revisions | `workspace/idea-stage-1127/IDEA_09_LAYER-1127/` |
| 10 | PROBE | CoT 步表示探针对抗的不可解码擦除 | 8 | 6 | 8 | 7 | 7 | **7.20** | Accept with Revisions | `workspace/idea-stage-1127/IDEA_10_PROBE-1127/` |

每份 `proposal.md` 均含：Problem Anchor、与旧 ideas + 2026 近邻的 delta、LaTeX 形式化方法、R-TOFU 接口、决定性验证实验（T0/T1/T2 + 预注册判据）、五维评分表与裁决、失败模式与局限。机器可读排名见 `workspace/idea-stage-1127/IDEA_RANKING-1127.json`。

---

## 4. 淘汰记录（6 个未达标 idea，保留为反重复记忆）

| ID | 方向 | 裁决 | 淘汰理由 |
|---|---|---|---|
| X1 | 推理时 TTT/ICL 轨迹调控遗忘 | Reject and Pivot | F1 红海：STaR / GUARD / DRAGON / ICUL / TTRL 已占「无参数推理时轨迹调控」，与旧 REDaCT 参考 arm 也近 |
| X2 | 「R-TOFU+++ 更细粒度 step-wise 新基准」 | Reject and Pivot | F1 红海：R-TOFU 已带 step-wise 指标，且首轮报告 X1 已因同一理由淘汰 |
| X3 | 梯度符号门控 / PCGrad 冲突几何 | Reject and Pivot | F1 红海：SAGO (ACL 2026) + GRU + OrthoGrad + LUR + OFMU 已成片；与旧 ORTHO 子空间轴也部分重叠 |
| X4 | 无保留集 token 级自蒸馏 | Reject and Pivot | F1 红海：SHRED (NeurIPS 2026) 已占；且本仓库 `docs/LCES_FULL_RESULTS.md` 实测 token 级压制致 0.485 幻觉 / 0.240 拒答，data-refuted |
| X5 | 量化链路 CoT 复活联合保证 | Reject and Pivot | 与旧 RRBU 恢复攻击轴近，且撞用户自有 L3 多模态 / D3 压缩格点遗忘方向，避免自我重复 |
| X6 | 模型合并的持续遗忘去冲突 | Reject and Pivot | F1 红海：CATA (2026) / MCU (2026) 已占；且旧 SEQU 已覆盖序列漂移校正 |

---

## 5. 与首轮 10 个 ideas 的去重说明

首轮（2026-11-26）10 个：`CSDA`（步级因果承重）、`EAS`（纠缠度调度）、`RRBU`（恢复攻击双层）、`DISCO`（解码策略 min-max）、`ORTHO`（推理子空间正交）、`SCOPE`（CoT 步级等价类）、`SEQU`（序列漂移）、`CFA`（反事实侧信道）、`RIPPLE`（多跳涟漪）、`BOUND`（承诺边界早停）。

本轮划界对照：

| 本轮 | 最像的旧 idea | 不同轴（至少一条） |
|---|---|---|
| BUDGET | DISCO（解码策略 min-max） | 旧：离散策略集合 $\mathcal P$；本：**连续推理预算** $b$，加预算单调性约束（机制是序结构，不是策略枚举） |
| PIVOT | SCOPE（等价类） | 旧：同语言内改写等价类；本：**跨语言 + 语言内 CoT 枢轴**（输入语言 ≠ 推理语言），机制是去枢轴化 |
| CONSENSUS | RRBU（relearning 攻击） | 旧：参数级 relearning / SAE steering；本：**多采样 majority-vote 共识**当恢复信号与训练目标 |
| DRIFT | SEQU（漂移校正） | 旧：多次请求间的状态漂移；本：**单次遗忘后**的可恢复漂移，分解到 CoT 步级做事后回补 |
| REVERSE | RRBU / ORTHO | 旧：抗参数级攻击 / 梯度投影；本：**单一推理时方向的可逆半径**作为训练期约束 |
| FAITH | SCOPE / CSDA | 旧：压 CoT 步；本：**CoT 不忠实**前提下改压答案级依赖，忠实度作门控变量 |
| GRAPH | SCOPE | 旧：人工 paraphrase 库（且被记为不可复现）；本：**最小锚点自生成**步级监督（corpus-free，无外部库） |
| ATTRIB | CSDA（步级因果） | 旧：activation patching 的步级因果贡献；本：**关键词 prompt attribution** 的恢复-破坏两难闭环 |
| LAYER | DISCO / ORTHO | 旧：输出跨策略不变 / 子空间正交；本：**擦除深度**的跨策略一致定位与证书（结构层非行为层） |
| PROBE | RRBU / CFA | 旧：恢复攻击 / 风格侧信道；本：**步表示 probe-adversarial** 的不可解码性目标 |

`fuxian` 分支 `docs/METHOD_STATUS.md` 的 UPBD / CCPD / causal concept-direction ablation、以及 `origin/Auto` 的 N02/N05/N06/N07/N10/N12 与 BOUND/DISCO/SEQU 相邻，已在首轮划界；本轮不复活。

---

## 6. 下一步

- [ ] 对 Top-1 `BUDGET` 跑其 `proposal.md §5` 的 T0/T1/T2（需服务器 GPU + R-TOFU forget10 + 多预算生成/激活 patching 脚本）。
- [ ] 取得真实跨家族评审线程（当前仅 hy4 = 用户指定 override，provisional），把 `evidence_gate` 从 `BLOCKED` 解锁。
- [ ] 对通过的 idea 调 `/experiment-plan` → `/experiment-bridge`；对未过的按 proposal §7 失败模式转向。
- [ ] 复用 `workspace/implementations/` 的 10 个核心实现 + `hooks/` 接线层作为起点（第二轮方法可作为新模块接入 `diagnostic_train_hooked.py`）。

> 本报告为候选初筛，不是投稿级结论。所有分数为机制分，标「待验证实验」，无实测数字。R-TOFU 全量诊断（首轮 `FULL_EVAL_TABLE.md`）显示 0 个 PASS、`evidence_gate=BLOCKED`，本轮 ideas 同样需先过决定性实验。
