# UPBD 统一遗忘框架 — Idea Discovery 与五维评估报告

**日期**：2026-11-28  
**方向**：保留 UPBD 的“边界继承 + 行为修复”，将其从单一 post-hoc 配方体系化为完整、统一的 LRM 遗忘框架。  
**实际流程**：deep-research / research-lit → idea-creator（本地发散，外部生成不可用）→ 多源 novelty-check → idea-evaluator → 排名与六份 proposal。  
**产物**：6 个最终候选、6 份独立 proposal；淘汰与重生成记录见下文。  
**证据状态**：未运行新实验；所有高分均为机制分；外部评审两次超时，`REVIEW_UNAVAILABLE`；`evidence_gate = BLOCKED`。

---

## 1. Executive Summary

推荐首位是 **TCR**：把 UPBD 继承的单个 checkpoint 扩展为行为、表示、参数三层契约，并将修复写成可容许集成员问题。它不是“更强的遗忘损失”，而是让任意上游 unlearner 都能输出可继承边界、接入同一行为修复与审计接口。

六个最终方案均达到用户指定门槛：五维平均分不低于 7，且裁决均为 **Accept with Revisions**。没有给 Strong Accept，因为每个方案都有需要实际修改或验证的 MAJOR 缺陷，且没有新实验数据。排名表示**研究决策价值与可执行性**，不是投稿接收概率。

## 2. 研究问题与范围

- **RQ1**：现有强方法把遗忘落在哪一层，UPBD 还能保留什么真正不同的对象？
- **RQ2**：如何把边界继承与行为修复写成完整训练—组合—审计闭环，而非 retain SFT 的新名字？
- **RQ3**：哪些实验能区分边界继承、普通行为监督、初始化与真正的修复算子贡献？

非目标：不宣称参数级知识擦除；不声称全局 MIA/relearning 安全；不把固定拒答率当作遗忘充分条件；不单独优化某个上游遗忘算法。

## 3. Literature Landscape

### 3.1 强方法与框架占据的落点

| 方向 | 已核实代表工作 | 对本任务的约束 |
|---|---|---|
| LRM 基准与 CoT 泄漏 | [R-TOFU, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.265/) | 必须逐步审计 CoT，并覆盖 ZeroThink/LessThink 等解码变化 |
| 表示级遗忘 | [R2MU, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.220/) | 是 UPBD 的继承边界来源，不可把其表示误导写成新贡献 |
| 自然退出轨迹 | [GUARD, EMNLP 2026](https://arxiv.org/abs/2609.21677) | safe-exit distillation 已被占据；新方案必须有额外不变量或约束对象 |
| 反事实轨迹 | [CiPO, ACL 2026](https://aclanthology.org/2026.acl-long.143/) | 反事实 CoT + 偏好优化不能作为独立 delta |
| 固定约束下恢复效用 | [PTP-U, 2026](https://arxiv.org/abs/2610.11915) | “先遗忘、后修复”作为一般思想已有直接先例 |
| 子空间/post-hoc 投影 | [OBU](https://arxiv.org/abs/2610.27490)、[JANUS](https://arxiv.org/abs/2609.19985) | LoRA 投影、零空间修复本身不新；需要新的边界对象与组合语义 |
| 约束多目标 | [HAMU, ICML 2026](https://arxiv.org/abs/2606.02119) | constrained unlearning 与不可调和停止已存在 |
| 路由式遗忘 | [CRU, 2026](https://arxiv.org/abs/2609.34475) | 冻结基座 + query-time routing 已存在，普通 router 不足够 |
| 动态审计 | [The Unlearning Mirage, COLM 2025](https://arxiv.org/abs/2603.11266) | 改写、别名、多跳探针是必要审计，不应包装成新的训练机制 |
| 多跳路径切断 | [DUMIC, 2026](https://arxiv.org/abs/2609.34442) | “切断所有恢复路径”已被占据，CLOSURE 因而淘汰 |
| 诚实遗忘 | [Unlearners Can Lie, ACL 2026](https://arxiv.org/abs/2605.08765) | 固定 IDK 与稳定拒答可能是浅层/欺骗性行为，必须单独测诚实性与结构有效性 |
| 重训练参照蒸馏 | [UNDO, NeurIPS 2025](https://arxiv.org/abs/2506.06278) | retain-only oracle 作为蒸馏目标已出现，REF-TWIN 因而淘汰 |
| 后续微调不变性 | [ILU, ICML 2025](https://arxiv.org/abs/2506.01339) | BLESS 只能主张良性算子族上的不变集/组合裕度，不能泛称 invariance 首创 |
| 流式约束修复 | [BLADE](https://arxiv.org/abs/2608.22557)、[PDU](https://arxiv.org/abs/2506.05314) | STREAM 的 delta 只能是流式状态、累计违反和不可行语义 |
| OT 遗忘 | [Opt-Out, ACL 2025](https://arxiv.org/abs/2406.12329)、[UOT-Unlearn](https://arxiv.org/abs/2603.16489) | TMOT 必须落在离散 CoT 行为、禁运支撑与三边际，而非“OT + unlearning” |
| 请求/参数冲突 | [CLUE](https://arxiv.org/abs/2509.20977)、[CATA](https://arxiv.org/abs/2605.18610)、[ICCU](https://arxiv.org/abs/2605.27138) | CROWN 不能主张首次 SAT、权重冲突或顺序无关；只保留请求级最小让步与组合重验 |

### 3.2 统一后的真正空位

文献并非没有“修复”，而是缺少一个同时满足以下条件的 LRM 框架：

1. 上游 unlearner 产生可继承、可重验的边界对象；
2. 行为修复同时覆盖 CoT、retain、aux 与非披露参考；
3. 遗忘结论限定于声明的探针族、攻击族和请求范围；
4. 对未见问法、后续良性学习、连续请求与组合冲突有明确失效语义；
5. retain-only 与 target-start 控制用于归因，而非可选附录。

本地 `papers/`、`literature/` 未发现 PDF；本地 awesome-lrm-unlearning 仅作线索，正式引用均回到论文原页或官方 proceedings 核实。检索结论是“在这些关键词下未检索到直接重叠”，不等于全球首创。

## 4. 本地证据能支持什么

- 可支持：UPBD 是一种低成本的边界继承 + 行为修复实现，已有 pilot 与全量运行产物。
- 不可支持：全量 `holdout=0` 的 forget/retain 行同时用于训练和评测，不能证明 held-out 泛化。
- 不可支持：pilot 与全量使用不同 judge/样本口径，不能把数字差异解释为规模效应。
- 不可支持：高 AFE/CFE 与高拒答不能区分知识擦除、拒答策略和 collapse。
- 因此六个方案的评分**不使用 pilot 小样本增益作为方法有效性的直接证据**，只把已有管线视为可行性基础。

## 5. 共享统一接口

所有最终候选共享四元接口，但各自只引入一个主机制：

$$
\mathfrak F=(B,U,A,G),
$$

- $$B$$：可检验的 forget boundary；
- $$U$$：retain/aux 的 target 行为与结构有效性；
- $$A$$：探针族、攻击族、置信区间与覆盖范围明确的审计证据；
- $$G$$：未见问法、后续学习或请求组合下的适用范围与失效状态。

共同训练原则：从已有 unlearned checkpoint 出发；forget 侧学结构有效、非披露、非固定模板的行为；retain/aux 侧恢复 target 行为；LoRA 更新必须通过独立 calibration 才能接受。

共同归因控制：

- **C1 retain-only**：从同一边界做等数据、等步数、等参数面的 retain/aux SFT；
- **C2 target-start**：从 target 做同一完整配方；
- **M 完整方法**：只有相对 C1 有独立增益、且不复制 C2 的泄漏行为，才可归因给新机制。

共同审计：按实体与问题隔离 train/calibration/test；覆盖直接问、改写、别名、多跳、解码变体和良性 relearning；judge 固定 Hy4；分别报告 leak、collapse、hallucination、refusal、held-out utility 与置信区间。

## 6. Ranked Ideas

| 排名 | Idea | 主机制 | H/F/S/C/B | 均分 | Novelty gate | 裁决 | Proposal |
|---:|---|---|---|---:|---|---|---|
| **1** | **TCR** | 行为/表示/参数三层契约的可容许修复 | 7/8/8/8/9 | **8.00** | 7 · PROCEED | Accept with Revisions | [IDEA_01_TCR](IDEA_01_TCR/proposal.md) |
| **2** | **BLESS** | 良性后续学习算子族下的不变集与组合裕度 | 7/7/9/7/8 | **7.60** | 6 · CAUTION | Accept with Revisions | [IDEA_02_BLESS](IDEA_02_BLESS/proposal.md) |
| **3** | **RCBF** | 参数—屏障共适应的 CoT 前向安全集 | 8/6/9/7/8 | **7.60** | 6 · CAUTION | Accept with Revisions | [IDEA_03_RCBF](IDEA_03_RCBF/proposal.md) |
| **4** | **STREAM** | 流式原始—对偶修复、累计违反与不可行输出 | 7/8/8/7/8 | **7.60** | 5.5 · CAUTION | Accept with Revisions | [IDEA_04_STREAM](IDEA_04_STREAM/proposal.md) |
| **5** | **TMOT** | forget 禁运、retain 守恒、unknown 受控的三边际 OT | 8/6/8/6/9 | **7.40** | 6 · CAUTION | Accept with Revisions | [IDEA_05_TMOT](IDEA_05_TMOT/proposal.md) |
| **6** | **CROWN** | 请求级语义冲突、最小让步与组合后重验 | 7/7/8/7/8 | **7.40** | 6 · CAUTION | Accept with Revisions | [IDEA_06_CROWN](IDEA_06_CROWN/proposal.md) |

### 排名依据

1. **TCR**：最接近统一母框架，2×H20 上原型成本最低，且三层消融能直接检验是否只是工程串联。
2. **BLESS**：问题重要、与 UPBD 的自然冲突最强；先验证局部裕度是否预测真实恢复，再决定是否值得做闭包。
3. **RCBF**：科学上限高，但“策略改变后 barrier 失效”是明显理论风险，因此排在 BLESS 后。
4. **STREAM**：工程价值高，但 BLADE/PDU/OLU 已占据算法零件，必须先证明不可行语义与累计违反有新增信息。
5. **TMOT**：三边际结构有潜力，但序列 OT 与 LoRA 蒸馏的近似链最容易失真。
6. **CROWN**：问题新颖性较好，但冲突数据与策略权重依赖人工定义，且须放弃参数交换性的强话术。

## 7. 六个候选的逐一评估摘要

### 7.1 TCR

- **一句话故事**：把继承的参数点变成可组合的三层契约，行为修复只能在契约交集内移动。
- **MAJOR**：PTP-U/OBU/JANUS 近邻强；单 checkpoint 不能识别真实语义边界。
- **Lifecycle**：Innovative Technique，8–12 周；fit=Yellow。
- **Paradigm**：First Principles、Elephant、Technology Cycle、Hamming 均为 Yes，但只有概念潜力。
- **Top actions**：边界读取稳定性 T0；C1/C2；三层逐一消融。

### 7.2 BLESS

- **一句话故事**：交付后的良性微调也属于遗忘系统的作用域，修复必须保留组合安全裕度。
- **MAJOR**：ILU/Margin Cliff 近邻；有限算子族不能证明全局闭包。
- **Lifecycle**：Frontier Exploration，8–10 周；fit=Yellow。
- **Paradigm**：First Principles、Elephant、Technology Cycle、Hamming 为 Yes。
- **Top actions**：UPBD 修复后的敏感差异 T0；去裕度/去约束 T2；留出整族 T3。

### 7.3 RCBF

- **一句话故事**：CoT 遗忘是轨迹安全集的前向不变性，修复策略和 barrier 必须共同更新。
- **MAJOR**：BarrierSteer/CBF-LLM 已有核心工具；旧动力学 barrier 对新策略可能无效。
- **Lifecycle**：Frontier Exploration，10–14 周；fit=Yellow。
- **Paradigm**：First Principles、Elephant、Technology Cycle、Hamming 为 Yes。
- **Top actions**：冻结 barrier 的失效 T0；共适应对照 T2；正裕度/参数邻域消融 T3。

### 7.4 STREAM

- **一句话故事**：连续删除需要继承历史约束并允许不可行，而不是每轮独立宣称成功。
- **MAJOR**：BLADE/PDU/OLU 近邻；非凸 LRM 上 no-regret 不可直接成立。
- **Lifecycle**：Data-Intensive / Innovative Technique，10–12 周；fit=Yellow。
- **Paradigm**：Elephant、Technology Cycle、Hamming 为 Yes；First Principles 为 No。
- **Top actions**：最小不可行请求 T0；流式基线对照 T1；对偶继承消融 T2。

### 7.5 TMOT

- **一句话故事**：把遗忘实现为行为概率质量的禁运运输，而非压到固定拒答点。
- **MAJOR**：Opt-Out/UOT-Unlearn/Cheng 近邻；高维序列 OT 与蒸馏误差可能使约束失真。
- **Lifecycle**：Cross-Disciplinary，10–14 周；fit=Yellow。
- **Paradigm**：First Principles、Elephant、Technology Cycle、Hamming 为 Yes。
- **Top actions**：禁运质量数值检查 T0；三边际消融 T2；禁运改软惩罚 T3。

### 7.6 CROWN

- **一句话故事**：多请求遗忘首先是语义可满足性与让步问题，然后才是参数组合。
- **MAJOR**：CLUE/CATA/ICCU 近邻；有状态继承与严格交换性冲突。
- **Lifecycle**：Innovative Technique / Novel Problem，8–12 周；fit=Yellow。
- **Paradigm**：First Principles、Elephant、Hamming 为 Yes；Technology Cycle 为 No。
- **Top actions**：显式冲突最小例 T0；请求流基线 T1；固定优先级对照 T2。

完整五维证据、可行性、失败模式与转向条件见各 proposal 第 7–8 节。

## 8. Novelty Verification 与淘汰—重生成记录

### 8.1 第一轮

| 原候选 | 核心问题 | 处理 | 后续 |
|---|---|---|---|
| RISC | Conformal Unlearning 已占据双侧风险界，SUITE 已占据组合探针；五维均分 6.8，低于 7 | **Reject and Pivot** | 不再以 conformal/router 为贡献；转向新的动力学与请求语义对象 |
| CLOSURE | DUMIC 已包含“sever all recovery paths”；闭包/min-cut 是对偶而非新不变量 | **Reject and Pivot（CRITICAL F1，指名 DUMIC）** | 重生成 RCBF，把对象从静态路径图改为轨迹动力学安全集 |
| REF-TWIN | UNDO 已有 retain-only retrain oracle 蒸馏基线，SHIFT/FADE 也接近 | **Reject and Pivot（CRITICAL F1，指名 UNDO）** | 重生成 TMOT，放弃 oracle 行为匹配，改为禁运/三边际运输 |
| CAUSAL-SCAFFOLD | CSD、Selective Forgetting、PACT 分别覆盖实体替换、CoT slot 替换和成对反事实；均分约 6.2 | **Reject and Pivot** | 不再以实体替换反事实作为独立创新 |
| BLESS | ILU 近邻强，但“良性算子族不变集 + 组合裕度”仍有可证伪差异 | 收窄后保留 | 最终 #2 |
| CROWN | 严格交换性与有状态继承冲突 | 放弃参数交换性，只保留规范层代数与最小让步 | 最终 #6 |
| TCR | PTP-U/OBU/JANUS 强近邻 | 收窄到三层 checkpoint 契约与组合接口 | 最终 #1 |

### 8.2 第二轮

| 原候选 | 核心问题 | 处理 | 后续 |
|---|---|---|---|
| COVER | SUITE/CluCERT/Margin Cliff 三方夹击，Lipschitz 覆盖界可能空洞；均分约 6.0 | **Reject and Pivot** | 不进入最终六选 |
| META | L2UL/UnHype 已占据学习型 unlearning 算子与 hypernetwork LoRA；均分约 6.6 | **Reject and Pivot** | 不以“学习修复算子”充数 |
| DATA | RASLIK/GRACE 已覆盖影响/正交 retain 选择；均分约 6.2 | **Reject and Pivot** | 数据选择不独立成篇 |
| DARWIN | 分型、算子选择、停止证书分别被既有工作覆盖；全局 no-further-repair 不可判定 | **Reject and Pivot** | 不将 taxonomy + bandit + stopping 拼装为框架 |
| RCBF | BarrierSteer/CBF-LLM 近邻强 | 加入策略—屏障共适应、正裕度和有限经验证书后保留 | 最终 #3 |
| STREAM | 原始—对偶与 LoRA repair 已有 | 只保留流式状态、累计违反、凸代理 regret 与不可行语义 | 最终 #4 |
| TMOT | OT 遗忘已有多种实现 | 只保留离散 CoT 行为上的禁运支撑与三边际 | 最终 #5 |

机械去重还排除了：BUDGET 连续预算、DISCO 解码 min-max、CFA 风格侧信道、RRBU relearning unroll、SCOPE 等价类路由、PROBE 表示不可解码等本地已有方向；它们没有通过改名进入最终候选。

## 9. External Critical Review

- Hy4 发散生成：240 秒超时，无可用响应。
- Hy4 六候选批判审查：300 秒超时，无可用响应。
- 状态：**REVIEW_UNAVAILABLE**。未获得可核验的外部接受，不声称跨家族通过。
- 审计轨迹：
  - `.aris/traces/idea-discovery/2026-11-28_upbd/`
  - `.aris/traces/research-review/2026-11-28_upbd/`

本报告的五维裁决是执行者按 `idea-evaluator` 标准进行的本地初筛，不是外部 reviewer verdict。

## 10. 下一步与证据边界

1. 所有方案先跑共同 C1/C2 与严格 held-out 切分，再运行各自 T0；不要先堆完整系统。
2. 推荐首先做 TCR 的边界读取稳定性与三层消融；BLESS 与 RCBF 共享生成审计，可复用产物。
3. 任何“证书”都必须写明探针族、攻击族、扰动范围与数值近似；超出范围输出未知/不可行，不写全局保证。
4. 若新方法只提高固定模板拒答，或增益被 retain-only/target-start 解释，立即按各 proposal 的转向规则缩减或放弃。
5. 未找到 ARIS `run_state.py` / `idea_discovery_gate.py` 帮助器，且没有新实验与外部评审，因此本报告不作为已通过的 idea-discovery evidence gate receipt。
