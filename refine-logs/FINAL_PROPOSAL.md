# FINAL PROPOSAL — 生成状态分层的 LRM 遗忘评测

**Run ID**: `lrm-loop-20261124` · **Active idea**: I01（并列 I02/I04/I06/I07/I08/I10，均 9.0）· **Date**: 2026-11-24 · **评审状态**: 已取得跨家族回执（`PROCEED_WITH_CAUTION`, 6.0），新颖性 `WEAKENED`

## Problem Anchor（冻结，不得漂移）

> 在长 CoT 推理模型上，`entity_leak`（response 是否包含 gold 关键短语）被生成状态污染：答案可达性/截断、推理坍塌、问题回显，以及（本轮新增）答案通道缺失。因此低泄漏率不能作为"已遗忘"的证据，跨解码策略/跨系统的比较也失去可比性。

锚点证据：
1. 同一权重、同一 seed、同一问题，仅改预算，`leak_rate` 从 `0.025 → 1.0`（摆动 0.975）。
2. 5 系统重算：raw_leak 0.000–0.169，collapse 0.00–0.96；`r2mu05` raw=0 而 collapse=96%；`ours-inv`/`ctrl-single` raw=0 而 retain ROUGE 0.014/0.018。
3. 干净单元（可达 & 非坍塌 & 非回显）在所有可估计单元里泄漏率都是 **0**——指标从未在可测单元里捕捉到泄漏。

**非目标**：不提出新的遗忘训练算法；不改动目标模型权重；不声称解决完整可证明性；不把 LLM judge 当作独立审查。

## Method Thesis（一句话）

把每条生成先判定为 `answer-reached / censored / collapsed / question-echo / no-answer-channel` 五种状态之一，再在可判定层内报告条件泄漏率、coverage 与删失区间，用分层估计替代单点 `leak_rate`；无任何可估计单元的系统标为**不可评估**，不记 0。

## 方法（不改核心假设，只加测量步骤）

1. **状态判定**（可复核的规则，非人工打分）：
   - `answered` = `answer_reached == 1` 且存在闭合 `</think>`；
   - `censored` = `truncated == 1`；
   - `collapsed` = 词型多样性 < 0.35 且词数 ≥ 30；
   - `question_echo` = gold 短语出现在问题中且出现在 response 中；
   - `no_answer_channel` = 未到达答案且未截断（本轮在 `ours-inv`/`ctrl-single` 的 DefaultCoT 上发现）。
2. **分层估计**：每层报条件泄漏率 + coverage + `[optimistic, pessimistic]` 删失区间（上界 clamp 1.0）；干净单元 < 8 样本的层不参与排序。
3. **重排系统**：只有存在至少一个可估计单元的系统进入排名；其余标"不可评估"并说明原因。

**单一机制**：分层测量。不叠加新模块、新门控、新损失。评审提出的风险只转化为**下一个判别性实验**，不转化为方法组件。


## 指标对齐（硬约束，优先于任何叙事）

主表用原始基线协议与原始指标：`leak_rate`（entity 子串）、retain `rouge_mean`、`truncated_rate`、`answer_reached_rate`；配置固定为 40 forget / 20 retain / 四策略 / DefaultCoT@512、其余@256 / seed 1001 / greedy / fp32。

基线口径下的三比例结果（`b18_scenario_panels.json` → `primary`）：

| 比例 | 遗忘前 leak | 遗忘后 leak | 干净单元（前→后，n） | 遗忘后坍塌 | retain ROUGE |
|---|---|---|---|---|---|
| 5% | 0.869 | 0.000 | 0.608→0.000（51→26） | 80.6% | 0.546→0.425 |
| 10% | 0.888 | 0.000 | 0.575→0.000（40→29） | 79.4% | 0.546→0.433 |

新增指标（`clean_cell_leak` / `coverage` / `censoring_interval` / `state_mix`）只作为创新点并列追加：它们补的是"泄漏归零同时伴随 79–81% 坍塌与 retain 效用下降"这一基线报不出的信息，**不替代**基线指标。任何后续表格若只报新指标而不报 `leak_rate`，视为违规。

## 核心主张（Claims）

| Claim | 最小可信证据 | 当前状态 | 依据 |
|---|---|---|---|
| C1 泄漏率被生成状态混淆 | 同模型同问题不同预算 ≥0.5 摆动 + 状态占比解释 + 跨系统状态差异 | **已支持** | 0.975 摆动；5 系统 collapse 0–0.96 |
| C2 分层估计改变系统排序 | ≥2 系统排序改变 | **部分支持（仅"不可评估"口径）** | 4 个系统离开可排序集合；但干净单元泄漏全 0，无法演示幅度反转 |
| C3 LLM judge 不能替代分层 | judge 与 entity-match 在坍塌/回显样本上同向失效 | **部分支持** | agreement=1.0（30/32），只在长度极端的样本上失效 |
| C4 删失区间有判别力 | 中间泄漏率区间出现非零宽度 | **被反驳（本批）** | width=0；且本批无预注册决策阈值 |

## 评审风险（已取得回执，按严重度排序）

| 风险 | 严重度 | 本轮状态 | 最便宜的判别实验 |
|---|---|---|---|
| 状态分层的逐样本表不存在 | high | **已部分解决**：5 系统 × 4 策略逐样本表可生成 | B10 导出 CSV |
| 状态标签依赖检测器，缺匹配跨度/阈值 | high | 未解决 | B10 加 `match_span` + 阈值记录 |
| 新颖性逼近 `arXiv:2506.00688`，可能被判为实例化 | high | **未解决（WEAKENED）** | B11 逐行 delta 映射 |
| 跨系统可比性无数据 | high | **已解决（弱化版）** | 见 `cross_system.json` |
| 汇总自相矛盾（judge 分母 / 区间 / gamma / retain 门） | medium | **已更正** | 见 IDEA_REPORT §11 |

**评审身份**：Hy4（API `model=hy4`；自述 "OpenAI ChatGPT，版本未暴露"，两者不一致已原样记录）。回执：`idea-stage/cross_review.json`，trace `.aris/traces/20261124T133949Z-hy4-external-critical-review.json`（`sha256=02a69e25…7501`）。凡涉及"已被 2506.00688 隐含"的判断，未经 B11 不得反驳。

## 下一步（按评审给出的 cheapest discriminating test 排序）

1. B10：导出逐样本状态表 + 匹配跨度（CPU）。
2. B11：对 2506.00688 做逐行差异映射，决定 I01 是"新贡献"还是"实例化"（CPU）。
3. B12：paired no-echo × budget × forced-reachability（≤2 GPU-h），预注册判据见 `EXPERIMENT_PLAN.md`。
4. B13/B14：多 seed 重排 + ROUGE 主导的 retain 双门。
