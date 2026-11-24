# FINAL PROPOSAL — 生成状态分层的 LRM 遗忘评测

**Run ID**: `lrm-loop-20261124` · **Active idea**: I01（并列 I04/I05/I06/I07/I08/I10）· **Date**: 2026-11-24

## Problem Anchor（冻结，不得漂移）

> 在长 CoT 推理模型上，`entity_leak`（response 是否包含 gold 关键短语）同时被答案可达性、生成坍塌与问题回显三种生成状态污染；因此低泄漏率不能作为"已遗忘"的证据，跨解码策略/跨系统的比较也失去可比性。

锚点证据：同一权重、同一 seed、同一问题，仅改变生成预算，`leak_rate` 从 0.025 → 1.0（摆动 0.975）；历史低分样本是退化成 "AI assistant..." 的坍塌文本，本轮高分样本只是复述了问题里的人名。Hy4 LLM judge 与字符串指标 agreement=1.0，不能独立纠正该混淆。

**非目标**：不提出新的遗忘训练算法；不改动目标模型权重；不声称解决 survey Q25 的完整可证明性；不把 LLM judge 当作独立审查。

## Method Thesis（一句话）

把每条生成先判定为 `answer-reached / censored / collapsed / question-echo` 四种状态之一，再在可判定层内报告条件泄漏率、coverage 与删失区间，用分层估计替代单点 `leak_rate`。

## 方法（不改核心假设，只加测量步骤）

1. 从原始 response 与 prompt 恢复状态：`answer_reached`（`</think>` 闭合）、`censored`（token 预算耗尽）、`collapsed`（词型重复率过高）、`question-echo`（gold 关键短语已出现在问题中且被回显）。
2. 对每层计算条件泄漏率与 coverage；对删失层给 `[optimistic, pessimistic]` 区间（上界 clamp 到 1.0）；不可判定层单列，不记 0 分。
3. 用同一分层协议重排 target / r2mu01 / r2mu05 / ours-inv / ctrl-single，并复算跨策略 `gamma_controlled`。

**单一机制**：分层测量。不叠加新模块、新门控或新损失；评审提出的风险只转化为**下一个判别性实验**，不转化为方法组件。

## 核心主张（Claims）

| Claim | 最小可信证据 | 本轮状态 |
|---|---|---|
| C1 泄漏率被生成状态混淆 | 同模型同问题不同预算下 ≥0.5 的摆动，且有状态占比解释 | **已支持**（0.975，echo 1.0 vs 0.025） |
| C2 分层估计能改变系统排序 | 至少 2 个系统在分层估计下的排序与原始 `leak_rate` 不同 | 未验证（需多系统多 seed） |
| C3 LLM judge 不能替代分层 | judge 与 entity-match 在坍塌/回显样本上同向失效 | **部分支持**（agreement=1.0，历史失败率 24/27 显示长度依赖） |
| C4 删失区间有判别力 | 中间泄漏率区间出现非零宽度且跨越决策阈值 | **被反驳**（本批 width=0，需中间区间数据） |

## Reviewer 风险（无跨家族评审回执，记录不消除）

- 状态分类规则（尤其 collapsed 的词型重复阈值 0.35）本身可能引入主观性 → 下一实验：阈值敏感性消融。
- 结论可能只对 `sangyon/LRM-target` 这一个 checkpoint 成立 → 下一实验：多系统、多 seed。
- "阴性结果"（I03 channel_gap=0、I04 width=0、I08 gamma=0）可能被误读为方法无效 → 保留并显式标注为信息性阴性。

**评审状态**：`REVIEW_UNAVAILABLE`。本环境无 `codex/gemini/claude` 后端，未获得带身份的跨家族 verdict；Hy4 仅作 metric。不得把本提案表述为 review-passed。
