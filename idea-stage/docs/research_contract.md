# Research Contract — W1 → W1.5 Handoff

**Run ID**: `lrm-loop-20261124` · **Date**: 2026-11-24 · **Status**: ACTIVE，证据门禁 BLOCKED

## Selected Idea

**I01 — 生成状态分层泄漏审计**（`idea-stage/IDEA_REPORT.md` §5.1，v2 分数 9.0，排名 1）

并列高分方案：I04 / I05 / I06 / I07 / I08 / I10（均 9.0），作为同一诊断框架的组成实验，不是独立方法。

**选择理由**：唯一直接攻击已量化验证的 Problem Anchor（同模型泄漏率摆动 0.975）的方案；本轮已有判别性实测；不需要新增训练、新增模块或额外数据；阴性结果同样改变领域评测实践。

## Frozen Problem Anchor

`entity_leak` 是 response 对 gold 关键短语的子串匹配，被 answer-reachability / censoring / reasoning collapse / question echo 四种生成状态污染；低泄漏率不构成遗忘证据。

## Core Claims（实现与 result-to-claim 的唯一来源）

| Claim | 最小可信证据 | 当前 |
|---|---|---|
| C1 泄漏率被生成状态混淆 | 同模型同问题不同预算 ≥0.5 摆动 + 状态占比解释 | 已支持 |
| C2 分层估计改变系统排序 | ≥2 系统排序变化（多 seed/多系统） | 未验证 |
| C3 LLM judge 不能替代分层 | judge 与 entity-match 在坍塌/回显样本同向失效 | 部分支持 |
| C4 删失区间有判别力 | 中间区间非零宽度且跨阈值 | 被反驳（本批 width=0） |

## Minimum Convincing Evidence

1. B4：5 系统 × 3 seed × 4 策略的分层估计，检验 C2。
2. B8：collapsed 阈值敏感性消融，排除分类规则主观性。
3. B5/B6：补齐 I02/I09 的预注册指标，不再以描述性计数替代判别性测量。
4. B9：跨家族 reviewer 回执 + `idea_discovery_gate.py` 退出 0；在此之前不得宣称 PASS。

## Non-goals

不提出新遗忘训练算法；不改目标权重；不把 Hy4 judge 当独立审查；不放宽 rubric `cff0a224b94802cc` 或 8 分达标线。

## Next-step Pointer

- 实现入口：`/experiment-bridge` 按本契约实现 B4/B5/B6/B8。
- 结果判读：`/result-to-claim` 以 C1–C4 为 claims 来源；`/ablation-planner` 读取 B8。
- 复算：`python3 /usr/local/workspace/redact/idea-stage/auto_loop.py --stage all`
- 会话恢复：本文件是 ACTIVE idea 的 #2 恢复点；完整证据在 `idea-stage/IDEA_REPORT.md` 与同目录 JSON。
