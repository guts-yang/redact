# Research Contract — W1 → W1.5 Handoff

**Run ID**: `lrm-loop-20261124` · **Date**: 2026-11-24 · **Status**: ACTIVE；评审回执已取得，证据门禁仍 `BLOCKED`（ARIS helper 缺失）

## Selected Idea

**I01 — 生成状态分层泄漏审计**（`idea-stage/IDEA_REPORT.md` §5.1，v4 分数 9.0，排名 1）

并列高分：I02 / I04 / I06 / I07 / I08 / I10（均 9.0），是同一诊断框架的组成实验，不是独立方法。I09 已被外部查新淘汰（ShadowBench，NeurIPS 2026）。

**选择理由**：唯一直接攻击已量化验证的 Problem Anchor 的方案；本轮已有跨系统判别性实测（4 个"完美遗忘"系统离开可排序集合）；不需要新增训练、模块或数据；阴性结果同样改变领域评测实践。

## Frozen Problem Anchor

`entity_leak` 是 response 对 gold 关键短语的子串匹配，被 `answered / censored / collapsed / question_echo / no_answer_channel` 五种生成状态污染；低泄漏率不构成遗忘证据，无可用干净单元的系统必须标 `unevaluable` 而不是 0。

## Core Claims（实现与 result-to-claim 的唯一来源）

| Claim | 最小可信证据 | 当前 |
|---|---|---|
| C0 场景比例 | forget01/05/10 = 1%/5%/10% 三种遗忘比例；跨比例不可横向比较 | **条件已确认（非缺陷）**；上一版"未学过 forget01"的判断已撤回 |
| C1 泄漏率被生成状态混淆 | 同模型同问题不同预算 ≥0.5 摆动 + 状态占比解释 | **已支持（范围收窄）**：状态改变读数，但 no-echo 区间内预算无影响 |
| C1b 场景内测量 | no-echo 探针上干净单元泄漏非零且随策略变化 | **已支持**：γ@512=0.625（forget10）；forget01 上干净单元恒为 0 |
| C2 分层改变可评估性 | ≥2 系统无法产生 ≥8 干净样本 | **支持（仅可评估性口径）**：r2mu05 / r2mu10 / target 多格不可评估 |
| C3 LLM judge 不能替代分层 | judge 与 entity-match 在坍塌/回显样本同向失效 | 部分支持（agreement=1.0，30/32） |
| C4 删失区间有判别力 | 中间区间非零宽度且跨预注册阈值 | 被反驳（width=0，且本批无阈值） |
| C5 前后符号由状态决定 | 两个比例场景的遗忘前后 Δ 符号相反，且坍塌率变化可预测符号 | **支持（新，B18）**：forget01 +0.144 / forget10 −1.000 |

## Minimum Convincing Evidence

1. **B10**：5 系统 × 4 策略逐样本状态表（含 `match_span`、`prompt_hash`）——评审的最高危反对点。
2. **B11**：对 `arXiv:2506.00688` 的逐行 delta 映射，决定 I01 是"新贡献"还是"实例化"（可降级，不可抬升）。
3. **B12**：paired no-echo × 预算 × 强制可达性，预注册判据见 `EXPERIMENT_PLAN.md`（≤2 GPU-h）。
4. **B13/B14**：多 seed 排序反转、ROUGE 主导的 retain 门。
5. **B15**：安装 ARIS helper 并跑终检；在此之前不得宣称 PASS。

## Non-goals

不提出新遗忘训练算法；不改目标权重；不把 Hy4 judge 当独立审查（评审走独立高层调用）；不放宽 rubric `cff0a224b94802cc` 或 8 分达标线；不把 `2506.00688` 的贡献说成我们自己的。

## Next-step Pointer

- 实现入口：`/experiment-bridge` 按本契约实现 B10–B14。
- 结果判读：`/result-to-claim` 以 C1–C4 为 claims 来源；`/ablation-planner` 读取 B8。
- 复算：`python3 /usr/local/workspace/redact/idea-stage/auto_loop.py --stage all`
- 复审：`python3 /usr/local/workspace/redact/idea-stage/cross_review.py`
- 会话恢复：本文件是 ACTIVE idea 的 #2 恢复点；完整证据在 `idea-stage/IDEA_REPORT.md` 与同目录 JSON。
