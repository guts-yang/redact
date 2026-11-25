# EXPERIMENT TRACKER — Run `lrm-loop-20261124`（v4）

| Block | 开始 (UTC) | 结束 | GPU | 耗时 | 结果 | Signal | 产物 |
|---|---|---|---|---|---|---|---|
| B0 `loop24` | 2026-11-24 | 完成 | H20 | 40.8 s | 4 策略 `leak_rate=1.0`；DefaultCoT truncated=1.0/reached=0.0；短思考 reached=1.0 | POSITIVE (C1) | `eval_loop24.{jsonl,summary.json}` |
| B1 `loop24_budget` | 2026-11-24 | 完成 | H20 | ~1 min | 四预算均 1.0，truncated 1.0→0.75；区间 [1.0,1.0] width=0 | NEGATIVE (C4) | `eval_loop24_budget.*` |
| B2 `judge_loop24` | 2026-11-24 | 完成 | — | ~30 s API | 30/32 成功，2 失败，agreement=1.0（已判定分母），llm_leak=1.0 | WEAK NEGATIVE (C3) | `judge_loop24.*` |
| B3 `loop24_cot` | 2026-11-24 | 完成 | H20 | ~3 min | 8/8 闭合 `</think>`，`channel_gap=0.0` | NEGATIVE (I03) | `eval_loop24_cot.*` |
| B7 代码诊断 | 2026-11-24 | 完成 | — | CPU | 7/7 缺陷文件级复验 | POSITIVE (I10) | `code_diagnosis.json` |
| B4a 跨系统重算 | 2026-11-24 | 完成 | — | CPU ~3 s | 5 系统 raw 0.000–0.169 vs collapse 0.00–0.96；干净单元泄漏全 0；4 系统离开可排序集合 | POSITIVE (C1/C2弱) | `cross_system.json` |
| B5a 匹配预算 | 2026-11-24 | 完成 | — | CPU | 共享问题集 @256：gamma_matched=0.10 > gamma_unmatched=0.05 | NEGATIVE（方向相反） | `cross_system.matched_budget` |
| B9 跨家族评审 | 2026-11-24 | 完成 | — | API | Hy4 `PROCEED_WITH_CAUTION` 6.0；I01/I04 新颖性 `WEAKENED`；17 条证据指控 | MIXED | `cross_review.json` + `.aris/traces/…` |
| B10 逐样本状态表 | 2026-11-24 | 完成 | — | CPU | 1200 行 CSV，含状态标签与字符跨度 | DONE | `audit/state_table.csv` |
| B14 ROUGE 主导效用门 | 2026-11-24 | 完成 | — | CPU | 5/5 通过可达门、5/5 失败于 ROUGE 门 | POSITIVE | `b14_utility_gate.json` |
| B8 阈值敏感性 | 2026-11-24 | 完成 | — | CPU | 干净单元泄漏恒为 0；不可评估集合恒为 {target, r2mu05} | STABLE | `b08_threshold_sensitivity.json` |
| B12 no-echo 审计 | 2026-11-24 | 完成 | H20 | ~8 min | DefaultCoT 1.000 / SummaryThink 0.500 / LessThink 0.438 / ZeroThink 0.375；γ@512=0.625；预算无影响 | POSITIVE（并触发 split 更正） | `b12_noecho_audit.json`, `eval_b12_*` |
| B11 delta 映射 | — | — | — | — | — | TODO | — |
| B13 多 seed 排序 | — | — | — | — | 阻塞：需先完成 B16 | TODO | — |
| B18 三比例面板 | 2026-11-24 | 完成 | H20 | ~10 min | 1% +0.144（坍塌降） / 5% −1.000（坍塌 100%，clean_n=0） / 10% −1.000（坍塌 87.5%，clean_n=2） | POSITIVE（符号由状态决定） | `b18_scenario_panels.json` |
| B16 各系统自有比例重跑 | 2026-11-24 | 完成（1%/5%/10%） | H20 | 见 B18 | r2mu05/r2mu10 遗忘后不可评估 | MIXED | `eval_b19_*` |
| B17 掌握度前置门 | — | — | — | — | — | TODO | — |
| B15 证据门禁 | — | — | — | — | ARIS helper 仍缺失 | BLOCKED | `gate_status.json` |

## 评分迭代轨迹（rubric `cff0a224b94802cc`，达标线 8.0）

| 阶段 | 达标 | 均分 | 最低 | 变化来源 |
|---|---|---|---|---|
| v1 提案态 | 0/10 | 7.0 | 7.0 | 无实测锚点、无 pilot、未应用查新 |
| v2 证据态 | 10/10 | 8.7 | 8.0 | 新增 GPU/judge/代码复验证据 |
| v3 外部查新 + 评审回执 | 9/10 | 8.5 | 7.0 | **降分**：I09 被 ShadowBench 淘汰、I05 降为应用级；I01/I04 新颖性未获签 |
| v4 跨系统 + B12 no-echo | 9/10 | 8.6 | 7.0 | 仅 I02 因匹配预算可算而 8→9；B12 提供有效 split 证据但 rubric 维度已满 |

> ⚠️ **第 2 轮更正**：`cross_system.json` 的 forget01 数字只作状态演示（checkpoint 未学过该 split）；有效测量在 `b12_noecho_audit.json`。C2 已被改判为 `NOT_SUPPORTED_ON_A_VALID_SPLIT`。

期间四次提高严格度（均未降低标准）：D3 需判别性对照、D4 需外部查新 + 回执、零覆盖单元改标 `unevaluable`、修正概率越界/混样/结构性零值/坍塌样本误述。

## 已知限制（不隐藏）

- 单 seed（1001）为主；B13 未做，C2 只能停在"不可评估"口径。
- `r2mu05`/`ours-inv`/`ctrl-single` 的"零泄漏"与效用门冲突，尚未用 ROUGE 主导门复算（B14）。
- I03 与外部工作重叠；I05 技术属通用实践；I09 已淘汰。
- 评审 17 条中 5 条需新测量（B10/B12），2 条为口径降级，10 条已修正。
- 门禁 `BLOCKED`：评审阻塞已解除，helper 阻塞未解除。

## 复算方式

```bash
cd /usr/local/workspace/redact
python3 idea-stage/tools/auto_loop.py --stage all
python3 idea-stage/tools/cross_review.py     # 仅当需要新的评审回执
```
