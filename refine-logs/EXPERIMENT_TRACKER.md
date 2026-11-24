# EXPERIMENT TRACKER — Run `lrm-loop-20261124`

| Block | 开始 (UTC) | 结束 | GPU | 耗时 | 结果 | Signal | 产物 |
|---|---|---|---|---|---|---|---|
| B0 `loop24` | 2026-11-24 | 完成 | H20 | 40.8 s | 4 策略 `leak_rate=1.0`；DefaultCoT truncated=1.0/reached=0.0；短思考 truncated=0/reached=1.0；`gamma=0` | POSITIVE (C1) | `eval_loop24.{jsonl,summary.json}` |
| B1 `loop24_budget` | 2026-11-24 | 完成 | H20 | ~1 min | 该 tag 四预算均 1.0，truncated 1.0→0.75；区间 [1.0,1.0] width=0 | NEGATIVE (C4) | `eval_loop24_budget.*` |
| B2 `judge_loop24` | 2026-11-24 | 完成 | — | ~30 s API | 30/32 成功，2 失败，agreement=1.0，llm_leak=1.0 | WEAK NEGATIVE (C3) | `judge_loop24.{jsonl,summary.json}` |
| B3 `loop24_cot` | 2026-11-24 | 完成 | H20 | ~3 min | 8/8 闭合 `</think>`，`channel_gap=0.0`，answer_leak=1.0 | NEGATIVE (I03) | `eval_loop24_cot.*`, `judge_loop24_cot.*` |
| B7 代码诊断 | 2026-11-24 | 完成 | — | CPU | 7/7 缺陷文件级复验 | POSITIVE (I10) | `idea-stage/code_diagnosis.json` |
| B4 多系统多 seed | — | — | — | — | — | TODO | — |
| B5 跨策略匹配预算 | — | — | — | — | — | TODO | — |
| B6 改写探针配对 | — | — | — | — | — | TODO | — |
| B8 阈值敏感性 | — | — | — | — | — | TODO | — |
| B9 证据门禁 | — | — | — | — | `REVIEW_UNAVAILABLE` / helper 缺失 | BLOCKED | `idea-stage/gate_status.json` |

## 评分迭代轨迹（rubric `cff0a224b94802cc`，达标线 8.0）

| 阶段 | 达标 | 均分 | 最低 | 变化来源 |
|---|---|---|---|---|
| v1 提案态 | 0/10 | 7.0 | 7.0 | 无实测锚点、无 pilot、未应用查新 |
| v2 证据态 | 10/10 | 8.7 | 8.0 | 仅新增 GPU/judge/代码复验证据 |

期间三次提高严格度（均未降低标准）：D3 改为需判别性对照、D4 因外部查新不可封顶 1 分、修正概率越界/混样/结构性零值。最高分 9.0 是环境上限，不是满分。

## 已知限制（不隐藏）

- 单 seed（1001）、单 checkpoint（target）为主；B4 未完成。
- I02 预注册的 `gamma_budget`、I09 的 `recovery` 未测，两者 D3=1。
- I03 本地清单有 9 篇近邻，D4=0。
- 外部多源查新与跨家族评审均不可用；终检 `BLOCKED`。

## 复算方式

```bash
cd /usr/local/workspace/redact && python3 idea-stage/auto_loop.py --stage all
```

新产物落盘后重跑即可自动更新 `pilot_results.json` 与 `idea_scores.json`；rubric 与达标线不变。
