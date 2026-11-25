# idea-stage — 结构与接手说明

本目录只保留**当前用于候选训练复现的 8 个 idea**及其证据。已淘汰的方向在 `archive/`，历史报告保留在 `IDEA_REPORT.md`。

## 目录

```
idea-stage/
  README.md                 本文件（接手先读这里）
  IDEA_REPORT.md            历史主报告（含所有更正记录，v1–v4 + 第 13–17 节）
  candidates/               8 个入选 idea，每个一个 proposal.md（含遗忘算法实现）
  archive/                  已淘汰 idea 的归档说明（不删证据，只归档结论）
  tools/                    全部脚本（可一键复算）
  evidence/                 全部机器生成的证据（JSON/JSONL/CSV），由 tools 写入
  docs/                     研究契约 research_contract.md
```

## 8 个候选（按分数）

| ID | 方向 | 分数 | 一句话 |
|---|---|---|---|
| N08 | 检测器契约绑定复现 | 10 | 阈值/后端/数据版本绑定，否则不可复现 |
| N10 | 状态位移是遗忘的主产物 | 10 | 泄漏下降 79% 可由状态位移解释 |
| N02 | 删失区间强制上报 | 9 | 点估计在 95% 截断下无意义 |
| N03 | judge 失败率感知的 AFE | 9 | 分母口径差 0.167，失败率 7.5–15% |
| N07 | 效用-遗忘 Pareto 硬门 | 9 | retain 掉 0.112，必须设门 |
| N12 | 知识存在性前置门 | 9 | NLL 与复现两信号矛盾时不得评遗忘 |
| N05 | 泄漏片段定位 | 8 | 92.5% 的泄漏 span 不是"只报名字" |
| N06 | 跨策略状态配对重放 | 8 | 状态非不变率 0.300（未过 0.3 阈） |

淘汰：N01（STaR 的 MCS 已抢先）、N04（coverage-aware 已存在）、N09（K-Bench 六通道）、N11（REDaCT collapse-free）。详见 `archive/ROUND2_ELIMINATED.md`。

## 一键复算

```bash
cd /usr/local/workspace/redact
python3 idea-stage/tools/auto_loop.py --stage all        # 重算分数与门禁
python3 idea-stage/tools/round2_verify.py                # 严格 D3/D4 重算
python3 idea-stage/tools/metric_parity.py                # 基线对齐核查
python3 idea-stage/tools/idea_baseline_matrix.py rXX     # 异常检测新一轮
```

## 证据与口径（接手必看）

- **基线协议**：40 forget / 20 retain / 四策略 / DefaultCoT@512、其余@256 / greedy / seed 1001 / fp32。
- **主指标**：`leak_rate`（entity 子串）、retain `rouge_mean`、`truncated_rate`、`answer_reached_rate`；AFE/CFE 用仓库原有 prompt 与 parser（`scripts/judge_afe_rtofu.py`、`scripts/judge_cfe_rtofu.py`），只换 API 传输层。
- **新增指标**：`clean_cell_leak` / `coverage` / `censoring_interval` / `state_mix`，只并列追加，**不得替换基线指标**。
- **已知缺陷**：`decode_eval.py` 不记录 `--forget-file`（B20 未修）；`ours-inv`/`ctrl-single` adapter 挂在错误基座（E2 未修）；改写库是拒答改写（I09 不可复现）。
- **门禁**：`evidence_gate = BLOCKED`（缺 ARIS helper），评审回执已由 Hy4 提供，见 `evidence/cross_review.json`。
