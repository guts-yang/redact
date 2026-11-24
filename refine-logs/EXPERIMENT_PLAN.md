# EXPERIMENT PLAN — Run `lrm-loop-20261124`（v4，含评审驱动的 B10–B15）

**Claim-driven**。每个 block 只服务一个 claim；判定标准在跑之前写死；结果无论正负都记录。
资源：1× H20 96GB；venv `/jizhicfs/chenyliao/ops/venvs/r-tofu-ga`；fp32；`do_sample=False`；seed 1001；模型 `/jizhicfs/chenyliao/models/sangyon/LRM-target`（权重 sha256 见 `idea-stage/run_manifest.json`）。总 GPU 预算 ≤8h，单 pilot ≤2h。

## 运行顺序与预算

| Block | 对应 Claim / Idea | 配置 | 判定标准（预注册） | 预算 | 状态 |
|---|---|---|---|---|---|
| B0 | C1 / I01 | 4 策略 × (8 forget + 4 retain)，192 tok | 状态占比能解释 ≥0.5 的泄漏率摆动 | ~1 min | **DONE（POSITIVE）** |
| B1 | C1 / I04 | 8 forget × 预算 64/128/256/512 | 区间宽度 >0 | ~1 min | **DONE（NEGATIVE，width=0）** |
| B2 | C3 / I05 | Hy4 judge 对 B0 全样本 | agreement <0.9 或失败率 >0.05 | ~1 min API | **DONE（WEAK NEGATIVE，30/32）** |
| B3 | P1 / I03 | 8 forget × 1024 tok，DefaultCoT | ≥4 条闭合 `</think>`；`channel_gap>0.05` 为阳性 | ~3 min | **DONE（NEGATIVE，8/8 可测）** |
| B7 | I10 | 检测器阈值/后端/数据版本契约复验 | ≥3 项独立可复验缺陷 | CPU | **DONE（POSITIVE，7 项）** |
| B4a | C2 / I01+I07 | 5 系统 × 4 策略历史生成重算（单 seed） | ≥2 系统离开可排序集合 | CPU | **DONE（POSITIVE，4 系统）** |
| B5a | I02 | 共享问题子集 @256 的跨策略 gamma | `gamma_matched` 与 `gamma_unmatched` 差 >0.05 | CPU | **DONE（NEGATIVE：0.10 vs 0.05，差距变大）** |
| B10 | 评审风险 1/2 | 5 系统 × 4 策略逐样本状态表导出（CSV：state / leak / match_span / echo / prompt_hash / gold_phrase） | 每个系统每策略 ≥1 行可复核；匹配跨度可定位到字符区间 | CPU | **DONE（1200 行，含跨度）** |
| B11 | 评审风险 3（新颖性） | 对 `arXiv:2506.00688` 做逐行 delta 映射（它的跨格式矩阵 vs 本方法的状态分层 + 删失 + coverage） | 若映射显示其已隐含 ≥3/5 要素 → 主动把 I01 降级为"实例化"，D4 保持 1 | CPU | TODO |
| B12 | C1 的决定性检验 | paired no-echo 探针 × 4 策略 × {192,512,1024} | (a) 每个匹配格 ≥8 干净样本；(b) 干净标签在预算间不变而 raw 改变；(c) 至少一条 no-echo 样本生成出 gold 短语 | ≤2 GPU-h | **DONE（8 格，约 8 分钟 GPU）**：DefaultCoT 16/16、ZeroThink 6/16、γ@512=0.625；判据 (a) 在 @192 失败、(b) 空成立、(c) 通过 |
| B13 | C2（排序反转） | seed {1001,1002,1003} × 各系统**自己的** forget split | ≥2 系统排序反转（不是"不可评估"） | ≤3 GPU-h | TODO（依赖 B16 先修 split） |
| B14 | I07（效用门） | 以 ROUGE-L recall 为主门（≥0.5）重算 5 系统 retain | `ours-inv`/`ctrl-single` 若 ROUGE <0.5 则确认"完美遗忘"= 废掉答案 | CPU | **DONE（5/5 可达门通过、5/5 ROUGE 门失败）** |
| B8 | C1 稳健性 | collapsed 阈值 {0.25,0.35,0.45} 敏感性 | 结论在 3 个阈值下一致 | CPU | **DONE（实质结论稳定；0.45 时 1 个单元跨过 8 样本线）** |
| B16 | 修场景 | 三个比例场景各跑一次前后对照（1% 用原有 40 题，5%/10% 用 no-echo 探针） | 同比例内比较；必须报 clean_n | ≤1 GPU-h | **DONE**：1% 0.025→0.169；5% 1.000→0.000（clean_n 0）；10% 1.000→0.000（clean_n 2） |
| B17 | 评测前置门 | 把"checkpoint 是否掌握 forget 集"做成门（NLL + 生成复现率双阈值） | shipped target 在 forget01 上 NLL 5.47 → 必须判"不可用于遗忘评测" | CPU | TODO |
| B15 | 门禁 | 安装 ARIS helper（`run_state.py`/`idea_discovery_gate.py`/`reviewer_route.py`）并跑终检 | 终检退出 0 | 依赖外部 | **BLOCKED（helper 缺失）** |
| B9 | 评审 | 跨家族 reviewer 回执 | 取得带身份正面回执 | API | **DONE（Hy4，PROCEED_WITH_CAUTION）** |


## 指标对齐约束（用户 2026-11-24 指定，硬性）

- 训练与评估一律用原始基线协议：`n_forget=40`、`n_retain=20`、四策略、DefaultCoT@512 / 其余@256、`do_sample=False`、`seed=1001`、`fp32`。
- 主表**先报**基线指标：`leak_rate`、`answer_leak_rate`、`rouge_mean`（retain）、`truncated_rate`、`answer_reached_rate`。
- 新增指标（`clean_cell_leak` / `coverage` / `censoring_interval` / `state_mix`）只能并列追加，且必须写明"基线报不出什么"；**不得替换或单报新指标**。
- 任何偏离基线的 run（no-echo 探针、非默认预算、预算网格、小样本 pilot）必须在 `metric_parity.json` 里被判为 `DEVIATION` 并在表格里标注，不得与基线 run 混排。
- 核查入口：`python3 idea-stage/metric_parity.py`（37 runs：9 ALIGNED / 28 DEVIATION，全部有标注）。

| B20 | 元数据 | 让 `decode_eval.py` 在 `summary.json` 记录 `--forget-file` 路径与 sha256 | 任取一个 run，能从 summary 反推出探针集 | CPU | TODO |

## 已完成 block 的真实产物

```
/jizhicfs/chenyliao/models/sangyon/LRM-target/*          (权重；sha256 见 run_manifest.json)
/usr/local/workspace/R-TOFU/workspace/results/eval_{target,r2mu01,r2mu05,ours-inv,ctrl-single}.jsonl
/usr/local/workspace/R-TOFU/workspace/results/*_budget.jsonl        (DefaultCoT 64/128/256/512)
/usr/local/workspace/R-TOFU/workspace/results/eval_loop24{,_budget,_cot}.jsonl
/usr/local/workspace/R-TOFU/workspace/results/judge_loop24{,_cot}.jsonl
/usr/local/workspace/redact/idea-stage/{run_manifest,cross_system,problem_validation,code_diagnosis,
    novelty_scan,external_novelty,pilot_results,cross_review,idea_scores,gate_status}.json
/usr/local/workspace/redact/.aris/traces/20261124T133949Z-hy4-external-critical-review.json
```

## 最小复现命令

```bash
cd /usr/local/workspace/redact
python3 idea-stage/auto_loop.py --stage all      # 无 GPU，含跨系统重算与四阶段评分
python3 idea-stage/cross_review.py               # 跨家族评审；写 trace + receipt

# GPU 部分（仅在需要新生成时）
cd /usr/local/workspace/R-TOFU/workspace
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/jizhicfs/chenyliao/ops/venvs/r-tofu-ga/bin/python
$PY code/decode_eval.py --model target --tag loop24 --n-forget 8 --n-retain 4 \
    --strategies DefaultCoT,LessThink,ZeroThink,SummaryThink --max-new-tokens 192 --batch-size 4
$PY code/llm_judge.py --tags loop24 --limit-per-group 8 --concurrency 4
```

## 约束

- 不混样：预算曲线按 tag/system 分开统计；跨策略匹配必须**同时**共享问题子集与预算（B5a 已按此实现）。
- 概率合法：删失上界 clamp 到 1.0。
- 不用结构性零值当证据：CoT 通道只能用闭合 `</think>` 的样本（B3）。
- 零覆盖/零样本的单元**不得**报 0：标 `unevaluable`（B4a 已按此实现，4 个系统因此离开排序）。
- LLM judge 是 metric，不是 reviewer。评审回执只能由 `cross_review.py` 的独立调用产生。
- 评分 rubric `cff0a224b94802cc` 与 8 分线冻结；新证据只允许通过 `auto_loop.py` 重算分数。
- B11 的结论可以**降低** I01 的定位（新贡献 → 实例化），不允许反过来抬高。
