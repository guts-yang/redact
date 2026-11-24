# EXPERIMENT PLAN — Run `lrm-loop-20261124`

**Claim-driven**。每个 block 只服务一个 claim；判定标准在跑之前写死；结果无论正负都记录。
资源：1× H20 96GB；venv `/jizhicfs/chenyliao/ops/venvs/r-tofu-ga`；fp32；`do_sample=False`；seed 1001；模型 `/jizhicfs/chenyliao/models/sangyon/LRM-target`。总 GPU 预算 ≤8h，单 pilot ≤2h。

## 运行顺序与预算

| Block | 对应 Claim / Idea | 配置 | 判定标准（预注册） | 预算 | 状态 |
|---|---|---|---|---|---|
| B0 | C1 / I01 | 4 策略 × (8 forget + 4 retain)，192 tok | 状态占比能解释 ≥0.5 的泄漏率摆动 | ~1 min | **DONE** |
| B1 | C1 / I04 | 8 forget × 预算 64/128/256/512 | 区间宽度 >0 且跨决策阈值 | ~1 min | **DONE（NEGATIVE）** |
| B2 | C3 / I05 | Hy4 judge 对 B0 全样本 | agreement <0.9 或失败率 >0.05 视为需分层 | ~1 min API | **DONE（WEAK NEGATIVE）** |
| B3 | P1 / I03 | 8 forget × 1024 tok，DefaultCoT | 至少 4 条闭合 `</think>`；`channel_gap>0.05` 为阳性 | ~3 min | **DONE（NEGATIVE，8/8 可测）** |
| B4 | C2 / I01+I08 | 5 系统 × 3 seed × 4 策略，分层估计 | ≥2 系统排序改变 | ~40 min | TODO |
| B5 | I02 | 4 策略 × 4 匹配预算，跨策略 `gamma_budget` | `gamma_budget` 与 `gamma_unmatched` 差异 >0.05 | ~15 min | TODO |
| B6 | I09 | 400 条改写探针 × GPU 生成配对 | 零名字泄漏下 recovery >0 | ~20 min | TODO |
| B7 | I10 | 检测器阈值/后端/数据版本契约复验 | ≥3 项独立可复验缺陷 | CPU | **DONE（POSITIVE，7 项）** |
| B8 | C2 稳健性 | collapsed 阈值 {0.25,0.35,0.45} 敏感性 | 结论在 3 个阈值下一致 | CPU | TODO |
| B9 | 门禁 | 跨家族 reviewer + ARIS helper | 取得正面回执且 `idea_discovery_gate.py` 退出 0 | 依赖外部 | **BLOCKED** |

## 已完成 block 的真实产物

```
/usr/local/workspace/R-TOFU/workspace/results/eval_loop24.jsonl             (48 rows)
/usr/local/workspace/R-TOFU/workspace/results/eval_loop24_budget.jsonl
/usr/local/workspace/R-TOFU/workspace/results/eval_loop24_cot.jsonl         (8 rows, 8/8 closed)
/usr/local/workspace/R-TOFU/workspace/results/judge_loop24.jsonl             (32 rows)
/usr/local/workspace/R-TOFU/workspace/results/judge_loop24_cot.jsonl         (8 rows)
/usr/local/workspace/redact/idea-stage/{problem_validation,code_diagnosis,novelty_scan,pilot_results,idea_scores}.json
```

## 最小复现命令

```bash
cd /usr/local/workspace/R-TOFU/workspace
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/jizhicfs/chenyliao/ops/venvs/r-tofu-ga/bin/python
$PY code/decode_eval.py --model target --tag loop24 --n-forget 8 --n-retain 4 \
    --strategies DefaultCoT,LessThink,ZeroThink,SummaryThink --max-new-tokens 192 --batch-size 4
$PY code/llm_judge.py --tags loop24 --limit-per-group 8 --concurrency 4
cd /usr/local/workspace/redact && python3 idea-stage/auto_loop.py --stage all
```

## 约束

- 不混样：预算曲线必须按 tag/system 分开统计（B1 已修正早期混样错误）。
- 概率合法：删失上界 clamp 到 1.0（B1 已修正 1.25 的越界值）。
- 不用结构性零值当证据：CoT 通道只能用闭合 `</think>` 的样本（B3）。
- LLM judge 是 metric，不是 independent reviewer；不得用它生成 review receipt。
- 评分 rubric `cff0a224b94802cc` 与 8 分线冻结；新证据只允许通过 `auto_loop.py` 重算分数。
