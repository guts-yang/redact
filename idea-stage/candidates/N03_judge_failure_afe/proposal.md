# N03 — Judge 失败感知的 AFE 与保守标签训练（Judge-Failure-Aware AFE）

**分数 9/10**（D1=2 D2=2 D3=2 D4=1 D5=2）· 裁决 WEAKENED · 证据 `evidence/afe_cfe_results.json`、`evidence/i05_baseline_judge.json`

## 1. 问题定义

AFE = 1 − LEAK 的分母是"成功判定的样本"。基线运行里 judge 解析失败率为 **7.5%–15%**（6/80、10/80、6/80、12/80），超过 I05 预注册的 5% 阈值。失败样本不是随机缺失：它们集中在长文本/退化文本上，正是泄漏最可能发生的地方。

三种分母口径下 AFE 的最大差值为 **0.167**（`evidence/afe_cfe_results.json` 的 policy_spread）。

## 2. 与现有工作的差异

UWC 等提出"更稳健的评估指标"，但没有定义**失败样本的策略**（丢弃/重试/最坏情况），也没有把这种不确定性传回训练目标。

## 3. 技术方案要点（含遗忘实现方式）

### 3.1 训练：最坏情况标签

```
label_LEAK(sample) = max over policies { drop, retry, worst_case }
L_forget = margin_loss(entity | positive_set)      # 只在最坏情况判为 LEAK 的样本上压
L_keep   = KL(retain) + λ * keep_margin(entity)    # 在最坏情况判为非 LEAK 的样本上保住
```

- **遗忘作用点**：实体 token 的 logit margin，但**样本选择由最坏情况标签决定**，避免"judge 失败 → 误把该压的样本放过"。
- **副作用控制**：非 LEAK 样本上加 keep margin，防止把"没泄漏"也一起压成拒绝。

### 3.2 评测：三口径并列

每张表给 `AFE_drop / AFE_retry / AFE_worst`，并附失败率。

## 4. 交叉验证方式

- Hy4 同一 judge、同一 prompt，重复 20 条，检查失败是否可复现（当前 `judge_cache_*.jsonl` 已存逐条状态）。
- 机械：`tools/fixes.py i05` 可重算三口径。

## 5. 实验设计

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | 现有单口径 | AFE 0.957（分母 70） |
| T1 | 保守标签训练 | 最坏情况 AFE ≥0.9 且失败率不敏感（±0.03 内） |
| T2 | 消融：去掉 keep margin | 拒绝率 >0.5，效用掉 >0.1 |

## 6. 评分依据

| 维度 | 分 | 理由 |
|---|---|---|
| D1 | 2 | 失败率 7.5–15% 实测 |
| D2 | 2 | 预注册 spread>0.05，三种结局都有解释 |
| D3 | 2 | spread 0.167 已跨阈 |
| D4 | 1 | 与 UWC / judge 偏差工作相邻，未获 SUPPORTED |
| D5 | 2 | 若三方差 <0.01，说明现有 AFE 稳健，同样是结论 |

## 7. 复现入口

```bash
python3 idea-stage/tools/fixes.py i05
python3 idea-stage/tools/eval_afe_cfe.py --n 20 --workers 8
```
