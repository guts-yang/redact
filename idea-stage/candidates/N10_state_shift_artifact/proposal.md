# N10 — 状态位移优先的遗忘：把"坍塌"从结果里扣掉（Collapse-Accounted Unlearning）

**分数 10/10**（D1=2 D2=2 D3=2 D4=2 D5=2）· 裁决 SUPPORTED · 证据 `evidence/b18_scenario_panels.json`、`evidence/idea_loop_latest.json`

## 1. 问题定义

在 R-TOFU 上，遗忘前后 `leak_rate` 从 0.888 掉到 0.000，但同一批样本的**坍塌率从 0.0 升到 0.794、截断率从 0.08 升到 0.93**。也就是说"泄漏归零"里有多大比例来自"模型不再说话"，现有报告无法回答。

实测：状态位移可解释 **79.4%** 的 Δleak（10% 场景），5% 场景为 80.6%。

**要解决的是：让"遗忘"这个词只指知识层的变化，不指退化。**

## 2. 与现有工作的差异

| 工作 | 做法 | 缺口 |
|---|---|---|
| R-TOFU | 报 CFE/AFE，另附一个 collapse 列 | 不把 collapse 从 Δleak 里扣掉，也不给分解 |
| STaR | 关注跨解码一致性 | 沿解码轴，不沿生成状态轴 |
| 本文 | Δleak 分解为"状态位移分量 + 状态内分量" | 首次把分解做成训练目标 |

## 3. 技术方案要点（含遗忘实现方式）

### 3.1 遗忘损失：状态条件化的实体压制

对每个 forget 样本，按生成状态分派不同权重：

```
L_forget = Σ_i w(s_i) * max(0, logit_entity(a_i) - logit_entity(y_safe) + m)
w(s) = 1.0   if s == answered_clean
w(s) = 0.3   if s == censored
w(s) = 0.0   if s == collapsed        # 坍塌样本不施加遗忘梯度
```

- **遗忘作用位置**：只在 CoT span 内的实体 token 上做 logit margin（`BAN_MARGIN=20`），answer span 上另加一份拒答 SFT。
- **为什么按状态加权**：坍塌样本上继续压实体只会加剧退化——这正是 R2MU 出现 80% collapse 的机制假设。
- **坍缩约束**：

```
L_fluency = -log p(y_fluent | x)   on forget prompts,  y_fluent = pre-unlearning continuation
L = W_f * L_forget + W_c * L_fluency + W_r * KL(retain)
```

### 3.2 评测侧（本 idea 的一半）

报告三元组 `(Δleak_total, Δleak_state_shift, Δleak_within_state)`，后者才是遗忘证据；前者写进"行为副作用"。

## 4. 交叉验证方式

- Hy4 复核分解是否"分项之和 = 观测 Δ"（trace `20261124T171519Z-hy4-final-adjudication.json`）。
- 机械：`tools/analyze_b18.py` 可从原始 jsonl 重算每一步。
- 负控制：把 `w(collapsed)=1.0` 跑一次，若 collapse 上升而 clean-cell leak 不变，则说明加权有效。

## 5. 实验设计

| 组 | 方法 | 判据 |
|---|---|---|
| T0 | R2MU（现有基线） | 复现 0.888→0.000 与 79% collapse |
| T1 | 本方法（w 按状态） | clean-cell leak 降 ≥0.5 且 collapse 增幅 < 0.2 |
| T2 | 消融：不加 fluency 项 | collapse 增幅应回到 >0.5 |
| T3 | 消融：不加 retain KL | retain ROUGE 掉 >0.1 |

协议：40 forget / 20 retain / 4 策略 / DefaultCoT@512 / seed 1001 / LoRA r=8 / H20。
**预期**：T1 在 clean-cell 上达到 0 泄漏，同时 collapse < 0.2；T2 退化，证明 fluency 项必要。

## 6. 评分依据

| 维度 | 分 | 理由 |
|---|---|---|
| D1 | 2 | 79.4% 的分解比例是本地实测 |
| D2 | 2 | 预注册"collapse 增幅 <0.2"，三种结局都有解释 |
| D3 | 2 | 分解已完成，阈值 >50% 被跨过 |
| D4 | 2 | 无同对象工作，Hy4 签 SUPPORTED |
| D5 | 2 | 若状态位移只占 20%，同样推翻本文前提，两种结果都可发表 |

## 7. 复现入口

```bash
python3 idea-stage/tools/analyze_b18.py           # 三比例面板 + 分解
python3 idea-stage/tools/idea_baseline_matrix.py  # 状态矩阵
```
训练入口：`R-TOFU/workspace/idea_loop/train.py`（`LOSS_TYPE=LOGIT_BAN` 的加权变体）。
