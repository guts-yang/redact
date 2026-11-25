# N06 — 状态不变遗忘：四种解码前缀下同一问题必须同态（State-Invariant Unlearning）

**分数 8/10**（D1=2 D2=2 D3=1 D4=1 D5=2）· 裁决 WEAKENED · 证据 `evidence/idea_loop_latest.json`（逐策略状态分解）

## 1. 问题定义

现有报告按策略各算一个泄漏率，但**同一道题在不同策略下会进入不同状态**：DefaultCoT 常被截断/坍塌、ZeroThink 直接答题。实测 40 题中 **12 题（0.300）** 在四策略间状态不一致。

因此"策略 A 比 B 泄漏少"可能只是"A 比 B 更容易坍塌"。

## 2. 与现有工作的差异

STaR 的 MCS 比较的是**泄漏分数**的一致性；本文要求的是**生成状态**（answered / censored / collapsed / no-channel）的一致性，对象不同但相邻——这也是它只拿到 WEAKENED 的原因。

## 3. 技术方案要点（含遗忘实现方式）

### 3.1 训练：前缀条件一致性

```
for q in forget_set:
    for prefix p in {Default, Less, Zero, Summary}:
        s_p = state(model, q, p)
    L_state = Σ_{p≠p0} 1[s_p != s_p0] * (CE(refuse | q, p))     # 强制同态：都进入"干净拒答"
L = W_f * L_forget(entity margin) + λ_s * L_state + W_r * KL(retain)
```

- **遗忘作用点**：实体 token margin 负责"不说"；`L_state` 负责"在任何前缀下都干净地不说"，避免某一路径坍塌、另一路径泄漏。
- **状态目标**：统一到 `answered_clean + REFUSE`（可达、不坍塌、不泄漏）。
- **为什么不用更多策略**：四个已覆盖 R-TOFU 的三个构造前缀与默认路径。

### 3.2 评测

报告 `state_non_invariance` 与"每策略 clean-cell leak"两个数；前者下降是方法有效的直接证据。

## 4. 交叉验证方式

- Hy4 复核配对键（question 文本）与状态定义，并检查是否存在"用不同题目冒充同一题"。
- 机械：`tools/idea_baseline_matrix.py` 可从原始 rows 重算非不变率。

## 5. 实验设计

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | 现有（无一致性项） | 非不变率 0.300 |
| T1 | 加 `L_state` | 非不变率 <0.15 且 clean-cell leak ≤0.05 |
| T2 | 消融：只对 ZeroThink 施加 | 其他策略非不变率不降 |

**D3=1 的原因**：预注册阈值是 `>0.3`，实测恰为 **0.300（12/40）**，未严格跨过。把题目从 40 扩到 120 条（CPU，`decode` 复用已有生成不成立，需重跑或扩采样）即可判定；这是把它抬到 9 分的最便宜路径。

## 6. 评分依据

| 维度 | 分 | 理由 |
|---|---|---|
| D1 | 2 | 0.300 是实测 |
| D2 | 2 | 预注册阈值与三结局明确 |
| D3 | 1 | 0.300 未严格 >0.3 |
| D4 | 1 | 与 STaR MCS 相邻，未获签名 |
| D5 | 2 | 若状态本就同态，说明按策略分报已充分，也有价值 |

## 7. 复现入口

```bash
python3 idea-stage/tools/idea_baseline_matrix.py
```
