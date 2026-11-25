# N07 — 效用约束下的遗忘（Pareto-Constrained Unlearning）

**分数 9/10**（D1=2 D2=2 D3=2 D4=1 D5=2）· 裁决 WEAKENED · 证据 `evidence/b18_scenario_panels.json`

## 1. 问题定义

遗忘后 retain ROUGE 从 **0.546** 掉到 **0.425（5%）/ 0.433（10%）**，降幅 0.112–0.120；而泄漏指标只报"归零"。同类现象在 `ours-inv`/`ctrl-single` 上更极端（retain ROUGE 0.014/0.018）——那些系统的"完美遗忘"其实是把答案能力一起废掉。

## 2. 与现有工作的差异

效用通常作为**并列参考列**，不是准入条件。本文把"retain 效用不得低于基线 0.95×"写成**硬约束**，并给出违反时的判定。

## 3. 技术方案要点（含遗忘实现方式）

### 3.1 约束优化

```
max  Δforget
s.t. retain_rouge >= 0.95 * base_retain_rouge      (≈0.518)
L = W_f * L_forget(entity margin, CoT span)
  + W_r * KL(retain)
  + λ_t * max(0, τ_u - retain_rouge_batch)          # 违反即加罚，λ_t 单调上调
```

- **遗忘作用点**：CoT span 内实体 token 的 logit margin；answer span 用拒答 SFT。
- **效用保护**：每步在小批 retain 上估 ROUGE（或代理 NLL），低于阈值就把 `λ_t` 上调 10%，直到回补。这是本方法唯一的机制，不引入新模块。

### 3.2 评测

每行同时给 `(leak, retain_rouge)` 二维点；缺失任一维不得进表。

## 4. 交叉验证方式

- Hy4 检查约束用的是同一批 retain rows、阈值事先声明、报告了方差。
- 机械：`tools/analyze_b18.py` 可直接复算二维点。

## 5. 实验设计

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | 无约束（R2MU） | retain 0.433，leak 0.000 |
| T1 | 硬约束 0.95× | retain ≥0.518 且 leak ≤0.05 |
| T2 | 阈值放宽到 0.90× | retain ≥0.491，观察 leak 是否反弹 |

**预期**：T1 会牺牲部分"归零"（leak 0.05 左右）换回效用；若 T1 做不到，说明当前方法族存在硬权衡，这本身是可发表的结论。

## 6. 评分依据

| 维度 | 分 | 理由 |
|---|---|---|
| D1 | 2 | 0.112 的降幅是实测 |
| D2 | 2 | 预注册 0.95× 约束，三种结局齐备 |
| D3 | 2 | 降幅 >0.05 已跨阈 |
| D4 | 1 | 效用感知遗忘很常见（MUBENCH 过安全），硬门未被占但未获签名 |
| D5 | 2 | 若约束无代价，说明此前担心多余，也有价值 |

## 7. 复现入口

```bash
python3 idea-stage/tools/analyze_b18.py
```
