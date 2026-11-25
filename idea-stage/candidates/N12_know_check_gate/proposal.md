# N12 — 知识存在性前置门（Know-Check-Gated Unlearning）

**分数 9/10**（D1=2 D2=2 D3=2 D4=1 D5=2）· 裁决 WEAKENED · 证据 `evidence/b17_calibration_failure.json`、`evidence/b17_know_check_target_*.json`

## 1. 问题定义

如果 checkpoint 本来就不掌握某个 split，"泄漏率低"就没有任何含义。本仓库里三个信号对该问题的回答互相矛盾：

| 信号 | forget01 上的读数 | 隐含判断 |
|---|---|---|
| 生成（greedy，512 tok） | `entity_leak = 0.0`，97.5% 退化 | 不知道 / 说不出来 |
| teacher-forced NLL | **1.885**（< 2.0 门） | 知道 |
| 既有记录 `CHAMPION_LOGIC.md` | **5.4692** | 不知道 |

同时 `forget01 ⊂ forget10`（400/3600 题），所以两次 NLL 实际评的是同一批题——**这不是两次独立测量**。

## 2. 与现有工作的差异

基准默认"模型知道 forget set"；本仓库自己就撞上了反例。没有基准把知识存在性做成**运行前置断言**。

## 3. 技术方案要点（含遗忘实现方式）

### 3.1 训练：知识门控的遗忘

```
gate(x) = 1  iff  NLL(x, gold) <= 2.0  and  reproduction(x) >= 0.5
L = Σ_x gate(x) * L_forget(x)  +  W_r * KL(retain)
```

- **遗忘作用点**：仍是 CoT span 内实体 token 的 margin，但**只对通过门的样本施加**。
- **为什么重要**：对"本来就不会"的样本施加遗忘梯度，等价于噪声标签训练，会把模型推向退化——这与观察到的 80% collapse 一致。
- **门本身要冻结**：阈值（2.0 / 0.5）与 prompt 模板写进配置，不得按结果回调。

### 3.2 评测

未通过门的 split 一律标 `NOT EVALUABLE`，不产出 AFE/CFE。

## 4. 交叉验证方式

- Hy4 检查阈值是否事先声明、两信号是否用同一模板。
- 机械：`tools/know_check.py` 可对任意 (model, split) 重算。
- 已知冲突要保留：先冻结模板，再重算 NLL 与复现，两者一致才启用门。

## 5. 实验设计

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | 不设门 | forget01 上 AFE 看似 0.975，实为"不会" |
| T1 | 设门 | forget01 标 NOT EVALUABLE；forget05/10 正常评 |
| T2 | 模板统一后重算 | NLL 与复现一致（否则门不可用） |

## 6. 评分依据

| 维度 | 分 | 理由 |
|---|---|---|
| D1 | 2 | 三信号矛盾是实测记录 |
| D2 | 2 | 预注册"两信号不一致"，正/负/零都有解释 |
| D3 | 2 | 不一致已发生（跨过阈值） |
| D4 | 1 | UWC 的评估前置条件相邻；本仓库无外部签名 |
| D5 | 2 | 若两信号一致，说明现有基准安全，也有价值 |

## 7. 复现入口

```bash
python3 idea-stage/tools/know_check.py --model target --split forget01 --n 16
```
