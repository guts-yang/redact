# N05 — 片段级遗忘：压片段而不是压整个回答（Span-Level Unlearning）

**分数 8/10**（D1=2 D2=2 D3=1 D4=1 D5=2）· 裁决 WEAKENED · 证据 `evidence/judge_cache_*.jsonl`（含 `leak_span`）、`evidence/afe_cfe_results.json`

## 1. 问题定义

`entity_leak` 是二元的：只要出现 gold 短语就算泄漏。AFE judge 已经吐出**逐字 span**（≤15 词），本仓库缓存里有这些 span。实测 **只有 7.5% 的泄漏 span ≤2 词**——即绝大多数"泄漏"是带命题内容的句子，不是"只报了个名字"。

这条观测与 I09（lexical anchoring）方向相反：**名字只是入口，真泄漏是命题。**

## 2. 与现有工作的差异

ShadowBench 主张"删名字不等于删知识"；本文进一步要求**训练目标作用在片段上**，而不是只对实体 token 做压制。片段级目标在该方向的既有工作里没有实现。

## 3. 技术方案要点（含遗忘实现方式）

### 3.1 遗忘损失：span 级 unlikelihood

```
span = judge_leak_span(response)                # 逐字片段
L_forget = -Σ_{t in span} log(1 - p(t | context))          # unlikelihood
         + margin_loss(entity token)                        # 仍压实体，防止换名
```

- **遗忘作用点**：CoT span 内、被 judge 标出的**整段 token**，而非单个实体；实体 margin 作为兜底。
- **为什么需要两层**：只压实体 → 模型改用代词/别名继续叙述（ShadowBench 的发现）；只压片段 → 换个措辞又能说。二者叠加才封闭。

### 3.2 评测

新增 `span_length` 分布与 `entity_only_share` 作为并列指标；`entity_only_share` 高说明漏的是名字，低说明漏的是命题。

## 4. 交叉验证方式

- Hy4 抽样核对 span 是否逐字出现在 response 中（当前 92.5% 的 span >2 词，需人工抽检 20 条）。
- 机械：`tools/eval_afe_cfe.py` 的缓存可直接聚合 span 长度。

## 5. 实验设计

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | 只压实体（R2MU 类） | entity_only_share 0.075，换名后仍泄漏 |
| T1 | span + 实体双层 | 换名探针上泄漏降 ≥0.3 |
| T2 | 消融：只压 span | 实体 token 仍出现，泄漏回弹 |

**注意**：本 idea 的阈值（entity_only_share > 0.5）**未被跨过**（实测 0.075），因此 D3=1。若要升到 9 分，必须换一个真正被跨过的预注册判据（例如"span+实体双层在换名探针上降幅 >0.3"），并跑 T1/T2。

## 6. 评分依据

| 维度 | 分 | 理由 |
|---|---|---|
| D1 | 2 | span 长度分布是实测 |
| D2 | 2 | 预注册判据明确，三种结局齐备 |
| D3 | 1 | 阈值未跨过（0.075 < 0.5），只到描述性 |
| D4 | 1 | 与 ShadowBench 相邻，未获签名 |
| D5 | 2 | "泄漏是命题而非名字"翻转同样可发表 |

## 7. 复现入口

```bash
python3 idea-stage/tools/eval_afe_cfe.py --n 20 --workers 8
```
