# N08 — 契约绑定的推理时遗忘（Contract-Bound Inference-Time Unlearning）

**分数 10/10**（D1=2 D2=2 D3=2 D4=2 D5=2）· 裁决 SUPPORTED · 证据 `evidence/code_diagnosis.json`（7/7）

## 1. 问题定义

推理时遗忘（REDaCT 类：检测器触发 → 从越界句重生成 → 注入 soft token）的结论**不可复现**：
阈值持久化会"silently change the flagging rate, not error"（`PIPELINE.md:221`），judge 后端从 GPT-4o 漂到 DeepSeek，数据与检测器权重都不随仓库提供。同一份代码换台机器，AFE 就变。

**要解决的问题不是"忘不忘得掉"，而是"这个遗忘结果能不能被第三方复算"。**

## 2. 与现有工作的差异

| 工作 | 做了什么 | 没做什么 |
|---|---|---|
| REDaCT / ARIA | 报告推理时门控的 AFE/CFE 提升 | 未把阈值/后端/权重/数据绑定成可校验契约 |
| 模型卡 / 数据卡 | 声明版本 | 不把版本作为**运行前置断言**，也不失败即停 |
| 本文 | 契约文件 + 启动断言 + 违约计数 | — |

## 3. 技术方案要点（含遗忘实现方式）

### 3.1 遗忘本身（inference-time，不改权重）

```
for sentence_boundary in CoT:
    s = detector(ctx_before_boundary)          # 二分类 head，阈值 τ（来自契约）
    if s >= τ:
        regenerate from boundary
        inject soft_token (16 x 4096) at embedding layer via forward hook
```

- **遗忘发生在哪**：检测器命中点之后的所有 token 都由"纠正式 soft token"重新条件化，使实体名不再被生成；目标模型的权重**不动**。
- **soft token 怎么来**：在"越界点→拒答/中性续写"的配对数据上训练一个 16 向量的可学习嵌入（`scripts/train_soft_token.py`），只更新这 16 个向量。
- **检测器怎么来**：RoBERTa + 单二分类头（`redact/regen/binary_head_model.py`），在首个泄漏句标注上训练。

### 3.2 契约（本 idea 的真正贡献）

`contract.json` 必须包含并参与哈希：

```json
{"detector_sha256": "...", "threshold": 0.37, "judge_backend": "hy4",
 "judge_prompt_sha256": "...", "forget_file_sha256": "...",
 "soft_token_sha256": "...", "decoding": {"do_sample": false, "max_new_tokens": 512}}
```

启动时逐项断言：任一不符 → **抛异常而不是静默降级**。每次运行输出 `contract_violations` 计数。

### 3.3 与 AFE/CFE 的关系

契约里同时绑定 judge 的 prompt 哈希，保证 AFE/CFE 的分母口径不变（见 N03）。

## 4. 交叉验证方式

- 评审：Hy4 逐项核对契约字段是否都能从仓库文件复算（trace `.aris/traces/20261124T170058Z-hy4-final-adjudication.json`）。
- 机械：`tools/cross_review.py` 与 `evidence/code_diagnosis.json` 的 7 项断言可重跑。
- 负面控制：故意改一个阈值，契约必须报违约且 AFE 变化被记录。

## 5. 实验设计

| 组 | 配置 | 判据 |
|---|---|---|
| C0 | 原样运行 | 7 项断言全部通过 |
| C1 | 阈值 ±0.05 | 违约计数 ≥1，且 flagging rate 变化被记录 |
| C2 | 换 judge 后端（DeepSeek→Hy4） | 契约拒绝启动（后端哈希不符） |
| C3 | 删 `data/rtofu` | 启动即失败，不产出 AFE |

基线协议：40 forget / 20 retain / 四策略 / 默认预算 / seed 1001。

**预期**：C0 通过；C1–C3 全部"失败即停"，并给出可读的违约行。这是"可复现"与"不可复现"的判别实验。

## 6. 评分依据

| 维度 | 分 | 理由 |
|---|---|---|
| D1 问题锚定 | 2 | 7 项缺陷均为文件级复验，非文献主张 |
| D2 可证伪 | 2 | 预注册违约计数 ≥3，且有正/负/零三种解释 |
| D3 可复现 | 2 | 7 ≥ 3，阈值被跨过，可一键重跑 |
| D4 新颖性 | 2 | 外部检索无同对象工作；Hy4 签 SUPPORTED |
| D5 信息量 | 2 | 违约=0 也发表（说明该类方法其实可复现） |

## 7. 复现入口

```bash
python3 idea-stage/tools/auto_loop.py --stage diagnose   # 7 项断言
python3 idea-stage/tools/metric_parity.py                # 协议对齐
```
产物：`evidence/code_diagnosis.json`、`evidence/metric_parity.json`。
