# IDEA 01 · TCR — 三层边界契约与可容许修复
## Tri-layer Contract Repair for Unified LRM Unlearning

| 项 | 值 |
|---|---|
| 排名 | **#1** |
| 五维 (Higher/Faster/Stronger/Cheaper/Broader) | 7 / 8 / 8 / 8 / 9 |
| 均分 | **8.00** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| Novelty gate | **7/10 · PROCEED**（检索截至 2026-11-28） |
| 状态 | provisional；无实验验证；外部评审 `REVIEW_UNAVAILABLE` |
| 资源假设 | 2×H20、8B LRM、R-TOFU、现有 UPBD LoRA 管线 |
| 核心类型 | Novel Method + 统一框架 |

---

## 1. 理念与 Problem Anchor

> 遗忘不应只是一个 loss，而应是一份可继承、可检查、可组合的契约：上游 unlearner 负责产生边界，行为修复只能在边界允许的集合内移动。

UPBD 已包含“冻结 R2MU checkpoint + LoRA 行为修复”的雏形，但它只继承了一个参数点，无法回答修复是否改变遗忘语义。TCR 将该参数点提升为三类可检验对象：

1. **行为契约**：在声明的 forget 探针族上，披露风险不得超过预算；
2. **表示契约**：修复不得重新激活 target 与 unlearned 模型差异所标识的敏感方向；
3. **参数契约**：修复在高遗忘敏感参数方向上的能量必须受限。

TCR 不声称从单个 checkpoint 恢复“真实知识边界”。边界始终相对于探针族、表示统计和上游资产定义；这是一种可操作边界，而非参数级删除证明。

## 2. 最近邻与真正 delta

| 近邻 | 已占据的内容 | TCR 保留的差异 |
|---|---|---|
| [PTP-U, 2026](https://arxiv.org/abs/2610.11915) | 固定遗忘约束下恢复非目标能力的两阶段框架 | PTP-U 的边界是行为阈值；TCR 额外从 checkpoint 构造表示与参数契约，并面向 CoT 组合接口 |
| [OBU, 2026](https://arxiv.org/abs/2610.27490) | Projected LoRA、良性能力子空间的正交零空间 | OBU 是单层投影；TCR 的主对象是三类契约的交集与修复接受规则 |
| [JANUS, 2026](https://arxiv.org/abs/2609.19985) | post-hoc Jacobian 零空间矫正 | JANUS 保护历史任务损失，不表达 unlearning 的披露语义 |
| [HAMU, ICML 2026](https://arxiv.org/abs/2606.02119) | 约束多目标遗忘与不可调和判据 | HAMU 从原模型优化遗忘；TCR 从已有 unlearned checkpoint 继承并修复 |
| GUARD / CiPO | 学 safe-exit 或反事实 CoT | TCR 可把它们作为上游边界或行为参考，不与其争夺轨迹生成器 |

**不可写的 claim**：“首次两阶段遗忘”“首次约束优化”“首次 LoRA 投影”。  
**可检验的 claim**：同一修复算子能否在异质上游边界上保持三类契约，并在 retain-only 与 target-start 控制之外产生独立增益。

## 3. 整体框架

1. **Boundary Reader**：从 target 或已有 unlearned checkpoint 读取行为探针统计、表示敏感谱和参数敏感支持，输出边界对象

   $$
   B=\big(\mathcal C_\epsilon,\mathcal S_f,\mathcal W_f\big).
   $$

2. **Admissible Repair**：学习非披露行为并恢复 retain/aux 行为；依次执行梯度投影、小步 LoRA 更新、契约检查与必要回退。
3. **Contract Certificate + Held-out Audit**：输出最坏探针风险、投影残差、覆盖范围和归因对照。

行为目标不再局限于固定 `SAFE_EXIT`：forget 侧使用结构有效、非披露、措辞多样的退出轨迹；retain/aux 侧学习目标模型的推理结构与答案行为。固定模板仅作为 UPBD 消融臂。

## 4. 算法设计

令 $$\theta_u$$ 为继承的 unlearned checkpoint，$$\Delta$$ 为 LoRA 增量，$$\mathcal Q=\mathcal Q_{\mathrm{direct}}\cup\mathcal Q_{\mathrm{para}}\cup\mathcal Q_{\mathrm{alias}}\cup\mathcal Q_{\mathrm{hop}}\cup\mathcal Q_{\mathrm{dec}}$$ 为声明的探针族。定义披露风险 $$L_{\mathcal Q}(\theta)$$ 与效用 $$U(\theta)$$。

### 4.1 三层边界对象

$$
B=\big(\mathcal C_\epsilon,\mathcal S_f,\mathcal W_f\big),\qquad
\mathcal C_\epsilon=\{\theta:L_{\mathcal Q}(\theta)\le\epsilon\}.
$$

$$\mathcal S_f$$ 由 target/unlearned 在 forget 探针上的激活差或 Jacobian 谱估计；$$\mathcal W_f=(w_1,\ldots,w_p)$$ 是参数块敏感权重，而非“知识所在位置”的本体断言。

### 4.2 可容许修复集

$$
\mathcal A(B)=\left\{\Delta:
\begin{array}{l}
\theta_u+\Delta\in\mathcal C_\epsilon,\\
\|\mathcal P_{\mathcal S_f}J_\Delta\|_2\le\tau_r,\\
\sum_j w_j\|\Delta_j\|_F^2\le\tau_p
\end{array}\right\},
$$

其中 $$J_\Delta$$ 是修复对声明探针输出的局部 Jacobian。第二层限制敏感方向上的功能变化，第三层限制高敏感参数块上的更新能量。

### 4.3 投影—修复—接受

$$
\min_{\Delta}\quad
\mathcal L_{\mathrm{nd}}(D_f)+
\lambda_r\mathcal L_{\mathrm{target}}(D_r)+
\lambda_a\mathcal L_{\mathrm{target}}(D_a),
\qquad \Delta\in\mathcal A(B).
$$

每步先计算普通 SFT 梯度，再投影到局部可容许方向；只有在独立 calibration probes 上满足

$$
\widehat L_{\mathcal Q}(\theta_u+\Delta)\le\epsilon,
\qquad
\Delta U_{\mathrm{aux}}\ge-\delta_u,
\qquad
\mathrm{Collapse}\le\delta_c
$$

时才接受，否则减半步长。输出

$$
\mathrm{Cert}(B,\mathcal Q)=
(\widehat L_{\mathrm{worst}},\rho_{\mathrm{proj}},
\widehat\kappa_{\mathrm{cover}},\epsilon,\delta_u,\delta_c),
$$

其中所有量都是**指定探针族上的经验证书**，不是全称保证。

## 5. 训练与评测闭环

1. **Boundary reading**：分别从 R2MU、GA/NPO 等至少两类上游 checkpoint 构造 $$B$$；固定同一 tokenizer、解码与 Hy4 judge。
2. **严格切分**：forget 按实体与问题双重切分为 train/calibration/test；禁止 `--holdout 0` 后又评测同一批问题。
3. **行为修复**：LoRA 学非披露 forget 行为与 retain/aux target 行为，每轮重新读取边界残差。
4. **归因控制**：
   - C1：从 $$\theta_u$$ 做等预算 retain-only SFT；
   - C2：从 target 做同配方 target-start SFT；
   - M：完整 TCR；
   - 另设随机子空间与无参数契约消融。
5. **审计**：直接问、改写、别名、多跳、ZeroThink/LessThink、良性 relearning 分别报告；collapse、hallucination、refusal 与 leak 不合并。

## 6. 决定性实验与创新点

| 实验 | 对照 | 通过/证伪含义 |
|---|---|---|
| T0 | 同一 checkpoint 的边界读取跨 seed | 若三层统计不稳定，则“边界资产”不成立，停止 |
| T1 | TCR vs UPBD vs C1/C2 | 只有超过 C1 且不依赖 C2 的泄漏水平，修复算子才有独立贡献 |
| T2 | 分别删除行为/表示/参数契约 | 每一层必须对应一类可观测失败；否则该层是装饰 |
| T3 | 跨上游边界 leave-one-family-out | 同一 Repair 在新边界上仍有效，才支持“统一框架” |
| T4 | held-out 改写/别名/多跳/解码 | 若风险显著回弹，只能声称 in-scope 行为修复 |

**创新点**：

- 把 UPBD 的“冻结点”升级为**三层可操作边界资产**；
- 把“修复不伤遗忘”写成**可容许集成员关系与接受/回退规则**；
- 把上游 unlearner、行为修复与审计证书统一到同一接口；
- 用 C1/C2 与三层消融把继承、数据和修复的贡献分开。

## 7. Idea-evaluator 评估

### 致命缺陷

| 缺陷 | 严重度 | 防御 |
|---|---|---|
| F1：PTP-U/OBU/JANUS 已覆盖两阶段或投影修复 | MAJOR | 只主张 checkpoint 派生的三层契约与 CoT 组合；直接复现近邻并做三层消融 |
| 单 checkpoint 不能识别真实语义边界 | MAJOR | 把边界限定为声明探针与统计量下的操作对象；T0 不稳定则放弃 |

无 CRITICAL；现有 UPBD 数据未直接反驳三层契约机制。

### 五维依据

| 维度 | 分 | 依据与提分实验 |
|---|---:|---|
| Higher | 7 | 机制上同时约束披露与效用；尚无独立增益证据，T1 决定能否到 8 |
| Faster | 8 | 继承 checkpoint + LoRA，避免重跑上游遗忘；投影/SVD 增加常数开销 |
| Stronger | 8 | 三层约束直接针对修复重开；跨探针与跨边界仍待 T3/T4 |
| Cheaper | 8 | 复用现有 CSV、checkpoint 与 judge，无需精确重训；需生成动态探针 |
| Broader | 9 | 边界读取与修复接口可接 GA/NPO/R2MU/GUARD，并可迁移到非 LRM |

### 生命周期、范式与可行性

- 类别：Innovative Technique；预计 8–12 周。用户每周可投入小时数未声明，故 capability fit 为 **Yellow**。
- Paradigm probe：First Principles=Yes（遗忘从 loss 变为契约）；Elephant=Yes（修复重开与效用崩塌）；Technology Cycle=Yes（PEFT/激活工具成熟）；Hamming=Yes（若成立会改变方法组合方式）。概念上具 disruptive potential，但无实验证据。
- Compute=Medium；Data=Medium（需隔离动态探针）；Engineering=Medium；Timeline=Yellow。2×H20 足够做 8B LoRA，瓶颈是多种边界与生成审计。

**最终裁决：Accept with Revisions，worth pursuing, pending T0/T1/T2。**  
首要动作：先做 T0；再做 C1/C2；最后做三层逐一消融。任何一项失败都缩减或放弃对应 claim。

## 8. 失败即转向

- 若 retain-only 与 TCR 等效：把成果改写成“简单修复已足够”的归因报告，不保留方法主张。
- 若 target-start 也得到同样低泄漏：说明遗忘来自行为监督而非边界继承，放弃 TCR。
- 若只有 R2MU 有效、跨边界失败：降级为单边界工程改进，不再称统一框架。
- 若表示或参数契约消融无差异：删除该层，避免为框架完整性保留无效模块。
