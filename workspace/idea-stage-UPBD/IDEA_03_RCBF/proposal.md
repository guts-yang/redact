# IDEA 03 · RCBF — 参数—屏障共适应的 CoT 安全集遗忘
## Robust Control-Barrier Forgetting with Policy Co-adaptation

| 项 | 值 |
|---|---|
| 排名 | **#3** |
| 五维 (Higher/Faster/Stronger/Cheaper/Broader) | 8 / 6 / 9 / 7 / 8 |
| 均分 | **7.60** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| Novelty gate | **6/10 · PROCEED WITH CAUTION** |
| 状态 | provisional；无实验验证；外部评审 `REVIEW_UNAVAILABLE` |
| 核心类型 | Novel Method + 轨迹级训练约束 |

---

## 1. 理念与 Problem Anchor

> 遗忘不是“最后一个答案没有泄漏”，而是整条推理动力学始终停留在非披露安全集内；当 LoRA 修复改变动力学时，原来的安全判据也必须重新适应。

R-TOFU 关注 CoT 逐步泄漏，GUARD 学一条自然 safe-exit，UPBD 学固定退出串。这些都不回答一个更基础的问题：行为修复后，模型是否可能在中途离开安全区域，再走回一个表面正常的答案？

RCBF 将继承边界解释为 CoT 隐轨迹上的安全集，并引入**带正裕度、随策略共适应的离散 control barrier**。其核心不是把已有 CBF 不等式搬进 NLP，而是研究“策略参数更新后，barrier 何时仍有效”。

## 2. 最近邻与真正 delta

| 近邻 | 已占据的内容 | RCBF 的不同轴 |
|---|---|---|
| [BarrierSteer, 2026](https://arxiv.org/abs/2602.20102) | 推理期学习 hidden-state CBF 并 steering，不改参数 | RCBF 在训练期共适应 barrier 与 LoRA 修复，并针对 unlearning CoT |
| [CBF-LLM, 2025](https://arxiv.org/abs/2511.03121) | token 生成的离散 CBF 过滤条件 | 离散不等式本身完全沿用；差异是参数更新下的鲁棒裕度与训练内约束 |
| [GUARD, EMNLP 2026](https://arxiv.org/abs/2609.21677) | 将自然 safe-exit CoT 蒸馏进参数 | GUARD 优化轨迹质量；RCBF 优化整条轨迹的安全集不变性 |
| BOUND / BUDGET（本地） | 答案承诺边界 / 推理预算单调 | RCBF 的状态是 CoT 隐轨迹，不是答案后缀或预算标量 |
| REDaCT / STaR | 推理时检测并纠正/过滤 | RCBF 产出合并后的权重，不依赖在线监控器 |

**不可写的 claim**：“首次将 CBF 用于 LLM”“离散 CBF 不等式是新定理”。  
**可检验的 claim**：共适应与正安全裕度能否防止“旧策略上学到的 barrier 在修复后失效”。

## 3. 整体框架

1. 从继承的 unlearned checkpoint $$\theta_u$$ 采样 forget、retain 与 aux CoT 轨迹，用 Hy4 定位首次披露步与结构失效。
2. 学习非披露状态的安全集，要求安全裕度严格为正：

   $$
   \mathcal S_{\phi,\eta}=\{z:h_\phi(z)\ge\eta\},
   \qquad \eta>0.
   $$

3. 交替执行策略—屏障优化：固定策略并更新 barrier 与披露对照；固定 barrier 并在扰动策略族上约束 LoRA 修复；随后从新策略重新采样轨迹，进入下一轮。
4. 输出 merged LRM 以及有限轨迹族上的经验不变性报告。

非披露正例来自结构有效、内容安全的退出轨迹；负例包括直接泄漏、别名重建、多跳恢复与幻觉性替代。固定 `SAFE_EXIT` 仅作对照，避免把 barrier 学成模板检测器。

## 4. 算法设计

令 $$z_t^\theta$$ 为策略 $$\theta$$ 在第 $$t$$ 个推理步的完整上下文隐藏状态，$$h_\phi(z)$$ 为安全裕度函数。安全集为

$$
\mathcal S_{\phi,\eta}=\{z:h_\phi(z)\ge\eta\},\qquad \eta>0.
$$

### 4.1 离散前向不变条件

对声明的修复策略邻域

$$
\Theta_\rho=\{\theta_u+\Delta:\Delta\in\mathrm{LoRA}_r,
\ \|\Delta\|_F\le\rho\},
$$

要求采样轨迹满足

$$
\min_{\theta'\in\Theta_\rho}
\left[
 h_\phi(z_{t+1}^{\theta'})-(1-\gamma)h_\phi(z_t^{\theta'})
\right]\ge0,
\qquad \gamma\in(0,1).
$$

正裕度 $$\eta$$ 防止 $$h\to0$$ 仍被误称为安全；$$\Theta_\rho$$ 显式表达“修复会改变动力学”。

### 4.2 共适应训练

Barrier 损失使用安全状态 $$z^+$$、披露状态 $$z^-$$ 与边界状态 $$z^b$$：

$$
\mathcal L_h=
\mathbb E[\eta-h_\phi(z^+)]_+
+\mathbb E[h_\phi(z^-)]_+
+\lambda_b\,\mathbb E|h_\phi(z^b)-\eta|.
$$

LoRA 修复目标为

$$
\min_{\Delta}
\mathcal L_{\mathrm{beh}}+
\lambda_{\mathrm{cbf}}\,
\mathbb E\left[
 (1-\gamma)h_\phi(z_t^\theta)-h_\phi(z_{t+1}^\theta)
\right]_+
+\lambda_U\mathcal L_{\mathrm{retain/aux}},
$$

其中

$$
\mathcal L_{\mathrm{beh}}=
\mathcal L_{\mathrm{non\text{-}disclosure}}(D_f)
+\lambda_s\mathcal L_{\mathrm{structure}}.
$$

每轮必须从新 $$\theta$$ 重采样轨迹并更新 $$h_\phi$$；冻结旧 barrier 的版本作为关键消融。

### 4.3 经验证书的边界

输出

$$
\widehat{\mathrm{VI}}=
\mathbb E_{q\sim\mathcal Q_{\mathrm{test}}}
\left[\frac1T\sum_t
\mathbf 1\{h(z_{t+1})<(1-\gamma)h(z_t)\}\right],
$$

以及 $$h<\eta$$ 的进入率、自由生成 leak、collapse 与效用。它们只覆盖采到的轨迹和声明的 $$\Theta_\rho$$，不构成对所有 reachable states 的证明。若无法给出非空洞的扰动界，就只能称为 barrier-shaped regularizer。

## 5. 训练与评测闭环

1. 从 R2MU/GA 等继承边界生成 CoT，使用 Hy4 做逐步泄漏标注；训练、校准、测试按实体隔离。
2. 学初始 $$h_\phi$$，并用“模型是否仍知道答案”的探针避免把拒答风格当安全状态。
3. 交替更新 barrier 与 LoRA 修复；每轮加入新动力学产生的 hard negatives。
4. 在未见改写、别名、多跳、多轮与解码变体上计算 $$\widehat{\mathrm{VI}}$$ 和实际泄漏。
5. 保留 C1 retain-only、C2 target-start、无共适应、$$\eta=0$$、$$\rho=0$$ 五组对照。

## 6. 决定性实验与创新点

| 实验 | 配置 | 判读 |
|---|---|---|
| T0 | 旧策略学 barrier，直接用于修复后策略 | 若不变性显著破坏，证明共适应问题真实存在 |
| T1 | RCBF vs GUARD/UPBD/BarrierSteer | 比较 held-out 轨迹越界率、最坏 leak、retain 效用与推理开销 |
| T2 | 冻结 barrier vs 共适应 | 无差异则核心机制被驳倒 |
| T3 | $$\eta=0$$ 与无 $$\Theta_\rho$$ 鲁棒项 | 验证正裕度与参数邻域是否各自必要 |
| T4 | 长 CoT、ZeroThink/LessThink、多轮追问 | 只在单轨迹成立则降级为局部行为约束 |

**创新点**：

- 将 UPBD 的静态遗忘边界扩展为**CoT 动力学安全集**；
- 把“LoRA 修复改变动力学”纳入 barrier 的鲁棒条件与共适应过程；
- 用首次披露步而非最终答案构造训练信号；
- 明确区分经验不变性、局部鲁棒条件与不可获得的全称证书。

## 7. Idea-evaluator 评估

### 致命缺陷

| 缺陷 | 严重度 | 防御 |
|---|---|---|
| F1：BarrierSteer/CBF-LLM 已有 hidden barrier 与同类不等式 | MAJOR | 只主张训练期参数—屏障共适应；与推理期 barrier 做直接对照 |
| 旧动力学上的 barrier 对新策略无效，且完整上下文非马尔可夫 | MAJOR | 状态保留完整上下文；每轮重采样；正裕度 + 策略邻域；T0/T2 失败则降级为经验正则 |

无 CRITICAL；但若论文宣称形式化全局安全证书，该缺陷会升级为 CRITICAL，因此当前版本明确禁止此表述。

### 五维依据

| 维度 | 分 | 依据与提分实验 |
|---|---:|---|
| Higher | 8 | 逐步安全约束比只监督最终答案更直接覆盖 CoT 泄漏；待 T1 |
| Faster | 6 | 需多轮轨迹采样与 barrier 更新，明显慢于原始 UPBD |
| Stronger | 9 | 目标是跨推理步与解码的轨迹不变性；最高上限也最受 T0/T2 约束 |
| Cheaper | 7 | 复用已有生成和 Hy4 首漏标注，不需精确重训；新增 barrier 训练 |
| Broader | 8 | 可推广到 agent 轨迹、工具调用与一般安全策略修复 |

### 生命周期、范式与可行性

- 类别：Frontier Exploration；预计 10–14 周；每周投入未声明，fit=**Yellow**。
- Paradigm probe：First Principles=Yes（遗忘是动力学安全集）；Elephant=Yes（CoT 中途泄漏）；Technology Cycle=Yes（隐藏状态与控制工具成熟）；Hamming=Yes。潜力强但理论风险同样高。
- Compute=Medium-High；Data=Medium；Engineering=High（需逐步隐藏状态与轨迹回放）；Timeline=Yellow。2×H20 可做 8B 原型，但不适合一开始做多模型大规模扫描。

**最终裁决：Accept with Revisions，pending T0/T2/T3。**  
最优先不是写完整框架，而是证明“冻结 barrier 在修复后确实失效、共适应能修复该失效”这一对照。

## 8. 失败即转向

- 若冻结 barrier 与共适应无差异：删除共适应主张，改做轻量轨迹正则。
- 若 $$\widehat{\mathrm{VI}}$$ 很低但自由生成仍泄漏：barrier 没有语义效度，停止安全证书叙事。
- 若安全来自高拒答/高 collapse：判为退化解，不计为遗忘改善。
- 若 BarrierSteer 在相同效用下更便宜：转向推理期组合，而非坚持权重修复。
