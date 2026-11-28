# IDEA 05 · TMOT — 三边际最优传输行为修复
## Tri-Marginal Optimal Transport for Unified LRM Unlearning

| 项 | 值 |
|---|---|
| 排名 | **#5** |
| 五维 (Higher/Faster/Stronger/Cheaper/Broader) | 8 / 6 / 8 / 6 / 9 |
| 均分 | **7.40** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| Novelty gate | **6/10 · PROCEED WITH CAUTION** |
| 状态 | provisional；无实验验证；外部评审 `REVIEW_UNAVAILABLE` |
| 核心类型 | Novel Method + 跨领域数学框架 |

---

## 1. 理念与 Problem Anchor

> 遗忘不是把 forget 输出朝一个拒答点拉近，而是在行为分布中禁止概率质量流入披露区域；修复则是在 retain 边际不变、unknown 边际可控的条件下重新运输行为质量。

UPBD 的固定 `SAFE_EXIT` 将整个 forget 分布压到一个点，容易产生模板侧信道与结构退化。TMOT 保留 UPBD 的“继承边界 + 行为修复”，但把行为目标从单条文本换成**受约束的轨迹分布**：

- forget 边际：目标知识对应的行为支撑为禁运区；
- retain 边际：target 的有用行为质量必须守恒；
- unknown 边际：允许谨慎移动，但为大位移付费，避免过度遗忘。

这使“不披露”“保效用”“不过度拒答”成为同一运输问题的三个边际条件，而不是三个互相拉扯的 loss 权重。

## 2. 最近邻与真正 delta

| 近邻 | 已占据的内容 | TMOT 的不同轴 |
|---|---|---|
| [Opt-Out, ACL 2025](https://arxiv.org/abs/2406.12329) | OT/Wasserstein 用于 LLM unlearning 的参数偏移正则 | TMOT 的 OT 作用于 CoT+answer 行为分布，而非参数正则 |
| [UOT-Unlearn, 2026](https://arxiv.org/abs/2603.16489) | 图像一步生成中的 unbalanced OT 与质量重分配 | TMOT 处理离散多步推理、三边际语义和 LoRA 参数化 |
| [Cheng et al., ICLR 2026](https://arxiv.org/abs/2603.26569) | retain/forget 纠缠下一维 loss 分布的 $$W_2$$ 约束 | TMOT 学的是行为 transport plan，不是标量 loss 分布约束 |
| CiPO / GUARD | 反事实或 safe-exit CoT 轨迹 | TMOT 可用其轨迹构造参考分布，但核心贡献是禁运支撑与三边际运输 |
| PTP-U | 向原模型做 KL 恢复 | TMOT 比较 KL 与 OT，不以“换一个距离”本身为贡献 |

**不可写的 claim**：“首次将 OT 用于 unlearning”“首次用 Wasserstein 防反弹”。  
**可检验的 claim**：禁运支撑和三边际约束能否在离散 CoT 行为上产生 KL/固定模板无法得到的性质。

## 3. 整体框架

1. 分别获取继承模型分布 $$P_u$$、target 模型分布 $$P_t$$ 与非披露安全参考分布 $$P_s$$。
2. 构造轨迹统计 $$\Phi(Y)$$，其中 $$Y=(c,a)$$ 包含 CoT 与答案，并附加 scope 条件。
3. 求解三边际 OT：forget 边际禁止概率质量进入披露支撑，retain 边际保持 target 输出质量守恒，unknown 边际约束移动成本并允许弃权。
4. 得到修复参考分布 $$P_\star$$，随后蒸馏到 LoRA，并重验继承边界。

安全参考 $$P_s$$ 是结构有效、内容非披露、风格多样的轨迹分布，而不是一个固定字符串。可由规则、target 改写和 Hy4 审核共同构造；所有参考生成种子与过滤规则需冻结。

## 4. 算法设计

对提示 $$q$$，令 $$Y=(c,a)$$ 为完整 CoT 与答案。用可计算统计 $$\Phi(Y)$$ 表示轨迹，包括逐步隐藏状态摘要、答案分布、结构标记和非披露特征。OT 在该统计空间定义，而非声称直接计算任意序列分布的精确 Wasserstein 距离。

### 4.1 披露禁运区

令 $$\mathcal Y_f(q)$$ 为包含目标知识或可从其重建的行为集合。forget 运输计划 $$\pi_f$$ 必须满足

$$
\pi_f\in\Pi(P_u,P_s),\qquad
\pi_f\big(Y,\mathcal Y_f(q)\big)=0.
$$

因此任何来自 $$P_u$$ 的概率质量都不能被运输到披露支撑。

### 4.2 三边际约束

对 retain 提示，要求输出边际与 target 一致：

$$
\operatorname{marg}_{2}\pi_r=P_t,
\qquad \pi_r\in\Pi(P_u,P_t).
$$

对 unknown 提示，允许运输到弃权/安全替代，但限制位移：

$$
\int c(Y,Y')\,\mathrm d\pi_u(Y,Y')\le\epsilon_u.
$$

统一运输目标为

$$
P_\star=
\arg\min_{\{\pi_s\}}
\sum_{s\in\{f,r,u\}}w_s
\left[
\int c_s(Y,Y')\,\mathrm d\pi_s
+\tau\,\Omega(\pi_s)
\right],
$$

其中 $$\Omega$$ 为 entropic 正则，可用 Sinkhorn 求解；三类硬约束优先于权重 $$w_s$$。

### 4.3 LoRA 行为内化

得到参考 $$P_\star$$ 后，修复继承模型：

$$
\min_\Delta
\sum_{s\in\{f,r,u\}}
\mathbb E_{q\sim D_s}
\left[
\mathcal W_\tau\big(
P_{\theta_u+\Delta}(\cdot\mid q),
P_\star(\cdot\mid q)
\big)\right]
+\beta\|\Delta\|_F^2,
$$

并接受满足继承边界、held-out 效用和结构有效性约束的更新。实践中可用 sampled sequence KL 近似内层蒸馏，但必须报告从 OT 到序列蒸馏的近似误差，不能把近似项误称为精确 OT 保证。

### 4.4 可检验性质

- **支撑安全**：若 $$\pi_f(Y,\mathcal Y_f)=0$$ 且蒸馏近似误差为 $$\zeta$$，则目标分布对披露支撑的质量至多由 $$\zeta$$ 控制；
- **retain 守恒**：在精确边际条件下，retain 参考行为不因 forget 禁运而被平均化；
- **近似可审**：分别报告运输成本、边际违反、蒸馏误差，而非只报最终 leak。

## 5. 训练与评测闭环

1. 采样继承模型、target 模型和安全参考的 CoT+answer，按实体与问题隔离 train/calibration/test。
2. 在 $$\Phi(Y)$$ 空间用 Sinkhorn 构造三边际运输计划，记录边际违反与禁运质量。
3. 将 $$P_\star$$ 蒸馏到 LoRA；保留 UPBD 固定模板、KL-to-target、CiPO/GUARD 式轨迹监督作对照。
4. 在 held-out 改写、别名、多跳、解码变体与良性 relearning 下测泄漏。
5. 统一 Hy4 judge，另报 collapse、hallucination、retain 效用、unknown 过度拒答与运输/蒸馏误差。

## 6. 决定性实验与创新点

| 实验 | 配置 | 判读 |
|---|---|---|
| T0 | 直接检查 forget 运输计划的禁运质量 | 若禁运约束本身被数值求解破坏，不进入训练 |
| T1 | TMOT vs UPBD/KL/CIPO/GUARD | 同参考数据与预算下比较最坏 leak、retain 效用和风格多样性 |
| T2 | 去掉三边际，仅保留单一 OT/KL | 若结果相同，三边际不是核心机制 |
| T3 | 去掉禁运支撑或改为软惩罚 | 应观察到披露质量回弹，否则禁运约束只是形式装饰 |
| T4 | retain-only 与 target-start | 分离普通 SFT、初始化与 OT 修复的贡献 |

**创新点**：

- 把固定退出点替换为**带禁运支撑的非披露行为分布**；
- 用 forget/retain/unknown 三边际统一不披露、效用保持和过度遗忘；
- 将运输计划误差与 LoRA 内化误差分开报告，形成可证伪链条；
- 保留 UPBD 的边界继承与低成本行为修复，而非提出另一个上游遗忘损失。

## 7. Idea-evaluator 评估

### 致命缺陷

| 缺陷 | 严重度 | 防御 |
|---|---|---|
| F1：Opt-Out/UOT-Unlearn/Cheng 已覆盖 OT 与遗忘的多种形式 | MAJOR | 只主张离散 CoT 行为的三边际禁运运输；逐一复现/对比 KL 与 OT 近邻 |
| 高维序列 OT 与 LoRA 蒸馏可能使理论约束失真 | MAJOR | 限定在声明的 $$\Phi$$ 统计空间；报告边际/禁运/蒸馏误差；T0/T3 不过即停止 |

无 CRITICAL；若直接宣称序列分布上的精确遗忘保证，则转为 Reject。

### 五维依据

| 维度 | 分 | 依据与提分实验 |
|---|---:|---|
| Higher | 8 | 直接约束披露质量与 retain 边际，机制上覆盖固定模板的失败；待 T1/T3 |
| Faster | 6 | Sinkhorn 与多次轨迹采样增加成本，明显慢于普通 LoRA SFT |
| Stronger | 8 | 禁运与 unknown 边际可提升抗模板/过遗忘能力；未见探针仍待验证 |
| Cheaper | 6 | 不需精确重训，但需要大量参考轨迹与运输计算 |
| Broader | 9 | 三边际运输可迁移到多模态、agent 行为和一般选择性删除 |

### 生命周期、范式与可行性

- 类别：Cross-Disciplinary / Innovative Technique；预计 10–14 周；fit=**Yellow**。
- Paradigm probe：First Principles=Yes（遗忘作为禁运运输）；Elephant=Yes（模板化与过遗忘）；Technology Cycle=Yes（可微 OT 工具成熟）；Hamming=Yes。概念潜力强，实验与近似风险高。
- Compute=Medium-High；Data=Medium-High；Engineering=High；Timeline=Yellow。首个原型应在小样本与低维 $$\Phi$$ 上验证，不直接做全量多模型。

**最终裁决：Accept with Revisions，pending T0/T2/T3。**  
优先证明三边际与禁运约束相对 KL 有独立作用；若只是 OT 距离替换，按 Reject and Pivot 处理。

## 8. 失败即转向

- 若 OT 与 KL 在相同数据/预算下无差异：放弃方法线，不保留“OT 更优雅”作为贡献。
- 若禁运计划正确但自由生成仍泄漏：说明 $$\Phi$$ 丢失关键语义，停止运输证书主张。
- 若 retain 边际守恒导致 forget 泄漏回升：输出不可行区域，而非调权重掩盖冲突。
- 若安全仅来自模板多样性：归入 CFA/行为多样化，不声称 OT 遗忘。
