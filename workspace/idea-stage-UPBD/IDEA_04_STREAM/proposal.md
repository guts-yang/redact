# IDEA 04 · STREAM — 可行性感知的流式边界—行为修复
## Feasibility-Aware Streaming Boundary Repair

| 项 | 值 |
|---|---|
| 排名 | **#4** |
| 五维 (Higher/Faster/Stronger/Cheaper/Broader) | 7 / 8 / 8 / 7 / 8 |
| 均分 | **7.60** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| Novelty gate | **5.5/10 · PROCEED WITH CAUTION** |
| 状态 | provisional；无实验验证；外部评审 `REVIEW_UNAVAILABLE` |
| 核心类型 | New Setting + Novel Method |

---

## 1. 理念与 Problem Anchor

> 真实遗忘是一串带保留约束的请求流，而不是一次训练；系统必须持续继承历史边界、修复当前行为，并能够承认“当前请求不可同时满足”。

UPBD 处理单次边界修复；STREAM 把它提升为在线过程。每个新请求既带有 forget scope，也带有 retain/protect scope。系统 warm-start 上一轮 LoRA 与对偶状态，更新当前行为，同时记录历史约束的累计违反。

核心不是声称“非凸 8B LRM 上存在 no-regret 保证”，而是区分三个层次：

1. 凸代理上的在线分析；
2. 实际 LRM 上的经验累计违反；
3. 无可行解时的显式不可行/最小让步输出。

## 2. 最近邻与真正 delta

| 近邻 | 已占据的内容 | STREAM 的不同轴 |
|---|---|---|
| [BLADE, EMNLP 2026](https://arxiv.org/abs/2608.22557) | bilevel、增广拉格朗日、LoRA repair、连续请求 | BLADE 每轮重新做离线约束优化；STREAM 的对象是跨请求对偶状态、累计违反与可行性状态 |
| [PDU, NeurIPS 2025](https://arxiv.org/abs/2506.05314) | 原始—对偶约束遗忘 | 原始—对偶只是工具；STREAM 的增量在流式比较器与不可行语义 |
| [OLU, 2025](https://arxiv.org/abs/2505.08557) | 在线学习/删除与 regret | OLU 跟踪统计删除目标；STREAM 比较的是带历史行为约束的可行修复轨迹 |
| [COP, 2026](https://arxiv.org/abs/2610.40483) | 历史 forget 目标的激活投影 | STREAM 不靠累积正交投影，保留显式对偶与约束预算 |
| SEQU / CATA | 顺序漂移回补 / task-vector 冲突 | STREAM 不承诺参数交换或合并，直接处理在线可行域 |

**不可写的 claim**：“首次 primal-dual unlearning”“首次 continual LoRA repair”“非凸 LRM no-regret”。  
**可检验的 claim**：累计约束违反、相对批量可行解的代理 regret、以及不可行输出能否让流式修复更可审计。

## 3. 整体框架

1. 接收流式请求；第 $$t$$ 个请求表示为

   $$
   R_t=(D_{f,t},D_{r,t},\epsilon_t).
   $$

2. 继承上一轮的参数 $$\theta_{t-1}$$、历史约束集合 $$\mathcal H_{t-1}$$ 与对偶状态 $$\lambda_{t-1}$$。
3. 执行可行性检查：可行时 warm-start LoRA 行为修复并进行原始—对偶更新；不可行时输出 `INFEASIBLE`，或求最小让步 $$\delta$$。
4. 审计当前请求并重验全部历史边界，分别记录凸代理上的 regret 与真实 LRM 上的累计约束违反。

forget 侧学习结构有效的非披露行为，retain/aux 侧恢复 target 行为；每个请求的训练/校准/测试探针独立，当前测试行不得进入训练。

## 4. 算法设计

第 $$t$$ 个请求写为 $$R_t=(D_{f,t},D_{r,t},\epsilon_t)$$。继承参数 $$\theta_{t-1}$$，只训练当前 LoRA 增量 $$\Delta_t$$。

### 4.1 历史约束与当前修复

令 $$L_{k,t}$$ 为第 $$k$$ 个历史/当前边界在时刻 $$t$$ 的披露风险，约束为

$$
L_{k,t}(\theta_{t-1}+\Delta_t)\le\epsilon_k,
\qquad k\in\mathcal H_t.
$$

当前行为目标为

$$
\mathcal L_{\mathrm{beh},t}=
\mathcal L_{\mathrm{nd},t}
+\lambda_r\mathcal L_{\mathrm{ret},t}
+\lambda_a\mathcal L_{\mathrm{aux},t}.
$$

原始—对偶更新：

$$
\Delta_t\leftarrow
\operatorname{Proj}_{\mathcal D}
\left[\Delta_t-\eta_t
\nabla_\Delta\left(
\mathcal L_{\mathrm{beh},t}
+\sum_{k\in\mathcal H_t}\lambda_{k,t}L_{k,t}
\right)\right],
$$

$$
\lambda_{k,t+1}=
\left[\lambda_{k,t}+\mu_t(L_{k,t}-\epsilon_k)\right]_+.
$$

### 4.2 在线评价

真实系统记录累计约束违反

$$
V_T=\sum_{t=1}^{T}\sum_{k\in\mathcal H_t}
[L_{k,t}(\theta_t)-\epsilon_k]_+.
$$

在凸代理损失 $$\ell_t$$ 上，相对批量可行比较器定义

$$
\operatorname{Reg}_T=
\sum_{t=1}^{T}\ell_t(\theta_t)-
\min_{\theta\in\bigcap_{t=1}^{T}\mathcal C_t}
\sum_{t=1}^{T}\ell_t(\theta).
$$

只有当批量交集非空且比较器可计算时才报告 $$\operatorname{Reg}_T$$。对非凸 LRM 只报告经验 regret，不声称 $$O(\sqrt T)$$ 或 no-regret。

### 4.3 不可行与最小让步

若声明的凸代理或校准约束交集为空，输出 `INFEASIBLE`；若用户允许让步，则求

$$
\min_{\delta_k\ge0}\ \sum_k w_k\delta_k
\quad\text{s.t.}\quad
L_k(\theta)\le\epsilon_k+\delta_k,
\quad U(\theta)\ge u_{\min}.
$$

让步记录进入下一轮的请求规范；不静默修改原始隐私要求。

## 5. 训练与评测闭环

1. 构造含新实体、重复实体、跨请求保留冲突的请求流，至少包含 10 个请求。
2. 每轮继承上一模型与对偶状态，在独立 calibration 集上更新 LoRA。
3. 每轮重验全部历史 forget scope，而不是只测当前请求。
4. 对比 BLADE 式逐轮离线优化、无对偶继承、COP 式投影、UPBD 单请求独立版。
5. 统一 Hy4 judge；报告 held-out leak、retain utility、collapse、历史最坏违反、计算成本与不可行判定质量。

## 6. 决定性实验与创新点

| 实验 | 配置 | 判读 |
|---|---|---|
| T0 | 构造交集为空的 forget/retain 请求 | 系统必须输出不可行；若仍“成功遗忘”，说明评测掩盖冲突 |
| T1 | STREAM vs BLADE/PDU/COP | 同请求流与预算下比较 $$V_T$$、效用和成本 |
| T2 | 去掉对偶状态继承 | 若累计违反不变，在线对偶无贡献 |
| T3 | 去掉历史重验 | 应暴露旧边界恢复；若无差异，历史约束可能只是摆设 |
| T4 | retain-only 与 target-start | 排除普通 SFT 和初始化对每轮增益的解释 |

**创新点**：

- 将 UPBD 从单次 repair 扩展为带**历史约束状态**的流式遗忘；
- 把累计违反与可行性作为一等输出，而非只报最终平均指标；
- 严格区分凸代理 regret 与非凸 LRM 经验结果；
- 允许显式不可行和最小让步，避免在矛盾请求上制造虚假成功。

## 7. Idea-evaluator 评估

### 致命缺陷

| 缺陷 | 严重度 | 防御 |
|---|---|---|
| F1：BLADE/PDU/OLU 分别覆盖约束、对偶、在线 regret | MAJOR | 只主张流式可行域状态、累计违反与不可行语义；直接做同请求流对照 |
| 非凸 LRM 的 no-regret 与批量比较器可能不可定义 | MAJOR | 定理只限凸代理；交集为空时输出不可行，不计算伪 regret |

无 CRITICAL；若保留“非凸 no-regret”或“冲突时仍有批量可行解”表述，则必须转为 Reject。

### 五维依据

| 维度 | 分 | 依据与提分实验 |
|---|---:|---|
| Higher | 7 | 主要提升历史边界保持与请求可用性，不承诺单点 SOTA |
| Faster | 8 | warm-start LoRA 与对偶状态可避免每轮从零训练；历史审计仍有成本 |
| Stronger | 8 | 直接测跨请求恢复与累计违反；T1/T3 决定实际强度 |
| Cheaper | 7 | 复用边界与行为数据；需维护约束摘要和一定 replay |
| Broader | 8 | 可迁移到在线合规删除、持续安全与多租户策略更新 |

### 生命周期、范式与可行性

- 类别：Data-Intensive / Innovative Technique；预计 10–12 周；每周投入未声明，fit=**Yellow**。
- Paradigm probe：First Principles=No（OCO 是既有工具）；Elephant=Yes（请求冲突与历史恢复）；Technology Cycle=Yes（PEFT 使流式原型可行）；Hamming=Yes。为 possible disruptive potential。
- Compute=Medium-High；Data=High（需有标注语义的请求流）；Engineering=Medium-High；Timeline=Yellow。理论仅限凸代理，避免超出证据。

**最终裁决：Accept with Revisions，pending T0/T1/T2。**  
第一步是实现不可行请求的最小例子；如果系统连这个例子都无法正确拒绝，就不应进入大规模流式实验。

## 8. 失败即转向

- 若可行域比较器无法稳定定义：删除 regret 主张，保留纯经验在线协议。
- 若对偶继承无收益：退化为逐轮 BLADE 式优化，不称 STREAM 新方法。
- 若历史重验频繁失败：说明 UPBD 边界不能继承，应先回到 TCR 研究边界资产。
- 若冲突检测依赖人工真值：只能称 oracle-assisted prototype，不能称自动可行性判定。
