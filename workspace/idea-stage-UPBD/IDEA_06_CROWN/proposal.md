# IDEA 06 · CROWN — 冲突请求的最小让步与边界重验
## Conflict-aware Request Composition with Minimal Concession

| 项 | 值 |
|---|---|
| 排名 | **#6** |
| 五维 (Higher/Faster/Stronger/Cheaper/Broader) | 7 / 7 / 8 / 7 / 8 |
| 均分 | **7.40** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| Novelty gate | **6/10 · PROCEED WITH CAUTION** |
| 状态 | provisional；无实验验证；外部评审 `REVIEW_UNAVAILABLE` |
| 核心类型 | Novel Problem + 统一请求框架 |

---

## 1. 理念与 Problem Anchor

> 多用户删除请求的真正困难不是 adapter 如何平均，而是“谁要求忘、谁要求保”可能互相矛盾；系统必须显式解决语义冲突，再执行 UPBD 式边界继承与行为修复。

已有持续遗忘通常假设 forget 集彼此独立、retain 集固定。真实请求可能是：请求 A 要忘实体 $$X$$，请求 B 要保留与 $$X$$ 共享关系的事实，请求 C 又要求删除该关系的一条推导路径。简单顺序训练或 adapter merge 会隐藏冲突。

CROWN 将每次请求规范化为 forget/protect 范围与效用约束，先判断联合请求是否可满足，再求最小让步，最后执行行为修复并重验所有历史边界。它只承诺**请求规范层的交换/幂等**，不承诺参数更新的交换性。

## 2. 最近邻与真正 delta

| 近邻 | 已占据的内容 | CROWN 的不同轴 |
|---|---|---|
| [CLUE, ICLR 2026](https://arxiv.org/abs/2509.20977) | forget/retain circuit 转 CNF，SAT 定位 conflict neurons | CROWN 的对象是跨请求语义约束、最小让步和组合后重验；SAT 只可作为求解工具 |
| [CATA, 2026](https://arxiv.org/abs/2605.18610) | 持续遗忘中 task-vector 符号冲突聚合 | CROWN 处理请求级 forget/protect 语义冲突，不做权重符号平均 |
| [ICCU, 2026](https://arxiv.org/abs/2605.27138) | refusal rules 的 order-independent union | 规则并集天然交换但不做最小让步；CROWN 面向有状态边界与冲突请求 |
| HAMU | 单请求 forget/retain 不可调和时停止 | CROWN 处理跨请求规范、让步成本与历史组合 |
| SEQU（本地） | 按漂移阈值回补历史能力 | CROWN 的主对象是请求可满足性，不是漂移幅度 |

**不可写的 claim**：“首次用 SAT 做 unlearning”“首次实现顺序无关”“参数组合可交换”。  
**可检验的 claim**：请求级最小让步与组合重验能否减少语义冲突下的隐性行为违规。

## 3. 整体框架

1. 将第 $$i$$ 个请求规范化为

   $$
   r_i=(F_i,K_i,u_i),
   $$

   其中 $$F_i$$、$$K_i$$、$$u_i$$ 分别表示 forget scope、protect scope 与最低效用；作用域闭包只处理声明的关系和规则。
2. 构建联合约束并检测冲突：可满足时继承最近的已验证边界快照；不可满足时输出 `INFEASIBLE` 或进入最小让步流程。
3. 执行 UPBD 行为修复，同时学习非披露 forget 行为与 retain/aux target 行为。
4. 重验全部历史请求，并保留顺序置换审计与让步日志。

框架不把多个 repair adapter 直接相加，而是先合并请求规范，再从已验证快照执行一次受约束修复。这是避免顺序依赖的关键设计。

## 4. 算法设计

### 4.1 请求规范与组合

令 $$r_i=(F_i,K_i,u_i)$$，其中 $$F_i$$ 为 forget scope，$$K_i$$ 为 protect/retain scope，$$u_i$$ 为最低效用。规范化算子 $$\nu$$ 将别名、声明的等价改写与关系映射到规范范围。

请求组合为

$$
R=\bigsqcup_i\nu(r_i)=
\left(\bigcup_i F_i,\bigcup_i K_i,
\{U(\theta)\ge u_i\}_i\right).
$$

在规范层有

$$
\nu(r_i)\sqcup\nu(r_j)=\nu(r_j)\sqcup\nu(r_i),
\qquad \nu(r)\sqcup\nu(r)=\nu(r).
$$

这只是集合表达式的代数性质；参数轨迹仍允许顺序相关，论文必须分别报告规范等价与行为等价。

### 4.2 可满足性与冲突

每个 forget/protect 范围映射到行为约束 $$\mathcal C_i$$。联合可行域为

$$
\mathcal C(R)=\bigcap_i\mathcal C_i.
$$

若 $$F_i\cap K_j\ne\varnothing$$，或其声明的推导闭包导致约束矛盾，则标记冲突；不能简单把同一事实同时遗忘和保留。

### 4.3 最小让步修复

当 $$\mathcal C(R)=\varnothing$$ 且用户允许让步时，求

$$
\min_{\theta,\delta_i\ge0}
\sum_i w_i\delta_i
+\lambda\,\mathrm{Cost}_{\mathrm{repair}}(\theta,\theta_u)
$$

满足

$$
L_i(\theta)\le\epsilon_i+\delta_i,
\qquad
U_i(\theta)\ge u_i-\delta^u_i,
\qquad
\theta=\theta_u+\Delta,\ \Delta\in\mathrm{LoRA}_r.
$$

其中 $$w_i$$ 必须由请求方或策略预先指定，不能由模型自行选择谁被牺牲。若不允许让步，则输出 `INFEASIBLE`，不生成伪成功模型。

### 4.4 组合后重验

从最近已验证快照 $$\theta_S$$ 出发完成一次行为修复后，对每个历史请求计算

$$
V_i=\max_{q\in\mathcal Q_i^{\mathrm{test}}}
[L_i(\theta)-\epsilon_i]_+,
\qquad
V_{\mathrm{order}}=
\operatorname{Var}_{\sigma\in S_n}
[\mathcal M(\theta^{(\sigma)})],
$$

其中 $$V_{\mathrm{order}}$$ 只用于揭示实际参数/行为顺序依赖，不被“规范层可交换”掩盖。

## 5. 训练与评测闭环

1. 构造三类请求流：无冲突、显式冲突、经一跳/多跳关系形成的隐式冲突。
2. 对请求做规范化和冲突检测；冲突请求分别测试拒绝与最小让步模式。
3. 从同一已验证 UPBD 快照执行联合修复，避免直接 merge 多个独立 adapter。
4. 比较顺序 UPBD、CATA 式权重聚合、ICCU 式规则并集、CLUE 式冲突定位。
5. 统一 Hy4 judge；报告 held-out leak、历史最坏违反、retain 效用、让步成本、请求排列方差与计算开销。

## 6. 决定性实验与创新点

| 实验 | 配置 | 判读 |
|---|---|---|
| T0 | 同一事实同时 forget/protect | 必须输出不可行或显式让步；否则核心语义错误 |
| T1 | CROWN vs 顺序 UPBD/CATA/ICCU | 在冲突请求流上比较历史违反、效用和让步成本 |
| T2 | 去掉最小让步，改为固定优先级 | 若相同，最小让步无独立价值 |
| T3 | 去掉组合后历史重验 | 应暴露隐式恢复；无差异则说明重验无贡献 |
| T4 | 请求顺序置换 + retain-only/target-start | 分离规范层交换、参数顺序效应和普通 SFT 归因 |

**创新点**：

- 将 UPBD 统一框架扩展到**有冲突的多请求删除**，而非只处理顺序漂移；
- 严格区分请求规范的交换性与参数轨迹的顺序依赖；
- 以最小让步和不可行输出替代静默的 trade-off 调参；
- 组合的是请求与约束，不是 repair 权重；组合后所有历史边界必须重验。

## 7. Idea-evaluator 评估

### 致命缺陷

| 缺陷 | 严重度 | 防御 |
|---|---|---|
| F1：CLUE 已有 SAT 冲突，CATA 已有持续冲突，ICCU 已有顺序无关 | MAJOR | 只主张请求级最小让步与组合重验；SAT/集合并集只作工具 |
| 有状态继承与严格交换性一般冲突 | MAJOR | 交换性仅限规范化请求集；从共享快照做联合修复；参数顺序方差单独报告 |

无 CRITICAL；若恢复“参数可交换”或“首次 SAT”主张，则转为 Reject。

### 五维依据

| 维度 | 分 | 依据与提分实验 |
|---|---:|---|
| Higher | 7 | 主要减少冲突请求下的隐性违规，不承诺无冲突任务的 SOTA |
| Faster | 7 | 联合修复可避免逐请求重复训练，但规范求解与全历史审计有成本 |
| Stronger | 8 | 直接面向跨请求冲突与历史恢复；T1/T4 决定强度 |
| Cheaper | 7 | 复用 UPBD 边界与行为数据；需构造语义冲突请求，不能给 8 |
| Broader | 8 | 适用于多租户删除、策略冲突、持续合规与知识编辑组合 |

### 生命周期、范式与可行性

- 类别：Innovative Technique / Novel Problem；预计 8–12 周；fit=**Yellow**。
- Paradigm probe：First Principles=Yes（删除请求是可满足性问题）；Elephant=Yes（冲突请求常被忽略）；Technology Cycle=No；Hamming=Yes。具概念潜力，但是否改变领域取决于真实冲突基准。
- Compute=Medium；Data=High（冲突语义标注是瓶颈）；Engineering=Medium-High；Timeline=Yellow。最小原型可只用两条显式冲突请求验证 T0。

**最终裁决：Accept with Revisions，pending T0/T1/T2。**  
优先证明问题存在且现有顺序方法会静默违反请求；若真实请求流没有冲突，CROWN 的问题贡献不成立。

## 8. 失败即转向

- 若冲突只能由人工规则识别：称 oracle-assisted framework，不声称自动语义求解。
- 若 ICCU 规则并集在相同覆盖下更安全便宜：承认无状态规则更优，不保留有状态组合主张。
- 若顺序置换产生巨大行为差异：放弃顺序无关叙事，只保留冲突检测与最小让步。
- 若最小让步退化为手工权重：降级为策略接口，不称学习算法创新。
