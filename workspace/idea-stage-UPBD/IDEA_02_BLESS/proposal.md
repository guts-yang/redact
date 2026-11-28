# IDEA 02 · BLESS — 良性后续学习稳定的边界修复
## Benign-Learning-Stable Boundary Repair

| 项 | 值 |
|---|---|
| 排名 | **#2** |
| 五维 (Higher/Faster/Stronger/Cheaper/Broader) | 7 / 7 / 9 / 7 / 8 |
| 均分 | **7.60** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| Novelty gate | **6/10 · PROCEED WITH CAUTION** |
| 状态 | provisional；无实验验证；外部评审 `REVIEW_UNAVAILABLE` |
| 核心类型 | Novel Method + 部署期鲁棒性框架 |

---

## 1. 理念与 Problem Anchor

> 一个遗忘系统不只应在交付当天不泄漏，还应在用户继续做良性微调、适配新领域或合并正常能力时保持不泄漏。

UPBD 的 retain/aux SFT 证明“修复行为”与“后续重学习”只隔着数据与步数：同一种优化既可能恢复效用，也可能重开遗忘。BLESS 把这一矛盾变成主对象：构造一个在声明的**良性后续学习算子族**下保持的修复集合，而不是再做一个 relearning min-max。

核心不是“模型永远不能被恢复”，而是：

- 修复后为每个 forget 探针保留正的安全裕度；
- 估计每类良性算子对裕度的最大消耗；
- 在预算内选择不会耗尽裕度的 LoRA 修复；
- 对算子组合做逐步重验，超出声明族则明确失效。

## 2. 最近邻与真正 delta

| 近邻 | 已占据的内容 | BLESS 的不同轴 |
|---|---|---|
| [ILU, ICML 2025](https://arxiv.org/abs/2506.01339) | 用 invariance 正则提升下游微调稳健性 | BLESS 构造的是带裕度的**不变集与算子闭包**，不是平均意义上的正则项 |
| [FDCU, 2026](https://arxiv.org/abs/2609.39279) | Fisher/功能干预投影以抗 retraining | FDCU 在遗忘训练时做双掩码；BLESS 修复已有边界并约束未来良性算子 |
| [Margin Calibration, 2026](https://arxiv.org/abs/2607.27836) | forget margin 与重学习攻击预算 | 其内层是对抗恢复；BLESS 的算子只含 retain/aux/无关公共数据，强调部署后闭包 |
| [Jogging the Memory, ICLR 2025](https://arxiv.org/abs/2406.13356) | 良性公开数据微调可恢复遗忘 | BLESS 把这种攻击面转为训练时的可容许修复条件 |
| 本地 RRBU | 内层 $$K$$ 步重学习 unroll | BLESS 不使用 min-max 内层，不与其争“对抗训练”贡献 |

**可辩护定位**：不是“首次研究 relearning”，而是“把 UPBD 修复定义为在良性后续算子族下的不变集成员问题，并报告逐算子最坏裕度”。

## 3. 整体框架

1. 读取继承边界 $$\theta_u$$，并在 forget 探针上估计初始安全裕度。
2. 构造良性后续算子库 $$\Psi$$，覆盖 retain SFT、aux SFT、公共数据适配、LoRA 与全参小步更新；训练族与留出族严格分离。
3. 执行 **Margin-Budgeted Repair**：估计算子敏感方向，限制 LoRA 修复，并保留组合裕度。
4. 恢复 retain/aux 效用，逐一审计后续算子及其组合；超出声明算子族时拒绝给出安全结论。

forget 行为采用结构有效的非披露轨迹，而不是仅拟合固定 IDK。框架同时保留 retain/aux 行为修复，因此优化目标仍然是 UPBD 的两面：不披露与可用性。

## 4. 算法设计

### 4.1 边界裕度与良性算子

令 $$m_\theta(q)$$ 为探针 $$q$$ 上“非披露答案相对最佳披露答案”的 teacher-forced margin，越大越安全。定义

$$
M_{\mathcal Q}(\theta)=\min_{q\in\mathcal Q}m_\theta(q).
$$

良性后续学习算子为

$$
\mathcal T_\psi(\theta)=\operatorname{FineTune}
(\theta;D_\psi,\eta_\psi,K_\psi,\mathrm{param}_\psi),
\qquad \psi\in\Psi,
$$

其中 $$D_\psi$$ 只允许 retain、aux 或独立公共数据，不含 forget 目标答案。

### 4.2 不变集与可容许修复

对声明的算子族与步数预算 $$K$$，定义

$$
\mathcal I_{\tau,\Psi}=
\left\{\theta:
\min_{\psi_1,\ldots,\psi_k\in\Psi,\ k\le K}
M_{\mathcal Q}\big(\mathcal T_{\psi_k}\circ\cdots\circ
\mathcal T_{\psi_1}(\theta)\big)\ge\tau
\right\}.
$$

修复问题为

$$
\max_{\Delta}\ U(\theta_u+\Delta)
\quad\text{s.t.}\quad
\theta_u+\Delta\in\mathcal I_{\tau,\Psi},
\quad \|\Delta\|_F\le\rho.
$$

该集合通常不可精确计算，因此实现采用局部敏感上界与接受检验，不把有限采样称为全称证明。

### 4.3 一阶裕度消耗与修复约束

对单步算子估计

$$
r_\psi(\theta)=M_{\mathcal Q}(\theta)-
M_{\mathcal Q}(\mathcal T_\psi(\theta)),
\qquad
\widehat r_{\mathrm{rob}}=
\max_{\psi\in\Psi_{\mathrm{train}}}r_\psi(\theta)
+\beta\,\mathrm{std}_{\psi}(r_\psi).
$$

若局部模型满足 $$r_\psi\le\widehat r_{\mathrm{rob}}$$，并预留 $$K$$ 步裕度，则取

$$
M_{\mathcal Q}(\theta_u+\Delta)
\ge \tau+K\widehat r_{\mathrm{rob}}.
$$

LoRA 每轮在效用目标上更新后，检查该裕度与投影残差；违反则回退。这给出**在局部敏感假设与声明算子族内**的归纳式安全条件，而非对任意未来微调 universally robust 的承诺。

### 4.4 联合目标

$$
\min_\Delta
\mathcal L_{\mathrm{nd}}(D_f)+
\lambda_r\mathcal L_{\mathrm{ret}}+
\lambda_a\mathcal L_{\mathrm{aux}}
+\lambda_m[\widehat r_{\mathrm{rob}}-\rho_r]_+,
$$

并接受满足效用、结构有效性、边界裕度三重条件的更新。审计另报自由生成泄漏，防止 teacher-forced margin 与真实生成脱节。

## 5. 训练与评测闭环

1. 从 R2MU 与至少一种 GA/NPO 边界出发，建立相同初始裕度。
2. 将良性算子按**数据来源、参数化、学习率、步数**分组；训练族用于估计，完全未见的族用于测试闭包外推。
3. 进行 UPBD 行为修复，同时估计 $$\widehat r_{\mathrm{rob}}$$，只接受保留组合裕度的更新。
4. 测试单算子、重复算子、异质算子组合；报告最坏裕度而非只报平均值。
5. 所有遗忘判断使用 Hy4；保留 collapse、幻觉、拒答稳定性与 retain held-out 效用。

## 6. 决定性实验与创新点

| 实验 | 配置 | 判读 |
|---|---|---|
| T0 | 仅做 UPBD 修复，再施加良性算子 | 若裕度大幅下降，证明问题真实存在 |
| T1 | BLESS vs ILU/RRBU/UPBD/TCR | 在相同后续算子预算下比较最坏 forget margin 与 held-out utility |
| T2 | 去掉裕度预留或敏感约束 | 若结果不变，BLESS 的主机制无贡献，应淘汰 |
| T3 | 留出一整个算子族 | 只对训练族有效则声明为 family-specific，不能称闭包泛化 |
| T4 | retain-only 与 target-start 控制 | 排除普通 SFT 和初始化造成的伪增益 |

**创新点**：

- 把“修复会不会成为下一次重学习”纳入统一遗忘定义；
- 用**逐算子裕度消耗 + 组合预留**替代昂贵的 unroll min-max；
- 同时给出适用算子族、组合深度和失效条件，避免无限鲁棒性话术；
- 与 UPBD 的行为修复自然组合，也可作为 TCR 的时间维扩展。

## 7. Idea-evaluator 评估

### 致命缺陷

| 缺陷 | 严重度 | 防御 |
|---|---|---|
| F1：ILU 已占据“invariance 抗微调”，Margin Cliff 已占据 margin 界 | MAJOR | 只主张良性算子族下的不变集与最坏组合裕度；直接对比 ILU 与 Margin Cliff |
| 有限算子样本不能证明全局闭包 | MAJOR | 证书严格限定算子族与深度；留出整族检验，超出范围返回未知而非安全 |

无 CRITICAL；现有 UPBD 数字只说明直接提示上的行为，不反驳该机制。

### 五维依据

| 维度 | 分 | 依据与提分实验 |
|---|---:|---|
| Higher | 7 | 目标是保持修复效用并降低后续恢复；未有实测，不给 8 |
| Faster | 7 | 不需精确重训，也不必 unroll 全部攻击；但需执行多组良性算子 |
| Stronger | 9 | 主轴是直接优化部署后最坏恢复风险；T3 是强度证据的必要条件 |
| Cheaper | 7 | 复用已有数据与边界；新增算子 rollout 成本，不宣称零成本 |
| Broader | 8 | 可接任意上游遗忘，并可迁移到持续对齐、模型发布审计 |

### 生命周期、范式与可行性

- 类别：Frontier Exploration / Innovative Technique；预计 8–10 周；每周投入未声明，fit=**Yellow**。
- Paradigm probe：First Principles=Yes（遗忘被定义为后续算子下的不变性）；Elephant=Yes（良性重学习）；Technology Cycle=Yes（PEFT 让算子扫描可行）；Hamming=Yes。具概念颠覆潜力，但完全待验证。
- Compute=Medium-High；Data=Medium；Engineering=Medium；Timeline=Yellow。最大风险是局部裕度界在实际自由生成中空洞或失真。

**最终裁决：Accept with Revisions，pending T0/T2/T3。**  
优先动作：先证明 UPBD 修复后确有良性算子敏感差异，再决定是否需要不变集机制；若普通 UPBD 已经对所有算子稳定，BLESS 没有问题可解。

## 8. 失败即转向

- 若一阶 $$r_\psi$$ 与实际恢复不相关：停止“裕度证书”主张，退化为纯攻击评测。
- 若只在训练算子族有效：改写成 family-specific robustness，不使用 closure。
- 若 RRBU 以相同或更低成本占优：承认 unroll 更强，不保留 BLESS 方法线。
- 若修复后 leak 下降仅来自固定拒答风格：判为行为模板，不作为遗忘证据。
