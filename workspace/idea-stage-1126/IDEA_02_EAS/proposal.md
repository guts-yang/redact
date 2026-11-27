# IDEA 02 · EAS — 纠缠度自适应干预调度
## Entanglement-Adaptive Scheduling for Concept-Level Unlearning

| 项 | 值 |
|---|---|
| 排名 | **#2** |
| 五维 (H/F/S/C/B) | 8 / 8 / 8 / 8 / 7 |
| 均分 | **7.8** |
| 裁决 | **Strong Accept**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU forget01/05/10（覆盖不同纠缠度） |
| 基线 | 统一强度 R2MU / GA+KL、CUD 难度分层（仅预测侧） |

---

## 1. Problem Anchor（一句话故事）

> 同一个遗忘算法对所有样本用同一强度：对「孤立事实」用力过猛伤到 retain 效用，对「高纠缠概念」用力不足又漏；EAS 用每个目标的**知识纠缠度**决定干预强度与层级，把「预测泄漏」升级成「按难度分配算力」的闭环。

锚定问题：**LRM 遗忘的干预强度应是样本的函数，而不是超参数**。本仓库 `docs/LCES_FULL_RESULTS.md` 的 Interpretation 也指向「concept-level, utility-constrained unlearning, not token-level banning」。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| SKeB / Obliviate (arXiv:2510.25732) | 知识纠缠度**因果预测**遗忘后泄漏 | 只预测，不据此调度干预 |
| CUD (ACL 2026 Findings, arXiv:2601.09624) | 电路引导的 pre-unlearning **难度指标**（易/难样本） | 预测 / 分层，不做算法选择或强度闭环 |
| RUM / HAMU（历史记忆） | 元算法 / 难度感知的碎片前作 | 未统一成连续强度函数 |

**Delta**：把 $\kappa(x)$（纠缠度）与 $d(x)$（电路难度）作为 **per-sample 干预强度 $\alpha(x)$** 的输入，输出「预算–效用 Pareto 前沿」。这正是「难度驱动调度科学」方向的已登记空白（预测 → 闭环调度）。据不完全检索，未见在 LRM CoT 上做连续强度路由的工作。

---

## 3. 方法（LaTeX）

**3.1 纠缠度 / 难度特征**
$$
\kappa(x)\;=\;\frac{1}{|N_1(x)|}\sum_{q\in N_1(x)} \mathrm{sim}\big(\phi(x),\phi(q)\big)\cdot \mathrm{cooccur}(e_f, q),
\qquad
d(x)\;=\;\mathrm{depth}\big(\text{circuit}(x)\big)
$$
其中 $N_1(x)$ 为知识图中一跳邻居；$\kappa$ 大表示目标与保留知识高度纠缠，$d$ 大表示电路深、难遗忘。

**3.2 连续干预强度**
$$
\alpha(x)\;=\;\sigma\!\left(w^\top\big[\kappa(x),\,d(x),\,1\big]\right)\;\in\;[0,1],
\qquad
\mathcal{L}_{\text{repr}}(x)\;\text{(表示级, 重)},\ \ \mathcal{L}_{\text{out}}(x)\;\text{(输出级, 轻)}
$$
$$
\boxed{\;
\mathcal{L}_{\text{EAS}}(x)
=
\alpha(x)\,\mathcal{L}_{\text{repr}}(x)
\;+\;\big(1-\alpha(x)\big)\,\mathcal{L}_{\text{out}}(x)
\;+\;\lambda(\kappa)\,\mathrm{KL}\big(p_\theta(\cdot\mid x_r)\,\|\,p_{\theta_0}(\cdot\mid x_r)\big)
\;}
$$
其中 $\lambda(\kappa)=\lambda_0(1+\kappa)$：纠缠越高，retain 锚越强。

**3.3 预算约束下的 Pareto 目标**
$$
\max_{\theta,w}\ \ \mathrm{Utility}(\theta)
\quad\text{s.t.}\quad
\sum_{x\in D_f}\mathrm{cost}\big(\alpha(x)\big)\;\le\;B,
\qquad
\mathrm{Leak}(\theta)\;\le\;\varepsilon
$$
拉格朗日形式：
$$
\mathcal{L}=\mathcal{L}_{\text{EAS}}+\beta\Big[\sum_x \mathrm{cost}(\alpha(x))-B\Big]_+ + \gamma\big[\mathrm{Leak}-\varepsilon\big]_+
$$
$\mathrm{cost}(\alpha)=\alpha\cdot c_{\text{repr}}+(1-\alpha)\cdot c_{\text{out}}$，表示级干预（需前向 + patching）显著贵于输出级 SFT。

**3.4 与 LCES 失败的对应**
token 级无差别压制 $\Rightarrow$ 幻觉/坍塌（本仓库实测）；EAS 用 $\alpha$ 把「是否压到表示层」变成**可学习的样本级决策**，避免对低纠缠样本做重干预。

---

## 4. R-TOFU 接口

- 用 `forget01 ⊂ forget05 ⊂ forget10` 构造不同纠缠度分布；`config/kb_resolutions/forget10.json` 与 `knowledge_banks/` 提供实体–声明图，可据此算 $\kappa$。
- 指标：Leak、MU、Retain ROUGE-L、collapse，加本方法核心 **预算–效用曲线** $\{(B,\ \mathrm{Utility}^\*(B))\}$ 与 $\alpha(x)$ 分布直方图。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | 全样本统一表示级 / 全样本统一输出级 | 各自在一端失效（collapse 或 leakage） |
| T1 | EAS（$\alpha(x)$ 由 $\kappa,d$ 决定） | 在同等预算 $B$ 下 Utility 高于任一 T0 ≥ 0.05，且 Leak ≤ 0.05 |
| T2 | 消融：$\alpha$ 随机（打乱 $\kappa$ 与样本的配对） | T1 优势消失 → 证明「纠缠度驱动」而非「混合两种损失」 |
| T3 | 消融：固定 $\lambda$（去掉 $\lambda(\kappa)$） | 高纠缠 split 上 retain 掉 > 0.1 |

**决定性实验**：T1 vs T2（隔离调度信号本身）。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 8（机制分） | 把强度配给需要的样本，机制上同时改善 Leak/Utility；待 T1 |
| Faster | 8（机制分） | 低纠缠样本走轻量输出级，总梯度/前向开销下降；相对 exact-retrain 优势明确 |
| Stronger | 8（机制分） | 对 split 间纠缠度分布漂移鲁棒；待 forget01/05/10 全验证 |
| Cheaper | 8（机制分） | 显式预算约束 $B$，用更便宜的路径覆盖多数样本 |
| Broader | 7（机制分） | 调度框架可套到任意两档干预；未跨域 |

**裁决：Strong Accept**（四维 ≥8、零 CRITICAL）。机制分 → *pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. $\kappa(x)$ 依赖知识图完整性；R-TOFU 是虚构作者，KB 由 `kb_resolutions` 生成，可能低估纠缠 → 需与 teacher-forced NLL 难度做一致性校验。
2. $\alpha(x)$ 可能退化成常数（所有样本同档）→ 用 $\mathrm{Var}[\alpha]$ 作健康指标，过低即触发重新参数化。
3. 与仓库 `N07 效用 Pareto 门` 相邻：N07 是**硬阈值准入**，EAS 是**连续强度路由**，二者不重复但论文需主动划界。
4. 本地无 GPU / 无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/eas/entanglement.py`（复用 `config/kb_resolutions/` 与 `knowledge_banks/`）；训练在 `scripts/upbd/train_upbd.py` 骨架上把单损失替换为 $\mathcal{L}_{\text{EAS}}$。冻结协议：fp32 / seeds 1001, 2002 / 400 forget + 400 retain + 217 aux。
