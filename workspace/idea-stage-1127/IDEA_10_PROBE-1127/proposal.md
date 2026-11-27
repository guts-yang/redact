# IDEA 10 · PROBE — CoT 步表示探针对抗的不可解码擦除
## Probe-Adversarial Undecodable Erasure at CoT Step Representations

| 项 | 值 |
|---|---|
| 排名 | **#10（第二轮 1127）** |
| 五维 (H/F/S/C/B) | 8 / 6 / 8 / 7 / 7 |
| 均分 | **7.20** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU `forget01/05/10`，DeepSeek-R1-Distill-Llama-8B |
| 基线 | PARS (arXiv:2609.36612)、首轮 RRBU / CFA |

---

## 1. Problem Anchor（一句话故事）

> PARS 给出理论：输出抑制与表征擦除可分离 —— 模型嘴上不说了，权重里仍读得出。于是「LEAK=0」不等于「忘干净」。现有 LRM 遗忘几乎都在压输出/轨迹，没人把「从 CoT 步表示里还能不能解码出 forget 实体」当成训练目标。PROBE 在步级隐藏表示上挂一个对抗探针，要求无论怎么探测都读不出 $e_f$，把「不可解码」写成遗忘的主目标。

锚定问题：**LRM 遗忘应在 CoT 步表示层达到「不可解码」，而不只是行为层不泄漏；否则知识仍驻留、可被下游探针/微调/steering 取回**。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| PARS / Hidden-State Leakage (arXiv:2609.36612, 2026) | 证明输出抑制 ≠ 表征擦除可分离；提 probe-adversarial 目标 | 普通 LLM、句/序列级表征；未下沉到 LRM **CoT 步级**，未接 R-TOFU |
| RRBU（首轮 #3） | 抗 relearning 恢复攻击 | 攻击在参数空间；非「步表示探针可读性」 |
| CFA（首轮 #8） | 风格特征侧信道对抗（GRL） | 审计「输出风格 → 成员身份」；非「步表示 → forget 实体」 |
| CRISP (ACL 2026) | SAE 特征激活焊死 | 特征级抑制，但目标是生成行为；非 probe 不可解码证书 |

**Delta**：把 probe-adversarial 从「序列级是否擦除」下沉到 **LRM 的每一个 CoT 步表示**，用步级探针 AUC 作主判据，并与行为层 LEAK 联合作为「真擦除」证据。据现有检索，未见在 LRM CoT 步级做 probe-adversarial 不可解码擦除的工作。

**与首轮 10 个的划界**：最像 `RRBU`（抗恢复）与 `CFA`（对抗探针）。不同轴：RRBU 抗**权重 relearning**；PROBE 抗**冻结步表示上的轻量探针**。CFA 的探针读「输出风格→成员」，PROBE 的探针读「步表示→forget 实体」，且作用在轨迹内部每一步。

---

## 3. 方法（LaTeX）

设 $\phi_\theta^{(\ell)}(r_t)$ 为步 $t$ 的隐藏表示，$e_f$ forget 实体，探针 $q_\psi:\mathbb R^d\to[0,1]$ 试图从步表示预测「该步是否属于 forget 作用域 / 是否编码 $e_f$」。

**3.1 步级探针与不可解码量**
$$
\hat m_t=q_\psi\big(\phi_\theta^{(\ell)}(r_t)\big),\qquad
\mathrm{AUC}_\theta^{(\ell)}=\mathrm{AUC}\Big(\{\hat m_t\},\ \{m_t^{\text{true}}\}\Big),
$$
$$
\mathrm{UDec}_\theta=\max_{\psi}\ \mathrm{AUC}_\theta^{(\ell)}\in[0.5,1],
\qquad \text{目标 } \mathrm{UDec}\to 0.5\ (\text{不可解码}).
$$

**3.2 对抗不可解码目标（核心，min-max / GRL）**
$$
\min_\theta\ \max_\psi\
\Big[\,\mathcal L_{\text{probe}}\big(q_\psi(\phi_\theta(r_t)),\,m_t\big)-\gamma\,\Big]_+
\;-\;\text{（对 }\psi\text{）}\ \lambda\,\mathcal L_{\text{probe}}
\;+\;\rho\,\mathrm{KL}(p_\theta\|p_{\theta_0})_{D_r}
\;+\;\mu\sum_{t:\,m_t=0}\|\phi_\theta(r_t)-\phi_{\theta_0}(r_t)\|_2^2 .
$$
实现用梯度反转层（GRL）：对 $\psi$ 最小化探针损失、对 $\theta$ 最大化之，使步表示「信息上不含 $e_f$」。第三项保 retain，第四项保非 forget 步结构（避免 collapse）。

**3.3 行为-表征双证据判据（机制推断，待验证）**
$$
\mathrm{LEAK}_\theta\le\varepsilon_{\text{beh}}
\quad\wedge\quad
\mathrm{UDec}_\theta\le 0.5+\varepsilon_{\text{repr}},
$$
并报告二者在恢复攻击（relearning / steering）后的**联合回升**：真擦除应两个都不回升，行为抑制则 $\mathrm{UDec}$ 先升。定义
$$
\mathrm{Gap}_\theta=\mathrm{UDec}_\theta-\mathrm{LEAK}_\theta
$$
作为「浅层抑制残差」诊断量。

参数更新走 LoRA，base 冻结。

---

## 4. R-TOFU 接口

- 数据：`forget10/task_id=1` 400 forget + retain；步切分复用 `contextual_window.py`（步 = CoT 句子）。
- 探针：$\ell$ 取中层（借 CRISP/ICML 2026 的「共享语义层」结论），线性 + MLP + n-gram 三类探针取最坏 $\mathrm{AUC}$（复用 CFA 的 `linear_probe_auc` 骨架）。
- 指标：核心 $\mathrm{UDec}$、$\mathrm{Gap}$、R-TOFU step-wise、LEAK、relearning/steering 后双回升、collapse、Retain ROUGE-L。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | R2MU / CoT-only GA | $\mathrm{LEAK}\approx 0$ 但 $\mathrm{UDec}\ge 0.8$（输出抑制、表征仍可读，复现 PARS） |
| T1 | PROBE（步级对抗不可解码） | $\mathrm{UDec}\le 0.55$ 且 $\mathrm{LEAK}\le 0.05$ 且 relearning 后双回升 $\le 0.10$ |
| T2 | 消融：序列级探针（等价 PARS），不按步 | 某中间步 $\mathrm{AUC}$ 仍高 → 证明「步级」必要 |
| T3 | 消融：去掉 GRL 对抗项，只压 LEAK | $\mathrm{UDec}$ 不降 → 证明「不可解码目标」必要 |

**决定性实验**：T1 vs T3（隔离对抗目标）与 T1 vs T2（隔离步级粒度）。若均无差异，核心机制被驳倒 → 转向。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 8（机制分） | 把「真擦除」写成可优化目标，机制上逼近不可逆；待 T1 |
| Faster | 6（机制分） | 每步挂探针 + 对抗更新；相对 exact-retrain 仍快 |
| Stronger | 8（机制分） | 对 relearning / steering / 下游探针恢复鲁棒；待跨 split |
| Cheaper | 7（机制分） | 探针轻量、无新标注；复用 CFA 探针骨架 |
| Broader | 7（机制分） | 「不可解码」可迁移到任意白盒可取的推理模型；未跨域 |

**裁决：Accept with Revisions**。降档理由：① F1=MAJOR（PARS 近邻强，须主动划界「LRM 步级 + R-TOFU 联合证据 vs 普通 LLM 序列级」）；② 探针族选择可能低估最坏可读性（F6 可验证性，须报三类探针取 max）；③ 白盒表示可取是前提（F7）。机制分 → *pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. 探针能力有限，$\mathrm{UDec}\approx 0.5$ 可能只说明「线性探针读不出」而非「信息不在」→ 用 MLP / n-gram / 微调探针取最坏值，声明威胁模型边界，不夸成「信息论擦除」。
2. GRL 对抗训练不稳定 → 用 CFA 已验证的「分离优化器 + detached 特征」协议。
3. 步级不可解码可能伤正常步表示的语义 → 用 $\mu$ 结构保持与 Retain ROUGE-L 兜底。
4. 本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/probe/step_probe.py`（步级三类探针）与 `scripts/probe/undecodable_loss.py`（GRL 对抗）；复用 CFA 的 `cfa_hook.py` 探针骨架与 CSDA 步切分。冻结：fp32 / seeds 1001, 2002。
