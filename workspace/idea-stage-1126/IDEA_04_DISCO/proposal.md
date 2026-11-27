# IDEA 04 · DISCO — 解码不变的参数级抑制
## Decoding-Invariant Suppression of Context-free Revival

| 项 | 值 |
|---|---|
| 排名 | **#4** |
| 五维 (H/F/S/C/B) | 8 / 7 / 9 / 7 / 7 |
| 均分 | **7.6** |
| 裁决 | **Strong Accept**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU（原生含 DefaultThink / ZeroThink / LessThink 解码设定） |
| 基线 | R-TOFU Reasoned IDK、STaR MCS（评估）、CoT-only GA |

---

## 1. Problem Anchor（一句话故事）

> R-TOFU 发现一个反直觉漏洞：正常 CoT 下看似忘了，一旦用 ZeroThink / LessThink 跳过推理，原始事实又冒出来。这说明知识**还在参数里**，只是被 CoT 出口盖住。DISCO 把「裸 prompt / 任意解码前缀下都不泄漏」写成 min-max 训练目标，逼迫参数级而非出口级擦除。

锚定问题：**LRM 遗忘必须对解码策略不变，而不是对某一条 CoT 轨迹有效**。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| R-TOFU (EMNLP 2025) | **发现** ZeroThink/LessThink 复活漏洞 | 只作为 limitation，无训练修复 |
| STaR (AAAI 2026) | **MCS 多解码一致性评分**（评估指标） | 评估，不做 min-max 训练目标 |
| PARS (arXiv:2609.36612) | 证明输出抑制与表征擦除可分离，做 probe-adversarial | 对抗的是 hidden-state probe，非「解码前缀」族的最坏情况 |
| 仓库 `N06 状态不变遗忘` | 让生成**状态**（answered/collapsed）同态 | 对象不同：DISCO 管「解码策略 + 裸 prompt 复活」，不管生成状态同态 |

**Delta**：把解码策略集合 $\mathcal{P}$ 显式写成对抗者，$\min_\theta\max_{p\in\mathcal{P}}$；并引入**上下文无关探针**（bare prompt）作为「参数级残留」的可微信号。据现有检索，MCS 与 ZeroThink 发现均停留在评估层，未见其训练闭环。

---

## 3. 方法（LaTeX）

**3.1 解码策略族**
$$
\mathcal{P}=\big\{\ \text{DefaultThink},\ \text{ZeroThink},\ \text{LessThink},\ \text{Summary},\ \text{bare-prompt}\ \big\},
\qquad p\ \text{对应前缀/预算}\ \pi_p
$$

**3.2 min-max 目标**
$$
\boxed{\;
\min_\theta\ \max_{p\in\mathcal{P}}\ \ \mathcal{L}_{\text{leak}}(\theta;\,p)
\;+\;\lambda\,\mathcal{L}_{r}(\theta)
\;}
$$
$$
\mathcal{L}_{\text{leak}}(\theta;p)
=
\mathbb{E}_{x\in D_f}\Big[-\log\big(1-p_\theta(e_f\mid x,\pi_p)\big)\Big]_{\text{margin}}
\quad\text{或}\quad
\max\big(0,\ \mathrm{logit}(e_f\mid x,\pi_p)-m\big)
$$

**3.3 上下文无关探针（核心机制）**
bare prompt 即剥掉全部 CoT 前缀，直接问 $x_{\text{bare}}$：
$$
\mathcal{L}_{\text{param}}(\theta)
=
-\mathbb{E}_{x\in D_f}\log\sigma\!\Big(-\big[\mathrm{logit}_\theta(e_f\mid x_{\text{bare}})-\tau\big]\Big)
$$
定义**分离量**：
$$
\Delta(\theta)
=
\underbrace{\mathrm{Leak}_{\text{CoT}}(\theta)}_{\approx 0\ \text{时看似成功}}
-
\underbrace{\mathrm{Leak}_{\text{bare}}(\theta)}_{\text{参数级残留}}
$$
目标是把 $\Delta\to 0$（并同时把两项都压到 $\le\varepsilon$），即「不看 CoT 出口」也成立。

**3.4 与 CSDA 的组合（可选）**
$$
\mathcal{L}=\mathcal{L}_{\text{CSDA}}+\beta\,\max_{p\in\mathcal{P}}\mathcal{L}_{\text{leak}}(\theta;p)+\gamma\,\mathcal{L}_{\text{param}}(\theta)
$$
CSDA 断承重步、DISCO 防绕过路径，二者互补（见 CSDA §7.2）。

**3.5 评估：解码一致性下界**
$$
\mathrm{MCS}_{\text{worst}}(\theta)=\min_{p\in\mathcal{P}}\big(1-\mathrm{Leak}(\theta;p)\big)\ \ge\ 1-\varepsilon
$$

---

## 4. R-TOFU 接口

- 直接复用 R-TOFU 自带的四策略解码协议；`redact/inference.py` / `inference_regen.py` 已含 `</think>` 边界与前缀构造，可生成 bare-prompt 分支。
- 指标：每策略 AFE/CFE/LEAK + 核心 $\mathrm{MCS}_{\text{worst}}$ 与 $\Delta(\theta)$（裸 prompt 复活率）。
- split：forget01/05/10 各跑，验证 $\Delta$ 随遗忘比例的变化。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | Reasoned IDK / CoT-only GA（单解码训练） | 复现 ZeroThink 下 LEAK 显著 > DefaultThink |
| T1 | DISCO（min-max + bare 探针） | $\mathrm{MCS}_{\text{worst}}\ge 0.95$ 且 $\Delta\le 0.05$，Retain ROUGE-L 不掉 > 0.02 |
| T2 | 消融：去掉 bare 探针项（$\gamma=0$） | $\Delta$ 回弹 → 证明参数级信号必要 |
| T3 | 消融：$\mathcal{P}$ 只留 DefaultThink | 退回 T0 表现 → 证明解码族必要 |

**决定性实验**：T1 vs T2（隔离「上下文无关探针」）。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 8（机制分） | 把「看似成功」中最脆弱的一支补上；待 T1 |
| Faster | 7（机制分） | 相对 exact-retrain 快；$|\mathcal{P}|$ 小、前缀可批量，但需多策略前向 |
| Stronger | 9（机制分） | **主轴**：对解码分布漂移 / ZeroThink 攻击族鲁棒，直接封闭 R-TOFU 已知漏洞；待 T3 |
| Cheaper | 7（机制分） | 复用既有解码协议与 CSV，无新数据 |
| Broader | 7（机制分） | 「解码不变」可迁移到任意带可控推理前缀的 LRM |

**裁决：Strong Accept**（S=9、H=8，零 CRITICAL）。机制分 → *pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. $\max_{p}$ 可能收敛到「所有策略都拒答」的平凡解（与 LCES 的 0.485 幻觉 / 0.24 拒答同类）→ 必须联合 retain 锚 + collapse 监控，并把「干净退出」而非「空白拒答」设为目标。
2. bare prompt 上的 logit margin 与真实生成泄漏未必单调对齐 → 用 teacher-forced NLL 与生成泄漏做一致性校验（借鉴 N12 know-check 的冻结模板纪律）。
3. 与 `N06` 状态不变、`N10` 状态位移相邻：DISCO 明确只管解码/前缀族，不管生成状态分解，论文需划界。
4. 本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/disco/decode_family.py`（复用 `redact/inference.py` 的 `</think>` 切分与四策略前缀）；训练在 `scripts/upbd/train_upbd.py` 骨架上加 min-max 与 bare 探针项。冻结：fp32 / 1024 tokens / seeds 1001, 2002。
