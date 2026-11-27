# IDEA 09 · LAYER — 跨解码策略一致的擦除深度定位与证书
## Decode-Consistent Erasure-Depth Localization with a Structural Certificate

| 项 | 值 |
|---|---|
| 排名 | **#9（第二轮 1127）** |
| 五维 (H/F/S/C/B) | 7 / 6 / 8 / 7 / 8 |
| 均分 | **7.20** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU `forget01/05/10`（4 解码策略），DeepSeek-R1-Distill-Llama-8B |
| 基线 | 首轮 DISCO、UDS / Activation Patching for Depth、首轮 ORTHO |

---

## 1. Problem Anchor（一句话故事）

> Multilingual Unlearning (ICML 2026) 证明：遗忘若只发生在「后段语言特化解码层」，就是浅层抑制、可被 steering 逆转；UDS 用 activation patching 测「遗忘深度」也只给诊断。DISCO（首轮）要求跨解码**输出**不变，但没问「擦除发生在哪一层、各策略下是否一致」。LAYER 把「擦除深度」本身做成跨解码策略一致的结构量：要求真正的知识擦除在共享语义层、且对 Default/Less/Zero/Summary 四种解码都落在同一层带，并给出可报告的深度证书。

锚定问题：**遗忘是否「足够深」不能只看输出，而应由跨解码策略一致的擦除深度（结构层位置）来判定与优化**。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| Multilingual Unlearning (ICML 2026) | 层-wise 定位「遗忘在后段」+ steering 可逆 | 诊断；不把深度写成训练目标、不要求跨解码一致 |
| UDS / Activation Patching for Depth (2026, 二手) | 白盒激活修补测遗忘深度 | 评测工具；非训练目标 |
| DISCO（首轮 #4） | 跨解码策略的**输出** min-max | 行为层不变；不定位/约束结构层位置 |
| ORTHO（首轮 #5） | 推理子空间正交保护（单一子空间） | 训练期投影；非「深度定位 + 跨策略一致性证书」 |
| CRISP (ACL 2026) | 选中层 SAE 特征焊死 | 固定层选择，无跨解码一致判据 |

**Delta**：定义擦除深度 $\mathrm{Depth}_\theta(p)$ 为「使泄漏回升所需的最浅修补层」，要求它在解码策略集合上一致且足够深（小/前中段层号），并发布 $\mathrm{DC}$ 证书。据现有检索，未见把「跨解码一致的擦除深度」形式化为主判据与训练目标的工作。

**与首轮 10 个的划界**：最像 `DISCO`（跨解码）与 `ORTHO`（结构层/子空间）。不同轴：DISCO 约束的是**输出分布**（行为层）；LAYER 约束的是**擦除发生的层位置**（结构层）及其跨策略一致性。ORTHO 是梯度投影，LAYER 是深度定位 + 证书。

---

## 3. 方法（LaTeX）

设层 $\ell=1..L$，解码策略 $p\in\mathcal P=\{$Default, LessThink, ZeroThink, SummaryThink$\}$。激活修补：把 $\ell$ 层 forget 轨迹的步表示换回 $\theta_0$ 的表示，测泄漏回升。

**3.1 擦除深度（核心量）**
$$
\mathrm{Depth}_\theta(p)
=\min\Big\{\ \ell:\ \mathrm{Leak}\big(\theta;\ \mathrm{do}(\phi^{(\ell)}\leftarrow\phi_{\theta_0}^{(\ell)});\ p\big)\ \ge\ \varepsilon\ \Big\},
$$
即「从哪一层往回修补，知识就回来了」——修补层越浅（$\ell$ 小）说明抑制越靠近解码端、越浅；目标把 $\mathrm{Depth}$ 推深（要求更靠前中段、共享语义层才擦得动）。

**3.2 跨策略深度一致性**
$$
\mathrm{DC}_\theta
=\max_{p,p'\in\mathcal P}\big|\,\mathrm{Depth}_\theta(p)-\mathrm{Depth}_\theta(p')\,\big|,
\qquad \text{目标 } \mathrm{DC}\to 0 \ \wedge\ \min_p\mathrm{Depth}_\theta(p)\le \ell_{\text{shared}} .
$$
若 ZeroThink 下 $\mathrm{Depth}$ 很浅而 Default 下很深，说明「少想就漏」正是因为擦除没沉到共享层（与 R-TOFU 复活现象一致）。

**3.3 深度引导训练目标**
$$
\mathcal L_{\text{LAYER}}
=\sum_{p\in\mathcal P}\sum_{\ell\le \ell_{\text{shared}}}
\Big[\,\mathrm{Leak}\big(\theta;\mathrm{do}(\phi^{(\ell)}\!\leftarrow\!\phi_{\theta_0}^{(\ell)});p\big)-\varepsilon\Big]_+
\;+\;\lambda\,\mathrm{DC}_\theta^{\text{(soft)}}
\;+\;\mu\,\mathrm{KL}(p_\theta\|p_{\theta_0})_{D_r},
$$
其中第一项惩罚「浅层一补就漏」，强制擦除下沉；第二项用各策略 $\mathrm{Depth}$ 的方差做软一致约束。$\mathrm{DC}^{\text{(soft)}}=\mathrm{Var}_p[\widehat{\mathrm{Depth}}_\theta(p)]$。

**3.4 证书（机制推断，待验证）**
$$
\mathrm{Cert}(\delta)
=\Pr_{p\in\mathcal P}\Big[\ \mathrm{Depth}_\theta(p)\le \ell_{\text{shared}}\ \wedge\ \mathrm{DC}_\theta\le \delta\ \Big],
$$
报告深度-一致性证书，作为「非表面抑制」的结构层证据（区别于行为层 LEAK）。

参数更新走 LoRA，base 冻结。

---

## 4. R-TOFU 接口

- 数据：`forget10/task_id=1` 400 forget + retain；4 解码策略复用 `redact/inference.py` 前缀。
- 激活修补：复用 CSDA 的 `step_attribution.py` / `patch_layer` 骨架，但扫描 $\ell$ 而非 $t$。
- 指标：核心 $\mathrm{Depth}_\theta(p)$、$\mathrm{DC}$、$\mathrm{Cert}$、R-TOFU step-wise、LEAK、steering 恢复率（借 ICML 2026）、collapse、Retain ROUGE-L。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | R2MU / DISCO | $\mathrm{DC}>0$ 且 ZeroThink 下 $\mathrm{Depth}$ 显著更浅（复活对应浅抑制） |
| T1 | LAYER（深度引导） | $\min_p\mathrm{Depth}\le \ell_{\text{shared}}$ 且 $\mathrm{DC}\le 1$ 层 且 steering 恢复率降 $\ge 0.3$ |
| T2 | 消融：去掉深度项，只留 DISCO 输出 min-max | $\mathrm{Depth}$ 仍浅、steering 恢复率不降 → 证明「结构层深度」必要 |
| T3 | 消融：单策略（Default）算深度 | ZeroThink 下 $\mathrm{DC}$ 回弹 → 证明「跨策略一致」必要 |

**决定性实验**：T1 vs T2（隔离深度轴）与 T1 vs T3（隔离跨策略轴）。若均无差异，核心机制被驳倒 → 转向。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 7（机制分） | 把「擦除深度」做成可优化量；但深度与泄漏非严格单调，上限保守 |
| Faster | 6（机制分） | 层扫描 $\times$ 策略扫描开销高；相对 exact-retrain 仍快 |
| Stronger | 8（机制分） | 对 ZeroThink 复活 + steering 可逆两种浅抑制证据做结构层闭环；待跨 split |
| Cheaper | 7（机制分） | 复用 patching 钩子，无新标注；算力主要在扫描 |
| Broader | 8（机制分） | 「深度证书」可迁移到任意分层可修补的推理/多语模型 |

**裁决：Accept with Revisions**。降档理由：① F1=MAJOR（Multilingual Unlearning 层定位 + UDS 深度诊断近邻强，须主动划界「训练目标/证书 vs 诊断」）；② $\ell_{\text{shared}}$ 的定义与层扫描算力是前置工程风险（F7）。机制分 → *pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. 激活修补的「泄漏回升」未必单调对应真知识（可能是修补扰动副作用）→ 用控制修补（无关层/打乱）校准并报对照。
2. 层扫描 $\times$ 4 策略成本大 → 用二分搜索定位 $\mathrm{Depth}$ 而非全扫，并缓存激活。
3. 深度推得太前可能伤共享语义能力 → 用 $\mu$ retain 锚与 Retain ROUGE-L 兜底。
4. 本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/layer/depth_scan.py`（层修补 + $\mathrm{Depth}$ 二分定位）与 `scripts/layer/depth_loss.py`；复用 CSDA patching 与 DISCO 策略批。冻结：fp32 / seeds 1001, 2002。
