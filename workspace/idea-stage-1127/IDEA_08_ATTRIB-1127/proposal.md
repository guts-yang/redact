# IDEA 08 · ATTRIB — 步级关键词归因与恢复-破坏两难阻断
## Step-Level Keyword Attribution and Recovery–Destruction Dilemma Closure

| 项 | 值 |
|---|---|
| 排名 | **#8（第二轮 1127）** |
| 五维 (H/F/S/C/B) | 8 / 7 / 8 / 8 / 6 |
| 均分 | **7.40** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU `forget01/05/10` + 关键词强调改写集，DeepSeek-R1-Distill-Llama-8B |
| 基线 | UnPact (arXiv:2509.24675)、首轮 CSDA / RRBU |

---

## 1. Problem Anchor（一句话故事）

> UnPact 用 prompt attribution 揭示：所谓「遗忘」常常只是破坏了模型对**关键词的关注**，而知识能被「把关键词强调回去」的提问恢复 —— 并给出 Recovery Rate / Destructive Rate 两难：压得越狠越伤效用，压得越轻越易恢复。但这只是审计发现。ATTRIB 把「关键词归因」下沉到 **CoT 步级**，并把两难写成训练闭环：要求「关键词强调改写」与「原题」在步级泄漏上不可区分。

锚定问题：**LRM 遗忘不能只是破坏关键词关注；必须在步级上让「强调关键词」这一最廉价恢复手段失效，且同时不触发破坏性过遗忘**。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| UnPact (arXiv:2509.24675, 2025) | prompt attribution 揭示遗忘=破坏关键词关注；Recovery/Destructive Rate 两难 | 普通 LLM、答案级审计；不产训练方法、不下沉到 CoT 步 |
| CSDA（首轮 #1） | activation patching 的步级因果贡献 $c_t$ | 因果归因对象是「步表示」；非「关键词注意力」；不做恢复-破坏闭环 |
| RRBU（首轮 #3） | 抗 relearning 攻击 | 攻击在参数空间；非关键词强调改写 |
| SCOPE（首轮 #6） | 同语言改写等价类 | 语义等价类；非「归因关键词强调」这一特定攻击 |

**Delta**：定义步级关键词归因 $\mathrm{KA}_t$（该步对 forget 关键词的注意力份额），构造「关键词强调改写」恢复集，写步级两难目标 $\min_\theta \max_{\text{emphasis}}$，并给出 $\mathrm{RD}$（恢复-破坏差）判据。据现有检索，未见把 UnPact 的关键词归因两难下沉到 LRM CoT 步级并训练闭环的工作。

**与首轮 10 个的划界**：最像 `CSDA`（步级因果）。不同轴：CSDA 的归因量是 $\mathrm{do}(\phi(r_t)\leftarrow\tilde\phi)$ 的**表示因果贡献**；ATTRIB 的归因量是**关键词注意力份额**，攻击是「强调关键词的改写问句」，目标是两难闭合（不靠全局更狠地压）。

---

## 3. 方法（LaTeX）

设 forget 实体关键词集 $\mathcal K(e_f)$，轨迹 $\mathcal T=(r_1,\dots,r_T;a)$，第 $\ell$ 层步 $t$ 对关键词的注意力份额 $\alpha_{t,k}$。

**3.1 步级关键词归因**
$$
\mathrm{KA}_t
=\sum_{k\in\mathcal K(e_f)}\alpha_{t,k}^{(\ell)},\qquad
\mathrm{KA}_t^{\theta_0}-\mathrm{KA}_t^{\theta}
=\text{该步被「破坏」的关键词关注度}.
$$
UnPact 的发现即：遗忘后 $\mathrm{KA}^{\theta}$ 大幅下降，但知识仍在，只是「没被注意到」。

**3.2 关键词强调恢复攻击（内层 $\max$）**
$$
\tilde x=\mathrm{Emph}(x;\mathcal K),\qquad
\rho_\theta^{\text{emph}}
=\mathbb E_{x\in D_f}\big[\mathbb 1[e_f\in\mathrm{Gen}_\theta(\tilde x)]\big],
$$
$$
\mathrm{RR}_\theta=\rho_\theta^{\text{emph}}-\rho_\theta^{\text{orig}}\quad(\text{Recovery Rate}),\qquad
\mathrm{DR}_\theta=\mathrm{Util}(\theta_0)-\mathrm{Util}(\theta)\quad(\text{Destructive Rate}).
$$

**3.3 两难闭合目标（核心）**
$$
\min_\theta\ \underbrace{\big[\mathrm{RR}_\theta\big]_+}_{\text{恢复不可行}}
\;+\;\eta\,\underbrace{\big[\mathrm{DR}_\theta-\delta\big]_+}_{\text{破坏有上界}}
\;+\;\lambda\,\mathrm{KL}(p_\theta\|p_{\theta_0})_{D_r}
\;+\;\mu\sum_t
\underbrace{\big\|\,\mathrm{KA}_t^{\theta}-\mathrm{KA}_t^{\text{safe}}\,\big\|_2^2}_{\text{步级关键词关注重定向（非全局压低）}},
$$
其中 $\mathrm{KA}_t^{\text{safe}}$ 为 retain CoT 上的关键词关注基线（把关注**重定向**到安全分布，而非像 GA 那样一味打压，避免 $\mathrm{DR}$ 飙升）。

**3.4 两难判据（机制推断，待验证）**
$$
\mathrm{RD}_\theta=\mathrm{RR}_\theta+\mathrm{DR}_\theta
\xrightarrow[]{\mathcal L\to 0}\ \text{同时小},
$$
即「恢复不了」与「没伤效用」同时成立。报告 $\mathrm{RD}$ 曲线相对 UnPact 基线的 Pareto 位移。

参数更新走 LoRA，base 冻结。

---

## 4. R-TOFU 接口

- 数据：`forget10/task_id=1` 400 forget + retain；恢复集用模板化关键词强调改写（大写/重复/「请问关于 X」），不引入人工标注。
- 归因：复用 `redact` 前向钩子取 CoT 步注意力（步 = 句子，`contextual_window.py` 切分）。
- 指标：核心 $\mathrm{RR}$、$\mathrm{DR}$、$\mathrm{RD}$、R-TOFU step-wise、LEAK、Retain ROUGE-L、collapse。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | R2MU / CoT-only GA | $\mathrm{RR}$ 显著 $>0$（关键词强调恢复存在，复现 UnPact） |
| T1 | ATTRIB（步级重定向 + 两难闭合） | $\mathrm{RR}\le 0.05$ 且 $\mathrm{DR}\le 0.02$ 且 Retain ROUGE-L $\ge$ T0 |
| T2 | 消融：去掉步级 $\mathrm{KA}$ 项，只做全局更狠压制 | $\mathrm{DR}$ 飙升（两难未解）→ 证明「步级重定向」必要 |
| T3 | 消融：随机选等量步重定向 | $\mathrm{RR}$ 显著劣于 T1 → 证明「关键词归因」而非「步数少」是机制 |

**决定性实验**：T1 vs T3（隔离归因轴）与 T1 vs T2（隔离两难闭合）。若均无差异，核心机制被驳倒 → 转向。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 8（机制分） | 同时压恢复率与破坏率，机制上改善两难；待 T1 |
| Faster | 7（机制分） | 需强调改写集前向，但可缓存；相对 exact-retrain 快 |
| Stronger | 8（机制分） | 对关键词强调改写鲁棒；待跨 split |
| Cheaper | 8（机制分） | **主轴之一**：强调改写由模板生成、无人工标注；归因复用前向钩子 |
| Broader | 6（机制分） | 关键词归因较依赖 tokenizer / 注意力可见性，跨域迁移弱于其它候选 |

**裁决：Accept with Revisions**。降档理由：① F1=MAJOR（UnPact 近邻强，须主动划界「步级闭环 vs 答案级审计」）；② 需注意力可见的白盒前提（F7 工程）；③ Broader 偏弱。机制分 → *pending the validation experiment*。

---

## 7. 失败模式与局限（F10）

1. 关键词集 $\mathcal K$ 不全（别名/翻译/间接指代）→ 报覆盖率并与 PIVOT 的跨语言轴显式互补。
2. 注意力归因未必等于因果（attention ≠ causation）→ 用两类控制改写做对照，声明是「归因闭环」而非严格因果。
3. 步级重定向可能把正常关键词关注也改掉 → 用 $\mathrm{KA}^{\text{safe}}$ 与 retain 锚兜底。
4. 本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/attrib/keyword_attrib.py`（步级注意力份额）与 `scripts/attrib/dilemma_loss.py`；强调改写复用 `paraphrases/` 模板（需先修复）。冻结：fp32 / seeds 1001, 2002。
