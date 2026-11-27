# IDEA 08 · CFA — 反事实侧信道审计与去关联
## Counterfactual Side-Channel Audit and De-correlation

| 项 | 值 |
|---|---|
| 排名 | **#8** |
| 五维 (H/F/S/C/B) | 7 / 7 / 8 / 7 / 8 |
| 均分 | **7.4** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU（构造 in-scope / out-of-scope 配对探针） |
| 基线 | Reasoned IDK、Direct IDK、CiPO（固定反事实，二手来源） |

---

## 1. Problem Anchor（一句话故事）

> 固定拒答模板 / 固定反事实本身就是信息：攻击者只要问「你是真不会还是被要求忘的」，就能从「一律 I don't know」反推哪些实体在 forget set 里。CiPO 已自承这个侧信道，但没解决。CFA 把「输出风格与成员身份的互信息」显式压到零。

锚定问题：**LRM 遗忘的输出策略不能泄露「什么被遗忘」这一成员身份**。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| CiPO (ACL 2026, arXiv:2604.15847, 二手) | 反事实轨迹替代拒答；**自承固定模式可成泄露通道** | 承认问题，未给去关联目标 / 审计协议 |
| Honesty in LLM Unlearning (2026, 二手) | 自我知识 vs 自我表达，检测「知道但拒答」 | 评估框架，非训练方法 |
| MIA / RULI (USENIX Security 2025) | 成员推断攻击与评估 | 通用 LLM，非 LRM 输出风格侧信道 |
| 仓库 UPBD | 固定 `SAFE_EXIT` 拒答串 | 设计文档已承认「可能是拒绝模板」，无去关联 |

**Delta**：定义风格–成员互信息 $I(S;M)$，用对抗者 + 梯度反转把它最小化，并发布「风格侧信道审计协议」。据现有检索，未见把「固定拒答/反事实模板」当作成员推断侧信道来攻防的工作。

---

## 3. 方法（LaTeX）

**3.1 侧信道形式化**
令 $M\in\{0,1\}$ 表示样本是否属于 forget set，$S$ 为输出风格特征（拒答长度、`</think>` 位置、模板概率、反事实措辞分布）：
$$
I(S;M)=H(M)-H(M\mid S),\qquad
\text{攻击者}\ \hat M=\mathcal{A}(S),\ \ \mathrm{AUC}_\mathcal{A}
$$
目标 $\mathrm{AUC}_\mathcal{A}\to 0.5$（不可区分）。

**3.2 对抗去关联**
$$
\boxed{\;
\min_\theta\ \mathcal{L}_{\text{forget}}(\theta)+\lambda\mathcal{L}_r(\theta)
\;-\;\beta\,\mathcal{L}_{\text{adv}}\big(\mathcal{A}_\psi(S(\theta))\big)
\;},\qquad
\max_\psi\ \mathcal{L}_{\text{adv}}(\mathcal{A}_\psi;M)
$$
其中 $\mathcal{A}_\psi$ 是判别「in-scope vs out-of-scope」的风格探针；$-\beta$ 项经梯度反转层（GRL）实现，逼 $S$ 与 $M$ 独立。

**3.3 多样性反事实替代固定模板**
$$
y_{\text{cf}}\sim p_{\text{cf}}(\cdot\mid x,\ z),\quad z\sim\mathcal{N}(0,I),\qquad
\mathrm{Var}\big[S(y_{\text{cf}})\big]\ \ge\ \sigma^2_{\min}
$$
用随机潜变量生成风格多样的安全推理，而非单一 `"I don't know."`。

**3.4 审计指标**
$$
\mathrm{SCA}=\sup_{\mathcal{A}\in\mathcal{A}_{\text{family}}}\big(\mathrm{AUC}_\mathcal{A}-0.5\big),
\qquad \text{目标 } \mathrm{SCA}\le 0.05
$$

---

## 4. R-TOFU 接口

- 探针构造：in-scope = forget10 原题 + 改写；out-of-scope = retain / real_authors / world_facts。风格特征从 `redact/inference_regen.py` 的生成结果抽取。
- 指标：常规 Leak/MU/Retain + 核心 $\mathrm{SCA}$（线性探针 / 小 MLP / n-gram 三类攻击者）+ 风格方差。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | Direct IDK / UPBD 固定 SAFE_EXIT | 风格探针 $\mathrm{AUC}_\mathcal{A}\ge 0.85$（侧信道存在） |
| T1 | CFA（对抗去关联 + 多样反事实） | $\mathrm{SCA}\le 0.05$ 且 Leak ≤ 0.05 |
| T2 | 消融：去掉对抗项（$\beta=0$） | $\mathrm{SCA}$ 回弹 → 证明确为去关联而非模板多样化 |
| T3 | 消融：用单模板反事实替代多样 $p_{\text{cf}}$ | 至少一类攻击者 AUC 回升 |

**决定性实验**：T1 vs T2（隔离对抗去关联）。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 7（机制分） | 主贡献是「隐私不泄露成员身份」，非直接提 MU；上限受限故不给 8 |
| Faster | 7（机制分） | 探针轻量、GRL 几乎零额外前向；相对 exact-retrain 快 |
| Stronger | 8（机制分） | 对成员推断 / 固定模板攻击族鲁棒；待 T2 |
| Cheaper | 7（机制分） | 复用既有生成与 CSV，无新标注 |
| Broader | 8（机制分） | 「输出风格侧信道」审计可迁移到任意拒绝/安全对齐系统 |

**裁决：Accept with Revisions**。降档理由：① F1=MAJOR（MIA/RULI、Honesty 框架近邻强，且 CiPO 已自承该问题，需严格划界到「LRM CoT 风格侧信道 + 训练期去关联」）；② 主贡献偏评估/安全，成文需与一个方法型 idea（如 DISCO / RRBU）合并增强。机制分 → *pending validation*。

---

## 7. 失败模式与局限（F10）

1. $\mathcal{A}_{\text{family}}$ 覆盖有限，未覆盖的攻击者仍可能区分 → 必须报告「攻击者族」并承认 $\sup$ 是经验上界。
2. 风格去关联可能与「干净拒答」冲突（风格太多样反而不像拒答）→ 用 collapse / 可读性作硬约束。
3. 与仓库 UPBD 的固定 SAFE_EXIT 直接相关：UPBD 设计文档已把此列为开放风险，CFA 是其针对性补强，非重复。
4. 二手来源（CiPO / Honesty）的 arXiv ID 待核验；本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/cfa/style_probe.py`（特征抽取 + 三类攻击者）与 `scripts/cfa/grl.py`（梯度反转）；训练在 `scripts/upbd/train_upbd.py` 骨架上接对抗项。冻结：fp32 / seeds 1001, 2002 / 探针与训练样本严格隔离。
