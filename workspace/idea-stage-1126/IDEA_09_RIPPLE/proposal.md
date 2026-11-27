# IDEA 09 · RIPPLE — 多跳涟漪局部化
## Multi-Hop Ripple Localization for Relational Erasure

| 项 | 值 |
|---|---|
| 排名 | **#9** |
| 五维 (H/F/S/C/B) | 8 / 7 / 8 / 7 / 7 |
| 均分 | **7.4** |
| 裁决 | **Accept with Revisions**（机制分，pending validation） |
| 状态 | provisional · evidence_gate = BLOCKED |
| 数据集 | R-TOFU 知识库（`knowledge_banks/` + `config/kb_resolutions/forget10.json`） |
| 基线 | GA/NPO、RippleBench（评估基准，二手来源） |

---

## 1. Problem Anchor（一句话故事）

> 删掉一个虚构作者会顺带伤到「同出版社 / 同国籍 / 同主题」的邻近知识，也会留下「多跳推理仍能反推回他」的关系链。RippleBench 只**测**涟漪效应；RIPPLE 用语义距离函数把影响半径约束进一个球，并剪断跨跳推导路径。

锚定问题：**LRM 遗忘应局部化到语义邻域，且切断多跳关系重建**。综述把 `Multi-Hop Relational Erasure` 列为未解方向。

---

## 2. 为什么是空白（近邻与 delta）

| 近邻 | 做了什么 | 没做什么 |
|---|---|---|
| RippleBench (Rinberg et al., 2025, 二手来源) | 用语义距离函数 + MCQ **量化**涟漪效应 | 评估基准，不给局部化训练目标 |
| SKeB / 知识纠缠度 (arXiv:2510.25732) | 纠缠度预测泄漏 | 预测，不做半径约束 / 跳数剪断 |
| 综述「Multi-Hop Relational Erasure」(二手) | 列为 future direction（因果中介 + 知识图） | 无具体方法 |
| EAS (IDEA 02) | 纠缠度驱动**干预强度** | 不做空间局部化 / 多跳剪断（互补而非重复） |

**Delta**：把涟漪写成可优化约束 $\mathrm{dist}(\phi(x),\phi(x_f))\le\rho \Rightarrow$ 保留，$\mathrm{hop}(x,x_f)\ge H \Rightarrow$ 剪断推导路径。据现有检索，未见「语义半径 + 跳数」双约束的 LRM 训练方法。

---

## 3. 方法（LaTeX）

**3.1 涟漪半径与多跳集**
$$
\mathcal{N}_\rho(x_f)=\{x:\ d_{\text{sem}}(\phi(x),\phi(x_f))\le\rho\},\qquad
\mathcal{H}_H(x_f)=\{x:\ \mathrm{hop}_{\mathcal{G}}(x,x_f)\le H\}
$$
$d_{\text{sem}}$ 由知识图嵌入 + `kb_resolutions` 解析出的实体–声明关系给出。

**3.2 局部化约束损失**
$$
\boxed{\;
\mathcal{L}_{\text{RIPPLE}}
=
\mathcal{L}_{\text{forget}}(x_f)
+
\lambda_1\!\sum_{x\in\mathcal{N}_\rho}\!\mathrm{KL}\big(p_\theta(\cdot\mid x)\,\|\,p_{\theta_0}(\cdot\mid x)\big)
+
\lambda_2\!\sum_{x\in\mathcal{H}_H}\!\mathcal{L}_{\text{pathcut}}(x,x_f)
\;}
$$
$$
\mathcal{L}_{\text{pathcut}}(x,x_f)=
\max\Big(0,\ \mathrm{sim}\big(\phi_\theta(\mathrm{CoT}(x)),\ \phi_{\theta_0}(\mathrm{CoT}(x_f))\big)-\tau\Big)
$$
即：邻近知识保持原分布（$\lambda_1$），而多跳推导路径不得再「走回」forget 的推理表示（$\lambda_2$）。

**3.3 涟漪度量**
$$
\mathrm{Ripple}(\rho)=\frac{1}{|\mathcal{N}_\rho|}\sum_{x\in\mathcal{N}_\rho}
\big|\ \mathrm{Utility}_\theta(x)-\mathrm{Utility}_{\theta_0}(x)\ \big|,
\qquad
\mathrm{Recon}(H)=\mathbb{P}\big[\ \mathcal{G}\text{-multi-hop}\ \Rightarrow\ e_f\ \big]
$$
目标：$\mathrm{Ripple}(\rho)$ 小、$\mathrm{Recon}(H)$ 小。

**3.4 与 CSDA 的区别**：CSDA 选**单条链内**的承重步；RIPPLE 约束**跨样本 / 跨跳**的关系半径。二者可叠加。

---

## 4. R-TOFU 接口

- 知识图：`knowledge_banks/forget{01,05,10}/claims.json` 与 `config/kb_resolutions/forget10.json`（含 `original_claim`/`resolved_claim` / 共指 notes）直接提供实体–声明关系。
- 评估：RippleBench 式 MCQ（语义邻近 / 无关两组）+ 多跳重建探针 + 常规 Leak/Retain/MU/collapse。

---

## 5. 决定性验证实验（预注册）

| 组 | 配置 | 判据 |
|---|---|---|
| T0 | GA / NPO（无局部化） | $\mathrm{Ripple}(\rho)$ 大、$\mathrm{Recon}(H)$ 显著 > 0 |
| T1 | RIPPLE | $\mathrm{Ripple}(\rho)$ 降 ≥ 0.3 且 $\mathrm{Recon}(H)\le 0.05$，Leak ≤ 0.05 |
| T2 | 消融：只留 $\lambda_1$（无 pathcut） | $\mathrm{Recon}(H)$ 回弹 → 证明多跳剪断必要 |
| T3 | 消融：把 $\mathcal{N}_\rho$ 换成随机等距集 | $\mathrm{Ripple}$ 不降 → 证明语义半径特异 |

**决定性实验**：T1 vs T2（隔离 pathcut）。

---

## 6. 五维评分与裁决

| 维度 | 分 | 依据 |
|---|---:|---|
| Higher | 8（机制分） | 同时保邻近效用 + 断多跳重建；待 T1 |
| Faster | 7（机制分） | 图预计算一次性，训练增两项轻量正则；相对 exact-retrain 快 |
| Stronger | 8（机制分） | 对语义邻近 / 多跳攻击鲁棒；待 T2 |
| Cheaper | 7（机制分） | 复用仓库 KB 资产，无需新标注 |
| Broader | 7（机制分） | 「语义半径」可迁移，但依赖知识图 → 普适性略受限 |

**裁决：Accept with Revisions**。降档理由：F1=MAJOR（RippleBench / SKeB / EAS 近邻均强，需严格划界到「训练期双约束」）；且依赖 KB 质量（虚构作者关系可能不全）→ 数据风险需前置核验。机制分 → *pending validation*。

---

## 7. 失败模式与局限（F10）

1. 虚构作者知识图不完整 → $d_{\text{sem}}$ / $\mathrm{hop}$ 估计偏差；需用 `kb_resolutions` 的共指 notes 增广并报告覆盖率。
2. 与 EAS（IDEA 02）都用纠缠度：EAS 调**强度**，RIPPLE 调**空间范围**，论文必须划界，否则被判重复。
3. pathcut 可能误伤通用多跳推理 → 用 retain 上的多跳数学题作对照。
4. 二手来源（RippleBench）元数据待核验；本机无 GPU、无跨家族评审；`evidence_gate = BLOCKED`。

---

## 8. 复现入口（服务器侧，待部署）

新增 `scripts/ripple/kgraph.py`（从 `knowledge_banks/` + `kb_resolutions/` 构图）与 `scripts/ripple/pathcut.py`；训练在 `scripts/upbd/train_upbd.py` 骨架上加两项正则。冻结：fp32 / seeds 1001, 2002 / 固定 $\rho,H$ 预注册。
