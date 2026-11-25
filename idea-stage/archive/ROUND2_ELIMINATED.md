# 归档：第 2 轮淘汰的 4 个 idea

评分文件：`evidence/round2_scores_strict.json`；裁决：`evidence/cross_review_round2.json`（trace 在 `.aris/traces/*-hy4-round2-novelty.json`）。

| ID | 方向 | 分数 | 裁决 | 淘汰理由 | 撞车来源 |
|---|---|---|---|---|---|
| N01 | 跨策略坍塌校正泄漏曲线 | 8 | REFUTED | 跨解码一致性评分已发表 | STaR `arXiv:2601.9281` 的 Multi-Decoding Consistency Score（Default/Zero/LessThink） |
| N04 | 覆盖率门控 | 8 | REFUTED | "coverage-aware" 已被占用，且语义不同（语言覆盖 vs 答案可达） | *Linguistic Loopholes in LLM Unlearning* 的 Coverage-Aware Unlearning |
| N09 | 通道分解（CoT vs answer） | 7 | REFUTED | 本地 11 篇近邻 + 六通道基准 | K-Bench `arXiv:2609.12808`、`arXiv:2605.18891` 的 `thk_leak` |
| N11 | 拒答/坍塌分离 AFE | 6 | REFUTED | 方法名就是 Collapse-Free，且基线表已单列 collapse | REDaCT（本地目录） |

## 保留的价值

- **N01 的价值转入 N10**：跨策略差在干净单元内仍有 0.636–0.643，这个数字作为 N10 的输入保留，但不再主张"跨策略曲线"是新的。
- **N04 的价值转入 N06/N07**：可达率作为门（N07 的效用门、N06 的状态配对）的一部分使用，不单独成篇。
- **N09 的价值转入 N05**：通道不再作为独立贡献，只作为 span 定位的一个维度。
- **N11 的价值转入 N03**：collapse 率作为 AFE 的分母口径输入。

## rubric 漏洞记录（必须随档案保留）

D4=0（新颖性被否）时其余四项仍可加满 **8 分**，因此"被否"的 idea 在数值上仍能过线。本轮 8 个入选全部**手工排除 REFUTED**，没有让 N01/N04 以 8 分混入。若后续修订 rubric，应把 D4=0 的满分上限降到 7。
