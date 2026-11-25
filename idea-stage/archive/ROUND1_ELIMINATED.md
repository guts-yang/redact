# 归档：第 1 轮 idea（I01–I10）

保留原因：证据链与更正记录在 `IDEA_REPORT.md` §5/§13–§17，证据文件在 `evidence/idea_scores*.json`。
本文件只归档**结论**，不再作为候选推进。

| ID | 方向 | 最终状态 | 归档理由 |
|---|---|---|---|
| I01 | 生成状态分层泄漏审计 | 诊断性，不作证据 | 干净单元对后处理变量取条件（collider）；Hy4 v2 判 REJECT |
| I02 | 等预算解码对比 | **并入 N01**，N01 又被 STaR MCS 抢先 | 新颖性归零 |
| I03 | CoT-only 通道审计 | 淘汰 | 本地 9 篇近邻 + K-Bench 六通道 |
| I04 | 删失评测区间 | **升级为 N02** | 被新编号继承 |
| I05 | judge-指标分歧分层 | **升级为 N03**（分母口径） | 被新编号继承 |
| I06 | 强制答案可达性探针 | 淘汰 | 无独立新颖性，被 N04/N06 吸收后仍受覆盖率工作抢先 |
| I07 | retain 效用硬门 | **升级为 N07** | 被新编号继承 |
| I08 | 受控不变性重评 | 否证 | 后测 γ=0 是坍塌/截断产物 |
| I09 | 复述恢复 / lexical anchoring | 淘汰 | 改写库是拒答改写，不可复现；ShadowBench 已发表 |
| I10 | 检测器阈值复现审计 | **升级为 N08** | 被新编号继承 |

## 继承关系

```
I04 -> N02   (删失区间)
I05 -> N03   (judge 分母口径)
I07 -> N07   (效用门)
I10 -> N08   (契约审计)
I02 -> N01 -> 淘汰（STaR）
```

## 不再推进但保留的资产

- `evidence/idea_scores_v1..v4.json`：四阶段评分轨迹（rubric 哈希 `cff0a224b94802cc` 未变）
- `evidence/b10_state_table.json` + `evidence/audit/state_table.csv`：1200 行逐样本状态表
- `evidence/b17_calibration_failure.json`：知识存在性三信号矛盾的记录（N12 的动机）
