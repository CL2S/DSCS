# KG 当前状态与下一步计划

## 1. 当前已经完成的部分

目前已经具备以下能力：

- 基于 LMKG 裁剪脓毒症子图
- 合并指南增强关系
- 建立 eICU 变量到 KG 节点的显式映射
- 生成 `kg_feature_vector`
- 把 KG 接入 donor 重排
- 增加 `knowledge-safe write`
- 增加 donor hard filter
- 增加 `generated_best` 候选干预模式

## 2. 当前仍然没有完成的部分

以下能力还没有真正实现：

- 多 donor 融合干预
- 多步治疗序列搜索
- 完整 action space 规划
- 时间窗、剂量、成本等完整治疗约束
- 严格因果识别层

所以当前系统的定位仍然是：

- donor 检索增强器
- 干预候选修正器
- 临床一致性约束层

不是完整的自动治疗规划器。

## 3. 文档收口后的建议阅读顺序

现在知识图谱部分收口为 4 个文件，建议按下面顺序阅读：

1. [KG_01_OVERVIEW.md](/docs/03_knowledge_graph/KG_01_OVERVIEW.md)
2. [KG_02_INTEGRATION.md](/docs/03_knowledge_graph/KG_02_INTEGRATION.md)
3. [KG_03_COUNTERFACTUAL_EVAL.md](/docs/03_knowledge_graph/KG_03_COUNTERFACTUAL_EVAL.md)
4. [KG_04_STATUS_AND_PLAN.md](/docs/03_knowledge_graph/KG_04_STATUS_AND_PLAN.md)

## 4. 建议的下一步

如果继续推进，我建议优先级如下：

1. 把 `generated_best` 扩到正式规模评估，而不只停留在 smoke
2. 扩充可生成动作，不只修复抗菌药和升压药
3. 引入多 donor 候选池，而不是只从单 donor 出发
4. 最后再做显式的治疗规划搜索
