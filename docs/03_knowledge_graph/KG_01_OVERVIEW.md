# 脓毒症知识图谱总览

## 1. 目标与边界

当前知识图谱是一个面向脓毒症场景的静态医学知识层，主要服务三件事：

- 组织脓毒症相关医学概念与关系
- 把 eICU 变量映射成病例级知识特征
- 为反事实 donor 检索与干预评估提供临床约束

它不是患者级动态病程图，也不是独立的因果图模型。

## 2. 核心输入

当前图谱的底座来自：

- [LMKG_released.json](/input/knowledge/LMKG_released.json)

增强层主要来自：

- SSC 2021 成人脓毒症指南
- CDC Hospital Sepsis Program 资料
- eICU 变量到 KG 节点的映射表

相关构图脚本：

- [build_sepsis_kg_mvp.py](/input/knowledge/scripts/build_sepsis_kg_mvp.py)
- [build_sepsis_kg_guideline_enhanced.py](/input/knowledge/scripts/build_sepsis_kg_guideline_enhanced.py)

## 3. LMKG 是什么

LMKG 是现成的大规模医学知识图谱资源。在本项目里，它不是直接拿来整张使用，而是作为基础知识底座：

- 先从 LMKG 中识别脓毒症相关种子实体
- 再保留高价值关系，如症状、检查、病原体、治疗、并发症
- 最后裁出脓毒症子图，形成本项目可控、可追溯的静态 KG

这样做的优点是起步快、结构稳定、来源明确；缺点是它本身偏静态知识，不包含患者时间序列和干预时序。

## 4. 构图方法

当前构图链路分为两层。

第一层是 MVP 子图：

1. 从 LMKG 中识别 `sepsis / 脓毒症 / 败血症 / septic shock` 等种子概念。
2. 保留高价值关系类型。
3. 从实体字段中补充一部分检查、部位等关系。
4. 去重并保留来源字段。

第二层是指南增强：

1. 把整理后的指南规则写成结构化关系。
2. 合并进 MVP 子图。
3. 为 donor 约束增加 `guideline alignment`、监测要求、关键治疗要求等信息。

## 5. 当前产物

MVP 产物目录：

- [sepsis_kg_mvp](/input/knowledge/13_processed_ready/sepsis_kg_mvp)

增强版产物目录：

- [sepsis_kg_guideline_enhanced](/input/knowledge/13_processed_ready/sepsis_kg_guideline_enhanced)

常用文件：

- [nodes.csv](/input/knowledge/13_processed_ready/sepsis_kg_guideline_enhanced/nodes.csv)
- [edges.csv](/input/knowledge/13_processed_ready/sepsis_kg_guideline_enhanced/edges.csv)
- [guideline_relations.csv](/input/knowledge/13_processed_ready/sepsis_kg_guideline_enhanced/guideline_relations.csv)
- [build_summary.json](/input/knowledge/13_processed_ready/sepsis_kg_guideline_enhanced/build_summary.json)

## 6. 当前定位

这套 KG 现在最适合做四件事：

- 作为 eICU 到临床概念层的映射底座
- 作为 patient state 的附加知识特征来源
- 作为 donor 检索时的临床一致性约束
- 作为反事实解释与规则审计层

如果你要继续往下看，推荐顺序是：

1. [KG_02_INTEGRATION.md](/docs/03_knowledge_graph/KG_02_INTEGRATION.md)
2. [KG_03_COUNTERFACTUAL_EVAL.md](/docs/03_knowledge_graph/KG_03_COUNTERFACTUAL_EVAL.md)
3. [KG_04_STATUS_AND_PLAN.md](/docs/03_knowledge_graph/KG_04_STATUS_AND_PLAN.md)





## 7. 新添加的知识图谱文件大致分成 4 层：资料层、设计层、产物层、说明层。

1. 原始知识资料层

在 input/knowledge 下面，按用途分文件夹：

00_inbox_raw
原始临时放置区。适合先丢下载下来的杂项文件，还没整理归类前先放这里。

01_guidelines_consensus
指南和专家共识，主要是脓毒症治疗和诊断规则来源，比如 SSC、CDC 这类内容。

02_drug_database
药物数据库，主要给抗菌药、升压药等药物实体和标准名做支持。

03_terminology_ontology
术语和本体，如 MeSH、SNOMED 类资源。主要用于标准化概念和做实体对齐。

04_reviews_papers
综述和论文，偏背景、方法参考和补充证据。

05_hospital_workflows
医院流程和项目资料，偏 sepsis program、流程要求、执行框架。

11_resources_shortcuts
资源入口和快捷索引，不一定是正式知识文件，更像链接导航。

12_manual_downloads
手动下载区，放自动脚本无法稳定抓取、需要人工处理的文件。

14_archive_versions
历史版本归档，防止覆盖旧资源。

2. 结构设计与方法层


这些文件夹不是“知识本体”，而是“怎么组织知识”的设计层。

06_schema_design
核心 schema 和映射设计。现在这里最关键的文件有：

eicu_to_kg_node_mapping.csv：eICU 变量到 KG 节点的映射表
sepsis_guideline_curated_relations.json：整理后的指南关系
kg_counterfactual_feature_schema.json：反事实相关 KG 特征定义
07_data_processing
数据清洗、格式转换、抽取前处理相关资料。

08_extraction_alignment
实体抽取、关系抽取、实体对齐相关资料。偏方法层，不是最终图谱文件。

09_graph_storage_api
图存储、查询和 API 设计参考，偏 Neo4j / RDF / 图服务侧。

10_evaluation
图谱质量评估和下游评估参考资料。

3. 已生成的知识图谱产物层

真正可直接用的图谱产物在 13_processed_ready：

sepsis_kg_mvp
第一版 MVP 脓毒症图谱。里面主要是：

nodes.csv：节点表
edges.csv：关系表
seed_entities.csv：种子实体
knowledge_sources.csv：知识来源登记
build_summary.json：构图统计摘要
sepsis_kg_guideline_enhanced
在 MVP 基础上加了指南增强关系的版本。比 MVP 多一个：

guideline_relations.csv
