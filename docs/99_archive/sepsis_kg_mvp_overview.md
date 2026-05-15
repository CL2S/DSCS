# 脓毒症知识图谱 MVP 详细说明

## 1. 文档目的

本文档用于详细介绍当前已经生成的脓毒症静态知识图谱 MVP，包括：

- 图谱的目标和边界
- 输入材料与数据来源
- 具体构造方法
- 输出文件结构
- 局部子图示例
- 当前版本的局限与下一步迭代方向

当前图谱产物位于：

- [sepsis_kg_mvp](/input/knowledge/13_processed_ready/sepsis_kg_mvp)

当前构建脚本位于：

- [build_sepsis_kg_mvp.py](/input/knowledge/scripts/build_sepsis_kg_mvp.py)

## 2. 图谱目标与边界

本图谱是一个“脓毒症静态医学知识图谱”，目标是组织脓毒症相关的通用医学知识，而不是记录某个患者的时间序列病程。

## 3. 输入材料

### 3.1 直接参与构图的主数据

当前真正用于生成三元组和节点的主数据文件是：

- [LMKG_released.json](/input/knowledge/LMKG_released.json)

该文件是当前版本图谱的核心来源，包含大规模医学实体和关系。脚本实际做的是：从这个大图中裁剪出一个“脓毒症中心子图”。

### 3.2 已下载并登记、但尚未深度对齐入图的数据

为了后续增强，当前项目中还准备了多类知识文件。这些文件已被登记到：

- [knowledge_sources.csv](/input/knowledge/13_processed_ready/sepsis_kg_mvp/knowledge_sources.csv)

当前登记到的知识文件共 37 个，分布如下：

- `01_guidelines_consensus`: 14 个
- `02_drug_database`: 2 个
- `03_terminology_ontology`: 3 个
- `12_manual_downloads`: 18 个

这些文件包括：

- SSC 2021 成人脓毒症指南及补充材料
- CDC 医院脓毒症项目核心要素与评估工具
- RxNorm 药品标准数据
- MeSH 术语本体
- DailyMed 药品标签数据

当前版本里，这些文件主要作为“已准备知识资产清单”，还没有被系统性解析为新增节点和新增关系。

## 4. 构图总体思路

### 4.1 不是从零抽取，而是从大图裁剪

这版图谱不是从 PDF、DOCX、XML 原文逐句抽实体和关系，而是先利用现成的大型医学知识图谱 `LMKG_released.json`，再从中抽取脓毒症相关子图。

这样做的原因很直接：

- 当前已有 LMKG，先复用能更快得到可用结果
- 先把图谱“搭起来”，比一开始就做全量文本抽取更稳妥
- 后续可以再把指南和术语体系逐步对齐进来，形成第二阶段增强版

### 4.2 以“种子实体”驱动子图扩展

脚本首先找出脓毒症相关的种子实体，然后围绕这些种子扩展邻居节点与邻接关系。

种子识别规则分两类：

- 精确匹配
  - `败血症`
  - `脓毒症`
  - `小儿感染性休克`
  - `新生儿败血症`
  - `sepsis`
  - `septic shock`
- 模式匹配
  - `sepsis-associated`
  - `sepsis-related`
  - `due to sepsis`
  - `脓毒症相关`

这一步的结果是确定了 24 个种子实体。

### 4.3 关系类型受限，不是全部照搬

虽然 LMKG 很大，但当前脚本没有把所有关系都拿进来，而是限定在更贴近设计文档的高价值关系集合中，例如：

- `Associated Symptom`
- `Associated Exam`
- `Associated Indicator`
- `Treatment`
- `Causative Agent`
- `Complication`
- `Finding Site`
- `Admission`
- `Measurement Method`

这样做的好处是：

- 图谱语义更集中
- 关系噪声更低
- 后续做临床解释和查询更容易

### 4.4 两类边共同组成当前图谱

当前边分为两类：

1. `LMKG` 原生关系边  
来自 `LMKG_released.json` 已有三元组。

2. 实体字段推断边  
当实体自身字段中出现“检查项目”“发病部位”等结构化信息时，脚本会补出一部分边。例如：

- `败血症 -> 降钙素原`，关系为 `Associated Exam`
- `败血症 -> 全身`，关系为 `Finding Site`

这类边的 `extraction_methods` 标记为 `entity_field_inferred`。

### 4.5 边去重与来源合并

同一对实体同一关系可能同时出现在：

- 种子一跳扩展结果中
- 内部子图关系中

脚本最终对边做了去重，只保留一条边，并把来源方式合并到 `extraction_methods` 字段中。例如：

- `["lmkg_relation_internal", "lmkg_relation_seed_hop"]`

这使得最终图谱可以同时保留“为什么被纳入”和“来自哪种抽取过程”。

## 5. 输出文件说明

当前图谱输出了 6 个核心文件。

### 5.1 节点表

- [nodes.csv](/input/knowledge/13_processed_ready/sepsis_kg_mvp/nodes.csv)

主要字段包括：

- `node_id`: 图中的节点 ID
- `entity_id`: 原始 LMKG 实体 ID，合成节点为空
- `name`: 节点主名称
- `node_class`: `real` 或 `synthetic`
- `primary_type`: 主实体类型
- `types`: 该节点的类型列表
- `aliases`: 别名列表
- `source_tags`: 来源标记，如 `Snomed`、`aplus`、`dayi`
- `description_excerpt`: 简介摘录
- `is_seed`: 是否是种子节点
- `seed_reason`: 成为种子的原因

### 5.2 关系表

- [edges.csv](/input/knowledge/13_processed_ready/sepsis_kg_mvp/edges.csv)

主要字段包括：

- `source_id`
- `target_id`
- `relation_type`
- `source_name`
- `target_name`
- `relation_sources`: 原始知识来源标签
- `extraction_methods`: 该边是如何进入图谱的

### 5.3 种子实体表

- [seed_entities.csv](/input/knowledge/13_processed_ready/sepsis_kg_mvp/seed_entities.csv)

这个文件可以直接回答两个问题：

- 当前图谱是围绕哪些核心脓毒症实体构造的
- 每个核心实体为什么被视为种子

### 5.4 知识源清单

- [knowledge_sources.csv](/input/knowledge/13_processed_ready/sepsis_kg_mvp/knowledge_sources.csv)

这个文件记录了当前工程中已准备好的指南、术语、本体、药品和手工下载资源。

### 5.5 构建摘要

- [build_summary.json](/input/knowledge/13_processed_ready/sepsis_kg_mvp/build_summary.json)

这里保存了图谱规模、关系分布、节点类型分布、构建方式计数等汇总信息。

### 5.6 简要说明

- [README.md](/input/knowledge/13_processed_ready/sepsis_kg_mvp/README.md)

## 6. 当前图谱规模

基于 `build_summary.json`，当前版本规模如下：

- 种子实体数：24
- 节点数：979
- 关系数：16588
- 合成节点数：9

关系分布如下：

- `Associated Symptom`: 8036
- `Finding Site`: 3185
- `Complication`: 2587
- `Associated Exam`: 2317
- `Admission`: 182
- `Treatment`: 170
- `Causative Agent`: 75
- `Associated Indicator`: 30
- `Measurement Method`: 6

节点类型分布如下：

- `Disease`: 826
- `Body Structure`: 90
- `Symptom`: 20
- `Laboratory Exam`: 19
- `Medicinal Substance`: 13
- `Organism`: 4
- `Biochemical Indicator`: 3
- `Procedure`: 3
- `Medical Department`: 1

这说明当前版本本质上还是一个“以疾病实体为中心的脓毒症近邻疾病图”，症状、检查、药物、病原体等是围绕它附着的。

## 7. 关键节点示例

### 7.1 败血症节点

主节点之一：

- `node_id=100`
- 名称：`败血症`
- 主类型：`Disease`
- 类型列表：`["Disease", "Laboratory Exam", "Symptom"]`
- 种子原因：`exact:败血症`

简介摘录：

> 败血症（septicemia），又称脓毒症。是由于病原微生物及其毒性产物持续存在于血液内所引起的一种急性全身性感染。

该节点的来源标记已经清理为：

- `["Snomed", "aplus", "dayi"]`

### 7.2 其他核心节点

当前种子中比较重要的核心节点包括：

- `73642`: `脓毒症`
- `48796`: `小儿感染性休克`
- `52016`: `新生儿败血症`
- `253407`: `Sepsis-associated myocardial dysfunction`
- `258397`: `Sepsis-associated lung injury`

这说明当前图谱不仅包含“脓毒症本体”，还包含“脓毒症相关器官损伤”和部分特殊人群实体。

## 8. 局部子图示例

下面给出几个围绕 `败血症` 的局部关系示例，帮助理解这张图实际长什么样。

### 8.1 与检查相关

从 `edges.csv` 中可以直接看到如下关系：

```text
Associated Exam | 实验室检查 | 败血症
Associated Exam | 病原学检查 | 败血症
Associated Exam | 白细胞 | 败血症
Associated Exam | 胆红素 | 败血症
Associated Exam | 血C-反应蛋白（CRP） | 败血症
```

这说明在当前图谱里，败血症与实验室检查、病原学检查、炎症与器官功能相关指标存在直接连接。

此外，还有从实体字段补出来的推断关系：

```text
Associated Exam | 败血症 | C-反应蛋白
Associated Exam | 败血症 | 降钙素原
Associated Exam | 败血症 | 血液常规
Associated Exam | 败血症 | 血氧
```

这些边不是 LMKG 原始三元组，而是从 `败血症` 节点自身字段 `检查项目` 中补出来的。

### 8.2 与症状相关

局部症状关系示例：

```text
Associated Symptom | 不适 | 败血症
Associated Symptom | 乳酸 | 败血症
Associated Symptom | 休克 | 败血症
Associated Symptom | 低氧血症 | 败血症
```

这里可以看到一个当前版本的现实问题：部分节点在 LMKG 中被建模得不够严格，比如 `乳酸` 出现在 `Associated Symptom` 下，这在医学语义上并不理想，更适合当作指标或检查相关概念。这也是后续清洗的重点之一。

### 8.3 与治疗相关

当前图谱中可直接看到若干药物或药物类别与败血症的治疗关系：

```text
Treatment | 万古霉素 | 败血症
Treatment | 喹诺酮 | 败血症
Treatment | 壁霉素 | 败血症
Treatment | 头孢噻吩 | 败血症
```

这些关系说明图中已经具备了最基础的“疾病-治疗”连边能力，但药物治疗知识仍然是粗粒度的，距离指南级的精准推荐还有差距。

### 8.4 与病原体相关

病原体关系示例：

```text
Causative Agent | 败血症 | Infectious agent
Causative Agent | 败血症 | Microorganism
Causative Agent | 败血症 | Virus
```

这部分提示了脓毒症的病原学来源，但还没有细化到充分可用的病原菌谱层级。后续如果把微生物本体或指南数据接入，这部分会更强。

### 8.5 与并发症相关

并发症关系示例：

```text
Complication | Acute kidney injury due to sepsis (disorder) | 败血症
Complication | Chronic kidney disease due to systemic infection (disorder) | 败血症
Complication | Sepsis syndrome | 败血症
```

这部分是当前图谱最有价值的区域之一，因为它把脓毒症与器官损伤、系统性恶化状态连接了起来。

### 8.6 与部位相关

部位关系示例：

```text
Finding Site | 全身 | 败血症
Finding Site | 关节 | 败血症
Finding Site | 动脉 | 败血症
```

其中 `全身` 是合理的，而部分具体解剖结构关系可能混有噪声，后续需要借助术语本体进一步清洗。

## 9. 合成节点的含义

当前图中有 9 个 `synthetic` 合成节点，例如：

- `降钙素原`
- `C-反应蛋白`
- `病原菌检查`
- `血液常规`
- `血氧`
- `生化`

它们出现的原因是：

- 这些概念在脓毒症实体字段中明确存在
- 但脚本在 LMKG 中未找到一个足够稳妥的实体可以直接对齐
- 为了不丢失这部分高价值知识，先临时建立合成节点

这是一种工程上的折中。优点是能保留医学信息，缺点是后续需要做实体对齐与去重。

## 10. 为什么当前图谱是合理的

从工程视角看，这版图谱是合理的，因为它同时满足了三个要求：

1. 已经真正形成了可查询、可导入、可扩展的图结构  
不是停留在“设计方案”层面。

2. 图谱中心紧扣脓毒症  
不是把整张 LMKG 全搬进来，而是做了种子驱动的脓毒症裁剪。

3. 输出格式清晰  
节点、边、种子、来源、统计分开输出，便于下一步接入 Neo4j、NetworkX 或自定义检索流程。

## 11. 当前版本的主要局限

这版图谱能用，但还明显不是最终版。主要局限如下。

### 11.1 三元组主体仍主要来自 LMKG

已下载的 SSC、CDC、MeSH、RxNorm、DailyMed 等资料，目前只是“进入工程”，还没有大规模转化成新三元组。

### 11.2 存在语义噪声

例如：

- 某些指标被建模成了症状
- 某些同义词体系把不完全等价的概念放到同一个名称簇里
- 某些“sepsis-related” 实体存在边界模糊现象

### 11.3 存在重复近义节点

例如：

- `败血症`
- `脓毒症`
- `Sepsis-associated myocardial dysfunction`
- 同义或近义的多版本英文实体

这些需要在下一阶段做实体归一与标准化映射。

### 11.4 治疗知识还不够指南化

当前 `Treatment` 多数仍是 LMKG 中已有的药物级连边，还没有体现出：

- 液体复苏策略
- 血培养先行
- 乳酸复测
- 升压药首选去甲肾上腺素
- 感染源控制策略

这些内容应该来自 SSC 2021 和 CDC 文件的进一步结构化抽取。

## 12. 下一步推荐增强路线

如果继续迭代，建议按下面顺序做。

### 12.1 第一步：术语对齐

把下列资源正式对齐进图谱：

- MeSH
- RxNorm
- DailyMed
- SPECIALIST / 词汇工具

目标是减少：

- 同义节点重复
- 中英混杂
- 药物名称不统一

### 12.2 第二步：把指南抽成规则化知识

重点处理：

- SSC 2021 成人指南
- CDC 医院脓毒症项目文件

建议新增的关系类型包括：

- `Assessed By`
- `Suggests`
- `Monitored By`
- `Risk Factor For`
- `Recommended First-line Therapy`
- `Recommended Within Time Window`

### 12.3 第三步：引入质量分层

每条边增加：

- 来源文档
- 来源机构
- 来源年份
- 是否指南推荐
- 证据等级
- 是否人工审核

这样图谱才能进入真正的临床可解释场景。

## 13. 如何重新生成

如果要重新生成当前版本图谱，可在项目根目录运行：

```powershell
python .\memory_mvp_project\input\knowledge\scripts\build_sepsis_kg_mvp.py
```

生成目录为：

```text
memory_mvp_project/input/knowledge/13_processed_ready/sepsis_kg_mvp
```

## 14. 总结

当前这版脓毒症知识图谱的本质是：

- 以 LMKG 为基础
- 以脓毒症实体为中心
- 通过种子实体扩展得到的静态医学子图

它已经具备以下能力：

- 查看脓毒症相关疾病簇
- 查看脓毒症的检查、症状、并发症、病原体、治疗关系
- 为后续 Neo4j 导入、检索增强、知识问答和临床解释打基础

它还不具备以下能力：

- 完整承载指南级推荐逻辑
- 表达患者时序病程
- 提供严格清洗后的标准术语层级

因此，当前版本最适合定位为：

- “脓毒症静态知识图谱第一版底座”

而不是最终临床知识库。
