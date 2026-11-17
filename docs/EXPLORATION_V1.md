# 10 篇固定文本探索实验 v1

> 历史协议：2026-11-16 起协议改为双模型评分，后续以[双模型 v2](DUAL_MODEL_V2.md)为准。旧批次原始记录与当时的人工计划保留，不补填 AI 人工标签。

这是一项小样本探索，不是完整评测体系验证。10 篇英文文本由 AI 生成，每篇约 310–335 词，覆盖交通、教育、回收、远程办公、公共文化、隐私、能源、体育、博物馆、消费信息。文本包含有意设计的语法、逻辑、相关性与论据缺陷；这些层次是设计意图，不是人工真值。没有引用外部数据集，不冒充真实学生作文或独立基准。

固定输入为 data/exploration_v1.jsonl。全部为 dev，未使用 test，没有泛化或确认性统计主张。设计层次与哈希记录在 exploration_v1_manifest.json，不应向标注者提供该清单，避免先入为主。

## 冻结协议

- 使用同一批 10 篇初稿与 essayloop-rubric-v3。版本变化是去掉旧相关性标准中仅针对 AI 话题的文字；旧 v2 的两篇运行不合并到本实验。
- 生成器与裁判按现有配置使用 qwen3.8-27b；生成温度 .3，裁判温度 0；后者不保证稳定。
- 四维权重 relevance .3 / grammar .2 / structure .2 / depth .3；所有人类与模型使用相同标准和锚点。
- 可信度：每篇固定原文评分 3 次，共 30 个任务，每任务四个裁判，上界 120 次调用。
- 消融：no_rewrite 仅评分 1 次；其他三条件从相同初稿开始，固定两轮评分和一次重写。每条件每篇 1 次，共 40 个任务，上界 310 次调用。
- 四组消融中的“不重写”是原稿对照，不能为凑两轮而改成重写；其余三组处理强度一致。
- 总授权上界 430 次；每节点总尝试 1 次，超时 180 秒。失败保留，不自动扩大预算或择优重跑。任务顺序在样本配对块内按固定 seed 随机化。
- 主比较预先指定 all_feedback vs no_feedback；其他比较为探索性。不预设提升，不调整 prompt 来隐藏负结果。

## 人工评分（必须由真人完成）

prepare 命令先于模型实验生成 human_reference/rater1.html 与 rater2.html。每人独立拿到自己的文件，用浏览器离线打开；不会联网。也可使用 JSONL 或阅读版 Markdown。

表单含每篇题目、原文、四维标准与 0–1 输入框，不含模型评分、原始 sample_id 或设计质量层次。每人填写自己的标注者 ID、评分、理由、日期，并如实声明是否本人评分、是否未查看模型/另一人分数。可导出草稿，关页面前须保存；草稿不能参与统计。表单不会冒充或自动生成真人意见。

两人应先独立完成并保存原始结果，再讨论分歧。不要先交流统一分数。只有一名标注者时报告这一限制。两人时以共同完成条目的均值作参考，并保留每人原分、逐维差值与备注；不覆盖分歧。也可单独人工裁决，但须保留原始资料，不在 test 上回调模型。

参考分析命令（替换真实 study 目录和已导出的 JSONL 路径）：

~~~powershell
.venv/Scripts/python -B -m src.annotation analyze-reference --run-dir runs/STUDY/reliability --packet runs/STUDY/human_reference --ratings PATH/rater1_completed.jsonl PATH/rater2_completed.jsonl --output runs/STUDY/human_reference_report.json
~~~

只有一人时 --ratings 后仅给一个文件。模板、空分、模型代评或没有真人声明的文件会被拒绝。声明只记录提供者的陈述，并不证明独立性；研究者应确认实际流程。

## 消融盲评

模型任务结束后生成 human_ablation/rater1.html 与 rater2.html。向标注者隐藏变体、模型评分、私有映射。完全相同的题目/文本只标注一次，再映射回多个 job，避免重复评分制造额外噪声。即使模型末轮评测失败，只要保留了文本也会进入包；失败任务在主分析中仍作为失败报告，不能包装成成功。

~~~powershell
.venv/Scripts/python -B -m src.annotation analyze-ablation --run-dir runs/STUDY/ablation --packet runs/STUDY/human_ablation --ratings PATH/rater1_completed.jsonl PATH/rater2_completed.jsonl --output runs/STUDY/human_ablation_report.json
~~~

以同样本同条件实际人类分计算配对差值；重复时先按样本平均。报告改善、持平、退化数量及比例，数值相等容差 1e-12 只处理浮点误差，不等于实践等效阈值。单列失败配对、未完成配对和未标注配对。成功配对的结果仍存在存活偏差，须结合全部失败率解释。

## 指标与限制

模型重复评分：每篇四维样本标准差（ddof=1）、总分 min/max/range/SD、成功和失败次数、已完成任务失败率；缺失与失败分开。仅成功维度参与该维统计，可能存在选择偏差。

人机：模型先按原稿平均重复分，再与人工参考计算 MAE 和并列秩修正的 Spearman；常数序列相关性 null。两人：逐维与加权总分的相互 MAE、Spearman，以及两向绝对一致性单人/双人均值 ICC(A,1)/(A,2)。这是当前两位研究标注者的结果，不能推断任意人群标注者；负 ICC 保留，零方差或不足条目返回 null。Spearman 衡量排序，不等于绝对一致，故同时给 MAE/ICC。

MAE/Spearman 和质量均值差可给样本级 percentile bootstrap 区间；n=10 区间不稳定，只有 3 次重复也不足以精确估计每篇波动。对原稿各样本独立抽样；同一原稿的重复或变体不当独立样本。消融人际一致性基于相关变体条目，默认不计算其 bootstrap 区间。多项比较未作显著性校正，主比较之外不做确认性结论。

模型自评分均值仅描述，不能代替人工效果。未提供真实人工标签时，MAE/Spearman、人际一致性与人类改善/退化数保持待完成，不填零或编造。没有价格表不推算金额；用量缺失不填零。

## 执行与归档

~~~powershell
.venv/Scripts/python -B -m src.study prepare --output-dir runs/EXPLORATION_NEW
.venv/Scripts/python -B -m src.study run --study-dir runs/EXPLORATION_NEW --execute --max-calls 430
.venv/Scripts/python -B -m src.study status --study-dir runs/EXPLORATION_NEW
~~~

每任务立即归档。不要再次运行 run 命令制造重复收费；它会拒绝已有任务目录。中断时用 status 检查完成量，未完成部分需单独评估剩余预算，不能直接重跑全部。源码变动后重新生成计划，不手工绕过哈希检查。

最终包含固定输入、协议、计划、环境与源码指纹、全部原始结果、实际失败、双人原始标注、分歧和分析报告。所有报告明确“AI 编写的 10 篇便利样本探索”，不得表述为评测体系全面有效或泛化已证实。
