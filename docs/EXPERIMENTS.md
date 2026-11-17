# 实验步骤与指标解释

运行与本地归档复算入口见 [复现说明](REPRODUCIBILITY.md)。下文的人类参考接口为可选功能，已执行双模型批次没有使用人工评分。

以下是可执行协议，不是已获得的实验结论。在项目根目录执行命令。plan/analyze/export-blind 不调用模型；run --execute 会产生真实调用与费用。

## 0. 固定数据和环境

data/examples.jsonl 只是格式演示。真实数据每行一个 JSON：

~~~json
{"sample_id":"essay-001","topic":"AI and employment","essay":"Actual authorized essay text...","split":"dev","source":"Actual origin and permission","human_scores":null}
~~~

独立人工标注后用四维数值对象替换 human_scores:null。样本 ID 唯一，文本/来源非空。相同主题的近似改写不跨 split。先在 dev 校准 rubric、权重和阈值，再冻结 test。样本量依据差异和标注资源规划，不预设小样本足以证明效果。

用 requirements-lock.txt 安装环境，保存源码快照与服务端模型版本。计划保存实际源码/运行环境指纹，代码或环境改变后需重新生成计划。

## 1. 验证重写闭环

app 使用 --stop-policy fixed --max-rounds 2，建议提供自己的固定初稿。检查两轮 history、第二轮 generation_history.feedback_used、输入输出哈希和裁判反馈是否确实进入修订。失败先修复 API，不直接扩大实验。哈希变化只证明改动；效果需盲评。

## 2. 可信度

~~~powershell
.venv/Scripts/python -B -m src.experiments plan --dataset data/examples.jsonl --kind reliability --model "qwen3.8-27b" --repeats 3 --max-attempts 1 --output runs/plans/reliability.json
~~~

模型名由本地服务配置决定，不保证其他账户可用。固定原文重复评分，报告每维有效判断数、原文内标准差、MAE、Spearman。没有人工参考时准确性为 null。稳定并不意味着正确。

## 3. 反馈消融

~~~powershell
.venv/Scripts/python -B -m src.experiments plan --dataset data/examples.jsonl --kind ablation --model "qwen3.8-27b" --max-rounds 2 --max-attempts 1 --repeats 3 --output runs/plans/ablation.json
~~~

从完全相同初稿出发，比较 no_rewrite、no_feedback、relevance_feedback、all_feedback。后三者 fixed 轮数；none 仍自行重写，以隔离反馈作用。自动报告各变体相对不重写，以及全反馈相对无反馈/相关性反馈的直接配对差异。指定主比较，避免事后挑显著结果。

## 4. 串并行与退出策略

~~~powershell
.venv/Scripts/python -B -m src.experiments plan --dataset data/examples.jsonl --kind efficiency --model "qwen3.8-27b" --repeats 3 --max-attempts 1 --output runs/plans/efficiency.json
.venv/Scripts/python -B -m src.experiments plan --dataset data/examples.jsonl --kind early_exit --model "qwen3.8-27b" --max-rounds 3 --threshold 0.85 --repeats 3 --max-attempts 1 --output runs/plans/early_exit.json
~~~

效率实验使用固定原文仅评分，控制提示、温度、模型、重试与输出上限。seed 随机化配对块中的执行顺序，不能消除远端负载干扰。报告 P50/P95、失败率、实际调用、配对耗时差与 speedup；不能从四路推导四倍加速。

退出策略比较 threshold 与 fixed；阈值由 dev 确定。必须同时看盲评质量、退化比例和调用/Token/成本变化；失败不能当提前退出成功。

## 5. 离线演示与显式执行

先生成计划，再读取保守上界：

~~~powershell
$p = Get-Content -Raw -Encoding UTF8 runs/plans/ablation.json | ConvertFrom-Json
$p.max_call_attempts
.venv/Scripts/python -B -m src.experiments run --plan runs/plans/ablation.json --output-dir runs/ablation-demo --demo --max-calls $p.max_call_attempts
.venv/Scripts/python -B -m src.experiments analyze --run-dir runs/ablation-demo --output runs/ablation-demo/report.json
~~~

演示文章与评分是脚本输出，不能报告为实验结果。真实运行需显式执行下列命令：

~~~powershell
.venv/Scripts/python -B -m src.experiments run --plan runs/plans/ablation.json --output-dir runs/ablation-live --execute --max-calls $p.max_call_attempts
~~~

max-calls 是调用次数上界，不是金额预算。执行前重算各任务上界、校验变体/配对/样本指纹，核对源码与环境。既有输出不覆盖。逐任务执行并即时落盘；四维内部可并行，不同时跑多个 job。中断后已完成任务可分析，complete=false；当前不自动续跑，避免重复收费。

## 6. 盲评

~~~powershell
.venv/Scripts/python -B -m src.experiments export-blind --run-dir runs/ablation-live --output-dir runs/ablation-blind
~~~

按 ANNOTATION_GUIDE.md 填写 annotation.jsonl 的四维分数，私有映射独立保存：

~~~powershell
.venv/Scripts/python -B -m src.experiments analyze --run-dir runs/ablation-live --annotations runs/ablation-blind/annotation.jsonl --mapping runs/ablation-blind/private_mapping.json --output runs/ablation-live/report.json
~~~

报告按样本平均重复差值，再做样本级 bootstrap 95% 区间；少于两个有效配对样本时区间 null。缺失标签/失败不补分，同时报告覆盖和有效配对数。质量比较基于成功且有标签的配对，需结合失败率解读存活偏差。模型自身的平均分仅作描述。

## 7. 用量与费用

用量来自每调用服务端 prompt_tokens/completion_tokens，失败、截断和解析重试也进入日志。任一调用缺失用量时严格 total_tokens=null，已知部分仍单列。未知用量不能填零制造节省。

价格表需根据实际适用价格填写。下列 null 是不可用的结构模板，校验会拒绝，不能当服务商报价：

~~~json
{"currency":"实际币种","as_of":"2026-11-09","source":"真实适用价格来源和版本","models":{"实际模型名称":{"input_per_million":null,"output_per_million":null}}}
~~~

填入真实非负数，analyze 增加 --prices <文件路径>。多模型均需对应价格。缺价或缺 usage 时 complete_cost=null；observed_cost 仅已知部分。计算不含阶梯、缓存折扣、税费或账单核销。

## 8. 交付

归档授权数据与 split、原始标注/裁决、冻结计划、源码快照与依赖锁、完整调用日志、报告、运行命令和失败分析。固定 seed 不能固定远端模型；记录服务端版本与时间。没有真实结果时不报告提升百分比、加速倍数或成本节省。
