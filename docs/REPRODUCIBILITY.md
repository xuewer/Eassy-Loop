# 复现与核对

## 离线检查已有实验

安装 requirements-lock.txt 后，在仓库根目录执行：

~~~powershell
.venv/Scripts/python -B -m pytest -q -p no:cacheprovider
.venv/Scripts/python -B tools/verify_dual_results.py
~~~

第二条命令只读取 experiments/dual_v2_20261116，不调用模型，也不覆盖归档文件。它检查调用唯一性和预算、任务计划、历史源码快照、原文、原始评分 JSON、引用、提示哈希、加权总分、MAE、Spearman、配对差值、区间和失败分母。所有归档数字都是历史已发生记录，离线核对不产生新的实验结果。

指定另一个同格式归档目录或输出新的核对记录：

~~~powershell
.venv/Scripts/python -B tools/verify_dual_results.py --run-dir experiments/dual_v2_20261116 --output runs/verification-new.json
~~~

output 的父目录须存在，文件须不存在。--check-working-source 额外要求当前 src 与旧实验源码完全一致，仅适用于旧源码环境；当前维护版已有命名和可观测性修复，默认核对归档快照。

## 新运行

单次 demo 不联网。真实模型运行必须提供本地密钥，并确认账户支持所用模型。不要复用已完成批次目录；使用新目录，保存新源码指纹、配置、调用预算和结果。

src.experiments 提供 reliability、ablation、efficiency 和 early_exit 计划。src.dual_study 是 10 篇样本、350 次尝试上限的固定协议工具，细节见 [双模型协议](DUAL_MODEL_V2.md)。更换样本量、提示、模型或评分标准应使用新的协议版本。

旧批次先由第一模型评测、修订，再由第二模型评分最终文本。第二模型不看条件和已有分数。若要区分模型差异与提示结构差异，需要额外控制提示布局；该实验尚未完成。

## 归档结构

公开证据从本地 runs 通过 tools/export_evidence.py 的白名单导出。MANIFEST.json 保存每个复制文件的 SHA256，原始 JSON 和日志按字节保留。没有复制 .env、IDE 设置、助手会话或审阅转录。历史 provenance 中可能保留项目路径和旧显示名，这些记录不随仓库改名重写。

原始结果包含文本与模型返回；使用自己的数据运行时，应先确认许可，再决定公开哪些材料。公开清单校验只能证明已列出的文件一致，不能证明服务商内部模型身份、计费完整性或客观文本质量。
