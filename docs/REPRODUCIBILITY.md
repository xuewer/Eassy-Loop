# 运行与复算

## 检查源码分发是否完整

在仓库根目录安装 requirements-lock.txt 后执行：

~~~powershell
python -B tools/verify_repository.py
python -B -m pytest -q -p no:cacheprovider
python -B -m src --demo --max-rounds 2 --stop-policy fixed
~~~

这些检查不需要 .env 或真实模型。完整性检查核对本地模块导入、必要配置、固定样本和文档链接；pytest 禁止网络连接。demo 使用脚本模型，只检查流程。

## 运行新实验

src.experiments 支持 reliability、ablation、efficiency 和 early_exit 计划。先生成计划，再选择 --demo 离线演示或 --execute 真实执行，并提供 --max-calls 上限。示例见 [实验工具](EXPERIMENTS.md)。真实调用需要自行配置服务账户，计划中的模型标识必须由该账户支持。

src.dual_study 实现十篇文本的固定双模型消融协议，见 [双模型协议](DUAL_MODEL_V2.md)。更换数据、提示、模型或评分标准应生成新计划，使用新的输出目录，不覆盖旧批次。固定 seed 只控制任务顺序与 bootstrap，不保证远端模型确定复现。

## 本地归档复算

原始模型返回、调用日志、历史源码快照和个人报告不随源码上传。已有归档可显式传给复算工具：

~~~powershell
python -B tools/verify_dual_results.py --run-dir runs/dual_v2_20261116
~~~

该工具专门核对已完成的 dual-v2 历史批次，包括固定数量、调用唯一性、快照哈希、评分与配对指标，不是任意规模实验的通用分析器。默认只读、不写入结果、不调用模型；--output 可写入尚不存在的新文件，父目录须存在。没有该归档时不必执行此命令，新实验用 src.dual_study analyze 或 src.experiments analyze 分析。

归档按原始内容保存，历史来源、失败与负结果不随项目命名或文档调整而改写。模型间 MAE 与 Spearman 不是客观准确率；完整用量与费用依赖服务端 usage 和适用价表。
