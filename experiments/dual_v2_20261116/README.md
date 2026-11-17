# 2026-11-16 双模型消融证据

本目录归档已执行批次，不是 demo。数据为 10 篇多主题合成开发集英文文本；第一模型请求标识 qwen3.8-27b，第二模型 kimi-k3。

- 40 个工作流，31 成功、9 失败；34 篇去重文本接受第二模型评分，33 成功、1 次截断。
- 共 320 次请求尝试，包括 4 次初始鉴权失败。没有补写成功结果，也没有将失败当零分。
- 主比较全反馈减无反馈 n=6，另有 4 个失败配对；两个裁判的判断方向不一致。

结果解释见 [实验报告](../../docs/RESULTS.md)，离线检查命令：

~~~powershell
python -B tools/verify_dual_results.py
~~~

|文件或目录|内容|
|---|---|
|MANIFEST.json|白名单导出文件的 SHA256|
|batch_plan.json、ablation/plan.json|固定配置、原始样本和任务顺序|
|ablation/job-*/result.json|各轮文本、反馈、评分、调用与失败|
|journals/|320 条请求记录，74 个任务日志|
|judge_plan.json、judge_results/|去重文本、评分目标及第二模型原始返回|
|DUAL_MODEL_REPORT.json|汇总指标及逐样本配对差值|
|SCORE_COMPARISON.jsonl|同文本四维评分对照|
|source_snapshot/|执行实验时的源码与依赖快照|
|verification_20261117/pytest.xml|实验后离线回归记录|

所有公开原始文件均按字节复制，保留旧项目显示名及 provenance 路径。当前源码已继续维护，因此默认只验证历史源码快照，不要求当前源码回退。MANIFEST 不包括本说明文件。
