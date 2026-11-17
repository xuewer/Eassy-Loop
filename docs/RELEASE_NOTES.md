# 2026-11-17 发布整理

项目显示名由 EssayLoop 改为 Eassy Loop，仓库使用 Eassy-Loop。历史实验仍保留旧显示名与源码哈希。

- 新增 python -m src 入口、.env.example 和公开复算说明。
- CLI 输出的 run_id 与结果目录名一致。
- generation_history.text_changed 区分文本是否改变，summary.rewrites_changed_text 汇总真实改动次数；原 rewrites_completed 仍记录成功重写调用数。
- 兼容 Windows 实验索引中的反斜杠路径，并保持越界路径检查。
- 发布固定批次原始结果、调用日志与历史源码快照，保留失败及负结果。
- README 和终端说明改为可复用文档，移除会话叙述与本机绝对路径；密钥、IDE 文件和助手审阅日志不进入发布内容。

本地验证：149 项离线测试通过；40 个工作流、34 个第二裁判记录、320 次调用的归档指标核对通过。此次整理没有新增模型调用或训练。该测试数量不代表覆盖率，CI 的实际结果应以 GitHub 运行记录为准。
