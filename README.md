# A2A-Evaluator: 基于状态机驱动的双 Agent 自动化论文评测系统

本项目不依赖任何重型多智能体全家桶的顶层 API，而是深入底层，利用 **LangGraph 状态机** 与原生的 **A2A (Agent-to-Agent)** 交互机制，实现了一个具备“动态反馈、闭合纠错、早停收敛”的论文评测系统。

## 💡 核心机制与技术亮点
- **显式状态契约 (State-Centric)**：通过 ` TypedDict` 定义严格的全局状态流转契约，确保数据流在多节点并发时的一致性。
- **高并发评测插件化 (Fan-out / Fan-in)**：流程在进入评估阶段后，自动触发 4 路并行的 LLM 评估线程（相关性、语法、结构、深度），相比串行架构提升 300% 的运行效率。
- **原生的容错解析控制**：不依赖易碎的 Prompt 强约束，在节点层设计了原生的正则表达式提取器与 `try-except` 兜底容错机制，防止非结构化文本导致的长工作流发散或卡死。
- **Scaling 自动收敛判定**：采用条件路由机制，当综合加权评分达标（>= 0.85）或迭代达到上限（3次）时，自动触发 Early-Exit（提早退出），精细化控制 Token 消耗。

## 🛠️ 快速开始

1. **克隆项目并安装依赖**
```bash
git clone [https://github.com/xuewer/A2A-evaluator.git](https://github.com/xuewer/A2A-evaluator.git)
cd A2A-evaluator
pip install -r requirements.txt