import os

# 关键环隔离：防止在启用了本地全局科学上网时导致阿里云报错 ProxyError
os.environ["NO_PROXY"] = "aliyuncs.com,127.0.0.1,localhost"

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from src.nodes import (
    EvaluationState, generator_node, check_relevance,
    check_grammar, analyze_structure, evaluate_depth, calculate_final_score
)

load_dotenv()


def route_decision(state: EvaluationState) -> str:
    """动态条件路由"""
    if state["final_score"] >= 0.85 or state["current_round"] >= 3:
        return "accept"
    return "reject"


# 组装状态机
workflow = StateGraph(EvaluationState)

workflow.add_node("generator", generator_node)
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("check_grammar", check_grammar)
workflow.add_node("analyze_structure", analyze_structure)
workflow.add_node("evaluate_depth", evaluate_depth)
workflow.add_node("calculate_final_score", calculate_final_score)

# 拓扑连线
workflow.add_edge(START, "generator")
workflow.add_edge("generator", "check_relevance")
workflow.add_edge("generator", "check_grammar")
workflow.add_edge("generator", "analyze_structure")
workflow.add_edge("generator", "evaluate_depth")

workflow.add_edge("check_relevance", "calculate_final_score")
workflow.add_edge("check_grammar", "calculate_final_score")
workflow.add_edge("analyze_structure", "calculate_final_score")
workflow.add_edge("evaluate_depth", "calculate_final_score")

workflow.add_conditional_edges(
    "calculate_final_score",
    route_decision,
    {
        "accept": END,
        "reject": "generator"
    }
)

app = workflow.compile()

# ==================== 终审报告输出 ====================
if __name__ == "__main__":
    test_topic = "The Impact of Artificial Intelligence on Future Employment"
    print(f"🚀 系统启动，评测主题: {test_topic}")

    # 运行图，拿到全套最终的状态数据
    res = app.invoke({"topic": test_topic, "current_round": 0})

    print("\n" + "=" * 20 + " 📝 智能体撰写的短文内容 " + "=" * 20)
    print(res["essay"])  # 👈 这一行负责把写好的完整文章打印出来
    print("=" * 65)

    print("\n" + "=" * 21 + " 📊 各维度详细评测得分 " + "=" * 21)
    print(f"📌 给定主题: {res['topic']}")
    print(f"🔄 迭代轮数: {res['current_round']} 轮")
    print("-" * 65)
    # 👈 依次打印出各个具体维度的核心得分
    print(f"🎯 核心指标 [相关性得分 (Relevance)]: {res['relevance_score']:.2f} / 1.00")
    print(f"✍️ 核心指标 [语法表达得分 (Grammar)  ]: {res['grammar_score']:.2f} / 1.00")
    print(f"🧱 核心指标 [篇章结构得分 (Structure)]: {res['structure_score']:.2f} / 1.00")
    print(f"💡 核心指标 [论证深度得分 (Depth)    ]: {res['depth_score']:.2f} / 1.00")
    print("-" * 65)
    print(f"🏆 最终综合加权得分 (Final Score)      : {res['final_score']:.2f} / 1.00")

    # 根据分数给出一个人类可读的最终结论
    if res['final_score'] >= 0.85:
        print("🟢 综合评定结论: 该短文质量优秀，通过多智能体自动化评测系统！")
    else:
        print("🟡 综合评定结论: 未能达到0.85分，已触发最大迭代保护强行退出。")
    print("=" * 65)