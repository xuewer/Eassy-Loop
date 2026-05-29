#核心节点
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from src.utils import extract_score

# 1. 定义全局状态
class EvaluationState(TypedDict):
    topic: str
    essay: str
    feedback: str
    relevance_score: float
    grammar_score: float
    structure_score: float
    depth_score: float
    final_score: float
    current_round: int

# 2. LLM
llm = ChatTongyi(model="qwen-plus", temperature=0.3)

def generator_node(state: EvaluationState) -> EvaluationState:
    """写手智能体：负责初稿撰写与后期重写优化"""
    current_round = state.get("current_round", 0) + 1
    if not state.get("essay"):
        prompt = ChatPromptTemplate.from_template(
            "Write a short essay on the topic: {topic}. Focus on quality and deep analysis."
        )
        chain = prompt | llm
        essay = chain.invoke({"topic": state["topic"]}).content
    else:
        prompt = ChatPromptTemplate.from_template(
            "Original Essay: {essay}\nFeedback: {feedback}\nRewrite to improve the score."
        )
        chain = prompt | llm
        essay = chain.invoke({"essay": state["essay"], "feedback": state["feedback"]}).content
    print(f"\n [Round {current_round}] Generator 已提交最新论文修改稿。")
    return {"essay": essay, "current_round": current_round}

def check_relevance(state: EvaluationState) -> EvaluationState:
    """专家1：相关性审查"""
    prompt = ChatPromptTemplate.from_template(
        "Analyze relevance of essay to topic '{topic}'. Start with 'Score: ' (0-1), then explain.\n\nEssay: {essay}"
    )
    result = (prompt | llm).invoke({"topic": state["topic"], "essay": state["essay"]})
    try: score = extract_score(result.content)
    except: score = 0.5
    return {"relevance_score": score, "feedback": f"Relevance feedback: {result.content}"}

def check_grammar(state: EvaluationState) -> EvaluationState:
    """专家2：语法与语言表达审查"""
    prompt = ChatPromptTemplate.from_template("Score grammar (0-1) for essay. Start with 'Score: '.\n\nEssay: {essay}")
    result = (prompt | llm).invoke({"essay": state["essay"]})
    try: score = extract_score(result.content)
    except: score = 0.5
    return {"grammar_score": score}

def analyze_structure(state: EvaluationState) -> EvaluationState:
    """专家3：篇章布局结构审查"""
    prompt = ChatPromptTemplate.from_template("Score structure (0-1) for essay. Start with 'Score: '.\n\nEssay: {essay}")
    result = (prompt | llm).invoke({"essay": state["essay"]})
    try: score = extract_score(result.content)
    except: score = 0.5
    return {"structure_score": score}

def evaluate_depth(state: EvaluationState) -> EvaluationState:
    """专家4：论证深度审查"""
    prompt = ChatPromptTemplate.from_template("Score depth (0-1) for essay. Start with 'Score: '.\n\nEssay: {essay}")
    result = (prompt | llm).invoke({"essay": state["essay"]})
    try: score = extract_score(result.content)
    except: score = 0.5
    return {"depth_score": score}

def calculate_final_score(state: EvaluationState) -> EvaluationState:
    """裁判节点：动态加权汇总"""
    final_score = (
        state["relevance_score"] * 0.3 +
        state["grammar_score"] * 0.2 +
        state["structure_score"] * 0.2 +
        state["depth_score"] * 0.3
    )
    print(f"当前评估总分: {final_score:.2f} (相关性:{state['relevance_score']}, 语法:{state['grammar_score']})")
    return {"final_score": final_score}