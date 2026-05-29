#工具函数
import re

def extract_score(content: str) -> float:
    """从大模型的非结构化文本中精准提取符合 'Score: X.XX' 格式的数字分数"""
    match = re.search(r'Score:\s*(\d+(\.\d+)?)', content)
    if match:
        return float(match.group(1))
    raise ValueError(f"无法从文本中解析出 Score 标记: {content}")