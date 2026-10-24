"""本地工具实现与白名单分发；论文数据仅用于演示。"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

PAPERS = [
    {
        "title": "LiST: Local-Simplex Test-Time LoRA Fusion",
        "year": 2026,
        "summary": "研究无标签测试时 LoRA 融合，通过相关适配器检索和局部权重搜索实现动态适配。",
        "keywords": "LoRA fusion test-time adaptation adapter 融合 检索",
    },
    {
        "title": "EC-LoRA: Energy-Driven Continual LoRA Implicit Generation",
        "year": 2026,
        "summary": "研究基于条件能量模型的持续学习 LoRA 隐式生成与参数优化。",
        "keywords": "LoRA continual learning energy model adapter 持续学习 能量模型",
    },
    {
        "title": "HMEPR: A Hierarchical Multimodal Emotion Recognition Framework With Redundancy-Aware Feature Refinement",
        "year": 2026,
        "summary": "研究多模态情绪识别中的跨模态对齐、冗余抑制与特征融合。",
        "keywords": "multimodal emotion recognition sentiment 多模态 情绪识别",
    },
]

BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

def calculator(expression: str) -> dict[str, Any]:
    """安全地计算 + - * / 和括号组成的表达式。"""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression 必须是非空字符串")
    if len(expression) > 80:
        raise ValueError("表达式过长，最多 80 个字符")

    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPS:
            result = BINARY_OPS[type(node.op)](visit(node.left), visit(node.right))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
            result = UNARY_OPS[type(node.op)](visit(node.operand))
        else:
            raise ValueError("仅支持数字、括号和 + - * / 运算")
        if not math.isfinite(result) or abs(result) > 1e12:
            raise ValueError("计算结果超出允许范围")
        return result

    return {"expression": expression, "result": visit(tree.body)}

def search_paper(query: str) -> dict[str, Any]:
    """从内置示例论文库中按关键词匹配，不连接互联网。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    if len(query) > 200:
        raise ValueError("检索词过长")

    # 为英文术语和中文关键词提供最小的、可理解的匹配逻辑。
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", query.lower())
    matches = []
    for paper in PAPERS:
        content = " ".join(str(value) for value in paper.values()).lower()
        score = sum(1 for token in set(tokens) if token in content)
        if score:
            matches.append((score, paper))

    matches.sort(key=lambda item: item[0], reverse=True)
    return {
        "source": "本地示例论文库（非实时、未联网）",
        "query": query,
        "total": len(matches),
        "papers": [paper for _, paper in matches[:3]],
    }

def execute_tool(name: str, arguments: Any) -> dict[str, Any]:
    """白名单分发工具，并把执行错误变成可供模型观察的结果。"""
    try:
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        if name == "calculator":
            if set(arguments) != {"expression"}:
                raise ValueError("calculator 参数只允许 expression")
            return {"ok": True, "data": calculator(arguments["expression"])}
        if name == "search_paper":
            if set(arguments) != {"query"}:
                raise ValueError("search_paper 参数只允许 query")
            return {"ok": True, "data": search_paper(arguments["query"])}
        raise ValueError(f"未知工具: {name}")
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
