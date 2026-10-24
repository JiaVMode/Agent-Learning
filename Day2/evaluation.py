"""演示性评测：工具路由、回答关键词和结构化输出。"""

from __future__ import annotations

import json

from agent import run_agent

EVAL_CASES = [
    ("计算 123 * 456，给出准确结果。", {"calculator"}, "56088"),
    ("在本地示例论文库搜索 LiST 和 EC-LoRA。", {"search_paper"}, "LiST"),
    ("计算 (12 + 8) / 5，并在本地示例库检索 HMEPR。", {"calculator", "search_paper"}, "HMEPR"),
    ("在本地示例论文库查询 QuantumPaper-DoesNotExist-999。", {"search_paper"}, None),
    ("你好，请介绍你能做什么。", set(), None),
]


def evaluate_online() -> None:
    """A starter eval: tool routing and structured-output success, not factual benchmark."""
    records = []
    for i, (question, expected_tools, token) in enumerate(EVAL_CASES, start=1):
        actual = run_agent(question, verbose=False)
        used = set(actual["used_tools"])
        tool_ok = used == expected_tools
        answer_ok = token is None or token.lower() in actual["answer"].lower()
        output_ok = actual.get("final_from_model", False)
        records.append({"case": i, "status": actual["status"], "used_tools": sorted(used),
                        "tool_route_pass": tool_ok, "keyword_pass": answer_ok, "schema_pass": output_ok})
        print(json.dumps(records[-1], ensure_ascii=False))
    print("初步工具路由通过率:", f"{sum(x['tool_route_pass'] for x in records)}/{len(records)}")
    print("说明：这是演示性 smoke test，不能证明事实准确率或综合 Agent 能力。")
