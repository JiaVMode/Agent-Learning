"""Agent 循环：模型决策、执行工具、回传结果和生成答案。"""

from __future__ import annotations

import json
from typing import Any

from llm_client import request_model
from tool_schemas import TOOLS
from tools import execute_tool

MAX_STEPS = 6  # 一个 step 代表一次模型决策，避免无限循环


def run_agent(question: str, max_steps: int = MAX_STEPS) -> str:
    """模型选择工具 -> Python 执行 -> 结果回传 -> 模型继续判断。"""
    instructions = (
        "你是一个严谨的中文科研助理。遇到数学运算请使用 calculator；"
        "需要从本地论文库找论文时请调用 search_paper。"
        "search_paper 只能搜索内置演示记录，绝不能声称联网、覆盖最新论文或查到了不存在的文章。"
        "工具返回错误时可以修改参数重试，不能伪造执行结果。"
        "在信息充分后及时给出最终答案。"
    )
    history: list[dict[str, Any]] = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        print(f"\n[Step {step}] 模型开始决策")
        try:
            message = request_model(history, TOOLS)
        except RuntimeError as exc:
            return str(exc)
        except Exception as exc:
            return f"vLLM 请求失败：{type(exc).__name__}：{exc}"

        # vLLM 的 Chat Completions 通过 tool_calls 返回工具请求。
        calls = message.get("tool_calls") or []
        if not calls:
            return message.get("content") or "模型未继续调用工具，也没有返回文本。"

        history.append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": calls,
        })

        for call in calls:
            function = call["function"]
            name = function["name"]
            try:
                args = json.loads(function["arguments"])
            except json.JSONDecodeError as exc:
                result = {"ok": False, "error": "无法解析 JSON 参数", "detail": str(exc)}
                print("工具参数解析失败:", name)
            else:
                result = execute_tool(name, args)
                print("调用工具:", name)
                print("工具结果:", result)

            # 将真实执行结果交回模型，tool_call_id 对应模型发起的调用。
            history.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })

    return f"已达到最多 {max_steps} 次模型决策上限，任务停止；不能确认已完成。"
