"""工具决策、真实执行记录和最终结构化回答。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

# Day1 与 Day2 是同级目录，共用 Day1 的请求与工具实现。
learning_dir = str(Path(__file__).resolve().parent.parent)
if learning_dir not in sys.path:
    sys.path.insert(0, learning_dir)

from Day1.llm_client import request_completion
from Day1.tool_schemas import TOOLS
from Day1.tools import execute_tool
from output_schema import OUTPUT_FORMAT, validate_answer
from prompts import FINALIZE_INSTRUCTION, INSTRUCTIONS

MAX_STEPS = 6


def _result(
    answer: dict[str, Any], trace: list[dict[str, Any]], *, from_model: bool
) -> dict[str, Any]:
    """执行记录由 Python 填写，模型只生成业务回答的四个字段。"""
    return {
        **answer,
        "trace": trace,
        "used_tools": [event["name"] for event in trace if event["executed"]],
        "final_from_model": from_model,
    }


def _failure(
    message: str, trace: list[dict[str, Any]], status: str = "failed"
) -> dict[str, Any]:
    return _result({
        "status": status,
        "answer": message,
        "evidence": [],
        "limitations": [message],
    }, trace, from_model=False)


def run_agent(
    question: str,
    *,
    request_fn: Callable[..., dict[str, Any]] | None = None,
    max_steps: int = MAX_STEPS,
    verbose: bool = True,
) -> dict[str, Any]:
    """先处理工具调用，再单独使用 JSON Schema 约束最终回答。"""
    if not isinstance(question, str) or not question.strip():
        return _failure("问题不能为空", [])
    if max_steps < 1:
        return _failure("最大决策步数必须大于 0", [])
    if request_fn is None:
        request_fn = request_completion

    history: list[dict[str, Any]] = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": question},
    ]
    trace: list[dict[str, Any]] = []
    finalizing = False

    for step in range(1, max_steps + 1):
        if verbose:
            phase = "生成结构化回答" if finalizing else "模型开始决策"
            print(f"\n[Step {step}] {phase}")
        try:
            if finalizing:
                response = request_fn(history, response_format=OUTPUT_FORMAT)
            else:
                response = request_fn(history, tools=TOOLS)
            choice = response["choices"][0]
            message = choice["message"]
        except Exception as exc:
            return _failure(f"vLLM 请求失败（{type(exc).__name__}）：{exc}", trace)

        if choice.get("finish_reason") not in ("stop", "tool_calls"):
            return _failure("模型回复未完成，无法确认最终答案。", trace)
        calls = message.get("tool_calls") or []

        if finalizing:
            if calls:
                return _failure("结构化回答阶段意外返回了工具调用。", trace)
            text = message.get("content") or ""
            if not text:
                return _failure("模型没有提供可解析的最终文本。", trace)
            try:
                answer = validate_answer(json.loads(text))
            except (ValueError, TypeError) as exc:
                return _failure(f"最终输出未通过 JSON 验证：{type(exc).__name__}", trace)
            return _result(answer, trace, from_model=True)

        if not calls:
            text = message.get("content") or ""
            if not text:
                return _failure("模型没有提供工具请求或最终文本。", trace)
            history.extend([
                {"role": "assistant", "content": text},
                {"role": "user", "content": FINALIZE_INSTRUCTION},
            ])
            finalizing = True
            continue

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
            except json.JSONDecodeError:
                args = None
                executed = False
                result = {"ok": False, "error": "工具参数不是有效 JSON"}
            else:
                result = execute_tool(name, args)
                executed = True
                if verbose:
                    print("调用工具:", name)
                    print("工具结果:", result)
            trace.append({
                "step": step, "name": name, "arguments": args,
                "result": result, "executed": executed,
            })
            history.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })

    return _failure(f"达到最大决策步数 {max_steps}，未确认任务完成。", trace, "partial")
