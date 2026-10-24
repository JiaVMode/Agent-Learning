"""离线自测：用假模型响应检查流程，不请求真实服务。"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from agent import run_agent
from output_schema import validate_answer


def _response(
    content: str = "", calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    return {"choices": [{
        "message": {"role": "assistant", "content": content, "tool_calls": calls or []},
        "finish_reason": finish_reason,
    }]}


def _call(expression: str, call_id: str = "call_1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "calculator", "arguments": expression},
    }


class _FakeModel:
    def __init__(self, responses: list[dict[str, Any]]):
        self.pending = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.requests.append(deepcopy({"messages": messages, **kwargs}))
        if not self.pending:
            raise RuntimeError("没有更多假响应")
        return self.pending.pop(0)


def self_test() -> None:
    """保留格式、工具回传、真实 trace、错误和步数限制检查。"""
    good = {
        "status": "completed", "answer": "56088",
        "evidence": ["calculator: 56088"], "limitations": [],
    }
    assert validate_answer(good) == good
    for bad in (
        {**good, "status": "unknown"},
        {**good, "extra": 1},
        {**good, "evidence": "wrong"},
    ):
        try:
            validate_answer(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("坏结果未被拒绝")

    fake = _FakeModel([
        _response(calls=[_call('{"expression":"123 * 456"}')], finish_reason="tool_calls"),
        _response("计算完成。"),
        _response(json.dumps(good, ensure_ascii=False)),
    ])
    result = run_agent("计算 123 * 456", request_fn=fake, verbose=False)
    assert result["status"] == "completed" and result["final_from_model"]
    assert result["used_tools"] == ["calculator"]
    assert result["trace"][0]["result"]["data"]["result"] == 56088
    observation = fake.requests[1]["messages"][-1]
    assert observation["role"] == "tool" and observation["tool_call_id"] == "call_1"
    assert json.loads(observation["content"])["data"]["result"] == 56088
    assert "response_format" not in fake.requests[0]
    assert fake.requests[2]["response_format"]["json_schema"]["strict"] is True
    assert "tools" not in fake.requests[2]

    limited = _FakeModel([
        _response(calls=[_call('{"expression":"1/0"}')], finish_reason="tool_calls"),
    ])
    stop = run_agent("计算 1/0", request_fn=limited, max_steps=1, verbose=False)
    assert stop["status"] == "partial" and not stop["final_from_model"]
    assert stop["trace"][0]["result"]["ok"] is False
    assert stop["used_tools"] == ["calculator"]

    invalid = _FakeModel([
        _response(calls=[_call("{")], finish_reason="tool_calls"),
    ])
    rejected = run_agent("错误参数", request_fn=invalid, max_steps=1, verbose=False)
    assert rejected["used_tools"] == []
    assert rejected["trace"][0]["executed"] is False

    truncated = _FakeModel([
        _response("准备回答。"),
        _response(json.dumps(good), finish_reason="length"),
    ])
    unfinished = run_agent("计算结果", request_fn=truncated, verbose=False)
    assert unfinished["status"] == "failed" and not unfinished["final_from_model"]

    empty = _FakeModel([])
    assert run_agent(" ", request_fn=empty, verbose=False)["status"] == "failed"
    assert not empty.requests
    print("离线自测通过：JSON 校验、工具回传、真实 trace、两阶段输出、除零、非法参数、截断和步数限制。")
