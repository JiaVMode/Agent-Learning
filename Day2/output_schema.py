"""最终回答的 JSON Schema、vLLM 请求格式和本地校验。"""

from __future__ import annotations

from typing import Any

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["completed", "partial", "failed"]},
        "answer": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "answer", "evidence", "limitations"],
    "additionalProperties": False,
}

OUTPUT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "research_agent_answer",
        "strict": True,
        "schema": ANSWER_SCHEMA,
    },
}


def validate_answer(value: Any) -> dict[str, Any]:
    """Validate the essential properties offline, even after schema-constrained generation.

    JSON Schema validates shape, not factual accuracy. Runtime trace stays outside
    model-generated JSON so it is always based on *executed* calls.
    """
    if not isinstance(value, dict):
        raise ValueError("回答必须是 JSON 对象")
    required = {"status", "answer", "evidence", "limitations"}
    if set(value) != required:
        raise ValueError(f"字段不匹配，必须恰好包含 {sorted(required)}")
    if value["status"] not in ("completed", "partial", "failed"):
        raise ValueError("status 必须是 completed / partial / failed")
    if not isinstance(value["answer"], str):
        raise ValueError("answer 必须是字符串")
    for key in ("evidence", "limitations"):
        if not isinstance(value[key], list) or not all(isinstance(x, str) for x in value[key]):
            raise ValueError(f"{key} 必须是字符串数组")
    return value
