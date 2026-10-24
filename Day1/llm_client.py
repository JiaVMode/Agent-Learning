"""请求服务器内的 vLLM Chat Completions 服务，无需 API Key。"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
MODEL = os.getenv("VLLM_MODEL", "Qwen3.5-9B")


def request_model(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    """发送对话和工具定义，返回模型的 assistant 消息。"""
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)["choices"][0]["message"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM 请求失败（HTTP {exc.code}）：{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"无法连接 vLLM 服务 {BASE_URL}：{exc.reason}"
            "（检查服务器上的 vLLM 进程和端口）"
        ) from exc
