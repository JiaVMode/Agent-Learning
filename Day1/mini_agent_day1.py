"""Mini Agent 入门练习：Function Calling + Agent Loop。

环境：Python 3.10+；仅使用标准库，无需 SDK 或 API Key。
直接在 gmy_2A100 服务器上运行，无需端口转发。
默认连接 http://127.0.0.1:8000/v1 的 Qwen3.5-9B vLLM 服务。
可通过 VLLM_BASE_URL、VLLM_MODEL 覆盖服务地址和模型名。
先执行 python3 -B mini_agent_day1.py --self-test，再执行 python3 -B mini_agent_day1.py。

注意：search_paper 仅搜索内置的三条演示论文记录，绝不联网。
"""

from __future__ import annotations

import sys

from agent import run_agent
from tools import calculator, execute_tool, search_paper


def self_test() -> None:
    """无需模型和 API Key；先验证工具本身正确且异常处理生效。"""
    assert calculator("123 * 456")["result"] == 56088
    assert calculator("(12 + 8) / 5")["result"] == 4.0
    assert search_paper("LoRA")["total"] >= 2
    assert execute_tool("calculator", {"expression": "1 / 0"})["ok"] is False
    assert execute_tool("calculator", {"expression": "__import__('os')"})["ok"] is False
    assert execute_tool("unknown_tool", {})["ok"] is False
    print("离线自测全部通过：计算、论文检索、除零错误、非法表达式、未知工具。")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        question = (
            "请计算 123 * 456，再从本地论文库中查找 LoRA 相关论文，"
            "告诉我找到哪些，并明确说明检索是否联网。并且告诉我调用了哪些工具。"
        )
        if len(sys.argv) > 1:
            question = " ".join(sys.argv[1:])
        print("[User]", question)
        print("\n[Final Answer]\n", run_agent(question))
