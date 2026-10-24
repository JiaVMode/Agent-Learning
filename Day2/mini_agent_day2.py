"""Day2 运行入口：提示词、结构化回答和演示评测。

Day1 与 Day2 目录需要保持同级，共用 Day1 的工具和 vLLM 请求。
无需 API Key，模型服务的启动方式与 Day1 相同。

    python3 -B mini_agent_day2.py --self-test
    python3 -B mini_agent_day2.py --demo
    python3 -B mini_agent_day2.py --eval
    python3 -B mini_agent_day2.py "在本地示例库查找 LiST，并计算 123 * 456"
"""

from __future__ import annotations

import argparse
import json

from agent import run_agent

DEFAULT_QUESTION = '在本地示例论文库查找 LiST，并计算 123 * 456。'


def main() -> None:
    parser = argparse.ArgumentParser(description="Day2：vLLM 工具调用与结构化回答")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-test", action="store_true", help="离线检查流程")
    modes.add_argument("--demo", action="store_true", help="运行默认任务")
    modes.add_argument("--eval", action="store_true", help="运行五条演示评测")
    parser.add_argument("question", nargs="*", help="要处理的问题")
    args = parser.parse_args()

    if args.self_test:
        from self_test import self_test
        self_test()
    elif args.eval:
        from evaluation import evaluate_online
        evaluate_online()
    else:
        question = DEFAULT_QUESTION if args.demo or not args.question else " ".join(args.question)
        print(json.dumps(run_agent(question), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
