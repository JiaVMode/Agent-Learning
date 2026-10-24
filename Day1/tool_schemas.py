"""提供给模型的工具名称、说明与参数结构，不执行工具。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "准确计算四则运算表达式，支持加、减、乘、除和括号。数学运算应优先使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "如 (12 + 8) / 5"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_paper",
            "description": "只在本地示例论文库中检索论文；不能搜索互联网或实时数据库。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "检索关键词，如 LoRA"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]
