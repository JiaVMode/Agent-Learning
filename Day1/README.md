# Day1：运行一个会调用工具的 Mini Agent

这个练习让 Qwen3.5-9B 根据问题选择工具，由 Python 执行工具，再让模型根据结果回答。

目前有两个工具：

- `calculator`：计算加、减、乘、除和括号组成的表达式。
- `search_paper`：搜索 `tools.py` 中的三条演示论文记录，不联网。

本文使用 [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) 演示。Day1 代码使用 Python 3.10+ 标准库，无需安装 OpenAI SDK；示例中的模型服务采用无 API key 的配置。

## 1. 创建并检查运行环境

以下命令以有可用 NVIDIA GPU 和驱动的 Linux 机器为例，统一使用 Python 3.12 和 vLLM 0.18.0。先在终端打开包含 `Day1` 文件夹的课程目录。

已经有兼容环境时，激活它后直接做第 1.4 节的检查。首次准备环境时，按下面的顺序执行。

### 1.1 安装环境管理工具 uv

uv 用来创建 Python 虚拟环境和安装依赖。先检查是否已经安装：

```bash
uv --version
```

如果提示找不到命令，按 [uv 官方安装方式](https://docs.astral.sh/uv/getting-started/installation/) 执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

`export PATH=...` 让当前终端能找到刚安装的 `uv` 命令。

### 1.2 创建并激活虚拟环境

在包含 `Day1` 文件夹的课程目录执行：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

第一行在课程目录下创建名为 `.venv` 的环境，用于单独保存 Python 和依赖。第二行让当前终端使用这个环境。

### 1.3 安装 vLLM

保持环境已激活，执行：

```bash
uv pip install "vllm==0.18.0" --torch-backend=auto
```

`==0.18.0` 表示固定 vLLM 版本；`--torch-backend=auto` 让 uv 根据 NVIDIA 驱动选择 PyTorch 后端。这种安装方式参照 [vLLM 官方快速开始](https://docs.vllm.ai/en/v0.18.0/getting_started/quickstart/)。安装需要联网，等待命令完成后再继续。

### 1.4 检查是否安装成功

```bash
python --version
vllm --version
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

应能看到 Python 3.12、vLLM 0.18.0，以及 `CUDA available: True`。如果显示 `False`，先用 `nvidia-smi` 检查 GPU 和驱动是否可用，再检查当前环境中的 PyTorch。

### 1.5 下次使用时怎样进入环境

环境只需要创建和安装一次。每次重新打开终端，在课程目录执行：

```bash
source .venv/bin/activate
```

退出当前环境时执行：

```bash
deactivate
```

下面会用两个终端：终端 A 运行模型服务，终端 B 运行 Day1。两个终端都在同一台机器上操作。

模型服务和 Day1 在同一台机器上运行，所以使用 `127.0.0.1`。它表示运行命令的这台机器。

## 2. 启动 vLLM 模型服务

### 2.1 先检查服务是否已经启动

在运行模型服务的机器上执行：

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

如果返回内容中有 `"id":"Qwen3.5-9B"`，说明模型接口可用。JSON 中有没有空格不影响含义。

复用服务时，还需要启用了 `--enable-auto-tool-choice` 和 `--tool-call-parser qwen3_coder`。第 2.2 节的启动命令包含这两项。可以用下面的命令查看服务进程的启动参数：

```bash
pgrep -af 'vllm.*(serve|api_server)'
```

确认模型名和工具调用参数后，直接进入第 3 节。

如果连接失败，可能是尚未启动，也可能是模型正在加载。已经执行过启动命令时，先等待加载完成，不要重复启动。

### 2.2 尚未启动时，在终端 A 执行

先查看 GPU 使用情况：

```bash
nvidia-smi
```

然后启动 Qwen3.5-9B：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-9B \
  --served-model-name Qwen3.5-9B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --language-model-only \
  --enforce-eager
```

每行结尾的 `\` 表示命令还没有结束；最后一行不加。整段一起复制执行即可。

`Qwen/Qwen3.5-9B` 是模型仓库名称。模型未缓存时，启动过程需要联网下载；已有模型文件时，可以把这个位置换成自己机器上的模型目录。`--served-model-name` 指定客户端使用的名称，后面的 Day1 代码会使用它。

这里使用标准的 `vllm serve` 命令启动服务，相关用法见 [vLLM 官方快速开始](https://docs.vllm.ai/en/v0.18.0/getting_started/quickstart/)。启动前确认所在环境中能运行 `vllm` 命令，服务采用无 API key 的配置，不设置 `--api-key` 或 `VLLM_API_KEY`。

模型加载需要一些时间，**保持终端 A 打开**。这是前台服务，运行时终端不会返回命令提示符。

再打开同一台机器上的终端 B，执行第 2.1 节的 `curl` 命令。确认能返回模型名后，再运行 Day1。

### 2.3 启动参数分别表示什么

上面的启动命令包含以下设置：

| 参数 | 示例值 | 含义 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0` | 使用 GPU 0 |
| `vllm serve` 后的模型参数 | `Qwen/Qwen3.5-9B` | 模型仓库名称，也可以换成已下载模型的目录 |
| `--served-model-name` | `Qwen3.5-9B` | 客户端请求中填写的模型名 |
| `--host` / `--port` | `127.0.0.1` / `8000` | 在本机的 8000 端口提供接口 |
| `--dtype` | `bfloat16` | 使用 BF16 数值精度 |
| `--tensor-parallel-size` | `1` | 模型使用一张 GPU |
| `--gpu-memory-utilization` | `0.85` | 按 GPU 总显存的 85% 设置 vLLM 显存预算 |
| `--max-model-len` | `32768` | 一次请求的输入和输出合计最多 32768 个 token |
| `--max-num-seqs` | `4` | vLLM 每次调度最多处理 4 条序列，与 Agent 的 Step 数量无关 |
| `--enable-auto-tool-choice` | 启用 | 允许模型自行决定是否请求工具调用 |
| `--tool-call-parser` | `qwen3_coder` | 将模型生成的工具调用解析为 `tool_calls` |
| `--reasoning-parser` | `qwen3` | 解析模型输出中的思考内容与回答内容 |
| `--language-model-only` | 启用 | 使用文本输入，禁用图片和视频输入 |
| `--enforce-eager` | 启用 | 使用即时执行模式，关闭 CUDA Graph |

token 是模型处理文字时使用的单位，与汉字、英文单词不一一对应。上下文包括问题、历史消息、工具定义、工具返回内容和本次生成的输出。

这些参数值是示例，实际可用的上下文长度和并发数量取决于模型与显存。显存不足时，先检查 GPU 占用，并根据情况缩短上下文或降低并发数量。

模型服务启动后会持续等待请求。Day1 程序结束后，模型服务仍可供下一次运行使用。用本节命令在前台启动的服务，可以在终端 A 按 `Ctrl+C` 停止。

## 3. 运行 Day1

在终端 B 中，先打开包含 `Day1` 文件夹的课程目录，激活第 1 节创建的环境，再进入 Day1：

```bash
source .venv/bin/activate
cd ./Day1
python3 --version
```

终端 A 激活的环境不会自动作用到终端 B，所以新终端需要再激活一次。如果使用的是其他兼容环境，激活对应环境即可。Day1 本身只用标准库，Python 3.10 或更高版本都可以运行；这里统一使用刚创建的 Python 3.12 环境。

### 3.1 先检查工具是否正常

```bash
python3 -B mini_agent_day1.py --self-test
```

看到“离线自测全部通过”即可。这一步只检查 Python 工具，不请求模型；即使 vLLM 没有启动，也可以运行。

`-B` 表示不生成 `__pycache__` 缓存文件。

### 3.2 运行默认问题

```bash
python3 -B mini_agent_day1.py
```

默认问题要求计算 `123 * 456`、查找 LoRA 论文，并说明检索是否联网、使用了哪些工具。

### 3.3 在命令行输入自己的问题

```bash
python3 -B mini_agent_day1.py "请使用 calculator 计算 123 * 456，最后告诉我结果。"
```

引号中的文字会替换代码中的默认问题。下面是一种预期输出，最终回答的措辞可能不同：

```text
[Step 1] 模型开始决策
调用工具: calculator
工具结果: {'ok': True, 'data': {'expression': '123 * 456', 'result': 56088}}

[Step 2] 模型开始决策

[Final Answer]
123 × 456 = 56088
```

“调用工具”和“工具结果”由 Python 在 `execute_tool(...)` 返回后打印。`ok: True` 表示工具执行成功；`ok: False` 时需要看返回的错误内容。

## 4. 五个 Python 文件各自负责什么

| 文件 | 负责的事情 |
| --- | --- |
| [mini_agent_day1.py](mini_agent_day1.py) | 运行入口：读取问题、处理 `--self-test`、打印最终答案 |
| [agent.py](agent.py) | Agent 循环：请求模型、处理工具调用、执行工具、回传结果 |
| [llm_client.py](llm_client.py) | 将对话和工具定义发送给 vLLM，读取模型回复 |
| [tool_schemas.py](tool_schemas.py) | 告诉模型有哪些工具，以及各工具的名称、用途和参数 |
| [tools.py](tools.py) | 实现计算器、论文检索和 `execute_tool` 白名单分发 |

建议先看入口文件，再看 `agent.py`，遇到模型请求和工具执行时，再分别看 `llm_client.py` 和 `tools.py`。

## 5. 一次工具调用到底怎样完成

以“计算 `123 * 456`”为例：

1. `mini_agent_day1.py` 将问题交给 `run_agent`。
2. `agent.py` 把问题和工具定义交给 `request_model`。
3. `llm_client.py` 请求 vLLM 的 `/v1/chat/completions` 接口。
4. 模型返回 `tool_calls`，其中包含工具名和参数。
5. `agent.py` 解析参数，调用 `execute_tool`。
6. `tools.py` 中的 Python 函数执行计算，返回结果。
7. Agent 将结果加入对话，再请求模型生成回答。

### 5.1 工具定义与工具实现

`tool_schemas.py` 是给模型看的工具说明。例如，`calculator` 的参数定义要求传入一个名为 `expression` 的字符串。

`tools.py` 才包含实际计算逻辑。`execute_tool` 根据工具名选择对应函数，并检查参数。计算器使用 AST 解析表达式，只允许规定的数字和运算符。

新增工具时，需要同时完成三件事：在 `tools.py` 实现函数，在 `execute_tool` 中加入分发逻辑，在 `tool_schemas.py` 中补充同名工具的说明和参数。

### 5.2 为什么要用 `json.loads`

模型返回的工具参数在 `arguments` 字段中，是一个 JSON 字符串。例如：

```python
import json

arguments = '{"expression": "123 * 456"}'
args = json.loads(arguments)
# args 现在是 Python 字典：{"expression": "123 * 456"}
```

Agent 中实际执行和打印的核心代码是：

```python
name = function["name"]
args = json.loads(function["arguments"])
result = execute_tool(name, args)

print("调用工具:", name)
print("工具结果:", result)
```

如果 JSON 解析失败，代码会记录参数错误，将错误结果交给模型处理。

### 5.3 工具结果怎样交回模型

`history` 保存本次运行的对话。每次请求都会把它发给模型，其中可能包含四种角色：

| `role` | 内容 |
| --- | --- |
| `system` | Agent 的行为要求，例如遇到计算时使用计算器 |
| `user` | 用户的问题 |
| `assistant` | 模型的工具调用请求或最终回答 |
| `tool` | Python 执行工具后返回的结果 |

工具结果使用 `json.dumps(result, ensure_ascii=False)` 转成 JSON 文本，放入 `tool` 消息。`tool_call_id` 对应模型发起调用时的 `id`，用于说明“这是哪一次工具调用的结果”。

因此，需要先保存模型发起的工具调用，再保存对应工具结果。下一次模型请求带上这些消息，模型才能根据真实结果继续处理。

## 6. 为什么会有多个 Step

**一个 Step 代表请求模型一次。** 同一个 Step 可以返回多个工具调用；当前代码会按 `for` 循环逐一执行它们。

默认任务通常分成两轮：

| Step | 模型与 Python 分别做什么 |
| --- | --- |
| Step 1 | 模型请求计算和论文检索，Python 执行两个工具并保存结果 |
| Step 2 | 模型读取工具结果，给出最终回答，循环结束 |

模型返回 `tool_calls` 时，Agent 执行工具后继续下一轮；模型没有返回工具调用时，Agent 返回它的文本回答。

### 6.1 让任务包含前后依赖

可以运行下面的问题：

```bash
python3 -B mini_agent_day1.py "先用 search_paper 查找 LoRA 相关论文。收到检索结果后，再用 calculator 计算论文数量乘以 10。两次工具调用分两轮进行，最后汇总结果。"
```

期望流程是：

1. Step 1：检索，得到 2 篇论文。
2. Step 2：读取检索结果，计算 `2 * 10`，得到 `20`。
3. Step 3：读取计算结果，生成最终回答。

实际 Step 数量取决于模型返回的内容。这个问题明确要求第二次调用等待第一次的结果，便于观察多轮流程。

### 6.2 最多允许几轮

`agent.py` 中设置了：

```python
MAX_STEPS = 6
```

需要更高上限时，可以改成 `10`，然后重新运行程序。它只控制最多请求模型几次；模型给出最终回答时就会提前结束。

最后一次用于生成最终回答的请求也占一轮。达到上限时，程序会返回“不能确认已完成”的提示。

## 7. 常改的配置

### 7.1 更换默认问题

修改 `mini_agent_day1.py` 中的 `question`。临时尝试问题时，直接使用第 3.3 节的命令行方式即可。

### 7.2 修改服务地址或模型名

`llm_client.py` 默认读取：

```python
BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
MODEL = os.getenv("VLLM_MODEL", "Qwen3.5-9B")
```

也可以在运行命令前设置环境变量，例如明确指定当前服务：

```bash
VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
VLLM_MODEL=Qwen3.5-9B \
python3 -B mini_agent_day1.py
```

客户端中的模型名应与 `/v1/models` 返回的 `id` 一致。改变客户端地址和名称后，还需要有对应的模型服务可用。

### 7.3 理解每次模型请求的设置

这些设置在 `llm_client.py` 的 `payload` 中：

| 设置 | 当前值 | 含义 |
| --- | --- | --- |
| `tool_choice` | `"auto"` | 由模型决定是否请求工具调用 |
| `temperature` | `0` | 使用较确定的生成方式，便于观察流程 |
| `max_tokens` | `1024` | 每次模型回复最多新生成 1024 个 token |
| `enable_thinking` | `False` | 本练习关闭模型的思考模式 |

`max_tokens` 控制本次回复长度，服务端的 `max-model-len` 控制输入加输出的总长度，两者需要同时满足。

## 8. 常见问题

| 现象 | 先做什么 |
| --- | --- |
| 提示无法连接 vLLM / `Connection refused` | 在同一服务器执行第 2.1 节的 `curl`，确认服务已启动并加载完成 |
| 提示 `Address already in use` | 8000 端口已被占用；先用 `curl` 确认是否就是需要的模型服务，已有服务时直接复用 |
| 启动时显存不足 | 用 `nvidia-smi` 查看 GPU 占用，确认是否已有模型服务；根据情况调整显存预算、缩短上下文或降低并发。模型本身过大时，需要换用更小的模型 |
| 提示模型名不存在 | 对照 `/v1/models` 返回的 `id` 检查 `VLLM_MODEL` |
| 提示找不到 `agent`、`tools` 等模块 | 保持五个 Python 文件在同一目录，进入 Day1 后运行入口文件 |
| 没有出现“调用工具” | 查看模型是否返回了工具请求；先用第 3.3 节明确要求计算的例子，并确认服务启用了自动工具调用和对应解析器 |
| 提示超过上下文长度 | 缩短问题或工具返回内容，减少累积的对话长度，并给输出保留 token 空间 |
| 提示达到最多 Step 上限 | 查看任务是否仍需要工具结果；可以简化任务或调整 `MAX_STEPS`，上限提示表示程序尚未确认完成 |

`search_paper` 始终只检索代码中的演示记录。返回论文数量表示匹配到了多少条内置记录。
