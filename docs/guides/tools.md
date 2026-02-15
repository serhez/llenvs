# Tools

llenvs provides first-class support for tool/function calling, enabling models to interact with environments through structured tool calls.

## Overview

Tool calling allows models to:
- Execute code (Python)
- Search for information
- Interact with external services
- Submit final answers

## Defining Tools

### Manual Definition

```python
from llenvs.core import (
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
)

# Define a tool
weather_tool = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a city",
    parameters=(
        ToolParameter(
            name="city",
            type=ToolParameterType.STRING,
            description="The city name",
        ),
        ToolParameter(
            name="units",
            type=ToolParameterType.STRING,
            description="Temperature units",
            required=False,
            enum=("celsius", "fahrenheit"),
        ),
    ),
)

# Terminal tools end the episode when called
submit_tool = ToolDefinition(
    name="submit_answer",
    description="Submit the final answer",
    parameters=(
        ToolParameter("answer", ToolParameterType.STRING, "The answer"),
    ),
    is_terminal=True,
)

# Convert to API schemas
openai_schema = weather_tool.to_openai_schema()
anthropic_schema = weather_tool.to_anthropic_schema()
```

### From Python Functions

`ToolDefinition.from_callable()` generates a tool definition by inspecting a Python function's signature and docstring:

```python
from llenvs.core import ToolDefinition

def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: The city name.
        units: Temperature units (celsius or fahrenheit).
    """
    ...

# Auto-generate definition from function
weather_tool = ToolDefinition.from_callable(get_weather)
# name="get_weather", description="Get the current weather for a city."
# city: STRING, required=True, description="The city name."
# units: STRING, required=False, description="Temperature units (celsius or fahrenheit)."

# Override name/description, mark as terminal
submit_tool = ToolDefinition.from_callable(
    submit_answer, name="submit", is_terminal=True
)
```

Type mapping: `str`→STRING, `int`→INTEGER, `float`→NUMBER, `bool`→BOOLEAN, `list`→ARRAY, `dict`→OBJECT. Generic types (`list[int]`, `dict[str, Any]`) map via their origin. Unannotated parameters default to STRING. Google-style `Args:` docstring sections are parsed for parameter descriptions.

## Using generate_with_tools

```python
from llenvs.inference.backends import OpenAIBackend
from llenvs.inference import ChatMessage, SamplingParams

backend = OpenAIBackend(model="gpt-4o")

messages = [
    ChatMessage(role="user", content="What's the weather in Paris?"),
]

result = backend.generate_with_tools(
    messages,
    tools=[weather_tool],
    params=SamplingParams(temperature=0.0),
    tool_choice="auto",  # "auto", "none", "required", or tool name
)

if result.has_tool_calls:
    for call in result.tool_calls:
        print(f"Tool: {call.name}")
        print(f"Args: {call.arguments}")
else:
    print(f"Response: {result.text}")
```

## Handling Tool Results

```python
from llenvs.core import ToolCall, ToolResult
from llenvs.inference import ChatMessage

# After receiving tool calls
call = result.tool_calls[0]

# Execute the tool (your implementation)
weather_data = get_weather(call.arguments["city"])

# Create tool result
tool_result = ToolResult.success(
    call_id=call.id,
    tool_name=call.name,
    output=weather_data,
)

# Build messages for next turn
messages = [
    ChatMessage(role="user", content="What's the weather in Paris?"),
    ChatMessage(
        role="assistant",
        content=result.text,
        tool_calls=result.tool_calls,
    ),
    ChatMessage.tool_result(tool_result),
]

# Continue conversation
final_result = backend.generate_with_tools(messages, tools, params)
```

## Tool Executors

### SimpleToolExecutor

For synchronous Python functions:

```python
from llenvs.core import SimpleToolExecutor, ToolCall

executor = SimpleToolExecutor({
    "add": lambda a, b: str(a + b),
    "multiply": lambda a, b: str(a * b),
})

call = ToolCall(id="1", name="add", arguments={"a": 5, "b": 3})
result = executor.execute(call)
print(result.output)  # "8"
```

### AsyncToolExecutor

For I/O-bound tools with parallel execution:

```python
from llenvs.core import AsyncToolExecutor
import asyncio

async def fetch_weather(city: str) -> str:
    # Async HTTP call
    ...

def calculate(expr: str) -> str:
    return str(eval(expr))  # Sync functions work too

executor = AsyncToolExecutor(
    tools={"get_weather": fetch_weather, "calculate": calculate},
    timeout=30.0,
)

# Execute multiple calls in parallel
calls = (
    ToolCall(id="1", name="get_weather", arguments={"city": "Paris"}),
    ToolCall(id="2", name="get_weather", arguments={"city": "London"}),
)

results = asyncio.run(executor.execute_batch_async(calls))
```

### MCPToolExecutor

Connect to MCP (Model Context Protocol) servers:

```python
from llenvs.core import MCPToolExecutor, MCPServerConfig

config = MCPServerConfig(
    command="npx",
    args=("-y", "@modelcontextprotocol/server-filesystem", "/tmp"),
    timeout=30.0,
)

async with MCPToolExecutor(config) as executor:
    # Discover available tools
    tools = await executor.list_tools()
    print(f"Tools: {[t.name for t in tools]}")

    # Execute tool call
    call = ToolCall(id="1", name="read_file", arguments={"path": "/tmp/test.txt"})
    result = await executor.execute_async(call)
```

## Tool Environments

Environments with built-in tools expose tool definitions via `available_tools` on the observation. Create them with `environment_registry.get()` and pass `tool_types` to specify which tools to enable:

```python
from llenvs.core.registry import environment_registry
from llenvs.core import Action, ToolCall

# Create tool-enabled environment
env = environment_registry.get(
    name="math:GSM8K",
    adapter="gem",
    tool_types=("python",),
)

state, _ = env.reset(options={"task_index": 0})
print(f"Tools: {[t.name for t in state.observation.available_tools]}")
# ['python', 'submit_answer']

# Use Python tool
call = ToolCall(id="1", name="python", arguments={"code": "print(0.15 * 80)"})
action = Action(tool_calls=(call,))
result = env.step(state, action)

print(f"Output: {result.info['tool_results'][0].output}")
# '12.0'
```

## Tool-Specific Rewards

```python
from llenvs.core import ToolValidityReward, ToolEfficiencyReward

# Reward for valid tool calls (1.0 if all valid)
validity_reward = ToolValidityReward()

# Penalizes excess/duplicate calls
efficiency_reward = ToolEfficiencyReward(
    max_calls_per_step=5,
    penalty_per_excess=0.1,
    duplicate_penalty=0.2,
)
```

Both reward classes accept a `_weight` parameter. With `_weight=0.0`, the signal appears in the `SignalBundle` for inspection but contributes nothing to the total. This is used for auto-monitoring (see below).

### Auto-Monitoring

All tool environments (GEM, verifiers, OpenEnv) auto-attach `ToolValidityReward` and `ToolEfficiencyReward` with `weight=0.0`. These monitoring rewards:

- Track tool call validity and efficiency on every step
- Contribute nothing to the total reward (weight=0)
- Are accessible via `signal_bundle.by_type(RewardType.STEP)` for inspection

No configuration is needed — monitoring is always on.

## Running Tool Environments

`TrajectoryRunner` auto-detects tool environments via `available_tools` on the observation and handles tool calling automatically:

```python
from llenvs.core.registry import environment_registry
from llenvs.evaluation import TrajectoryRunner

env = environment_registry.get(name="math:GSM8K", adapter="gem", tool_types=("python",))
backend = OpenAIBackend(model="gpt-4o")

runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0),
    system_prompt="Use tools to solve problems. Submit your final answer.",
)

result = runner.run_trajectory(task_index=0)
print(f"Success: {result.success}")
```

## Text-Based Tool Calling

API backends (OpenAI, Anthropic) support native function calling. Local GPU backends (vLLM, HuggingFace) do not. When a tool environment is used with a non-function-calling backend, the runner needs a `ToolCallParser` to bridge the gap.

Without a parser, tools are silently ignored and a warning is logged:

```
WARNING: Environment provides 2 tools but backend 'VLLMBackend' does not
support function calling and no tool_call_parser is configured. Tools will
be ignored.
```

### HermesToolCallParser

The built-in `HermesToolCallParser` implements the Hermes format used by many open-source tool-calling models (NousResearch Hermes, Qwen, and other fine-tunes):

```python
from llenvs.core import HermesToolCallParser
from llenvs.evaluation import TrajectoryRunner

parser = HermesToolCallParser()

runner = TrajectoryRunner(
    environment=env,
    backend=vllm_backend,  # No native function calling
    tool_call_parser=parser,
)

# Also works with run_evaluation()
from llenvs.evaluation.runner import run_evaluation

result = run_evaluation(
    environment=env,
    backend=vllm_backend,
    tool_call_parser=parser,
)
```

The parser:
1. **Injects** formatted tool definitions into the system message as XML (`<tools>...</tools>`)
2. **Generates** using standard text generation (`generate_chat`)
3. **Parses** tool calls from `<tool_call>...</tool_call>` blocks in the response

### How the Runner Chooses

The runner uses a three-way priority:

1. **Native function calling** — if the backend supports it (`capabilities.supports_function_calling`), native tool calling is used regardless of whether a parser is configured
2. **Text-based parsing** — if tools are available and a `tool_call_parser` is set, tools are formatted as text and parsed from the response
3. **No tools** — if neither is available, tools are ignored and a warning is logged

### Custom Parsers

Implement the `ToolCallParser` protocol to support other formats:

```python
from llenvs.core.tool_parsing import ToolCallParser, ParsedToolResponse

class MyCustomParser:
    def format_tools(self, tools: tuple[ToolDefinition, ...]) -> str:
        # Format tools as text for the prompt
        ...

    def parse(self, text: str, available_tools: tuple[ToolDefinition, ...]) -> ParsedToolResponse:
        # Parse tool calls from model output
        ...
```

## Next Steps

- **[GEM Tool Environments](../adapters/gem.md#tool-enabled-environments)**
- **[Tools Reference](../reference/tools.md)**
