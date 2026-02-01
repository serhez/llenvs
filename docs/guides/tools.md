# Tools & Function Calling

llenvs provides first-class support for tool/function calling, enabling models to interact with environments through structured tool calls.

## Overview

Tool calling allows models to:
- Execute code (Python)
- Search for information
- Interact with external services
- Submit final answers

## Defining Tools

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

For environments with built-in tools, use `ToolEnvironment`:

```python
from llenvs.adapters import create_gem_tool_environment
from llenvs.core import AgentAction, ToolCall

# Create tool-enabled environment
env = create_gem_tool_environment(
    "math:GSM8K",
    tool_types=("python",),
)

state, _ = env.reset(options={"task_index": 0})
print(f"Tools: {[t.name for t in state.observation.available_tools]}")
# ['python', 'submit_answer']

# Use Python tool
call = ToolCall(id="1", name="python", arguments={"code": "print(0.15 * 80)"})
action = AgentAction(tool_calls=(call,))
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

## ToolEpisodeRunner

For running tool-enabled evaluations:

```python
from llenvs.evaluation import ToolEpisodeRunner

env = create_gem_tool_environment("math:GSM8K")
backend = OpenAIBackend(model="gpt-4o")

runner = ToolEpisodeRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0),
    system_prompt="Use tools to solve problems. Submit your final answer.",
)

result = runner.run_episode(task_index=0)
print(f"Success: {result.success}")
```

## Next Steps

- **[GEM Tool Environments](../adapters/gem.md#tool-enabled-environments)**
- **[Tools Reference](../reference/tools.md)**
