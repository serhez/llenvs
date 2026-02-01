# Tools API Reference

This document covers tool/function calling support in llenvs.

## Tool Definitions

**Location**: `llenvs/core/tools.py`

### ToolParameterType

```python
class ToolParameterType(Enum):
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
```

### ToolParameter

```python
@dataclass(frozen=True)
class ToolParameter:
    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    enum: tuple[str, ...] | None = None

    def to_json_schema(self) -> dict[str, Any]: ...
```

### ToolDefinition

```python
@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: tuple[ToolParameter, ...] = ()
    is_terminal: bool = False  # Calling this tool ends the episode

    def to_openai_schema(self) -> dict[str, Any]: ...
    def to_anthropic_schema(self) -> dict[str, Any]: ...
```

Example:

```python
from llenvs.core import ToolDefinition, ToolParameter, ToolParameterType

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
```

## Tool Calls and Results

### ToolCall

```python
@dataclass(frozen=True)
class ToolCall:
    id: str                    # Unique ID for matching results to calls
    name: str                  # Name of tool to call
    arguments: dict[str, Any]  # Arguments to pass
```

### ToolResult

```python
class ToolResultStatus(Enum):
    SUCCESS = auto()
    ERROR = auto()
    INVALID_TOOL = auto()
    INVALID_ARGUMENTS = auto()

@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    status: ToolResultStatus
    output: str | dict[str, Any] = ""
    error: str | None = None

    @classmethod
    def success(cls, call_id, tool_name, output) -> ToolResult: ...
    @classmethod
    def from_error(cls, call_id, tool_name, error_message, status=ERROR) -> ToolResult: ...

    @property
    def is_success(self) -> bool: ...
    @property
    def is_error(self) -> bool: ...
```

## Tool Executors

### ToolExecutor Protocol

```python
class ToolExecutor(Protocol):
    def execute(self, call: ToolCall) -> ToolResult: ...
    def execute_batch(self, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]: ...
```

### SimpleToolExecutor

Executor backed by dict of Python callables:

```python
from llenvs.core import SimpleToolExecutor, ToolCall

def get_weather(city: str, units: str = "celsius") -> str:
    return f"Weather in {city}: 22° {units}"

executor = SimpleToolExecutor(tools={"get_weather": get_weather})

call = ToolCall(id="1", name="get_weather", arguments={"city": "Paris"})
result = executor.execute(call)
print(result.output)  # "Weather in Paris: 22° celsius"
```

### AsyncToolExecutor

**Location**: `llenvs/core/async_executor.py`

Executor that runs tool calls asynchronously with parallel batch execution:

```python
from llenvs.core import AsyncToolExecutor, ToolCall

async def fetch_weather(city: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.weather.com/{city}") as resp:
            return await resp.text()

def calculate(expr: str) -> str:
    return str(eval(expr))  # Sync also works

executor = AsyncToolExecutor(
    tools={"get_weather": fetch_weather, "calculate": calculate},
    timeout=30.0,
)

# Execute multiple calls in parallel
calls = (
    ToolCall(id="1", name="get_weather", arguments={"city": "Paris"}),
    ToolCall(id="2", name="get_weather", arguments={"city": "London"}),
    ToolCall(id="3", name="calculate", arguments={"expr": "2+2"}),
)
results = await executor.execute_batch_async(calls)  # Runs in parallel
```

Key features:
- Supports both sync and async callables
- Parallel execution via `asyncio.gather()` in batch mode
- Configurable timeout per tool call
- Sync functions run in thread pool to avoid blocking

### MCPToolExecutor

**Location**: `llenvs/core/mcp_executor.py`

Executor that connects to MCP (Model Context Protocol) servers:

```python
from llenvs.core import MCPToolExecutor, MCPServerConfig, ToolCall

# Connect to an MCP filesystem server
config = MCPServerConfig(
    command="npx",
    args=("-y", "@modelcontextprotocol/server-filesystem", "/tmp"),
)

async with MCPToolExecutor(config) as executor:
    # Discover available tools
    tools = await executor.list_tools()
    # (ToolDefinition(name="read_file", ...), ToolDefinition(name="write_file", ...), ...)

    # Execute a tool call
    call = ToolCall(id="1", name="read_file", arguments={"path": "/tmp/test.txt"})
    result = await executor.execute_async(call)
    print(result.output)

# Sync usage
with MCPToolExecutor(config) as executor:
    tools = executor.list_tools_sync()
    result = executor.execute(call)
```

## Tool Environment

**Location**: `llenvs/core/tool_environment.py`

### ToolEnvironment Protocol

```python
@runtime_checkable
class ToolEnvironment(Protocol[HiddenT]):
    @property
    def available_tools(self) -> tuple[ToolDefinition, ...]: ...

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[AgentObservation, HiddenT], dict[str, Any]]: ...

    def step(
        self,
        state: State[AgentObservation, HiddenT],
        action: AgentAction,
    ) -> StepResult[AgentObservation, HiddenT]: ...

    def execute_tools(
        self,
        calls: tuple[ToolCall, ...],
    ) -> tuple[ToolResult, ...]: ...
```

Key differences from base `Environment`:
- Returns `AgentObservation` with `available_tools` and `tool_results`
- Accepts `AgentAction` with optional `tool_calls`
- Executes tools internally via `execute_tools()`

### BaseToolEnvironment

```python
@dataclass
class BaseToolEnvironment(Generic[HiddenT]):
    """Base implementation with tool execution logic."""
    _tools: tuple[ToolDefinition, ...]
    _executor: ToolExecutor | None

    @property
    def available_tools(self) -> tuple[ToolDefinition, ...]: ...

    def execute_tools(self, calls) -> tuple[ToolResult, ...]:
        # Validates tools, delegates to executor

    def _check_terminal_tools(self, calls) -> bool:
        # Returns True if any called tool is terminal

    def _build_next_observation(self, current_obs, action, tool_results) -> AgentObservation:
        # Builds observation with updated message history and tool results
```

## Tool-Specific Rewards

**Location**: `llenvs/core/tool_rewards.py`

```python
@dataclass
class ToolValidityReward:
    """1.0 if all tool calls are valid, partial credit otherwise."""
    _name: str = "tool_validity"
    _reward_type: RewardType = RewardType.STEP

@dataclass
class ToolEfficiencyReward:
    """Penalizes redundant/excessive tool calls."""
    _name: str = "tool_efficiency"
    max_calls_per_step: int = 5
    penalty_per_excess: float = 0.1
    duplicate_penalty: float = 0.2
```

## Using Tools with Backends

```python
from llenvs.inference.backends import OpenAIBackend
from llenvs.inference import ChatMessage, SamplingParams

backend = OpenAIBackend(model="gpt-4o")

# Generate with tools
result = backend.generate_with_tools(
    messages=[ChatMessage(role="user", content="What's the weather in Paris?")],
    tools=[weather_tool],
    params=SamplingParams(temperature=0.0),
    tool_choice="auto",
)

if result.has_tool_calls:
    for call in result.tool_calls:
        print(f"Tool: {call.name}, Args: {call.arguments}")

# Convert to AgentAction for tool environments
action = result.to_agent_action()
```
