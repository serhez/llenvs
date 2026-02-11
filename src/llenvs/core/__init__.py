"""Core abstractions for MDP-style environments."""

from llenvs.core.state import (
    State,
    StateMetadata,
    Observation,
    Action,
)
from llenvs.core.tools import (
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolExecutor,
    SimpleToolExecutor,
)
from llenvs.core.async_executor import AsyncToolExecutor
from llenvs.core.mcp_executor import MCPToolExecutor, MCPServerConfig, MCPConnectionError
from llenvs.core.tool_environment import BaseToolEnvironment
from llenvs.core.tool_rewards import ToolValidityReward, ToolEfficiencyReward
from llenvs.core.reward import RewardSignal, RewardBundle, RewardType, RewardFunction
from llenvs.core.trajectory import Trajectory, Transition, Checkpoint
from llenvs.core.environment import Environment, StepResult, EnvironmentSpec
from llenvs.core.extraction import (
    AnswerExtractor,
    TagBasedExtractor,
    RegexExtractor,
    GSM8KExtractor,
    MultipleChoiceExtractor,
    CompositeExtractor,
    RawGenerationExtractor,
    BoxedExtractor,
    NumericExtractor,
    LastLineExtractor,
    CodeBlockExtractor,
    PatternAnswerExtractor,
    CleanedExtractor,
    NativeExtractor,
)
from llenvs.core.registry import (
    Registry,
    environment_registry,
    answer_extractor_registry,
    backend_registry,
)
from llenvs.core.config import (
    EvalConfig,
    EnvironmentConfig,
    ModelConfig,
    InferenceConfig,
    EnvironmentFactory,
    BackendFactory,
)
from llenvs.core.segmentation import (
    Segmenter,
    SentenceSegmenter,
    LineSegmenter,
    PatternSegmenter,
    CompositeSegmenter,
    TokenSegmenter,
    LLMSegmenter,
    default_segment_parser,
)
from llenvs.core.segmented_environment import (
    SegmentedEnvironment,
    SegmentedHidden,
)

__all__ = [
    # State
    "State",
    "StateMetadata",
    "Observation",
    "Action",
    # Tools
    "ToolDefinition",
    "ToolParameter",
    "ToolParameterType",
    "ToolCall",
    "ToolResult",
    "ToolResultStatus",
    "ToolExecutor",
    "SimpleToolExecutor",
    "AsyncToolExecutor",
    "MCPToolExecutor",
    "MCPServerConfig",
    "MCPConnectionError",
    # Tool Environment
    "BaseToolEnvironment",
    # Tool Rewards
    "ToolValidityReward",
    "ToolEfficiencyReward",
    # Rewards
    "RewardSignal",
    "RewardBundle",
    "RewardType",
    "RewardFunction",
    # Trajectory
    "Trajectory",
    "Transition",
    "Checkpoint",
    # Environment
    "Environment",
    "StepResult",
    "EnvironmentSpec",
    # Extraction
    "AnswerExtractor",
    "TagBasedExtractor",
    "RegexExtractor",
    "GSM8KExtractor",
    "MultipleChoiceExtractor",
    "CompositeExtractor",
    "RawGenerationExtractor",
    "BoxedExtractor",
    "NumericExtractor",
    "LastLineExtractor",
    "CodeBlockExtractor",
    "PatternAnswerExtractor",
    "CleanedExtractor",
    "NativeExtractor",
    # Registry
    "Registry",
    "environment_registry",
    "answer_extractor_registry",
    "backend_registry",
    # Config
    "EvalConfig",
    "EnvironmentConfig",
    "ModelConfig",
    "InferenceConfig",
    "EnvironmentFactory",
    "BackendFactory",
    # Segmentation
    "Segmenter",
    "SentenceSegmenter",
    "LineSegmenter",
    "PatternSegmenter",
    "CompositeSegmenter",
    "TokenSegmenter",
    "LLMSegmenter",
    "default_segment_parser",
    "SegmentedEnvironment",
    "SegmentedHidden",
]
