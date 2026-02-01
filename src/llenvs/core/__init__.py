"""Core abstractions for MDP-style environments."""

from env_evals.core.state import (
    State,
    StateMetadata,
    TextObservation,
    TextAction,
    AgentObservation,
    AgentAction,
)
from env_evals.core.tools import (
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolExecutor,
    SimpleToolExecutor,
)
from env_evals.core.async_executor import AsyncToolExecutor
from env_evals.core.mcp_executor import MCPToolExecutor, MCPServerConfig, MCPConnectionError
from env_evals.core.tool_environment import ToolEnvironment, BaseToolEnvironment
from env_evals.core.tool_rewards import ToolValidityReward, ToolEfficiencyReward
from env_evals.core.reward import RewardSignal, RewardBundle, RewardType, RewardFunction
from env_evals.core.trajectory import Trajectory, Transition, Checkpoint
from env_evals.core.environment import Environment, StepResult, EnvironmentSpec
from env_evals.core.extraction import (
    AnswerExtractor,
    TagBasedExtractor,
    RegexExtractor,
    GSM8KExtractor,
    MultipleChoiceExtractor,
    CompositeExtractor,
    FallbackExtractor,
)
from env_evals.core.registry import (
    Registry,
    environment_registry,
    extractor_registry,
    backend_registry,
)
from env_evals.core.config import (
    EvalConfig,
    EnvironmentConfig,
    ModelConfig,
    InferenceConfig,
    EnvironmentFactory,
    BackendFactory,
)
from env_evals.core.segmentation import (
    Segmenter,
    SentenceSegmenter,
    LineSegmenter,
    PatternSegmenter,
    CompositeSegmenter,
    SemanticSegmenter,
)
from env_evals.core.segmented_environment import (
    SegmentedEnvironment,
    SegmentedHidden,
)

__all__ = [
    # State
    "State",
    "StateMetadata",
    "TextObservation",
    "TextAction",
    "AgentObservation",
    "AgentAction",
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
    "ToolEnvironment",
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
    "FallbackExtractor",
    # Registry
    "Registry",
    "environment_registry",
    "extractor_registry",
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
    "SemanticSegmenter",
    "SegmentedEnvironment",
    "SegmentedHidden",
]
