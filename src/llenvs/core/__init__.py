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
from llenvs.core.tool_parsing import ToolCallParser, ParsedToolResponse, HermesToolCallParser
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
    JudgeConfig,
    EnvironmentLLMConfig,
    EnvironmentFactory,
    BackendFactory,
)
from llenvs.core.judge import (
    JudgeReward,
    JudgePromptTemplate,
    JUDGE_TEMPLATES,
    extract_judge_score,
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
from llenvs.core.branching import (
    BranchManager,
    BranchHandle,
    BranchingStrategy,
    CheckpointHandle,
    DirectStrategy,
    ActionReplayStrategy,
    ProcessForkStrategy,
)
from llenvs.integrations.scoring import ScoringResult
from llenvs.integrations.dataset_provider import TaskItem
from llenvs.integrations.token_mask import TrajectoryMasker, MaskedTrajectory, TokenSpan

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
    # Tool Parsing
    "ToolCallParser",
    "ParsedToolResponse",
    "HermesToolCallParser",
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
    "JudgeConfig",
    "EnvironmentLLMConfig",
    "EnvironmentFactory",
    "BackendFactory",
    # Judge
    "JudgeReward",
    "JudgePromptTemplate",
    "JUDGE_TEMPLATES",
    "extract_judge_score",
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
    # Branching
    "BranchManager",
    "BranchHandle",
    "BranchingStrategy",
    "CheckpointHandle",
    "DirectStrategy",
    "ActionReplayStrategy",
    "ProcessForkStrategy",
    # Integrations
    "ScoringResult",
    "TaskItem",
    "TrajectoryMasker",
    "MaskedTrajectory",
    "TokenSpan",
]
