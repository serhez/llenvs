"""Core abstractions for MDP-style environments."""

from llenvs.core.async_executor import AsyncToolExecutor
from llenvs.core.branching import (
    ActionReplayStrategy,
    BranchHandle,
    BranchingStrategy,
    BranchManager,
    CheckpointHandle,
    DirectStrategy,
    ProcessForkStrategy,
)
from llenvs.core.config import (
    BackendFactory,
    CodeExecutionConfig,
    EnvironmentConfig,
    EnvironmentFactory,
    EnvironmentLLMConfig,
    EvalConfig,
    InferenceConfig,
    IterativeConfig,
    JudgeConfig,
    ModelConfig,
)
from llenvs.core.environment import Environment, EnvironmentSpec, StepResult
from llenvs.core.extraction import (
    AnswerExtractor,
    BoxedExtractor,
    CleanedExtractor,
    CodeBlockExtractor,
    CompositeExtractor,
    GSM8KExtractor,
    JSONFieldExtractor,
    LastLineExtractor,
    MultipleChoiceExtractor,
    NativeExtractor,
    NumericExtractor,
    PatternAnswerExtractor,
    RawGenerationExtractor,
    RegexExtractor,
    SingleLineExtractor,
    TagBasedExtractor,
)
from llenvs.core.judge import (
    JUDGE_TEMPLATES,
    JudgePromptTemplate,
    JudgeReward,
    extract_judge_score,
)
from llenvs.core.mcp_executor import MCPConnectionError, MCPServerConfig, MCPToolExecutor
from llenvs.core.registry import (
    Registry,
    answer_extractor_registry,
    backend_registry,
    environment_registry,
)
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle, StepPenalty
from llenvs.core.segmentation import (
    CompositeSegmenter,
    LineSegmenter,
    LLMSegmenter,
    PatternSegmenter,
    Segmenter,
    SentenceSegmenter,
    TokenSegmenter,
    default_segment_parser,
)
from llenvs.core.segmented_environment import (
    SegmentedEnvironment,
    SegmentedHidden,
)
from llenvs.core.state import (
    Action,
    ImageContent,
    Observation,
    ObservationContent,
    State,
    StateMetadata,
)
from llenvs.core.tool_environment import BaseToolEnvironment
from llenvs.core.tool_parsing import HermesToolCallParser, ParsedToolResponse, ToolCallParser
from llenvs.core.tool_rewards import ToolEfficiencyReward, ToolValidityReward
from llenvs.core.tools import (
    SimpleToolExecutor,
    ToolCall,
    ToolDefinition,
    ToolExecutor,
    ToolParameter,
    ToolParameterType,
    ToolResult,
    ToolResultStatus,
    format_tool_call,
    format_tool_result,
    format_tool_result_data,
    oai_tools_to_definitions,
)
from llenvs.core.trajectory import Checkpoint, Trajectory, Transition
from llenvs.integrations.dataset_provider import TaskItem
from llenvs.integrations.scoring import ScoringResult
from llenvs.integrations.token_mask import MaskedTrajectory, TokenSpan, TrajectoryMasker

__all__ = [
    # State
    "State",
    "StateMetadata",
    "Observation",
    "ObservationContent",
    "Action",
    "ImageContent",
    # Tools
    "ToolDefinition",
    "ToolParameter",
    "ToolParameterType",
    "ToolCall",
    "ToolResult",
    "ToolResultStatus",
    "ToolExecutor",
    "SimpleToolExecutor",
    "oai_tools_to_definitions",
    "format_tool_call",
    "format_tool_result",
    "format_tool_result_data",
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
    "Signal",
    "SignalBundle",
    "RewardType",
    "RewardFunction",
    "StepPenalty",
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
    "JSONFieldExtractor",
    "MultipleChoiceExtractor",
    "CompositeExtractor",
    "RawGenerationExtractor",
    "BoxedExtractor",
    "NumericExtractor",
    "LastLineExtractor",
    "CodeBlockExtractor",
    "PatternAnswerExtractor",
    "SingleLineExtractor",
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
    "IterativeConfig",
    "CodeExecutionConfig",
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
