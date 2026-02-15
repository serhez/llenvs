# LMRL-Gym

The LMRL-Gym adapter wraps [LMRL-Gym](https://github.com/abdulhaim/LMRL-Gym) text-based RL environments for language models. LMRL-Gym provides 8 environments spanning navigation, word games, strategy, information gathering, and negotiation — all through a unified TextEnv protocol.

## Installation

LMRL-Gym is not on PyPI. Install from GitHub:

```bash
pip install git+https://github.com/abdulhaim/LMRL-Gym.git
```

!!! note
    LMRL-Gym is not included as a formal optional dependency in llenvs because it pins legacy dependency versions. Install it separately in your environment. LMRL-Gym depends on JAX, Flax, and other ML libraries. Some environments have additional dependencies (e.g., `python-chess` and `stockfish` for chess, `nltk` for twenty questions).

## Quick Start

Create an LMRL-Gym `TextEnv` and wrap it with the adapter:

```python
from llenvs.adapters import LMRLAdapter
from llenvs.core import Action

# Create your TextEnv (e.g., Wordle)
from llm_rl_scripts.wordle.env.env import WordleEnvironment
from llm_rl_scripts.wordle.env.game import Vocabulary

vocab = Vocabulary(all_vocab=word_list, target_vocab=target_list)
wordle_env = WordleEnvironment(vocab=vocab)

# Wrap with llenvs adapter
adapter = LMRLAdapter()
env = adapter.get_environment("wordle", text_env=wordle_env, num_tasks=100)

state, _ = env.reset(seed=42)
print(state.observation.prompt)

result = env.step(state, Action(text="crane"))
print(result.next_state.observation.messages[-1]["content"])
# Color-coded feedback: <g>c</g><b>r</b><y>a</y>...
```

## Available Environments

| Environment | Description | Oracle Required | Max Steps |
|------------|-------------|-----------------|-----------|
| `wordle` | Wordle word guessing (5-letter words, 6 guesses) | No | 6 |
| `chess` | Chess against Stockfish engine | No | 400 |
| `chess:endgame` | Chess endgame positions (e.g., KQK) | No | 400 |
| `maze:double_t` | Double-T maze navigation | No | 100 |
| `maze:umaze` | U-maze navigation | No | 100 |
| `twenty_questions` | Twenty Questions guessing game | Yes (T5 oracle) | 20 |
| `guess_city` | Guess the city with questions | Yes (T5 oracle) | 20 |
| `car_dealer` | Car dealer negotiation | Yes (buyer LLM) | 50 |
| `text_nav` | TextWorld-based text navigation | No | 100 |

Standalone environments (wordle, chess, maze) can be created directly from LMRL-Gym. Oracle-based environments (twenty_questions, guess_city, car_dealer) require an external model to act as the oracle/opponent — create the TextEnv manually and pass it via `text_env=`.

## TextEnv Protocol

LMRL-Gym environments follow the `TextEnv` protocol:

```python
class TextEnv:
    def reset(self, seed=None, options=None) -> TextHistory: ...
    def step(self, text_history: TextHistory) -> tuple[TextHistory, float, bool]: ...
    def close(self) -> None: ...
```

Where `TextHistory` is a `tuple[Text, ...]` and `Text` has `.text: str` and `.is_action: bool` attributes. The adapter wraps any object following this protocol.

## Creating Environments

### Wordle

```python
from llm_rl_scripts.wordle.env.env import WordleEnvironment
from llm_rl_scripts.wordle.env.game import Vocabulary

vocab = Vocabulary(all_vocab=word_list, target_vocab=target_list)
wordle_env = WordleEnvironment(vocab=vocab)

env = adapter.get_environment("wordle", text_env=wordle_env)
```

Actions are 5-letter word guesses. Feedback uses color tags: `<g>` (green/correct position), `<y>` (yellow/wrong position), `<b>` (black/not in word).

### Chess

```python
from llm_rl_scripts.chess.env.env import FenChessHistoryEnv

chess_env = FenChessHistoryEnv(max_moves=400)
env = adapter.get_environment("chess", text_env=chess_env)
```

Actions are SAN chess moves with spaces between characters. Observations show FEN notation. Requires `python-chess` and the Stockfish binary.

### Chess Endgames

```python
from llm_rl_scripts.chess.env.env import FenChessHistoryEnv, large_piece_random_endgame

position = large_piece_random_endgame(["K", "Q", "k"])
chess_env = FenChessHistoryEnv(max_moves=400, from_position=position)
env = adapter.get_environment("chess:endgame", text_env=chess_env)
```

### Maze

```python
from llm_rl_scripts.maze.env.env import MazeEnv
from llm_rl_scripts.maze.env.mazes import double_t_maze

# Use the setup helpers from the LMRL-Gym scripts
maze_env = MazeEnv(
    maze=double_t_maze(),
    valid_goals=valid_goals,
    actions=actions_dict,
    max_steps=100,
    describe_function=describe_fn,
    reward_function=reward_fn,
    last_k=5,
)
env = adapter.get_environment("maze:double_t", text_env=maze_env)
```

Actions are directional moves (e.g., `"move left"`, `"move right"`, `"move up"`, `"move down"`). Observations describe walls and optionally the current position.

### Twenty Questions

```python
from llm_rl_scripts.twenty_questions.env.env import TwentyQuestionsPolicyEnvironment

# Requires an oracle model (e.g., T5)
twenty_q_env = TwentyQuestionsPolicyEnvironment(
    oracle=my_oracle,
    word_list=word_list,
    max_conversation_length=20,
)
env = adapter.get_environment("twenty_questions", text_env=twenty_q_env)
```

## Task Indexing

LMRL-Gym environments don't have a native task index concept — they use seeds for randomization. The adapter maps task indices to seeds when `num_tasks` is set:

```python
env = adapter.get_environment("wordle", text_env=wordle_env, num_tasks=100)
assert len(env) == 100

state, _ = env.reset(options={"task_index": 42})
# Internally uses seed=42
```

An explicit seed always takes priority over the task-index-derived seed:

```python
state, _ = env.reset(seed=7, options={"task_index": 42})
# Uses seed=7, not 42
```

## Rewards

The adapter exposes LMRL-Gym's native per-step scalar reward:

| Reward | Type | Value |
|--------|------|-------|
| `lmrl_reward` | `STEP` (intermediate) | Per-step reward from TextEnv |
| `lmrl_reward` | `OUTCOME` (terminal) | Cumulative reward across the episode |

```python
result = env.step(state, Action(text="crane"))
signal = result.rewards.by_name("lmrl_reward")
print(signal.reward)       # e.g., -0.1 (wrong guess penalty)
print(signal.reward_type)  # RewardType.STEP
```

Reward semantics vary by environment:

| Environment | Step Reward | Terminal Reward |
|------------|-------------|-----------------|
| Wordle | -1 per wrong guess, 0 for correct | Cumulative |
| Chess | 0 intermediate, ±1 for checkmate/loss | Cumulative |
| Maze | -1 per step, -4 for illegal move, 0 for goal | Cumulative |
| Twenty Questions | -1 per question | Cumulative |

Add extra reward signals:

```python
env = LMRLEnvironment(
    text_env=wordle_env,
    extra_rewards=(my_custom_reward,),
)
```

## Hidden State

```python
@dataclass(frozen=True)
class LMRLHidden:
    env_name: str              # environment identifier
    episode_step: int          # current step count
    last_action: str | None    # last action taken
    cumulative_reward: float   # cumulative reward so far
    text_history: tuple        # raw LMRL-Gym TextHistory
```

The `text_history` field stores the full conversation as LMRL-Gym `Text` objects, preserving the original data for inspection or replay.

## Using with TrajectoryRunner

```python
from llenvs.adapters import LMRLAdapter
from llenvs.inference.backends import OpenAIBackend
from llenvs.evaluation import TrajectoryRunner
from llenvs.inference import SamplingParams

adapter = LMRLAdapter()
env = adapter.get_environment("wordle", text_env=wordle_env, num_tasks=100)

backend = OpenAIBackend(model="gpt-4o")
runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    sampling_params=SamplingParams(temperature=0.0),
)

result = runner.run_trajectory(task_index=0)
print(f"Reward: {result.trajectory.total_reward}")
```

## Wrapping Custom TextEnv Instances

Any object following the TextEnv protocol works with the adapter:

```python
class MyCustomTextEnv:
    def reset(self, seed=None, options=None):
        return (Text("Welcome! What is 2+2?", is_action=False),)

    def step(self, text_history):
        answer = text_history[-1].text.strip()
        if answer == "4":
            obs = Text("Correct!", is_action=False)
            return text_history + (obs,), 1.0, True
        obs = Text("Wrong, try again.", is_action=False)
        return text_history + (obs,), 0.0, False

env = adapter.get_environment("my_task", text_env=MyCustomTextEnv())
```

## Limitations

- Non-pure (`pure_step=False`) — cannot branch with `DirectStrategy`; use `ActionReplay` or `ProcessFork` strategies
- LMRL-Gym has heavy dependencies (JAX, Flax) for its training infrastructure
- Oracle-based environments (twenty questions, guess city, car dealer) require an external LLM
- No native answer extraction — configure an `answer_extractor` via environment config if needed
- `text_nav` requires a custom TextWorld fork from the LMRL-Gym repository
