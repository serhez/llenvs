"""miles trainer integration.

Modules in this package are entry points that the miles training launcher
loads via dotted-path CLI flags; they are not meant to be imported through
package-level re-exports:

- ``agent``: ``--custom-agent-function-path llenvs.integrations.miles.agent.run``
- ``reward``: ``--custom-rm-path llenvs.integrations.miles.reward.reward_func``
- ``data``: prompt-data JSONL exporter (``python -m llenvs.integrations.miles.data``)
- ``source``: ``--data-source-path llenvs.integrations.miles.source.LLEnvsDataSource``
- ``postprocess``: ``--session-sample-postprocessor-path
  llenvs.integrations.miles.postprocess.postprocess``
- ``advantage``: ``--custom-advantage-function-path
  llenvs.integrations.miles.advantage.turn_grpo`` (requires the turn-level-credit
  miles fork)

Environment configuration is discovered through the ``LLENVS_MILES_CONFIG``
environment variable (see ``config``). See ``docs/guides/miles.md`` for launch
recipes.
"""
