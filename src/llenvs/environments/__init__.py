"""Native llenvs environments.

Out-of-the-box environments that combine existing building blocks
into ready-to-use configurations. Unlike adapters (which bridge
third-party libraries), these are llenvs-native environments.
"""

from llenvs.environments.coding import IterativeCodingEnvironment

__all__ = ["IterativeCodingEnvironment"]
