"""SkyRL task preparation, native text launch, and token-credit components.

This module has no runtime imports. Data, scoring, and text episode helpers do
not import the native stack; credit/trainer helpers require Torch. Native
launch bindings are loaded separately by ``entrypoint`` in the SkyRL runtime.
"""
