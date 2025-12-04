"""Callbacks Module - LangChain callback handlers for streaming"""

import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "The 'exaid_core.callbacks' module is deprecated and contains no functionality. "
    "AgentStreamingCallback has been moved to 'demos/cdss_example/callbacks/'. "
    "Please update your imports accordingly.",
    DeprecationWarning,
    stacklevel=2
)
