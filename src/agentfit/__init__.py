"""agentfit: token-aware message truncation for LLM context windows.

Fit a list of chat messages into a token budget without splitting individual
messages. Four strategies mirror the Rust crate (agentfit-rs):

* ``head``   — keep the earliest messages (good for summarization pipelines)
* ``tail``   — keep the most recent messages (default; preserves recency)
* ``middle`` — drop from the middle, keep head + tail (preserves framing)
* ``slide``  — sliding-window over the tail, always advances forward

The system message (``role="system"``) is always preserved regardless of
strategy and does not count toward the sliding window.

Zero external dependencies.  BYO tokenizer via the ``tokenizer`` parameter
(a callable ``str -> int``).  Default is ``len(text) // 4``.
"""

from __future__ import annotations

from .core import (
    FitResult,
    Message,
    Strategy,
    WindowConfig,
    fit_messages,
    token_count,
)

__all__ = [
    "FitResult",
    "Message",
    "Strategy",
    "WindowConfig",
    "fit_messages",
    "token_count",
]

__version__ = "0.1.0"
