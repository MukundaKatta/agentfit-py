"""Core truncation logic for agentfit."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

Strategy = Literal["head", "tail", "middle", "slide"]


def _default_tokenizer(text: str) -> int:
    """Approximate token count: chars // 4.  No external dependencies."""
    return max(1, len(text) // 4)


def token_count(
    text: str,
    tokenizer: Callable[[str], int] | None = None,
) -> int:
    """Return the estimated token count for *text*.

    Args:
        text: The string to count.
        tokenizer: Optional callable ``str -> int``.  Defaults to
            ``len(text) // 4``.

    Returns:
        Token count (>= 1 for any non-empty string, 0 for empty).
    """
    if not text:
        return 0
    fn = tokenizer or _default_tokenizer
    return fn(text)


@dataclass
class Message:
    """A single chat message with an optional pre-computed token count.

    Args:
        role: ``"system"``, ``"user"``, ``"assistant"``, or any custom role.
        content: The text content of the message.
        tokens: Pre-computed token count.  If ``None``, it is computed lazily
            from *content* using the ``WindowConfig`` tokenizer.
    """

    role: str
    content: str
    tokens: int | None = None

    def count_tokens(self, tokenizer: Callable[[str], int] | None = None) -> int:
        """Return token count, using cached value when available."""
        if self.tokens is not None:
            return self.tokens
        return token_count(self.content, tokenizer)


@dataclass
class WindowConfig:
    """Configuration for the truncation window.

    Args:
        max_tokens: Token budget for the message list (exclusive of any tokens
            that the caller allocates to the model's completion).
        strategy: One of ``"tail"`` (default), ``"head"``, ``"middle"``, or
            ``"slide"``.
        tokenizer: Optional callable ``str -> int``.  Defaults to
            ``len(content) // 4``.
        preserve_system: When ``True`` (default) the first message with
            ``role="system"`` is always kept and is not counted against the
            sliding window.
    """

    max_tokens: int
    strategy: Strategy = "tail"
    tokenizer: Callable[[str], int] | None = None
    preserve_system: bool = True


@dataclass
class FitResult:
    """Return value of :func:`fit_messages`.

    Attributes:
        messages: The truncated list of messages, ready to pass to the API.
        dropped: Number of messages that were dropped.
        total_tokens: Estimated token count of the returned messages.
        budget_tokens: The ``max_tokens`` value from the config.
        truncated: ``True`` when at least one message was dropped.
    """

    messages: list[Message]
    dropped: int
    total_tokens: int
    budget_tokens: int
    truncated: bool = field(init=False)

    def __post_init__(self) -> None:
        self.truncated = self.dropped > 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count(msgs: list[Message], tok: Callable[[str], int] | None) -> int:
    return sum(m.count_tokens(tok) for m in msgs)


def _split_system(
    messages: list[Message],
    preserve: bool,
) -> tuple[list[Message], list[Message]]:
    """Return (system_msgs, rest).

    When *preserve* is True the first system message (if any) is split off.
    """
    if not preserve:
        return [], list(messages)
    for i, m in enumerate(messages):
        if m.role == "system":
            return [m], messages[:i] + messages[i + 1 :]
    return [], list(messages)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _fit_tail(
    system: list[Message],
    rest: list[Message],
    budget: int,
    tok: Callable[[str], int] | None,
    original_count: int,
) -> FitResult:
    """Keep most-recent messages; drop oldest non-system messages."""
    sys_tokens = _count(system, tok)
    remaining = budget - sys_tokens
    kept: list[Message] = []
    tokens_kept = 0
    for msg in reversed(rest):
        t = msg.count_tokens(tok)
        if tokens_kept + t <= remaining:
            kept.insert(0, msg)
            tokens_kept += t
        # else: drop it (too big to fit)
    result = system + kept
    return FitResult(
        messages=result,
        dropped=original_count - len(result),
        total_tokens=sys_tokens + tokens_kept,
        budget_tokens=budget,
    )


def _fit_head(
    system: list[Message],
    rest: list[Message],
    budget: int,
    tok: Callable[[str], int] | None,
    original_count: int,
) -> FitResult:
    """Keep oldest messages; drop most-recent non-system messages."""
    sys_tokens = _count(system, tok)
    remaining = budget - sys_tokens
    kept: list[Message] = []
    tokens_kept = 0
    for msg in rest:
        t = msg.count_tokens(tok)
        if tokens_kept + t <= remaining:
            kept.append(msg)
            tokens_kept += t
        else:
            break
    result = system + kept
    return FitResult(
        messages=result,
        dropped=original_count - len(result),
        total_tokens=sys_tokens + tokens_kept,
        budget_tokens=budget,
    )


def _fit_middle(
    system: list[Message],
    rest: list[Message],
    budget: int,
    tok: Callable[[str], int] | None,
    original_count: int,
) -> FitResult:
    """Keep head + tail; drop from the middle.

    Tries to preserve a balanced split: roughly half the budget goes to
    messages from the start of *rest*, half to the end.  The exact split
    is determined greedily — we keep adding from each end until the budget
    is exhausted.
    """
    sys_tokens = _count(system, tok)
    remaining = budget - sys_tokens

    if not rest:
        return FitResult(
            messages=system,
            dropped=original_count - len(system),
            total_tokens=sys_tokens,
            budget_tokens=budget,
        )

    # Two-pointer: grow from both ends
    lo, hi = 0, len(rest) - 1
    head_msgs: list[Message] = []
    tail_msgs: list[Message] = []
    head_tokens = tail_tokens = 0

    while lo <= hi:
        # Try head
        t_head = rest[lo].count_tokens(tok)
        t_tail = rest[hi].count_tokens(tok) if lo != hi else 0

        if head_tokens + tail_tokens + t_head <= remaining:
            head_msgs.append(rest[lo])
            head_tokens += t_head
            lo += 1
        else:
            break

        if lo > hi:
            break

        if head_tokens + tail_tokens + t_tail <= remaining:
            tail_msgs.insert(0, rest[hi])
            tail_tokens += t_tail
            hi -= 1
        else:
            break

    kept = head_msgs + tail_msgs
    result = system + kept
    return FitResult(
        messages=result,
        dropped=original_count - len(result),
        total_tokens=sys_tokens + head_tokens + tail_tokens,
        budget_tokens=budget,
    )


def _fit_slide(
    system: list[Message],
    rest: list[Message],
    budget: int,
    tok: Callable[[str], int] | None,
    original_count: int,
) -> FitResult:
    """Sliding window: find the longest contiguous suffix that fits.

    Unlike ``tail``, which may include non-contiguous messages, ``slide``
    guarantees that the returned messages form a contiguous suffix of the
    input (after the system message).  This is useful when the model's
    coherence depends on seeing an unbroken recent history.
    """
    sys_tokens = _count(system, tok)
    remaining = budget - sys_tokens

    # Compute suffix token sums from the right
    window_tokens = 0
    start = len(rest)  # exclusive lower bound
    for i in range(len(rest) - 1, -1, -1):
        t = rest[i].count_tokens(tok)
        if window_tokens + t <= remaining:
            window_tokens += t
            start = i
        else:
            break

    kept = rest[start:]
    result = system + kept
    return FitResult(
        messages=result,
        dropped=original_count - len(result),
        total_tokens=sys_tokens + window_tokens,
        budget_tokens=budget,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_StrategyFn = Callable[
    [list[Message], list[Message], int, Callable[[str], int] | None, int],
    FitResult,
]

_STRATEGIES: dict[Strategy, _StrategyFn] = {
    "head": _fit_head,
    "tail": _fit_tail,
    "middle": _fit_middle,
    "slide": _fit_slide,
}


def fit_messages(
    messages: list[Message],
    config: WindowConfig,
) -> FitResult:
    """Truncate *messages* to fit within ``config.max_tokens``.

    The system message (``role="system"``) is always preserved when
    ``config.preserve_system`` is ``True`` (the default).

    Args:
        messages: The full conversation history.
        config: Token budget and strategy settings.

    Returns:
        A :class:`FitResult` containing the trimmed list and metadata.

    Raises:
        ValueError: If ``config.max_tokens`` is not positive, or if the
            strategy name is unknown.

    Example::

        from agentfit import Message, WindowConfig, fit_messages

        msgs = [
            Message("system", "You are a helpful assistant."),
            Message("user", "Hello"),
            Message("assistant", "Hi there!"),
            Message("user", "What is the capital of France?"),
        ]
        cfg = WindowConfig(max_tokens=20, strategy="tail")
        result = fit_messages(msgs, cfg)
        assert result.truncated or not result.truncated  # always returns
    """
    if config.max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {config.max_tokens}")
    if config.strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy {config.strategy!r}. Choose from: {sorted(_STRATEGIES)}"
        )

    if not messages:
        return FitResult(
            messages=[],
            dropped=0,
            total_tokens=0,
            budget_tokens=config.max_tokens,
        )

    system, rest = _split_system(messages, config.preserve_system)
    original_count = len(messages)
    fn = _STRATEGIES[config.strategy]
    return fn(system, rest, config.max_tokens, config.tokenizer, original_count)
