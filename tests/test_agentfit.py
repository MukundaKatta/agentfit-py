"""Tests for agentfit-py."""

from __future__ import annotations

import pytest

from agentfit import Message, WindowConfig, fit_messages, token_count

# ---------------------------------------------------------------------------
# token_count
# ---------------------------------------------------------------------------


def test_token_count_empty():
    assert token_count("") == 0


def test_token_count_default_approximation():
    # 40 chars → 10 tokens
    assert token_count("a" * 40) == 10


def test_token_count_custom_tokenizer():
    assert token_count("hello world", tokenizer=lambda s: len(s.split())) == 2


def test_token_count_short_text_rounds_up():
    # 1 char → max(1, 0) = 1
    assert token_count("x") == 1


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


def test_message_count_tokens_no_cache():
    m = Message(role="user", content="hello world")  # 11 chars → 2 tokens
    assert m.count_tokens() == 2


def test_message_count_tokens_cached():
    m = Message(role="user", content="hello world", tokens=99)
    assert m.count_tokens() == 99


def test_message_count_tokens_custom_tokenizer():
    m = Message(role="user", content="hello world")
    assert m.count_tokens(tokenizer=lambda s: 5) == 5


# ---------------------------------------------------------------------------
# fit_messages — edge cases
# ---------------------------------------------------------------------------


def test_fit_empty_messages():
    cfg = WindowConfig(max_tokens=100)
    result = fit_messages([], cfg)
    assert result.messages == []
    assert result.dropped == 0
    assert result.total_tokens == 0
    assert not result.truncated


def test_fit_raises_on_zero_budget():
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        fit_messages([], WindowConfig(max_tokens=0))


def test_fit_raises_on_negative_budget():
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        fit_messages([], WindowConfig(max_tokens=-1))


def test_fit_raises_on_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        fit_messages(
            [Message("user", "hi")],
            WindowConfig(max_tokens=100, strategy="random"),  # type: ignore[arg-type]
        )


def test_fit_no_truncation_when_within_budget():
    msgs = [
        Message("user", "a" * 40),  # 10 tokens
        Message("assistant", "b" * 40),  # 10 tokens
    ]
    result = fit_messages(msgs, WindowConfig(max_tokens=30))
    assert not result.truncated
    assert result.dropped == 0
    assert result.messages == msgs


def test_fit_result_truncated_flag():
    msgs = [Message("user", "a" * 40)] * 5  # 5×10 = 50 tokens
    result = fit_messages(msgs, WindowConfig(max_tokens=15, strategy="tail"))
    assert result.truncated
    assert result.dropped >= 1


def test_fit_preserves_system_by_default():
    msgs = [
        Message("system", "a" * 40),  # 10 tokens
        Message("user", "b" * 400),  # 100 tokens — too big
        Message("user", "c" * 40),  # 10 tokens
    ]
    result = fit_messages(msgs, WindowConfig(max_tokens=25))
    roles = [m.role for m in result.messages]
    assert "system" in roles


def test_fit_system_not_preserved_when_disabled():
    msgs = [
        Message("system", "a" * 40),
        Message("user", "b" * 40),
    ]
    # Budget only fits one; with preserve_system=False system could be dropped
    result = fit_messages(
        msgs,
        WindowConfig(max_tokens=12, strategy="tail", preserve_system=False),
    )
    # tail keeps most-recent; user message is last → system may be dropped
    assert result.messages[-1].role == "user"


def test_fit_system_tokens_count_toward_budget():
    # system = 10 tokens, user = 10 tokens, budget = 12 → user shouldn't fit
    sys_msg = Message("system", "a" * 40)
    user_msg = Message("user", "b" * 40)
    result = fit_messages(
        [sys_msg, user_msg],
        WindowConfig(max_tokens=12, strategy="tail"),
    )
    # budget=12, system=10 → remaining=2, user needs 10 → user dropped
    assert len(result.messages) == 1
    assert result.messages[0].role == "system"
    assert result.dropped == 1


# ---------------------------------------------------------------------------
# tail strategy
# ---------------------------------------------------------------------------


def test_tail_keeps_most_recent():
    msgs = [Message("user", f"msg{i}" * 10) for i in range(5)]
    # Each "msgN" * 10 = ~40-50 chars → ~10 tokens; budget = 25
    # Should keep last 2
    result = fit_messages(msgs, WindowConfig(max_tokens=25, strategy="tail"))
    assert result.messages[-1] is msgs[-1]
    assert result.messages[-2] is msgs[-2]


def test_tail_drops_oldest_first():
    msgs = [
        Message("user", "a" * 400),  # 100 tokens — old, gets dropped
        Message("user", "b" * 40),  # 10 tokens
        Message("user", "c" * 40),  # 10 tokens
    ]
    result = fit_messages(msgs, WindowConfig(max_tokens=25, strategy="tail"))
    assert all(m.role == "user" for m in result.messages)
    contents = [m.content for m in result.messages]
    assert "b" * 40 in contents
    assert "c" * 40 in contents
    assert "a" * 400 not in contents


def test_tail_exact_budget_fit():
    # 3 messages × 10 tokens = 30 → exactly fits budget=30
    msgs = [Message("user", "a" * 40)] * 3
    result = fit_messages(msgs, WindowConfig(max_tokens=30, strategy="tail"))
    assert not result.truncated
    assert len(result.messages) == 3


def test_tail_precomputed_tokens():
    msgs = [
        Message("user", "x", tokens=50),
        Message("user", "y", tokens=5),
    ]
    result = fit_messages(msgs, WindowConfig(max_tokens=10, strategy="tail"))
    # Only "y" (5 tokens) fits
    assert len(result.messages) == 1
    assert result.messages[0].content == "y"


# ---------------------------------------------------------------------------
# head strategy
# ---------------------------------------------------------------------------


def test_head_keeps_oldest():
    msgs = [Message("user", f"msg{i}" * 10) for i in range(5)]
    result = fit_messages(msgs, WindowConfig(max_tokens=25, strategy="head"))
    assert result.messages[0] is msgs[0]
    assert result.messages[1] is msgs[1]


def test_head_drops_newest_when_over_budget():
    msgs = [
        Message("user", "a" * 40),  # 10 tokens — kept
        Message("user", "b" * 40),  # 10 tokens — kept
        Message("user", "c" * 400),  # 100 tokens — dropped
    ]
    result = fit_messages(msgs, WindowConfig(max_tokens=25, strategy="head"))
    contents = [m.content for m in result.messages]
    assert "a" * 40 in contents
    assert "b" * 40 in contents
    assert "c" * 400 not in contents


def test_head_with_system_preserved():
    msgs = [
        Message("system", "a" * 40),  # 10 tokens — always kept
        Message("user", "b" * 40),  # 10 tokens
        Message("user", "c" * 400),  # 100 tokens — dropped
    ]
    result = fit_messages(msgs, WindowConfig(max_tokens=25, strategy="head"))
    assert result.messages[0].role == "system"
    assert len(result.messages) == 2


# ---------------------------------------------------------------------------
# middle strategy
# ---------------------------------------------------------------------------


def test_middle_keeps_head_and_tail():
    msgs = [
        Message("user", "first"),
        Message("user", "dropped1"),
        Message("user", "dropped2"),
        Message("user", "last"),
    ]
    # Give just enough budget for first + last
    first_tokens = msgs[0].count_tokens()
    last_tokens = msgs[3].count_tokens()
    budget = first_tokens + last_tokens + 1  # +1 so both fit
    result = fit_messages(msgs, WindowConfig(max_tokens=budget, strategy="middle"))
    contents = [m.content for m in result.messages]
    assert "first" in contents
    assert "last" in contents


def test_middle_empty_rest():
    msgs = [Message("system", "a" * 40)]
    result = fit_messages(msgs, WindowConfig(max_tokens=5, strategy="middle"))
    assert result.messages[0].role == "system"


def test_middle_all_fit():
    msgs = [Message("user", "x" * 4)] * 4  # 4 × 1 token = 4 tokens
    result = fit_messages(msgs, WindowConfig(max_tokens=10, strategy="middle"))
    assert not result.truncated
    assert len(result.messages) == 4


def test_middle_oversized_head_does_not_starve_tail():
    # An oversized message at the head must not prevent the tail from being
    # filled up to budget (and vice-versa). Regression for a bug where the
    # two-pointer loop broke entirely as soon as one end overflowed.
    msgs = [
        Message("user", "HUGE", tokens=1000),  # never fits — must be dropped
        Message("user", "s1", tokens=5),
        Message("user", "s2", tokens=5),
        Message("user", "s3", tokens=5),
    ]
    result = fit_messages(msgs, WindowConfig(max_tokens=20, strategy="middle"))
    assert [m.content for m in result.messages] == ["s1", "s2", "s3"]
    assert result.total_tokens == 15
    assert result.dropped == 1


def test_middle_oversized_tail_does_not_starve_head():
    msgs = [
        Message("user", "s1", tokens=5),
        Message("user", "s2", tokens=5),
        Message("user", "s3", tokens=5),
        Message("user", "HUGE", tokens=1000),  # never fits — must be dropped
    ]
    result = fit_messages(msgs, WindowConfig(max_tokens=20, strategy="middle"))
    assert [m.content for m in result.messages] == ["s1", "s2", "s3"]
    assert result.dropped == 1


# ---------------------------------------------------------------------------
# slide strategy
# ---------------------------------------------------------------------------


def test_slide_contiguous_suffix():
    msgs = [
        Message("user", "a" * 400),  # 100 tokens — old
        Message("user", "b" * 40),  # 10 tokens
        Message("user", "c" * 40),  # 10 tokens
    ]
    result = fit_messages(msgs, WindowConfig(max_tokens=25, strategy="slide"))
    # b and c fit; a does not
    contents = [m.content for m in result.messages]
    assert "b" * 40 in contents
    assert "c" * 40 in contents
    assert "a" * 400 not in contents


def test_slide_full_window_fits():
    msgs = [Message("user", "a" * 40)] * 3  # 3 × 10 = 30 tokens
    result = fit_messages(msgs, WindowConfig(max_tokens=30, strategy="slide"))
    assert not result.truncated


def test_slide_single_large_message_dropped():
    msgs = [
        Message("user", "a" * 400),  # 100 tokens
        Message("user", "b" * 40),  # 10 tokens
    ]
    result = fit_messages(msgs, WindowConfig(max_tokens=12, strategy="slide"))
    assert len(result.messages) == 1
    assert result.messages[0].content == "b" * 40


def test_slide_with_system():
    msgs = [
        Message("system", "s" * 40),  # 10 tokens
        Message("user", "a" * 400),  # 100 tokens — dropped
        Message("user", "b" * 40),  # 10 tokens — fits
    ]
    result = fit_messages(msgs, WindowConfig(max_tokens=25, strategy="slide"))
    roles = [m.role for m in result.messages]
    assert "system" in roles
    contents = [m.content for m in result.messages]
    assert "b" * 40 in contents


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------


def test_fit_result_budget_tokens():
    cfg = WindowConfig(max_tokens=50)
    result = fit_messages([Message("user", "hi")], cfg)
    assert result.budget_tokens == 50


def test_fit_result_total_tokens_accurate():
    msgs = [
        Message("user", "a" * 40, tokens=10),
        Message("user", "b" * 40, tokens=10),
    ]
    result = fit_messages(msgs, WindowConfig(max_tokens=100))
    assert result.total_tokens == 20


def test_fit_result_dropped_accurate():
    msgs = [Message("user", "x" * 40)] * 5  # 50 tokens
    result = fit_messages(msgs, WindowConfig(max_tokens=15, strategy="tail"))
    assert result.dropped == 5 - len(result.messages)


# ---------------------------------------------------------------------------
# Custom tokenizer
# ---------------------------------------------------------------------------


def test_custom_tokenizer_word_count():
    def word_tok(s):
        return len(s.split())

    msgs = [
        Message("user", "one two three four five"),  # 5 words
        Message("user", "six seven eight nine ten"),  # 5 words
        Message("user", "eleven twelve thirteen"),  # 3 words
    ]
    cfg = WindowConfig(max_tokens=8, strategy="tail", tokenizer=word_tok)
    result = fit_messages(msgs, cfg)
    # last two messages = 5+3=8 words → fits exactly
    assert len(result.messages) == 2
    assert result.total_tokens == 8


def test_custom_tokenizer_char_count():
    def char_tok(s):
        return len(s)

    msgs = [
        Message("user", "hello"),  # 5 chars
        Message("user", "world"),  # 5 chars
    ]
    cfg = WindowConfig(max_tokens=7, strategy="tail", tokenizer=char_tok)
    result = fit_messages(msgs, cfg)
    assert len(result.messages) == 1
    assert result.messages[0].content == "world"


def test_custom_tokenizer_applies_to_system_message():
    # System message tokens must be measured with the config tokenizer too.
    def char_tok(s):
        return len(s)

    msgs = [
        Message("system", "sys"),  # 3 chars
        Message("user", "hello"),  # 5 chars
        Message("user", "world"),  # 5 chars
    ]
    cfg = WindowConfig(max_tokens=8, strategy="tail", tokenizer=char_tok)
    result = fit_messages(msgs, cfg)
    # system (3) is preserved → remaining 5 → only "world" fits
    assert [m.content for m in result.messages] == ["sys", "world"]
    assert result.total_tokens == 8


# ---------------------------------------------------------------------------
# Additional public-API coverage
# ---------------------------------------------------------------------------


def test_middle_preserves_chronological_order():
    msgs = [Message("user", str(i), tokens=10) for i in range(6)]
    result = fit_messages(msgs, WindowConfig(max_tokens=30, strategy="middle"))
    contents = [m.content for m in result.messages]
    # Keeps a head + tail slice, and the kept messages stay in order.
    assert contents == sorted(contents, key=int)
    assert contents[0] == "0"
    assert contents[-1] == "5"


def test_dropped_and_truncated_are_consistent():
    msgs = [Message("user", "x", tokens=10) for _ in range(4)]
    result = fit_messages(msgs, WindowConfig(max_tokens=15, strategy="tail"))
    assert result.dropped == len(msgs) - len(result.messages)
    assert result.truncated is (result.dropped > 0)


def test_head_with_custom_tokenizer():
    msgs = [
        Message("user", "aa"),  # 2 chars
        Message("user", "bb"),  # 2 chars
        Message("user", "cccc"),  # 4 chars — overflows
    ]
    cfg = WindowConfig(max_tokens=4, strategy="head", tokenizer=len)
    result = fit_messages(msgs, cfg)
    assert [m.content for m in result.messages] == ["aa", "bb"]
    assert result.dropped == 1


def test_token_count_zero_for_whitespace_default_floor():
    # 3 spaces → 3 // 4 == 0, but floor keeps it at 1 for non-empty text.
    assert token_count("   ") == 1
