# agentfit-py

Token-aware message truncation for LLM context windows. Python port of [agentfit-rs](https://github.com/MukundaKatta/agentfit-rs).

Zero dependencies. BYO tokenizer. Four strategies.

Part of the [@mukundakatta agent-stack](https://github.com/MukundaKatta).

## Install

```bash
pip install agentfit
```

## Quickstart

```python
from agentfit import Message, WindowConfig, fit_messages

msgs = [
    Message("system", "You are a helpful assistant."),
    Message("user", "Tell me about the history of Rome."),
    Message("assistant", "Rome was founded in 753 BC ..."),
    Message("user", "What about the fall of Rome?"),
]

cfg = WindowConfig(max_tokens=30, strategy="tail")
result = fit_messages(msgs, cfg)

print(f"kept {len(result.messages)} / {len(msgs)} messages")
print(f"tokens used: {result.total_tokens} / {result.budget_tokens}")
print(f"truncated: {result.truncated}, dropped: {result.dropped}")
```

## Strategies

| Strategy | What it keeps | Use when |
| --- | --- | --- |
| `tail` (default) | Most-recent messages | Chat assistants; recency matters most |
| `head` | Oldest messages | Summarization pipelines; early context matters |
| `middle` | Head + tail; drops the middle | Preserve framing + recent context |
| `slide` | Contiguous suffix (sliding window) | Strict coherence; no gaps in history |

The system message (`role="system"`) is always preserved regardless of strategy.

## Custom tokenizer

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

cfg = WindowConfig(
    max_tokens=2048,
    strategy="tail",
    tokenizer=lambda s: len(enc.encode(s)),
)
```

Or pass any `Callable[[str], int]`:

```python
cfg = WindowConfig(max_tokens=2048, tokenizer=lambda s: len(s.split()))
```

Default is `len(text) // 4` (no external deps).

## Pre-computed token counts

If you already know the token count (e.g. from an API response), cache it on the message:

```python
msg = Message(role="assistant", content=reply_text, tokens=response.usage.output_tokens)
```

`fit_messages` uses the cached value and skips re-tokenizing.

## FitResult

```python
result.messages      # list[Message] — pass directly to your API
result.dropped       # int — number of messages removed
result.total_tokens  # int — token count of returned messages
result.budget_tokens # int — max_tokens from config
result.truncated     # bool — True when dropped > 0
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

40 tests. Covers all four strategies, edge cases, custom tokenizers, system-message preservation, pre-computed tokens, and `FitResult` metadata.

## License

MIT.
