"""Speedy 0.4.6 tool-call guards (each opt-in by an environment variable or the config's `env`).

- Constrained tool-call opening (STRATA_TOOLCALL_CONSTRAIN=1): after the model writes `<tool_call>`, the next tokens are
  "\\n<function=" (forced) and then one of the request's declared tool names and ">\\n", picked token by token by the
  model under a logit bias that only lets through tokens continuing a declared name.  Seen on UD-Q4_K_XL at 80-620K
  context: `<4>`, `<=edit>`, `<` and an undeclared name where `<function=NAME>` belongs.
- Old thinking dropped at long context (STRATA_DROP_THINKING_ABOVE=N tokens): the thinking of older assistant turns is
  left out of the prompt.  Which turns is fixed in blocks (STRATA_DROP_THINKING_STEP turns), so the prompt prefix only
  changes once per block and the conversation cache is not lost on every request.
- Repeat warning (STRATA_REPEAT_WARN=1): an assistant tool call identical (name and arguments) to one of the previous
  STRATA_REPEAT_WARN_WINDOW (3) calls gets a short note after its result.

The functions here are pure (no engine, no tokenizer state of their own) so they are unit tested on the CPU.
"""
from __future__ import annotations

import json
import os

CONSTRAIN_OPEN = "\n<function="       # forced right after `<tool_call>`, as the template writes a call
CONSTRAIN_CLOSE = ">\n"               # the name's end: part of the targets, so the model's own ">\n" token can be used
CONSTRAIN_BIAS = 100.0                # the bias of an allowed token (logits are within about +-40: a mask in effect)
CONSTRAIN_MAX_STEPS = 48
DROP_THINKING_STEP = 16
REPEAT_WINDOW = 3
REPEAT_NOTE = "You already made this exact call; its result is above. Do something different."


# ------------------------------------------------------------------------------------------------ settings
def env_value(name: str, cfg_env: dict | None = None):
    """The process environment first, then the config's `env` block (the engine ignores keys it does not know)."""
    v = os.environ.get(name)
    if v is None or v == "":
        v = (cfg_env or {}).get(name)
    return v


def flag(value) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def int_setting(value, default: int = 0, low: int = 0) -> int:
    try:
        return max(low, int(str(value).strip()))
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------------------------------------ 1: constrained opening
def vocab_index(tok) -> dict[str, list[int]]:
    """token text -> ids for the printable-ASCII tokens (and newline), the only ones a tool name and its closing ">\\n"
    can be spelled with.  Built once per tokenizer."""
    n = len(getattr(tok, "tokens", ()) or ())
    getb = getattr(tok, "token_bytes", None)
    if not n:
        n = 256                                     # ByteTokenizer: one id per byte
    out: dict[str, list[int]] = {}
    for i in range(n):
        try:
            b = getb(i) if getb is not None else tok.decode([i]).encode("utf-8")
        except Exception:                           # noqa: BLE001 - an id with no text
            continue
        if not b or any(c > 126 or (c < 32 and c != 10) for c in b):
            continue
        out.setdefault(b.decode("ascii"), []).append(i)
    return out


def reverse_index(index: dict[str, list[int]]) -> dict[int, str]:
    return {i: s for s, ids in index.items() for i in ids}


def name_targets(names) -> list[str]:
    return sorted({str(n) + CONSTRAIN_CLOSE for n in names if n})


def allowed_next(index: dict[str, list[int]], text: str, targets: list[str]) -> list[int]:
    """The ids whose text continues `text` towards one of `targets` (each target is NAME + ">\\n")."""
    want: set[str] = set()
    for tg in targets:
        if tg.startswith(text) and len(tg) > len(text):
            rest = tg[len(text):]
            want.update(rest[:k] for k in range(1, len(rest) + 1))
    ids: list[int] = []
    for s in want:
        ids.extend(index.get(s, ()))
    return sorted(set(ids))


def remaining(text: str, targets: list[str]) -> list[str]:
    return [tg for tg in targets if tg.startswith(text)]


def constrain_name(step, encode, index, names, max_steps: int = CONSTRAIN_MAX_STEPS, rev: dict | None = None):
    """Picks a declared tool name token by token.  `step(allowed_ids) -> id | None` asks the model for one token under
    the bias; `encode(text) -> ids` spells a forced remainder.  Returns (ids after "<function=", name) or None when
    the model or the vocabulary cannot finish a name (the caller then lets the model write freely)."""
    targets = name_targets(names)
    if not targets:
        return None
    text, ids = "", []
    for _ in range(max_steps):
        left = remaining(text, targets)
        if not left:
            return None
        if text in left:                            # NAME>\n written in full
            return ids, text[:-len(CONSTRAIN_CLOSE)]
        if len(left) == 1:                          # one name fits: the rest is forced
            rest = left[0][len(text):]
            return ids + list(encode(rest)), left[0][:-len(CONSTRAIN_CLOSE)]
        allowed = allowed_next(index, text, targets)
        if not allowed:
            return None
        t = step(allowed)
        if t is None or t not in set(allowed):
            return None
        if rev is None:
            rev = reverse_index(index)
        piece = rev.get(t)
        if piece is None:
            return None
        text += piece
        ids.append(t)
    return None


# ------------------------------------------------------------------------------------------------ 2: drop old thinking
def message_chars(messages) -> int:
    """About how much text the prompt holds (an estimate; the tokens are counted after the render)."""
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            total += sum(len(p.get("text") or "") for p in c if isinstance(p, dict))
        r = m.get("reasoning_content")
        if isinstance(r, str):
            total += len(r)
        for call in m.get("tool_calls") or ():
            fn = call.get("function", call) if isinstance(call, dict) else {}
            a = fn.get("arguments") if isinstance(fn, dict) else None
            total += len(a) if isinstance(a, str) else len(json.dumps(a, ensure_ascii=False, default=str)) if a else 0
    return total


def drop_boundary(n_assistant: int, step: int = DROP_THINKING_STEP) -> int:
    """Assistant turns with an index below this lose their thinking.  The last turn always keeps it, and the boundary
    moves only in whole steps, so the rendered prefix changes once per `step` turns instead of on every request."""
    step = max(1, int(step))
    if n_assistant <= 1:
        return 0
    return ((n_assistant - 1) // step) * step


def drop_old_thinking(messages, step: int = DROP_THINKING_STEP) -> list[str]:
    """Clears reasoning_content of the assistant turns below drop_boundary (in place).  Returns the dropped texts."""
    turns = [m for m in messages if m.get("role") == "assistant"]
    cut = drop_boundary(len(turns), step)
    dropped = []
    for m in turns[:cut]:
        r = m.get("reasoning_content")
        if isinstance(r, str) and r.strip():
            dropped.append(str(r))
            m["reasoning_content"] = ""
    return dropped


# ------------------------------------------------------------------------------------------------ 3: repeat warning
def call_signature(call) -> tuple[str, str] | None:
    fn = call.get("function", call) if isinstance(call, dict) else None
    if not isinstance(fn, dict) or not fn.get("name"):
        return None
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            pass
    return str(fn["name"]), json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def _append_note(msg: dict, note: str) -> None:
    c = msg.get("content")
    text = f"\n\n[Note: {note}]"
    if isinstance(c, list):
        msg["content"] = list(c) + [{"type": "text", "text": text}]
    else:
        msg["content"] = (c if isinstance(c, str) else "" if c is None else str(c)) + text


def mark_repeats(messages, window: int = REPEAT_WINDOW, note: str = REPEAT_NOTE) -> list[dict]:
    """Every assistant tool call identical to one of the `window` calls before it gets `note` after its result (the
    last tool message that follows that turn).  It is a function of the history alone, so the same turns get the same
    note on every request and the prompt prefix stays stable.  Returns the repeats of the latest assistant turn (the
    ones a caller logs: earlier ones were logged when they were the latest) as {"name", "arguments", "times"}."""
    window = max(1, int(window))
    history: list[tuple[str, str]] = []
    latest: list[dict] = []
    last_asst = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    for i, m in enumerate(messages):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        reps = []
        for call in m["tool_calls"]:
            sig = call_signature(call)
            if sig is None:
                continue
            if sig in history[-window:]:
                reps.append({"name": sig[0], "arguments": sig[1], "times": history.count(sig) + 1})
            history.append(sig)
        if not reps:
            continue
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            j += 1
        if j == i + 1:
            continue                                # no result yet: nothing to put the note after
        _append_note(messages[j - 1], note)
        if i == last_asst and all(messages[k].get("role") == "tool" for k in range(i + 1, len(messages))):
            latest = reps
    return latest
