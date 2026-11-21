"""Speedy tool-call guards (each opt-in by an environment variable or the config's `env`).

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
import re

CONSTRAIN_OPEN = "\n<function="       # forced right after `<tool_call>`, as the template writes a call
CONSTRAIN_CLOSE = ">\n"               # the name's end: part of the targets, so the model's own ">\n" token can be used
CONSTRAIN_BIAS = 100.0                # the bias of an allowed token (logits are within about +-40: a mask in effect)
CONSTRAIN_MAX_STEPS = 48
DROP_THINKING_STEP = 16
REPEAT_WINDOW = 3
REPEAT_NOTE = "You already made this exact call; its result is above. Do something different."
# Loop breaker (Speedy 0.5.0: on by default, STRATA_LOOP_BREAKER=0 turns it off): the note names the next action, and
# only a call that is the same call with the same result and no write in between counts (re-running tests after an edit
# is not a loop).  "Light": no thinking floor by default (STRATA_LOOP_BREAKER_FLOOR=N adds one); the repeated tool name
# is masked for one reply after the second loop turn in a row (STRATA_LOOP_BREAKER_MASK=0 turns that off).
LOOP_NOTE = ("This call returned the same result as before; calling it again will not change it. If the task is done, "
             "write the final answer now. Otherwise take a different step.")
LOOP_FLOOR = 0
TOOL_STEP_THINK_CAP = 6144     # Speedy 0.5.0: thinking tokens of a reply that follows a tool result (0 = no cap)
_WRITE_NAME = re.compile(r"write|edit|patch|replace|create|delete|remove|move|rename|insert|apply", re.I)
_WRITE_CMD = re.compile(r"(^|[^<>&0-9])>{1,2}(?!&)|\bsed\s+-i|\btee\b|\bmv\b|\bcp\b|\brm\b|\bpatch\b|git\s+(apply|checkout|restore|stash)|\btouch\b|\bmkdir\b")
# 0.5.1: a write whose result says it did nothing (not found / error / no match / unchanged) opens no epoch, so a
# failing edit repeated verbatim is a loop (TB2 fix-git: 566 identical "File ... not found" edits went unnoticed).
_FAILED = re.compile(r"^(?:error|failed)\b|\bnot found\b|no such file|does not exist|could not find|cannot find|"
                     r"\bno match|did not match|not unique|multiple matches|\bunchanged\b|no changes|"
                     r"must be different|are identical|permission denied", re.I)
# volatile parts of a result, replaced before results are compared: call ids, uuids, hex ids, timestamps, durations
_VOLATILE = re.compile(r"\bcall_[0-9A-Za-z_]+|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|"
                       r"\b(?=[0-9a-f]*\d)[0-9a-f]{7,}\b|\b0x[0-9a-f]+\b|"
                       r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d(?::\d\d)?(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?|\b\d\d:\d\d:\d\d(?:\.\d+)?\b|"
                       r"\b\d+(?:\.\d+)?\s*(?:ns|us|µs|ms|s|sec|secs|seconds|m|min|minutes)\b", re.I)
NEAR_STREAK = 3                # same tool name + same normalized result this many calls in a row: a loop (args may vary)
NEAR_ARGS_RATIO = 0.85        # "near-identical" arguments (difflib ratio) for a near repeat whose result is not a failure
REPEAT_BACKSTOP = 3            # with the loop breaker on, a name+args repeat seen this many times still gets the repeat note


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


def mark_repeats(messages, window: int = REPEAT_WINDOW, note: str = REPEAT_NOTE, min_times: int = 2,
                 skip_noted: bool = False) -> list[dict]:
    """Every assistant tool call identical to one of the `window` calls before it gets `note` after its result (the
    last tool message that follows that turn).  It is a function of the history alone, so the same turns get the same
    note on every request and the prompt prefix stays stable.  Returns the repeats of the latest assistant turn (the
    ones a caller logs: earlier ones were logged when they were the latest) as {"name", "arguments", "times"}.
    `min_times`: only a call seen at least this often in the conversation counts.  `skip_noted`: a turn whose last result
    already ends with a note (the loop breaker's) gets no second one (0.5.1 backstop)."""
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
            if sig in history[-window:] and history.count(sig) + 1 >= min_times:
                reps.append({"name": sig[0], "arguments": sig[1], "times": history.count(sig) + 1})
            history.append(sig)
        if not reps:
            continue
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            j += 1
        if j == i + 1:
            continue                                # no result yet: nothing to put the note after
        if skip_noted and _NOTE_TAIL.search(_raw_text(messages[j - 1])):
            continue
        _append_note(messages[j - 1], note)
        if i == last_asst and all(messages[k].get("role") == "tool" for k in range(i + 1, len(messages))):
            latest = reps
    return latest


# ------------------------------------------------------------------------------------------------ 4: loop breaker (1120)
_NOTE_TAIL = re.compile(r"\n\n\[Note: [^\]]*\]\s*$")


def _raw_text(msg) -> str:
    c = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c if isinstance(c, str) else "" if c is None else str(c)


def _result_text(msg) -> str:
    return _NOTE_TAIL.sub("", _raw_text(msg)).strip()


def failed_result(res: str | None) -> bool:
    """A tool result that says the call did nothing (checked on its first 300 characters)."""
    return bool(res) and bool(_FAILED.search(res.strip()[:300]))


def _norm_empty(res: str) -> str:
    return re.sub(r"[\s()]|no output", "", res.lower())


def _similar(a: str | None, b: str | None, ratio: float = NEAR_ARGS_RATIO) -> bool:
    import difflib
    return a is not None and b is not None and (a == b or difflib.SequenceMatcher(None, a, b).ratio() >= ratio)


def norm_result(res: str | None) -> str | None:
    """The result with volatile parts (ids, timestamps, durations) replaced, for comparing results."""
    return None if res is None else _VOLATILE.sub("#", res)


def is_write(sig: tuple[str, str]) -> bool:
    """A call that may change the workspace: a write/edit-like tool name, or a shell command with a redirect or a
    file-changing command in its arguments."""
    name, args = sig
    if _WRITE_NAME.search(name):
        return True
    if re.search(r"shell|bash|exec|run|terminal|command", name, re.I):
        try:
            a = json.loads(args)
        except ValueError:
            a = args
        cmd = " ".join(str(v) for v in a.values()) if isinstance(a, dict) else str(a)
        return bool(_WRITE_CMD.search(cmd))
    return False


def mark_loops(messages, window: int = REPEAT_WINDOW, note: str = LOOP_NOTE) -> list[dict]:
    """Like mark_repeats, stricter: a call is a loop repeat when one of the `window` calls before it has the same name
    and arguments AND the same result text (volatile ids/timestamps/durations normalized), with no successful write
    (is_write, result not a failure) between the two; or the same tool name returned the same result NEAR_STREAK calls
    in a row (arguments may differ).  Such a call's turn gets `note`
    after its last tool result.  A function of the history alone (cache-stable).  Returns the loop repeats of the latest
    assistant turn (when only tool results follow it) as {"name", "arguments", "times", "streak"}; streak = how many
    consecutive assistant turns, up to this one, held a loop repeat of that name."""
    window = max(1, int(window))
    history: list[tuple[tuple[str, str], str, int]] = []
    epoch = 0
    run: tuple[str, str] | None = None      # (tool name, normalized result) of the previous call
    run_n = 0
    run_args: str | None = None
    streak: dict[str, int] = {}
    latest: list[dict] = []
    last_asst = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    for i, m in enumerate(messages):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            j += 1
        results = messages[i + 1:j]
        by_id = {r.get("tool_call_id"): r for r in results if r.get("tool_call_id")}
        reps, names_now = [], set()
        for k, call in enumerate(m["tool_calls"]):
            sig = call_signature(call)
            if sig is None:
                continue
            res_msg = by_id.get(call.get("id")) if isinstance(call, dict) and call.get("id") in by_id else (
                results[k] if k < len(results) else None)
            raw = _result_text(res_msg) if res_msg is not None else None
            res = norm_result(raw)
            wrote = is_write(sig) and not failed_result(raw)
            # near repeat: same tool + same normalized result in a row, args different only slightly (or the result is
            # a failure); a successful write or an empty result ("(no output)" of searches) breaks the run
            key = (sig[0], res)
            same_run = (res is not None and key == run and not wrote and _norm_empty(res) != ""
                        and (failed_result(raw) or _similar(sig[1], run_args)))
            run_n = run_n + 1 if same_run else (0 if wrote or res is None else 1)
            run, run_args = (None, None) if wrote or res is None else (key, sig[1])
            exact = res is not None and any(h[0] == sig and h[1] == res and h[2] == epoch for h in history[-window:])
            near = run_n >= NEAR_STREAK
            if exact or near:
                names_now.add(sig[0])
                reps.append({"name": sig[0], "arguments": sig[1], "times": sum(1 for h in history if h[0] == sig) + 1})
            history.append((sig, res, epoch))
            if wrote:
                epoch += 1
        streak = {n: streak.get(n, 0) + 1 for n in names_now}
        for r in reps:
            r["streak"] = streak[r["name"]]
        if not reps or not results:
            continue
        _append_note(messages[j - 1], note)
        if i == last_asst and j == len(messages):
            latest = reps
    return latest


def masked_names(reps, min_streak: int = 2) -> list[str]:
    """Tier 2: the tool names to keep out of the next reply (a loop repeat seen on `min_streak` consecutive turns)."""
    return sorted({r["name"] for r in reps or () if r.get("streak", 0) >= min_streak})
