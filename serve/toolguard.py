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
# 0.5.3 adds edit cycles (mark_cycles below), whose mask holds for several replies.
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
                     r"must be different|are identical|permission denied|tool_use_error|has not been read|"
                     r"modified since", re.I)
# volatile parts of a result, replaced before results are compared: call ids, uuids, hex ids, timestamps, durations
_VOLATILE = re.compile(r"\bcall_[0-9A-Za-z_]+|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|"
                       r"\b(?=[0-9a-f]*\d)[0-9a-f]{7,}\b|\b0x[0-9a-f]+\b|"
                       r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d(?::\d\d)?(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?|\b\d\d:\d\d:\d\d(?:\.\d+)?\b|"
                       r"\b\d+(?:\.\d+)?\s*(?:ns|us|µs|ms|s|sec|secs|seconds|m|min|minutes)\b", re.I)
NEAR_STREAK = 3                # same tool name + same normalized result this many calls in a row: a loop (args may vary)
NEAR_ARGS_RATIO = 0.85        # "near-identical" arguments (difflib ratio) for a near repeat whose result is not a failure
REPEAT_BACKSTOP = 3            # with the loop breaker on, a name+args repeat seen this many times still gets the repeat note
# 0.5.3 edit cycles (STRATA_LOOP_BREAKER_CYCLE=0 turns them off): successful edits that go back and forth on one file
# with no other work between them (TB2: PLAN.md checkboxes "[ ]" -> "[~]" -> "[x]" -> "[ ]" for 400-990 requests; every
# edit succeeded and differed from the one before, so it was never "the same call + the same result").
CYCLE_REVISITS = 2             # edits that restore text an earlier edit replaced: a cycle
CYCLE_TOGGLES = 3              # different edits of the same lines that change only their checkbox marks: a cycle
CYCLE_CHAIN = 4                # edits in a row that each rewrite exactly the text the one before wrote: a cycle
CYCLE_PERIOD_MAX = 6           # the last calls repeat with a period of 2..this (same calls, same results, a write in them)
CYCLE_PERIOD_CALLS = 6         # ... over at least this many calls (and at least two full rounds)
CYCLE_FAILS = 4                # turns in a row whose edits all failed (any file, any error text), nothing else between
CYCLE_CHURN = 8                # turns in a row that only edit the session plan / todo list: a note (never a mask)
CYCLE_MASK_TURNS = 2           # the editing tools are masked from the second cycle turn of an episode on
CYCLE_HOLD = 2                 # ... and stay masked until this many turns of other work followed (doubled per relapse)
CYCLE_HOLD_MAX = 8
CYCLE_NOTE = ("You are rewriting the same part of {file} back and forth without doing any other work. Stop updating "
              "it. Do the next real step of the task now with another tool, for example run the command you planned "
              "or read or change another file. If the task is done, write the final answer.")
CYCLE_NOTE_FAIL = ("The last edits all failed, so nothing was changed. Stop sending edits like these. Read the file "
                   "again or do the next real step of the task with another tool. If the task is done, write the final "
                   "answer.")
CYCLE_NOTE_PERIOD = ("The last calls repeat the same sequence of steps with the same results; going around again will "
                     "not change anything. If the task is done, write the final answer now. Otherwise take a different "
                     "step.")
CYCLE_NOTE_CHURN = ("You have updated {file} many times in a row without doing any other work. Unless these edits are "
                    "the task itself, stop updating it and do the next real step of the task.")
CYCLE_PAUSED = " The editing tools are paused for the next few replies."
_PATH_KEYS = ("filePath", "file_path", "path", "filename", "file", "target_file", "notebook_path", "TargetFile")
_OLD_KEYS = ("oldString", "old_string", "old_str", "old_text", "oldText", "search", "old")
_NEW_KEYS = ("newString", "new_string", "new_str", "new_text", "newText", "replace", "new", "content", "contents",
             "file_text", "text")
_EDIT_WORDS = {"write", "edit", "editor", "patch", "replace", "multiedit", "notebookedit"}
_NAME_WORDS = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")
_EDIT_OK = re.compile(r"\s*(?:edit applied|wrote file|the file\b.*\bhas been (?:updated|created|written)|success|updated\b|"
                      r"done\b|ok\b)", re.I)
_TODO_TOOL = re.compile(r"todo|update_?plan", re.I)                 # the client's todo list (TodoWrite, update_plan)
_RUN_TOOL = re.compile(r"shell|bash|exec|run|terminal|command", re.I)     # as in is_write
_TODO_READ = re.compile(r"read|get|list|show", re.I)                # ... but not a tool that only reads it (todoread)
CYCLE_NOVEL = 8                # other work ends a mask only if the same call + result is not among the last calls
# the session plan / todo file by its exact name (PLAN.md, PLAN.session.md, TODO.md, todo.txt, ...): a longer name
# (PLAN-STREAMING.md) is a document someone is writing
_BOOK_FILE = re.compile(r"(?:^|[/\\])(?:plan|plan\.session|todo|todos|checklist)\.(?:md|markdown|txt)$", re.I)
_MARKS = re.compile(r"\[[ xX~\-/>!]\]")
_PATCH_FILE = re.compile(r"^\*\*\* (?:Update|Add) File: (.+)$", re.M)
TODO_TARGET = "the todo list"


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


# ------------------------------------------------------------------------------------------------ 5: edit cycles (0.5.3)
def is_edit_tool(name: str) -> bool:
    """A tool that changes a file it names, by a word of its name (edit, Write, MultiEdit, apply_patch,
    str_replace_editor; not dispatch_agent)."""
    return any(w.lower() in _EDIT_WORDS for w in _NAME_WORDS.findall(str(name)))


def edit_failed(res: str | None) -> bool:
    """An edit's result that says nothing was changed, judged on its first line only (a successful edit's result may
    go on with compiler / LSP diagnostics that read like a failure: "Edit applied successfully. ... Cannot find name")."""
    if res is None:
        return True
    first = res.strip().split("\n", 1)[0]
    return failed_result(first) and not _EDIT_OK.match(first)


def _first(d: dict, keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, str):
            return v
    return None


def file_op(sig: tuple[str, str]):
    """What an edit-like call changes: (target, old, new, bookkeeping) or None for any other call.  target: the file
    path (TODO_TARGET for a todo-list tool); old: the replaced text (None for a whole-file write and the todo list);
    new: the text written; bookkeeping: a plan / todo file or the todo list."""
    name, raw = sig
    try:
        a = json.loads(raw)
    except ValueError:
        a = None
    if _TODO_TOOL.search(name):
        return None if _TODO_READ.search(name) else (TODO_TARGET, None, raw, True)
    if not is_edit_tool(name) or not isinstance(a, dict):
        return None
    path = _first(a, _PATH_KEYS)
    old, new = _first(a, _OLD_KEYS), _first(a, _NEW_KEYS)
    edits = a.get("edits")
    if isinstance(edits, list) and edits and all(isinstance(e, dict) for e in edits):      # several edits in one call
        old = "\x00".join(_first(e, _OLD_KEYS) or "" for e in edits)
        new = "\x00".join(_first(e, _NEW_KEYS) or "" for e in edits)
    if path is None:                                    # apply_patch: the path and the lines are in the patch text
        text = next((v for v in a.values() if isinstance(v, str) and "*** " in v), None)
        m = _PATCH_FILE.search(text) if text else None
        if m is None:
            return None
        path = m.group(1).strip()
        lines = text.splitlines()
        old = "\n".join(ln[1:] for ln in lines if ln.startswith("-") and not ln.startswith("---"))
        new = "\n".join(ln[1:] for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
    if not path or new is None:
        return None
    return path, old, new, bool(_BOOK_FILE.search(path))


def _period(seq: list, wrote: list) -> int:
    """The period (2..CYCLE_PERIOD_MAX) with which the last calls repeat (same calls and results, a write among them,
    at least two rounds and CYCLE_PERIOD_CALLS calls), or 0."""
    n = len(seq)
    for k in range(2, CYCLE_PERIOD_MAX + 1):
        rounds = 1
        while (rounds + 1) * k <= n and seq[n - (rounds + 1) * k:n - rounds * k] == seq[n - k:]:
            rounds += 1
        if rounds >= 2 and rounds * k >= CYCLE_PERIOD_CALLS and any(wrote[n - k:]) \
                and len(set(seq[n - k:])) > 1 and all(r is not None for _, r in seq[n - k:]):
            return k
    return 0


def mark_cycles(messages, paused: str = CYCLE_PAUSED) -> dict:
    """Edit cycles: successful edits that go back and forth on one file while nothing else gets done.  "Other work" is
    any answered call that is not an edit, and a successful edit of another real file (running or reading a code file
    between its edits is other work: edit -> test cycles are not judged here).  For the session plan / todo list a call that names the file
    is not other work, and an edit of the plan or the todo list is never other work for another file.  Since the last
    other work, a call is a cycle call when it is
      - back-and-forth: the CYCLE_REVISITS-th edit that restores text an earlier edit replaced (whole-file write /
        todo list: writes a content already written),
      - marks: the CYCLE_TOGGLES-th different edit of the same lines that changes only their checkbox marks,
      - rewrite: the CYCLE_CHAIN-th edit in a row that rewrites exactly the text the edit before wrote,
      - period: the end of a sequence of calls repeating with a period of 2..CYCLE_PERIOD_MAX (same calls, same
        results, a write among them),
    and a turn is a cycle turn too when it is
      - failing: the CYCLE_FAILS-th turn in a row whose edits all failed (any file, any error text), or
      - churn: the CYCLE_CHURN-th turn in a row that only edits the session plan / todo list.
    While an episode runs, its files are judged at half these counts.
    A cycle turn gets a note after its last tool result, unless one is there already (run it after mark_loops: a turn
    that is also a plain loop keeps that note).  From the CYCLE_MASK_TURNS-th such turn of an episode the editing
    tools are to be masked (the note then ends with `paused`), until CYCLE_HOLD turns of other work followed (twice
    that after each relapse, up to CYCLE_HOLD_MAX; the same call with the same result as one of the CYCLE_NOVEL calls
    before it does not count); the episode is forgotten after twice the hold again, or when the
    user writes.  Churn, and a period whose round holds a shell / run call (edit -> test -> revert -> test) or no edit
    tool at all, get the note only: they never start or extend a mask.  A function of the history alone (cache-stable).
    Returns {"reps": the cycle calls of the latest assistant turn as {"name", "target", "kind", "hold" (counts for the
    mask), "edit" (an edit-tool call), "turn"},
    "mask": the tool names to keep out of the next reply (the episode's edit tools; cycle_mask_names adds their
    siblings), "hold": turns of other work still needed, "turns": cycle turns of the episode}."""
    files: dict[str, dict] = {}
    seq, wrote, edits = [], [], []          # every call as (signature, normalized result); a write?  a file_op?
    ep = {"turns": 0, "held": 0, "calm": 0, "relapses": 0, "names": set(), "targets": {}}     # targets: path -> plan?
    fail_turns = 0                          # turns in a row whose edits all failed
    latest: list[dict] = []
    prev_role = None

    def named(path: str, args: str) -> bool:
        """A call that names the session plan (reads it, rewrites it with the shell) is not other work for it."""
        return path != TODO_TARGET and json.dumps(path, ensure_ascii=False)[1:-1] in args

    def hold() -> int:
        return min(CYCLE_HOLD_MAX, CYCLE_HOLD * (1 + ep["relapses"]))

    def forget() -> None:
        ep.update(turns=0, held=0, calm=0, relapses=0, names=set(), targets={})

    i, n_msgs = 0, len(messages)
    while i < n_msgs:
        m = messages[i]
        role = m.get("role")
        if role == "user" and prev_role != "tool":       # the user wrote (text beside tool results is not a new turn)
            files.clear()
            del seq[:], wrote[:], edits[:]
            fail_turns = 0
            forget()
            latest = []
        if role != "assistant" or not m.get("tool_calls"):
            if role == "assistant":
                latest = []
            prev_role = role
            i += 1
            continue
        j = i + 1
        while j < n_msgs and messages[j].get("role") == "tool":
            j += 1
        results = messages[i + 1:j]
        by_id = {}
        for r in results:
            rid = r.get("tool_call_id") or r.get("_call_id")
            if rid:
                by_id[rid] = r
        reps, progress = [], False
        failed_ops, good_calls, touched = [], 0, {}
        for k, call in enumerate(m["tool_calls"]):
            sig = call_signature(call)
            if sig is None:
                continue
            cid = (call.get("id") or call.get("_call_id")) if isinstance(call, dict) else None
            res_msg = by_id[cid] if cid in by_id else (results[k] if k < len(results) else None)
            raw = _result_text(res_msg) if res_msg is not None else None
            op = file_op(sig)
            failed = edit_failed(raw) if op is not None else raw is None    # any answer to another call is work done
            seq.append((sig, norm_result(raw)))
            wrote.append(is_write(sig))
            edits.append(op is not None)
            kind, target = None, None
            if op is None:
                if not failed:                          # other work: every file starts over (the plan: unless named)
                    good_calls += 1
                    for path in [p for p in files if not (files[p]["book"] and named(p, sig[1]))]:
                        del files[path]
                    # for the mask: not about the episode's plan file, and not the same call + result once more
                    progress = progress or (seq[-1] not in seq[-1 - CYCLE_NOVEL:-1] and not any(
                        book_t and named(t, sig[1]) for t, book_t in ep["targets"].items()))
            else:
                target, old, new, book = op
                if failed:
                    if raw is not None:
                        failed_ops.append((sig[0], target))
                else:
                    good_calls += 1
                    if not book:                        # a real file changed: the other files start over
                        for path in [p for p in files if p != target]:
                            del files[path]
                        if target not in ep["targets"]:
                            progress = True             # ... and it is other work for the episode
                s = files.setdefault(target, {"turns": 0, "olds": set(), "news": set(), "prev": None, "revisits": 0,
                                              "toggles": {}, "chain": 0, "last": None, "book": book})
                touched[target] = sig[0]
                # a file of the running episode is judged at half the counts: a relapse is caught on its second edit
                div = 2 if target in ep["targets"] else 1
                # the very same edit as the one before on this file is a retry (its failure was not recognized as
                # one), not a change
                if not failed and old != new and (old is None or (old, new) != s["prev"]):
                    if old is None:
                        revisit = new in s["news"]
                    else:
                        revisit = bool(new.strip()) and new in s["olds"]
                    if revisit:
                        s["revisits"] += 1
                        if s["revisits"] >= CYCLE_REVISITS // div:
                            kind = "back-and-forth"
                    if old is not None:
                        key = _MARKS.sub("[?]", old)
                        if key != old and key == _MARKS.sub("[?]", new):
                            s["toggles"].setdefault(key, set()).add((old, new))    # a retried edit counts once
                            if len(s["toggles"][key]) >= -(-CYCLE_TOGGLES // div):
                                kind = kind or "marks"
                        s["chain"] = s["chain"] + 1 if s["last"] is not None and old == s["last"] else 0
                        if s["chain"] >= CYCLE_CHAIN // div:
                            kind = kind or "rewrite"
                        s["olds"].add(old)
                        s["prev"] = (old, new)
                    s["news"].add(new)
                    s["last"] = new
            held = op is not None                       # may this call start or extend a mask?
            if kind is None:
                k_per = _period(seq, wrote)
                if k_per:
                    # a round with a run in it (edit -> test -> revert -> test) gets the note only
                    kind, target = "period", target or sig[0]
                    held = any(edits[-k_per:]) and not any(_RUN_TOOL.search(c[0][0]) for c in seq[-k_per:])
            if kind is not None:
                reps.append({"name": sig[0], "target": target, "kind": kind, "hold": held, "edit": op is not None})
        # the turn as a whole: only failed edits / only the plan or the todo list
        fail_turns = fail_turns + 1 if failed_ops and not good_calls else 0 if good_calls else fail_turns
        if not reps and fail_turns >= CYCLE_FAILS // (2 if ep["held"] else 1):
            reps = [{"name": n, "target": t, "kind": "failing", "hold": True, "edit": True} for n, t in failed_ops]
        touched = {t: n for t, n in touched.items() if t in files}      # not started over later in this turn
        for t in touched:
            files[t]["turns"] += 1
        if not reps:
            churn = [t for t in touched if files[t]["book"]
                     and files[t]["turns"] >= CYCLE_CHURN // (2 if t in ep["targets"] else 1)]
            reps = [{"name": touched[t], "target": t, "kind": "churn", "hold": False, "edit": True}
                    for t in sorted(churn)]
        if reps:
            if ep["turns"] and ep["calm"]:
                ep["relapses"] += 1
            ep["turns"] += 1
            ep["calm"] = 0
            ep["held"] += any(r["hold"] for r in reps)
            for r in reps:
                r["turn"] = ep["turns"]
                if r["hold"] and r["edit"]:
                    ep["names"].add(r["name"])
                    if r["kind"] != "failing":
                        ep["targets"][r["target"]] = r["target"] == TODO_TARGET or bool(_BOOK_FILE.search(str(r["target"])))
            if results and not _NOTE_TAIL.search(_raw_text(messages[j - 1])):
                r0 = next((r for r in reps if r["hold"] and r["kind"] not in ("failing", "period")), None)
                kinds = {r["kind"] for r in reps}
                if r0 is not None or kinds == {"churn"}:
                    name = str((r0 or reps[0])["target"]).replace("\\", "/").rsplit("/", 1)[-1]
                    name = re.sub(r"[\[\]\s]+", " ", name).strip()[:80] or "that file"
                    text = (CYCLE_NOTE if r0 is not None else CYCLE_NOTE_CHURN).replace("{file}", name)
                elif "failing" in kinds:
                    text = CYCLE_NOTE_FAIL
                else:
                    text = CYCLE_NOTE_PERIOD
                if ep["held"] >= CYCLE_MASK_TURNS and any(r["hold"] for r in reps):
                    text += paused
                _append_note(messages[j - 1], text)
        elif ep["turns"] and progress:
            ep["calm"] += 1
            if ep["calm"] >= 2 * hold():
                forget()
        latest = reps
        prev_role = "tool" if results else "assistant"
        i = j
    masking = ep["held"] >= CYCLE_MASK_TURNS and ep["calm"] < hold()
    return {"reps": latest, "mask": sorted(ep["names"]) if masking else [],
            "hold": hold() - ep["calm"] if masking else 0, "turns": ep["turns"]}


def cycle_mask_names(state, declared) -> list[str]:
    """The declared tool names to keep out of the next reply while an edit cycle's mask holds: the tools the cycle
    used and every other file-editing or todo-list tool (masking only `edit` moved the same rewrite to `write`)."""
    names = set((state or {}).get("mask") or ())
    if not names:
        return []
    return sorted(n for n in declared if n and (n in names or is_edit_tool(n)
                                                or _TODO_TOOL.search(n) and not _TODO_READ.search(n)))
