"""Speedy 0.5.3: the tool-call opening after `<tool_call>` (STRATA_TOOLCALL_CONSTRAIN), "says X, does Y"
(STRATA_TOOLCALL_INTENT) and junk writes in the loop rules.

What went wrong before (0.4.6 .. 0.5.3-pre): the server stopped at every `<tool_call>`, wrote "\\n<function=" itself and
let the model pick the name's tokens.  Qwen's vocabulary writes `<function=bash>` as `<function` `=b` `ash` `>` (and
`=g` `lob`, `=g` `rep`, `=read`, `=t` `od` `ow` `rite`), so a forced bare `=` token is a place the model only knows
from the names written `=` + NAME (`write`, `edit`, `skill`, `task`, `webfetch`).  Measured on an agent run with ten
tools: 2,376 of 2,387 such openings named one of those five (write 2,306), 10 one of the other five; without the forced
opening the same model wrote bash in 12 of 28.

The engine here models that: it writes its own choice with the tokenizer's tokens, and after a bare `=` it prefers the
`=` + NAME family.

    python -m unittest serve.test_toolcall_constrain
"""
import json
import os
import tempfile
import unittest
import urllib.request
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from serve import toolguard as g
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parent.parent
MERGED = ["function", "=b", "=g", "=read", "ash", "lob", "rep", "write", "edit", "bash", "read", "glob", "grep"]


class MergedTokenizer(ByteTokenizer):
    """Bytes, the specials, and a few merged tokens the way Qwen's vocabulary has them: `=b`, `=g`, `=read` (the `=` of
    `<function=` goes into the name's first token) next to whole-word `write` / `edit` that follow a bare `=`."""
    SPECIALS = ByteTokenizer.SPECIALS + ["<tool_call>"]
    ALWAYS = ("<tool_call>",)
    BASE = 256 + len(SPECIALS)
    tokens = [None] * (BASE + len(MERGED))          # vocab_index reads its length

    def encode_marked(self, text, parse_special=False, plain=()):
        out, marks, i = [], [], 0
        while i < len(text):
            for k, s in enumerate(self.SPECIALS):
                if (parse_special or s in self.ALWAYS) and text.startswith(s, i) and not any(a <= i < b for a, b in plain):
                    out.append(256 + k)
                    i += len(s)
                    marks.append((i, len(out)))
                    break
            else:
                hit = max((m for m in MERGED if text.startswith(m, i)), key=len, default=None)
                if hit is not None:
                    out.append(self.BASE + MERGED.index(hit))
                    i += len(hit)
                else:
                    out.extend(text[i].encode("utf-8"))
                    i += 1
        return out, marks

    def token_bytes(self, t):
        if t >= self.BASE:
            return MERGED[t - self.BASE].encode()
        return self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])

    def decode(self, ids, errors="replace"):
        return b"".join(self.token_bytes(t) for t in ids).decode("utf-8", errors=errors)


def tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": {"arg": {"type": "string"}}, "required": ["arg"]}}}


NAMES = ["bash", "edit", "glob", "grep", "read", "write"]
TOOLS = [tool(n) for n in NAMES]
PARAMS = "<parameter=arg>\nv\n</parameter>\n</function>\n</tool_call>"
FAMILY = ("write", "edit")              # written `=` + NAME: what the model knows after a bare `=`


class NativeEngine(MockEngine):
    """A model that wants to call `want`.  It writes `first` (thinking, text, `<tool_call>` and whatever opening `first`
    holds); asked to go on after an opening it writes the call's parameters; asked for one token under a logit bias it
    ranks the allowed ids as the real model does: the next token of its own spelling of `want` when the opening so
    far is that spelling, else - after a bare `=` - the first name of the `=` + NAME family, else the lowest id."""

    def __init__(self, tok, first, want="bash"):
        super().__init__(tok, "x")
        self.first, self.want, self.calls = first, want, []
        self.info = {"logit_bias": 1}

    def _after(self, ids):
        call = self.tok.encode("<tool_call>")[0]
        if call not in ids:
            return None
        return list(ids[len(ids) - ids[::-1].index(call):])

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls.append((list(ids), max_new, dict(sampling or {})))
        tok = self.tok
        after = self._after(ids[self.base:]) if getattr(self, "base", None) is not None else None
        if getattr(self, "base", None) is None:
            self.base = len(ids)
        bias = (sampling or {}).get("logit_bias") or {}
        canon = tok.encode("\n<function=" + self.want + ">\n")
        if max_new == 1 and any(v > 0 for v in bias.values()):
            allowed = {int(k) for k, v in bias.items() if v > 0}
            nxt = canon[len(after)] if canon[:len(after)] == after and len(canon) > len(after) else None
            if nxt in allowed:
                yield nxt
            elif after and after[-1] == ord("="):
                fam = [tok.encode(n)[0] for n in FAMILY if tok.encode(n)[0] in allowed]
                yield fam[0] if fam else min(allowed)
            else:
                yield min(allowed)
            return
        if after is None:
            out = tok.encode(self.first, parse_special=True)
        elif tok.decode(after).endswith(">\n"):
            out = tok.encode(PARAMS)
        elif canon[:len(after)] == after:
            out = canon[len(after):] + tok.encode(PARAMS)
        elif after[-1] == ord("="):
            out = tok.encode(FAMILY[0] + ">\n" + PARAMS)
        else:
            out = tok.encode("???")
        for t in (out + tok.encode("<|im_end|>", parse_special=True))[:max_new]:
            if cancel.is_set():
                return
            yield t


def reply(name, think="I will look at it.", text=""):
    return f"{think}</think>\n\n{text}<tool_call>\n<function={name}>\n" + PARAMS


class Pure(unittest.TestCase):
    tok = MergedTokenizer()

    def setUp(self):
        self.index = g.vocab_index(self.tok)
        self.enc = lambda text: self.tok.encode(text)   # noqa: E731

    def test_the_vocabulary_merges_the_equals_sign_into_some_names(self):
        t = self.tok
        self.assertEqual([t.decode([i]) for i in t.encode("\n<function=bash>\n")], ["\n", "<", "function", "=b", "ash", ">", "\n"])
        self.assertEqual([t.decode([i]) for i in t.encode("\n<function=write>\n")], ["\n", "<", "function", "=", "write", ">", "\n"])

    def test_shared_opening_stops_before_a_merged_equals(self):
        dec = lambda ids: self.tok.decode(ids)          # noqa: E731
        self.assertEqual(dec(g.shared_opening(self.enc, NAMES)), "\n<function")
        self.assertEqual(dec(g.shared_opening(self.enc, ["write", "edit"])), "\n<function=")
        self.assertEqual(dec(g.shared_opening(self.enc, ["write"])), "\n<function=")     # never into the name
        self.assertEqual(g.shared_opening(self.enc, []), [])

    def test_opening_state(self):
        tg = g.opening_targets(NAMES, ("\n", ""))
        for text in ("", "\n", "\n<", "\n<function=", "<function=ba", "\n<function=B", "\n<function=read"):
            self.assertEqual(g.opening_state(text, tg), "open", text)
        for text in ("\n<function=bash>", "\n<function=bash>\n<parameter", "<function=Read>", "\n<function=glob> "):
            self.assertEqual(g.opening_state(text, tg), "done", text)
        for text in ("\n<4", "\n\n", "\n<function=x", "\n<function=bashx", "\n<function=reads", "{", "\n<function_"):
            self.assertEqual(g.opening_state(text, tg), "bad", text)
        self.assertEqual(g.opening_name("\n<function=Read>\n", tg), "read")
        self.assertEqual(g.opening_name("<function=grep>", tg), "grep")

    def test_no_choice_is_made_for_the_model(self):
        asked = []

        def step(allowed, new):
            asked.append((sorted(self.tok.decode([i]) for i in allowed), self.tok.decode(new)))
            return self.tok.encode("=b")[0]

        new, name, picks = g.constrain_opening(step, self.enc, self.index, NAMES)
        self.assertEqual((self.tok.decode(new), name, picks), ("\n<function=bash>\n", "bash", 1))
        # asked once, after the tokens every name shares, among every token that continues a name: the bare `=` of
        # write / edit and the merged `=b` / `=g` / `=read` side by side
        self.assertEqual(asked, [(["=", "=b", "=g", "=read"], "\n<function")])

    def test_a_bare_equals_is_the_models_own_pick(self):
        picks = iter(["=", "write"])
        new, name, n = g.constrain_opening(lambda allowed, new: self.tok.encode(next(picks))[0], self.enc, self.index, NAMES)
        self.assertEqual((self.tok.decode(new), name, n), ("\n<function=write>\n", "write", 2))

    def test_finishes_what_the_model_started(self):
        # "\n<function=g" written, then a token no name continues: glob / grep are left, the model picks between them
        ids = self.tok.encode("\n<function=g")
        seen = []

        def step(allowed, new):
            seen.append(sorted(self.tok.decode([i]) for i in allowed))
            return self.tok.encode("rep")[0]

        new, name, n = g.constrain_opening(step, self.enc, self.index, NAMES, text="\n<function=g", ids=ids)
        self.assertEqual((self.tok.decode(new), name, n), ("rep>\n", "grep", 1))
        self.assertEqual(seen, [["l", "lob", "r", "rep"]])
        # one name left: nothing is asked
        new, name, n = g.constrain_opening(lambda a, b: self.fail("asked"), self.enc, self.index, NAMES,
                                           text="\n<function=e", ids=self.tok.encode("\n<function=") + [ord("e")])
        self.assertEqual((self.tok.decode(new), name, n), ("dit>\n", "edit", 0))
        # another letter case is the model's: `Bash` is read as bash
        new, name, n = g.constrain_opening(lambda a, b: self.fail("asked"), self.enc, self.index, NAMES,
                                           text="\n<function=Ba", ids=self.tok.encode("\n<function=Ba"))
        self.assertEqual((self.tok.decode(new), name), ("sh>\n", "bash"))

    def test_nothing_fits(self):
        self.assertIsNone(g.constrain_opening(lambda a, b: None, self.enc, self.index, NAMES, text="\n<function=x"))
        self.assertIsNone(g.constrain_opening(lambda a, b: 99999, self.enc, self.index, NAMES))    # outside the mask
        self.assertIsNone(g.constrain_opening(lambda a, b: None, self.enc, self.index, []))


class Served(unittest.TestCase):
    ASK = [{"role": "user", "content": "go"}]

    def start(self, engine, **env):
        self.logdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.logdir.cleanup)
        p = mock.patch.dict(os.environ, {"STRATA_TOOLCALL_FAIL_LOG": os.path.join(self.logdir.name, "f.jsonl"), **env})
        p.start()
        self.addCleanup(p.stop)
        self.tok, self.engine = engine.tok, engine
        self.svc = Service(engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def post(self, stream=False, tools=TOOLS):
        body = {"model": "m", "messages": self.ASK, "tools": tools, "max_tokens": 400, "stream": stream}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        log = StringIO()
        with redirect_stdout(log), urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
        self.log = log.getvalue()
        if not stream:
            m = json.loads(raw)["choices"][0]["message"]
            return {"content": m.get("content") or "", "names": [t["function"]["name"] for t in m.get("tool_calls") or []],
                    "args": [json.loads(t["function"]["arguments"]) for t in m.get("tool_calls") or []]}
        content, names = "", []
        for line in raw.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                for ch in json.loads(line[6:])["choices"]:
                    d = ch.get("delta") or {}
                    content += d.get("content") or ""
                    names += [(c.get("function") or {}).get("name") for c in d.get("tool_calls") or []
                              if (c.get("function") or {}).get("name")]
        return {"content": content, "names": names}

    def biased(self):
        return [(self.tok.decode(p), s) for p, n, s in self.engine.calls if n == 1 and s.get("logit_bias")]

    # ---- the model's own valid choice is never replaced
    def test_the_models_bash_stays_bash(self):
        """Fails before 0.5.3: "\\n<function=" was forced, and after its bare `=` the pick was `write`."""
        self.start(NativeEngine(MergedTokenizer(), reply("bash")), STRATA_TOOLCALL_CONSTRAIN="1")
        c = self.post()
        self.assertEqual((c["names"], c["args"]), (["bash"], [{"arg": "v"}]))

    def test_a_valid_opening_costs_no_request(self):
        for name in NAMES:
            with self.subTest(name=name):
                self.start(NativeEngine(MergedTokenizer(), reply(name), want=name), STRATA_TOOLCALL_CONSTRAIN="1")
                self.assertEqual(self.post()["names"], [name])
                self.assertEqual(len(self.engine.calls), 1)             # one pass, nothing asked again
                self.assertNotIn("constrained tool call", self.log)

    def test_on_by_default(self):
        self.start(NativeEngine(MergedTokenizer(), reply("execute")))
        self.assertEqual(self.post()["names"], ["edit"])                # "e" was written: `edit` is what can come of it
        for off in ("0", "off", "false"):
            self.start(NativeEngine(MergedTokenizer(), reply("execute")), STRATA_TOOLCALL_CONSTRAIN=off)
            c = self.post()
            self.assertEqual(c["names"], [])
            self.assertEqual(len(self.engine.calls), 1)

    def test_streaming_shows_the_name_once(self):
        self.start(NativeEngine(MergedTokenizer(), reply("bash")), STRATA_TOOLCALL_CONSTRAIN="1")
        c = self.post(stream=True)
        self.assertEqual(c["names"], ["bash"])
        self.assertNotIn("<function", c["content"])

    def test_another_letter_case_and_no_newline_are_left_alone(self):
        for first in ("ok</think>\n\n<tool_call>\n<function=Bash>\n" + PARAMS, "ok</think>\n\n<tool_call><function=bash>\n" + PARAMS):
            with self.subTest(first=first):
                self.start(NativeEngine(MergedTokenizer(), first), STRATA_TOOLCALL_CONSTRAIN="1")
                self.assertEqual(self.post()["names"], ["bash"])
                self.assertEqual(len(self.engine.calls), 1)

    def test_long_mcp_names_and_their_aliases_pass_untouched(self):
        long = "mcp__playwright__browser_navigate_and_wait_for_selector_visible"      # more tokens than the watch holds
        tools = TOOLS + [tool(long), tool("fs.read_file")]
        for written in (long, long.replace("__", "_"), "fs__read_file", "read_file", "tools.bash"):
            with self.subTest(written=written):
                self.start(NativeEngine(MergedTokenizer(), reply(written), want=written), STRATA_TOOLCALL_CONSTRAIN="1")
                c = self.post(tools=tools)
                self.assertEqual(len(c["names"]), 1)
                self.assertEqual(self.biased(), [])                     # nothing was asked, nothing replaced
                self.assertNotIn("names no declared tool", self.log)

    def test_claude_code_names(self):
        tools = [tool(n) for n in ("Bash", "Read", "Edit", "Write", "Grep", "Glob", "TodoWrite")]
        for name in ("Bash", "Grep", "TodoWrite"):
            self.start(NativeEngine(MergedTokenizer(), reply(name), want=name), STRATA_TOOLCALL_CONSTRAIN="1")
            self.assertEqual(self.post(tools=tools)["names"], [name])
            self.assertEqual(len(self.engine.calls), 1)

    def test_plain_chat_and_a_tool_call_tag_in_prose_without_tools(self):
        self.start(NativeEngine(MergedTokenizer(), "ok</think>\n\nThe tag is <tool_call> and nothing else."))
        c = self.post(tools=None)
        self.assertIn("The tag is", c["content"])
        self.assertEqual(len(self.engine.calls), 1)

    # ---- never an undeclared name
    def test_a_broken_opening_is_finished_with_the_models_own_ranking(self):
        """`<4>` where `<function=NAME>` belongs: the `<` stays, `function` is the only way on, then the model picks
        among `=` / `=b` / `=g` / `=read` - and takes its own `=b`."""
        self.start(NativeEngine(MergedTokenizer(), "ok</think>\n\n<tool_call>\n<4>\n" + PARAMS), STRATA_TOOLCALL_CONSTRAIN="1")
        c = self.post()
        self.assertEqual(c["names"], ["bash"])
        self.assertNotIn("<4>", c["content"])
        asked = self.biased()
        self.assertEqual(len(asked), 1)
        self.assertTrue(asked[0][0].endswith("<tool_call>\n<function"))             # no bare `=` was put there
        allowed = sorted(self.tok.decode([int(k)]) for k, v in asked[0][1]["logit_bias"].items() if v > 0)
        self.assertEqual(allowed, ["=", "=b", "=g", "=read"])
        self.assertEqual({v for v in asked[0][1]["logit_bias"].values()}, {g.CONSTRAIN_BIAS})   # one bias for all: a mask
        self.assertEqual(asked[0][1].get("repetition_penalty"), 1.0)
        self.assertIn("names no declared tool", self.log)
        self.assertIn("[strata] constrained tool call: bash (1 picks", self.log)

    def test_an_undeclared_name_becomes_the_declared_one_it_started(self):
        self.start(NativeEngine(MergedTokenizer(), reply("grepper"), want="grep"), STRATA_TOOLCALL_CONSTRAIN="1")
        c = self.post()
        self.assertEqual(c["names"], ["grep"])
        self.assertEqual(self.biased(), [])                             # "=g" "rep" stood: only ">\n" was left to write

    def test_the_turn_ending_inside_the_opening(self):
        self.start(NativeEngine(MergedTokenizer(), "ok</think>\n\n<tool_call>\n<function=ba"), STRATA_TOOLCALL_CONSTRAIN="1")
        self.assertEqual(self.post()["names"], ["bash"])

    def test_the_requests_own_bans_stay(self):
        self.start(NativeEngine(MergedTokenizer(), "ok</think>\n\n<tool_call>\n<4>\n" + PARAMS), STRATA_TOOLCALL_CONSTRAIN="1")
        tok = self.tok
        body = {"model": "m", "messages": self.ASK, "tools": TOOLS, "max_tokens": 400,
                "logit_bias": {str(tok.encode("=b")[0]): -100}}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with redirect_stdout(StringIO()), urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        bias = self.biased()[0][1]["logit_bias"]
        self.assertEqual(bias[tok.encode("=b")[0]], -100)
        self.assertEqual(bias[tok.encode("=g")[0]], g.CONSTRAIN_BIAS)

    # ---- eager: the 0.4.6 flow, without the forced `=`
    def test_eager_asks_at_every_opening_and_still_gets_bash(self):
        self.start(NativeEngine(MergedTokenizer(), reply("bash")), STRATA_TOOLCALL_CONSTRAIN="eager")
        self.assertEqual(self.post()["names"], ["bash"])
        asked = self.biased()
        self.assertEqual(len(asked), 1)
        self.assertTrue(asked[0][0].endswith("<tool_call>\n<function"))

    # ---- tool call repair no longer forces the bare `=` either
    def test_repair_opening(self):
        """Fails before 0.5.3: the repair wrote "<tool_call>\\n<function=" and got `write`."""
        self.start(NativeEngine(MergedTokenizer(), "ok</think>\n\n<tool_call>\n<4>\n" + PARAMS),
                   STRATA_TOOLCALL_CONSTRAIN="0", STRATA_TOOLCALL_REPAIR="1")
        self.assertEqual(self.post()["names"], ["bash"])
        self.assertIn("tool call repair", self.log)

    # ---- says X, does Y
    SAID = "The write tool keeps being selected. Let me try to invoke the `bash` tool with a command."

    def test_said_bash_opened_write_is_written_again_as_bash(self):
        self.start(NativeEngine(MergedTokenizer(), reply("write", think=self.SAID), want="write"))
        c = self.post()
        self.assertEqual(c["names"], ["bash"])
        self.assertEqual(len(self.engine.calls), 2)                     # the reply, then on from "<function=bash>\n"
        self.assertTrue(self.tok.decode(self.engine.calls[1][0]).endswith("<tool_call>\n<function=bash>\n"))
        self.assertIn("[strata] tool intent: the reply said bash, the call opened as write", self.log)
        self.assertEqual(self.svc.totals.get("tool_calls_repicked"), 1)

    def test_said_bash_streaming_never_shows_write(self):
        self.start(NativeEngine(MergedTokenizer(), reply("write", think=self.SAID), want="write"))
        self.assertEqual(self.post(stream=True)["names"], ["bash"])

    def test_the_answer_text_is_what_was_said_last(self):
        # the thinking names bash, the answer's own sentence does not: nothing is replaced
        self.start(NativeEngine(MergedTokenizer(), reply("write", think=self.SAID, text="Writing the plan first.\n\n"), want="write"))
        self.assertEqual(self.post()["names"], ["write"])
        self.start(NativeEngine(MergedTokenizer(), reply("write", text="I'll invoke the `glob` tool.\n\n"), want="write"))
        self.assertEqual(self.post()["names"], ["glob"])

    def test_no_intent_no_change(self):
        for think in ("Let me write the plan first.", "I will look at it.", "Should I use the bash tool?",
                      "I should not call the bash tool here.", "I'll use the read tool and then the bash tool."):
            with self.subTest(think=think):
                self.start(NativeEngine(MergedTokenizer(), reply("write", think=think), want="write"))
                self.assertEqual(self.post()["names"], ["write"])
                self.assertEqual(len(self.engine.calls), 1)

    def test_same_tool_and_edit_family_are_not_mismatches(self):
        self.start(NativeEngine(MergedTokenizer(), reply("bash", think=self.SAID)))
        self.assertEqual(self.post()["names"], ["bash"])
        self.assertEqual(len(self.engine.calls), 1)
        self.start(NativeEngine(MergedTokenizer(), reply("write", think="Let me use the edit tool."), want="write"))
        self.assertEqual(self.post()["names"], ["write"])

    def test_intent_off(self):
        self.start(NativeEngine(MergedTokenizer(), reply("write", think=self.SAID), want="write"), STRATA_TOOLCALL_INTENT="0")
        self.assertEqual(self.post()["names"], ["write"])

    def test_only_the_first_call_of_a_reply(self):
        two = reply("bash", think=self.SAID) + "\n<tool_call>\n<function=write>\n" + PARAMS
        self.start(NativeEngine(MergedTokenizer(), two))
        self.assertEqual(self.post()["names"], ["bash", "write"])


if __name__ == "__main__":
    unittest.main()
