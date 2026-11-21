"""CPU-only tests for the opt-in loop breaker (loopbreak-1120): python -m unittest serve.test_loop_breaker."""
import copy
import json
import threading
import unittest

from serve import toolguard as g
from serve.test_reasoning_loop_recovery import Tokenizer, RecordingEngine
from serve.server import Service


def call(name, i, **args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def chain(steps):
    """steps: list of (name, args, result) -> user + alternating assistant/tool messages."""
    msgs = [{"role": "user", "content": "fix the bug"}]
    for i, (name, args, res) in enumerate(steps):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [call(name, i, **args)]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": res})
    return msgs


T = ("shell", {"command": "python3 run_tests.py"}, "1 failed")


class MarkLoops(unittest.TestCase):
    def test_same_call_same_result_flagged(self):
        m = chain([T, T])
        reps = g.mark_loops(m)
        self.assertEqual([(r["name"], r["streak"]) for r in reps], [("shell", 1)])
        self.assertIn(g.LOOP_NOTE, m[-1]["content"])
        self.assertNotIn("Note", m[2]["content"])

    def test_different_result_not_flagged(self):
        m = chain([T, ("shell", T[1], "0 failed")])
        self.assertEqual(g.mark_loops(m), [])
        self.assertNotIn(g.LOOP_NOTE, m[-1]["content"])

    def test_write_between_not_flagged(self):
        for w in (("edit", {"path": "a.py", "oldString": "x", "newString": "y"}, "ok"),
                  ("write", {"path": "a.py", "content": "y"}, "ok"),
                  ("shell", {"command": "sed -i s/x/y/ a.py"}, ""),
                  ("shell", {"command": "echo y > a.py"}, "")):
            m = chain([T, w, T])
            self.assertEqual(g.mark_loops(m), [], w)

    def test_read_between_still_a_loop(self):
        m = chain([T, ("read", {"path": "a.py"}, "code"), T])
        self.assertEqual([r["name"] for r in g.mark_loops(m)], ["shell"])

    def test_streak_and_mask(self):
        m = chain([T, T])
        self.assertEqual(g.masked_names(g.mark_loops(m)), [])      # 1st repeat: note + floor only
        m = chain([T, T, T])
        reps = g.mark_loops(m)
        self.assertEqual(reps[0]["streak"], 2)
        self.assertEqual(g.masked_names(reps), ["shell"])           # 2nd consecutive: mask

    def test_streak_resets(self):
        m = chain([T, T, ("read", {"path": "a.py"}, "code"), T])
        self.assertEqual([r["streak"] for r in g.mark_loops(m)], [1])   # turn 3 broke the run of loop turns

    def test_history_only_and_stable(self):
        base = chain([T, T, T])
        a, b = copy.deepcopy(base), copy.deepcopy(base)
        g.mark_loops(a)
        g.mark_loops(b)
        self.assertEqual(a, b)
        c = copy.deepcopy(base) + [{"role": "assistant", "content": "done"}, {"role": "user", "content": "more"}]
        self.assertEqual(g.mark_loops(c), [])                      # notes stay, nothing is "latest" any more
        self.assertEqual(c[:len(a)], a)                             # same prefix rendered as before

    def test_note_text_not_counted_as_result(self):
        m = chain([T, T])
        m[2]["content"] += "\n\n[Note: earlier note]"
        self.assertEqual(len(g.mark_loops(m)), 1)


class LoopFloor(unittest.TestCase):
    def run_case(self, breaker, flagged, floor=256, tools=(), info=None):
        tok = Tokenizer()
        ids = tok.encode('<|im_start|>user\nfix it<|im_end|>\n<|im_start|>user\n<tool_response>\n1 failed\n'
                         '</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n', parse_special=True)
        engine = RecordingEngine(tok, ['a' * 400 + '\n</think>\nok', 'more thinking.\n</think>\nanswer'])
        if info is not None:
            engine.info = info
        svc = Service(engine, tok, None)
        svc.answer_room = False
        svc.tool_think_floor = 0
        svc.loop_breaker, svc.loop_floor, svc.loop_mask = breaker, floor, True
        svc._think_close_id = 7
        cancel = threading.Event()
        cancel._strata_loop = [{"name": "shell", "arguments": "{}", "times": 3, "streak": flagged}] if flagged else []
        list(svc.run(ids, True, list(tools), 4000, {}, cancel))
        return engine

    def test_floor_only_on_flagged_turn(self):
        self.assertEqual(self.run_case(True, 1).sampling_calls[0].get("logit_bias"), {7: -100.0})
        self.assertNotIn("logit_bias", self.run_case(True, 0).sampling_calls[0])
        self.assertNotIn("logit_bias", self.run_case(False, 1).sampling_calls[0])

    def test_mask_drops_repeated_name(self):
        import serve.server as srv
        seen = []
        ob, os_ = srv.tnc_api.build, srv.tnc_api.engine_supports
        srv.tnc_api.build = lambda tok, names, thinking: (seen.append(list(names)), "REC")[1]
        srv.tnc_api.engine_supports = lambda e: True
        try:
            tools = [{"name": "shell", "parameters": {}}, {"name": "read", "parameters": {}}]
            e2 = self.run_case(True, 2, tools=tools)
            e1 = self.run_case(True, 1, tools=tools)
        finally:
            srv.tnc_api.build, srv.tnc_api.engine_supports = ob, os_
        self.assertIn(["read"], seen)                                # streak 2: shell masked
        self.assertEqual(e2.sampling_calls[0].get("_tool_name_constraint"), "REC")
        self.assertNotIn("_tool_name_constraint", e1.sampling_calls[0])   # streak 1: no mask

    def test_configure_from_env(self):
        svc = Service(RecordingEngine(Tokenizer(), ['x']), Tokenizer(), None)
        svc.configure_toolguard({"STRATA_LOOP_BREAKER": "1"})
        self.assertTrue(svc.loop_breaker)
        self.assertEqual((svc.loop_floor, svc.loop_mask), (g.LOOP_FLOOR, True))
        svc.configure_toolguard({"STRATA_LOOP_BREAKER": "1", "STRATA_LOOP_BREAKER_FLOOR": "0", "STRATA_LOOP_BREAKER_MASK": "0"})
        self.assertEqual((svc.loop_floor, svc.loop_mask), (0, False))
        svc.configure_toolguard({})
        self.assertTrue(svc.loop_breaker)                 # Speedy 0.5.0: on by default
        self.assertEqual((svc.loop_floor, svc.loop_mask), (0, True))
        svc.configure_toolguard({"STRATA_LOOP_BREAKER": "0"})
        self.assertFalse(svc.loop_breaker)


# ------------------------------------------------------------------ 0.5.1: failing writes, near repeats, volatile parts
import os
import random

FAIL = "File /tmp/opencode/placeholder.txt not found"


def edit_args(i):
    a = {"filePath": "/tmp/opencode/placeholder.txt", "oldString": "x", "newString": "y"}
    return a if i % 2 else {"filePath": a["filePath"], "newString": "y", "oldString": "x"}   # key order flips


def chain_ids(steps):
    """Like chain(), with fresh OpenCode-style call ids each turn."""
    rnd = random.Random(7)
    msgs = [{"role": "user", "content": "fix the repo"}]
    for name, args, res in steps:
        cid = "call_%016x_0" % rnd.getrandbits(64)
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": res})
    return msgs


class FailingWrites(unittest.TestCase):
    def test_failing_edit_loop_566(self):
        m = chain_ids([("edit", edit_args(i), FAIL) for i in range(566)])
        reps = g.mark_loops(m)
        self.assertEqual([r["name"] for r in reps], ["edit"])
        self.assertGreaterEqual(reps[0]["streak"], 2)
        self.assertEqual(g.masked_names(reps), ["edit"])
        self.assertNotIn("Note", m[2]["content"])          # the first failing edit is not a loop yet
        self.assertIn(g.LOOP_NOTE, m[4]["content"])        # the second one is

    def test_failing_edit_streak_by_turn(self):
        self.assertEqual(g.mark_loops(chain_ids([("edit", edit_args(i), FAIL) for i in range(1)])), [])
        r2 = g.mark_loops(chain_ids([("edit", edit_args(i), FAIL) for i in range(2)]))
        self.assertEqual([(r["name"], r["streak"]) for r in r2], [("edit", 1)])
        r3 = g.mark_loops(chain_ids([("edit", edit_args(i), FAIL) for i in range(3)]))
        self.assertEqual(g.masked_names(r3), ["edit"])

    def test_failure_kinds(self):
        for res in ("Error: oldString not found in content", "No changes to apply: oldString and newString are identical",
                    "oldString found multiple matches", "no such file or directory"):
            self.assertTrue(g.failed_result(res), res)
        for res in ("Edit applied successfully.", "Wrote file successfully.", ""):
            self.assertFalse(g.failed_result(res), res)

    def test_successful_write_opens_epoch(self):
        w = ("write", {"filePath": "a.py", "content": "x"}, "Wrote file successfully.")
        self.assertEqual(g.mark_loops(chain_ids([w, T, w, T])), [])

    def test_real_edit_test_cycles_not_loop(self):
        steps = []
        for k in range(6):
            steps.append(("edit", {"filePath": "a.py", "oldString": f"v{k}", "newString": f"v{k+1}"}, "Edit applied successfully."))
            steps.append(("bash", {"command": "pytest -q"}, f"{6 - k} failed in 0.{k}2s"))
        self.assertEqual(g.mark_loops(chain_ids(steps)), [])
        # same test failure after each edit (the edit changed the file, the failure stayed): still not a loop
        steps = []
        for k in range(6):
            steps.append(("edit", {"filePath": "a.py", "oldString": f"v{k}", "newString": f"v{k+1}"}, "Edit applied successfully."))
            steps.append(("bash", {"command": "pytest -q"}, "1 failed in 0.31s"))
        self.assertEqual(g.mark_loops(chain_ids(steps)), [])

    def test_successful_edits_in_a_row_not_loop(self):
        steps = [("edit", {"filePath": f"f{k}.py", "oldString": "a", "newString": "b"}, "Edit applied successfully.")
                 for k in range(5)]
        self.assertEqual(g.mark_loops(chain_ids(steps)), [])

    def test_near_identical_retries(self):
        steps = [("edit", {"filePath": "a.py", "oldString": f"def f({k}):", "newString": "def f():"},
                  "Error: oldString not found in content") for k in range(3)]
        self.assertEqual(g.mark_loops(chain_ids(steps[:2])), [])          # two differing tries: not yet
        reps = g.mark_loops(chain_ids(steps))
        self.assertEqual([r["name"] for r in reps], ["edit"])

    def test_near_streak_broken_by_other_call(self):
        e = lambda k: ("edit", {"filePath": "a.py", "oldString": f"x{k}", "newString": "y"}, "Error: oldString not found")
        self.assertEqual(g.mark_loops(chain_ids([e(0), e(1), ("read", {"filePath": "a.py"}, "1: x"), e(2)])), [])

    def test_near_success_results_similar_args(self):
        steps = [("bash", {"command": f"python3 check.py --seed {k}"}, "OK: 0 mismatches in 0.4s") for k in range(3)]
        self.assertEqual([r["name"] for r in g.mark_loops(chain_ids(steps))], ["bash"])

    def test_searches_with_no_output_not_loop(self):
        steps = [("bash", {"command": f"grep -rn {w} src"}, "(no output)") for w in ("alpha", "beta", "gamma", "delta")]
        self.assertEqual(g.mark_loops(chain_ids(steps)), [])
        steps = [("read", {"filePath": f"src/{w}.py"}, "1: import os") for w in ("alpha", "server", "util_helpers")]
        self.assertEqual(g.mark_loops(chain_ids(steps)), [])

    def test_volatile_ids(self):
        b = {"command": "./run.sh"}
        steps = [("bash", b, "job 3f9a2c41d7 failed after 12 ms at 2026-11-21T13:04:11Z"),
                 ("bash", b, "job 88be01aa42 failed after 9 ms at 2026-11-21T13:04:15Z")]
        self.assertEqual([r["name"] for r in g.mark_loops(chain_ids(steps))], ["bash"])
        self.assertEqual(g.norm_result("call_abc123_0 took 1.5s"), "# took #")

    def test_key_order(self):
        self.assertEqual(g.call_signature({"function": {"name": "t", "arguments": '{"a":1,"b":2}'}}),
                         g.call_signature({"function": {"name": "t", "arguments": '{"b":2,"a":1}'}}))

    def test_replay_fixture(self):
        """Trimmed synthetic replay shaped like the TB2 fix-git transcript rows (message deltas at msg_offset)."""
        path = os.path.join(os.path.dirname(__file__), "fixtures", "loop_replay_failing_edit.jsonl")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        hist, first_note, first_mask = [], None, None
        for t, row in enumerate(rows, 1):
            hist = hist[:row["msg_offset"]] + row["messages"]
            reps = g.mark_loops(copy.deepcopy(hist))
            if reps and first_note is None:
                first_note = t
            if g.masked_names(reps) and first_mask is None:
                first_mask = t
        self.assertEqual((first_note, first_mask), (6, 7))    # 2nd failing edit -> note, 3rd -> mask


class RepeatBackstop(unittest.TestCase):
    def test_backstop_needs_three(self):
        s = ("read", {"filePath": "a"}, "x")
        m = chain([s, s])
        self.assertEqual(g.mark_repeats(m, min_times=3, skip_noted=True), [])
        m = chain([s, s, s])
        self.assertEqual([r["times"] for r in g.mark_repeats(m, min_times=3, skip_noted=True)], [3])

    def test_no_double_note(self):
        m = chain_ids([("edit", edit_args(i), FAIL) for i in range(4)])
        g.mark_loops(m)
        g.mark_repeats(m, min_times=3, skip_noted=True)
        for msg in m:
            if msg["role"] == "tool":
                self.assertLessEqual(msg["content"].count("[Note:"), 1)

    def test_server_backstop_when_breaker_on(self):
        svc = Service(RecordingEngine(Tokenizer(), ['x']), Tokenizer(), None)
        svc.configure_toolguard({})
        self.assertTrue(svc.loop_breaker)
        # same call, result changes each time (the breaker sees no loop): the backstop note on the 3rd call only
        m = chain([("bash", {"command": "date"}, f"t{k}") for k in range(3)])
        svc._toolguard_prompt(m)
        self.assertNotIn("Note", m[2]["content"] + m[4]["content"])
        self.assertIn(g.REPEAT_NOTE, m[6]["content"])
        m = chain_ids([("edit", edit_args(i), FAIL) for i in range(4)])
        svc._toolguard_prompt(m)
        self.assertTrue(all(x["content"].count("[Note:") <= 1 for x in m if x["role"] == "tool"))
        self.assertIn(g.LOOP_NOTE, m[-1]["content"])


# ------------------------------------------------------------------ 0.5.3: edit cycles (back-and-forth edits of one file)
PLAN = "/work/PLAN.md"
OK = "Edit applied successfully."
STEP = "Create the service directory"


def mark(a, b, line=STEP, path=PLAN):
    return ("edit", {"filePath": path, "oldString": f"- [{a}] {line}", "newString": f"- [{b}] {line}"}, OK)


def cyc(steps):
    m = chain_ids(steps)
    return g.mark_cycles(m), m


class EditCycles(unittest.TestCase):
    def test_checkbox_cycle_noted_then_masked(self):
        ring = [mark("~", "x"), mark("x", " "), mark(" ", "~")]
        base = [("write", {"filePath": PLAN, "content": "- [ ] " + STEP}, "Wrote file successfully."), mark(" ", "~")]
        st, m = cyc(base + ring[:1])                 # [ ] -> [~] -> [x] with nothing between: fine
        self.assertEqual((st["reps"], st["mask"]), ([], []))
        st, m = cyc(base + ring[:2])                 # ... -> [ ]: the same line's mark a third time: the note, no mask yet
        self.assertEqual([(r["name"], r["kind"], r["turn"]) for r in st["reps"]], [("edit", "marks", 1)])
        self.assertEqual(st["mask"], [])
        self.assertIn("PLAN.md back and forth", m[-1]["content"])
        self.assertIn("Do the next real step", m[-1]["content"])
        st, m = cyc(base + ring)                     # the cycle goes on: the editing tools are held back
        self.assertEqual(st["reps"][0]["kind"], "back-and-forth")
        self.assertEqual((st["mask"], st["hold"], st["turns"]), (["edit"], g.CYCLE_HOLD, 2))
        self.assertIn("paused", m[-1]["content"])
        self.assertEqual(g.cycle_mask_names(st, ["bash", "edit", "read", "todowrite", "write", "MultiEdit"]),
                         ["MultiEdit", "edit", "todowrite", "write"])
        self.assertEqual(g.cycle_mask_names({"mask": []}, ["bash", "edit"]), [])
        self.assertEqual(g.cycle_mask_names(None, ["bash", "edit"]), [])

    def test_two_state_flip(self):
        a, b = mark(" ", "x"), mark("x", " ")
        self.assertEqual(cyc([a, b])[0]["reps"], [])                 # one undo is not a cycle
        self.assertEqual(cyc([a, b, a])[0]["reps"][0]["kind"], "back-and-forth")
        self.assertEqual(cyc([a, b, a, b])[0]["mask"], ["edit"])

    def test_mask_holds_until_other_work(self):
        a, b = mark(" ", "x"), mark("x", " ")
        loop = [a, b, a, b]
        work = lambda k: ("bash", {"command": f"make step{k}"}, f"built {k}")
        st, _ = cyc(loop + [work(1)])
        self.assertEqual((st["mask"], st["hold"]), (["edit"], 1))    # more than one reply: still held
        st, _ = cyc(loop + [("read", {"filePath": PLAN}, "- [ ] x")])
        self.assertEqual(st["hold"], 2)                              # looking at the plan again is not other work
        st, _ = cyc(loop + [("bash", {"command": f"sed -i s/x/y/ {PLAN}"}, "")])
        self.assertEqual(st["hold"], 2)                              # nor is rewriting it with the shell
        junk = ("skill", {"name": "some-skill"}, "<skill_content>")
        st, _ = cyc(loop + [junk, junk, junk])
        self.assertEqual(st["hold"], 1)                              # the same call + result again is not other work
        self.assertEqual(cyc(loop + [("todoread", {}, "[]")])[0]["hold"], 1)     # reading the todo list is a call like any
        st, _ = cyc(loop + [work(1), work(2)])
        self.assertEqual((st["mask"], st["turns"]), ([], 2))         # released, the episode is remembered
        st, m = cyc(loop + [work(1), work(2), a])                    # a legit update after real work: fine
        self.assertEqual((st["reps"], st["mask"]), ([], []))
        st, m = cyc(loop + [work(1), work(2), a, b])                 # relapse: caught on its second edit, longer hold
        self.assertEqual(st["reps"][0]["kind"], "back-and-forth")
        self.assertEqual((st["mask"], st["hold"]), (["edit"], 2 * g.CYCLE_HOLD))
        st, _ = cyc(loop + [work(k) for k in range(2 * g.CYCLE_HOLD)])
        self.assertEqual(st["turns"], 0)                             # forgotten after enough other work
        m = chain_ids(loop)
        m.append({"role": "assistant", "content": "Done."})
        m.append({"role": "user", "content": "now the next task"})
        self.assertEqual(g.mark_cycles(m), {"reps": [], "mask": [], "hold": 0, "turns": 0})   # the user wrote

    def test_write_same_content_and_todo_list(self):
        w = lambda c: ("write", {"filePath": "/work/run.sh", "content": c}, "Wrote file successfully.")
        self.assertEqual(cyc([w("a"), w("b"), w("a")])[0]["reps"], [])
        self.assertEqual(cyc([w("a"), w("b"), w("a"), w("b")])[0]["reps"][0]["kind"], "back-and-forth")
        self.assertEqual(cyc([w("a"), w("a"), w("a")])[0]["reps"][0]["kind"], "back-and-forth")
        t = lambda s: ("todowrite", {"todos": [{"content": STEP, "status": s}]}, "ok")
        st, m = cyc([t("pending"), t("in_progress"), t("pending"), t("in_progress")])
        self.assertEqual([(r["name"], r["target"]) for r in st["reps"]], [("todowrite", g.TODO_TARGET)])
        self.assertIn("the todo list back and forth", m[-1]["content"])

    def test_marks_only_and_rewrite_chain_and_churn(self):
        # the same line's mark changed a third time with nothing done between (no text ever restored exactly)
        e = lambda a, b: mark(a, b)
        st, _ = cyc([e(" ", "~"), e("~", "-"), e("-", "x")])
        self.assertEqual(st["reps"][0]["kind"], "marks")
        # a dummy edit chain x -> y -> z -> ... on one file
        ch = [("edit", {"filePath": "/tmp/p.txt", "oldString": chr(97 + k), "newString": chr(98 + k)}, OK)
              for k in range(6)]
        self.assertEqual(cyc(ch[:4])[0]["reps"], [])
        self.assertEqual(cyc(ch[:5])[0]["reps"][0]["kind"], "rewrite")
        # many turns that only edit the session plan: a soft note, never a mask
        many = [mark(" ", "x", line=f"step {k}") for k in range(g.CYCLE_CHURN + 6)]
        self.assertEqual(cyc(many[:g.CYCLE_CHURN - 1])[0]["reps"], [])
        st, m = cyc(many[:g.CYCLE_CHURN])
        self.assertEqual([(r["kind"], r["hold"]) for r in st["reps"]], [("churn", False)])
        self.assertIn("Unless these edits are the task itself", m[-1]["content"])
        st, m = cyc(many)
        self.assertEqual(st["mask"], [])
        self.assertNotIn("paused", m[-1]["content"])
        none = {"reps": [], "mask": [], "hold": 0, "turns": 0}
        code = [("edit", {"filePath": "/work/a.py", "oldString": f"v{k}", "newString": f"w{k}"}, OK) for k in range(30)]
        self.assertEqual(cyc(code)[0], none)                                                 # not a plan file
        doc = [("edit", {"filePath": "/work/PLAN-STREAMING.md", "oldString": f"v{k}", "newString": f"w{k}"}, OK)
               for k in range(30)]
        self.assertEqual(cyc(doc)[0], none)                                                  # a document being written
        # 18 edits of the plan in ONE turn (Claude Code sends them together) are one update
        m = [{"role": "user", "content": "revise the plan"},
             {"role": "assistant", "content": "", "tool_calls": [call("Edit", k, file_path=PLAN, old_string=f"a{k}",
                                                                      new_string=f"b{k}") for k in range(18)]}]
        m += [{"role": "tool", "tool_call_id": f"c{k}", "content": "updated"} for k in range(18)]
        self.assertEqual(g.mark_cycles(m), none)

    def test_failing_edits_of_changing_files(self):
        # TB2 custom-memory-heap-crash: edits of /tmp/.../probe1, probe2, ... (never the same call or the same error text)
        probe = lambda k: ("edit", {"filePath": f"/tmp/w/probe{k}", "oldString": str(k), "newString": str(k + 1)},
                           f"File /tmp/w/probe{k} not found")
        self.assertEqual(g.mark_loops(chain_ids([probe(k) for k in range(8)])), [])          # the 0.5.1 rules see nothing
        self.assertEqual(cyc([probe(k) for k in range(3)])[0]["reps"], [])
        st, m = cyc([probe(k) for k in range(4)])
        self.assertEqual([(r["kind"], r["turn"]) for r in st["reps"]], [("failing", 1)])
        self.assertIn(g.CYCLE_NOTE_FAIL, m[-1]["content"])
        st, m = cyc([probe(k) for k in range(5)])
        self.assertEqual(g.cycle_mask_names(st, ["bash", "edit", "write", "read"]), ["edit", "write"])
        self.assertTrue(m[-1]["content"].endswith(g.CYCLE_NOTE_FAIL + g.CYCLE_PAUSED + "]"))
        # five edits sent together that all fail for one reason (file not read yet), then the retry: one failed turn
        m = [{"role": "user", "content": "go"},
             {"role": "assistant", "content": "", "tool_calls": [call("Edit", k, file_path="/w/a.md", old_string=f"a{k}",
                                                                      new_string=f"b{k}") for k in range(5)]}]
        m += [{"role": "tool", "tool_call_id": f"c{k}", "content": "<tool_use_error>File has not been read yet. Read it "
               "first before writing to it.</tool_use_error>"} for k in range(5)]
        self.assertEqual(g.mark_cycles(m)["reps"], [])
        self.assertTrue(g.failed_result(m[-1]["content"]))
        # ... and with an error text the server does not know, the retry of the same edits is still no cycle
        for res in ("please open the file first", "The file must be opened before it is changed."):
            m = [{"role": "user", "content": "go"}]
            for rnd_, r in enumerate((res, "updated")):
                m.append({"role": "assistant", "content": "", "tool_calls": [
                    call("Edit", f"{rnd_}_{k}", file_path=PLAN, old_string=f"- [ ] s{k}", new_string=f"- [x] s{k}")
                    for k in range(5)]})
                m += [{"role": "tool", "tool_call_id": f"c{rnd_}_{k}", "content": r} for k in range(5)]
            m.append({"role": "assistant", "content": "", "tool_calls": [
                call("Edit", "z", file_path=PLAN, old_string="- [x] s1", new_string="- [~] s1")]})
            m.append({"role": "tool", "tool_call_id": "cz", "content": "updated"})
            self.assertEqual(g.mark_cycles(m), {"reps": [], "mask": [], "hold": 0, "turns": 0}, res)
        # failing shell commands in a row are ordinary debugging
        sh = [("bash", {"command": f"cc -c f{k}.c"}, f"error: f{k}.c not found") for k in range(8)]
        self.assertEqual(cyc(sh)[0]["reps"], [])

    def test_period_cycle_with_shell_writes(self):
        a = ("bash", {"command": "sed -i s/1/2/ conf.ini"}, "")
        b = ("bash", {"command": "sed -i s/2/1/ conf.ini"}, "")
        self.assertEqual(cyc([a, b, a, b])[0]["reps"], [])
        st, m = cyc([a, b, a, b, a, b])
        self.assertEqual([(r["kind"], r["hold"]) for r in st["reps"]], [("period", False)])
        self.assertEqual(st["mask"], [])                             # the shell is never masked by a cycle
        self.assertIn(g.CYCLE_NOTE_PERIOD, m[-1]["content"])
        # two files flipped in turn (each edit is "other work" for the other file): the period finds it
        p = [("edit", {"filePath": "/w/a.py", "oldString": "A", "newString": "B"}, OK),
             ("edit", {"filePath": "/w/b.py", "oldString": "C", "newString": "D"}, OK),
             ("edit", {"filePath": "/w/a.py", "oldString": "B", "newString": "A"}, OK),
             ("edit", {"filePath": "/w/b.py", "oldString": "D", "newString": "C"}, OK)]
        self.assertEqual(cyc(p + p[:3])[0]["reps"], [])
        st, m = cyc(p + p)
        self.assertEqual([(r["kind"], r["hold"]) for r in st["reps"]], [("period", True)])
        self.assertEqual(cyc(p + p + p[:1])[0]["mask"], ["edit"])
        # TB2 cobol-modernization: the same file written again and again with a junk call between (no run in the round)
        w = ("write", {"filePath": "/tmp/w/t.sh", "content": "echo 1"}, "Wrote file successfully.")
        junk = ("skill", {"name": "some-skill"}, "<skill_content>")
        st, m = cyc([w, junk] * 3)
        self.assertEqual([(r["kind"], r["hold"], r["edit"]) for r in st["reps"]], [("period", True, False)])
        st, m = cyc([w, junk] * 4)
        self.assertEqual(g.cycle_mask_names(st, ["bash", "write", "edit", "skill", "read"]), ["edit", "write"])
        self.assertTrue(m[-1]["content"].endswith(g.CYCLE_NOTE_PERIOD + g.CYCLE_PAUSED + "]"))
        # polling with changing output, or reads only, is not a cycle
        poll = []
        for k in range(6):
            poll += [("bash", {"command": "sleep 30"}, ""), ("bash", {"command": "tail -1 build.log > /tmp/l; cat /tmp/l"},
                                                             f"step {k}")]
        self.assertEqual(cyc(poll)[0]["reps"], [])
        reads = [("read", {"filePath": f"/w/{n}.py"}, n) for n in "abc"] * 4
        self.assertEqual(cyc(reads)[0]["reps"], [])

    def test_normal_work_not_flagged(self):
        none = {"reps": [], "mask": [], "hold": 0, "turns": 0}
        # building a file with many edits in a row (different regions, one replacement text used several times)
        build = [("write", {"filePath": "/w/app.py", "content": "def a():\n    TODO\n"}, "Wrote file successfully.")]
        build += [("edit", {"filePath": "/w/app.py", "oldString": f"    body{k}\n", "newString": "    pass\n"}, OK)
                  for k in range(12)]
        build += [("edit", {"filePath": "/w/app.py", "oldString": f"import m{k}\n", "newString": ""}, OK) for k in range(4)]
        self.assertEqual(cyc(build)[0], none)
        # edit -> test cycles: try a fix, the same test fails the same way, revert, try another (a test is other work)
        ed = lambda a, b: ("edit", {"filePath": "/w/app.py", "oldString": f"x = {a}", "newString": f"x = {b}"}, OK)
        test = lambda k: ("bash", {"command": "pytest -q"}, f"{k} failed in 0.{k}3s")
        debug = []
        for k in range(2, 8):
            debug += [ed(1, k), test(1), ed(k, 1), test(1)]
        self.assertEqual(cyc(debug)[0], none)
        e1, e2 = ed(1, 2), ed(2, 1)
        # the very same fix and revert with the very same test results, twice over: a note, but edit is never masked
        st, m = cyc([e1, test(2), e2, test(3), e1, test(2), e2, test(3)])
        self.assertEqual([(r["kind"], r["hold"]) for r in st["reps"]], [("period", False)])
        self.assertIn(g.CYCLE_NOTE_PERIOD, m[-1]["content"])
        st, m = cyc([e1, test(2), e2, test(3)] * 4)
        self.assertEqual(st["mask"], [])
        # one plan update per finished step (+ the todo list), real work between; then the wrap-up
        steps = []
        for k in range(7):
            steps += [mark(" ", "~", line=f"step {k}"), ("todowrite", {"todos": [{"content": f"step {k}", "status": "in_progress"}]}, "ok"),
                      ("bash", {"command": f"make step{k}"}, f"ok {k}"),
                      ("edit", {"filePath": f"/w/s{k}.py", "oldString": "a", "newString": "b"}, OK),
                      mark("~", "x", line=f"step {k}"), ("todowrite", {"todos": [{"content": f"step {k}", "status": "completed"}]}, "ok")]
        steps += [("edit", {"filePath": PLAN, "oldString": "## Next step\nbuild", "newString": "## Next step\nnone"}, OK),
                  ("edit", {"filePath": PLAN, "oldString": "## Decisions\n", "newString": "## Decisions\nkept the old API\n"}, OK)]
        self.assertEqual(cyc(steps)[0], none)
        # all checkboxes ticked at the end, one edit each (7 steps), and a multi-edit call
        self.assertEqual(cyc([mark(" ", "x", line=f"step {k}") for k in range(7)])[0], none)
        multi = [("MultiEdit", {"file_path": "/w/a.py", "edits": [{"old_string": f"a{k}", "new_string": f"b{k}"},
                                                                  {"old_string": f"c{k}", "new_string": f"d{k}"}]}, "ok")
                 for k in range(6)]
        self.assertEqual(cyc(multi)[0], none)
        # an edit that fails two or three times before the file is read again
        miss = lambda k: ("edit", {"filePath": "/w/app.py", "oldString": f"x  = {k}", "newString": "x = 9"},
                          "Could not find oldString in the file.")
        self.assertEqual(cyc([miss(1), miss(2), miss(3), ("read", {"filePath": "/w/app.py"}, "x = 1"), ed(1, 9)])[0], none)
        # one turn: an edit, then other work that starts the file over (parallel calls)
        m = [{"role": "user", "content": "go"},
             {"role": "assistant", "content": "", "tool_calls": [
                 call("Edit", 0, file_path="/w/a.py", old_string="a", new_string="b"),
                 call("Bash", 1, command="make"), call("Edit", 2, file_path="/w/b.py", old_string="a", new_string="b")]}]
        m += [{"role": "tool", "tool_call_id": f"c{k}", "content": "ok"} for k in range(3)]
        self.assertEqual(g.mark_cycles(m), none)
        # try a fix, run the file, revert, try another: the run names the file and is still other work
        run = lambda k: ("bash", {"command": "python /w/app.py"}, f"Traceback {k}")
        self.assertEqual(cyc([ed(1, 2), run(1), ed(2, 1), run(0), ed(1, 3), run(1), ed(3, 1), run(0), ed(1, 4)])[0], none)
        # successful edits whose results go on with diagnostics that read like failures, and test output that does
        lsp = [("edit", {"filePath": "/w/a.ts", "oldString": f"a{k}", "newString": f"b{k}"},
                "Edit applied successfully.\n\nLSP errors detected in this file, please fix:\nERROR [3:1] Cannot find name 'x'")
               for k in range(8)]
        self.assertEqual(cyc(lsp)[0], none)
        self.assertFalse(g.edit_failed(lsp[0][2]))
        self.assertTrue(g.edit_failed("Could not find oldString in the file. It must match exactly."))
        tst = lambda k: ("bash", {"command": "npm test"}, f"Error: module not found ({k})")
        self.assertEqual(cyc([ed(1, 2), tst(1), ed(2, 1), tst(2), ed(1, 3), tst(3), ed(3, 1), tst(4), ed(1, 5)])[0], none)
        # a single undo, then on with other things
        self.assertEqual(cyc([e1, e2, ("edit", {"filePath": "/w/app.py", "oldString": "y = 1", "newString": "y = 3"}, OK)])[0], none)

    def test_other_clients_shapes(self):
        # Claude Code (/v1/messages): calls and results carry _call_id, text may follow the results in a user message
        m = [{"role": "user", "content": "go"}]
        for k in range(4):
            a, b = (" ", "x") if k % 2 == 0 else ("x", " ")
            m.append({"role": "assistant", "content": "", "tool_calls": [{"_call_id": f"t{k}", "function": {
                "name": "Edit", "arguments": json.dumps({"file_path": PLAN, "old_string": f"- [{a}] s", "new_string": f"- [{b}] s"})}}]})
            m.append({"role": "tool", "_call_id": f"t{k}", "content": [{"type": "text", "text": "The file was updated."}]})
            m.append({"role": "user", "content": "<system-reminder>todo list is empty</system-reminder>"})
        st = g.mark_cycles(m)
        self.assertEqual((st["mask"], st["turns"]), (["Edit"], 2))
        self.assertIn("paused", m[-2]["content"][-1]["text"])
        self.assertEqual(g.cycle_mask_names(st, ["Bash", "Edit", "Write", "Read", "TodoWrite", "NotebookEdit", "Task"]),
                         ["Edit", "NotebookEdit", "TodoWrite", "Write"])
        # apply_patch (Codex-style): the path and the lines are in the patch text
        def patch(a, b):
            return ("apply_patch", {"input": f"*** Begin Patch\n*** Update File: PLAN.md\n@@\n-- [{a}] s\n+- [{b}] s\n*** End Patch"}, "Done!")
        st, m = cyc([patch(" ", "x"), patch("x", " "), patch(" ", "x"), patch("x", " ")])
        self.assertEqual(st["mask"], ["apply_patch"])
        self.assertEqual(g.file_op(("apply_patch", json.dumps({"input": "*** Update File: a/b.py\n-x\n+y"})))[:3], ("a/b.py", "x", "y"))
        self.assertIsNone(g.file_op(("bash", json.dumps({"command": "echo edit"}))))
        self.assertEqual([n for n in ("edit", "Write", "MultiEdit", "NotebookEdit", "apply_patch", "str_replace_editor",
                                      "edit_file", "dispatch_agent", "webfetch", "read", "Bash", "credit_check")
                          if g.is_edit_tool(n)],
                         ["edit", "Write", "MultiEdit", "NotebookEdit", "apply_patch", "str_replace_editor", "edit_file"])
        self.assertIsNone(g.file_op(("todoread", "{}")))
        self.assertIsNone(g.file_op(("edit", "not json")))

    def test_history_only_and_one_note(self):
        a, b = mark(" ", "x"), mark("x", " ")
        steps = [a, b, a, b, a, ("bash", {"command": "make"}, "ok"), a]
        full = chain_ids(steps)
        g.mark_loops(full)
        g.mark_cycles(full)
        g.mark_repeats(full, min_times=g.REPEAT_BACKSTOP, skip_noted=True)
        part = chain_ids(steps[:4])
        g.mark_loops(part)
        g.mark_cycles(part)
        g.mark_repeats(part, min_times=g.REPEAT_BACKSTOP, skip_noted=True)
        self.assertEqual(full[:len(part)], part)                     # earlier turns keep their text (cache-stable)
        self.assertTrue(all(x["content"].count("[Note:") <= 1 for x in full if x["role"] == "tool"))
        again = copy.deepcopy(full)                                  # a note already there: same result, no second note
        self.assertEqual(g.mark_cycles(again)["turns"], g.mark_cycles(chain_ids(steps))["turns"])
        self.assertEqual(again, full)

    def test_replay_fixture_plan_cycle(self):
        """Synthetic replay shaped like the TB2 PLAN.md rows: successful checkbox edits going round."""
        path = os.path.join(os.path.dirname(__file__), "fixtures", "loop_replay_plan_cycle.jsonl")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        hist, first_note, first_mask, old_notes, masked = [], None, None, 0, 0
        tools = ["bash", "edit", "glob", "grep", "read", "skill", "task", "todowrite", "webfetch", "write"]
        for t, row in enumerate(rows, 1):
            hist = hist[:row["msg_offset"]] + row["messages"]
            old_notes += bool(g.mark_loops(copy.deepcopy(hist)))
            st = g.mark_cycles(copy.deepcopy(hist))
            if st["reps"] and first_note is None:
                first_note = t
            names = g.cycle_mask_names(st, tools)
            if names:
                masked += 1
                first_mask = first_mask or t
                self.assertEqual(names, ["edit", "todowrite", "write"])
        self.assertEqual(old_notes, 0)                               # 0.5.1 rules: every edit succeeded, nothing seen
        self.assertEqual((first_note, first_mask), (7, 8))           # the mark's 3rd change -> note, the 4th -> mask
        self.assertEqual(masked, len(rows) - 7)                      # held for as long as the cycle goes on

    def test_server_note_mask_and_switch(self):
        import serve.server as srv
        ob, os_ = srv.tnc_api.build, srv.tnc_api.engine_supports
        seen = []
        srv.tnc_api.build = lambda tok, names, thinking: (seen.append(sorted(names)), "REC")[1]
        srv.tnc_api.engine_supports = lambda e: True
        try:
            tok = Tokenizer()
            a, b = mark(" ", "x"), mark("x", " ")

            def run(steps, env=None, force=None):
                engine = RecordingEngine(tok, ['thinking.\n</think>\nanswer'])
                svc = Service(engine, tok, None)
                svc.answer_room = False
                svc.configure_toolguard(env or {})
                svc.tool_step_think_cap = 0
                cancel = threading.Event()
                m = chain_ids(steps)
                svc._toolguard_prompt(m, cancel)
                ids = tok.encode('<|im_start|>user\nfix it<|im_end|>\n<|im_start|>assistant\n<think>\n', parse_special=True)
                tools = [{"name": n, "parameters": {}} for n in ("bash", "edit", "write", "read", "todowrite")]
                del seen[:]
                list(svc.run(ids, True, tools, 4000, {}, cancel, force=force))
                return m, engine, list(seen)

            m, e, names = run([a, b, a])
            self.assertIn("back and forth", m[-1]["content"])
            self.assertNotIn("paused", m[-1]["content"])
            self.assertNotIn("_tool_name_constraint", e.sampling_calls[0])        # first cycle turn: the note only
            m, e, names = run([a, b, a, b])
            self.assertIn("paused", m[-1]["content"])
            self.assertEqual(names, [["bash", "read"]])                           # edit, write, todowrite held back
            self.assertEqual(e.sampling_calls[0].get("_tool_name_constraint"), "REC")
            m, e, names = run([a, b, a, b, ("bash", {"command": "make"}, "ok")])
            self.assertEqual(names, [["bash", "read"]])                           # still held after one other step
            m, e, names = run([a, b, a, b, ("bash", {"command": "make"}, "ok"), ("bash", {"command": "make test"}, "ok")])
            self.assertEqual(names, [])
            self.assertTrue(all(x["content"].count("[Note:") <= 1 for x in m if x["role"] == "tool"))
            m, e, names = run([a, b, a, b], env={"STRATA_LOOP_BREAKER_CYCLE": "0"})
            self.assertEqual(names, [])
            self.assertNotIn("Note", m[-1]["content"])
            m, e, names = run([a, b, a, b], env={"STRATA_LOOP_BREAKER_MASK": "0"})
            self.assertEqual(names, [])
            self.assertIn("back and forth", m[-1]["content"])
            self.assertNotIn("paused", m[-1]["content"])                          # no mask: the note does not claim one
            m, e, names = run([a, b, a, b], env={"STRATA_LOOP_BREAKER": "0"})
            self.assertEqual((names, "Note" in m[-1]["content"]), ([], False))
        finally:
            srv.tnc_api.build, srv.tnc_api.engine_supports = ob, os_


if __name__ == "__main__":
    unittest.main()
