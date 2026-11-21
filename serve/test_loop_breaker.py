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


if __name__ == "__main__":
    unittest.main()
