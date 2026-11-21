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


if __name__ == "__main__":
    unittest.main()
