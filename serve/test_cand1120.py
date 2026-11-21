"""CPU-only tests for the cand-1120 opt-in knobs: python -m unittest serve.test_cand1120.
STRATA_TOOL_STEP_THINK_CAP / STRATA_THINK_FLOOR_CAP lower one reply's thinking budget; the loop note text and the
streak at which the mask starts can be set."""
import io
import contextlib
import threading
import unittest
from serve.test_reasoning_loop_recovery import Tokenizer, RecordingEngine
from serve.server import Service
from serve import toolguard


class ThinkCap(unittest.TestCase):
    def run_case(self, after_tool=True, outputs=None, sampling=None, **attrs):
        tok = Tokenizer()
        tail = ('<|im_start|>user\n<tool_response>\nFAIL 0 passed\n</tool_response><|im_end|>\n' if after_tool
                else '<|im_start|>user\nfix it<|im_end|>\n')
        ids = tok.encode('<|im_start|>user\nfix it<|im_end|>\n' + tail + '<|im_start|>assistant\n<think>\n', parse_special=True)
        engine = RecordingEngine(tok, outputs or ['EOF\n' * 200 + '</think>\nok', '\n</think>\nanswer', 'answer'])
        svc = Service(engine, tok, None)
        svc.answer_room = False
        svc._think_close_id = 7
        for k, v in attrs.items():
            setattr(svc, k, v)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            list(svc.run(ids, True, [], 4000, sampling or {}, threading.Event()))
        return engine, out.getvalue()

    def test_cap_on_tool_step(self):
        engine, log = self.run_case(tool_step_think_cap=100)
        self.assertIn("tool step think cap: 100", log)
        self.assertIn("thinking budget reached", log)
        self.assertGreaterEqual(len(engine.calls), 2)       # the wrap-up is another pass

    def test_cap_off_and_not_after_tool(self):
        for after, attrs in ((True, {"tool_step_think_cap": 0}), (False, {"tool_step_think_cap": 100}), (True, {"think_floor_cap": 100, "tool_step_think_cap": 0})):
            engine, log = self.run_case(after, **attrs)
            self.assertNotIn("tool step think cap", log)
            self.assertNotIn("thinking budget reached", log)
            self.assertEqual(len(engine.calls), 1)

    def test_floor_cap_only_on_floor_turn(self):
        engine, log = self.run_case(tool_think_floor=50, think_floor_cap=150, outputs=[
            'EOF\n' * 200, 'EOF\n' * 200 + '</think>\nok', '\n</think>\nanswer', 'answer'])
        self.assertIn("tool step think cap: 150", log)
        self.assertIn("floor turn", log)
        self.assertIn("thinking budget reached", log)

    def test_request_budget_is_honoured(self):
        # 0.5.0 (owner): a budget the request named itself is not capped after a tool result
        for key in ("reasoning_budget_tokens", "thinking_budget_tokens"):
            engine, log = self.run_case(tool_step_think_cap=100, sampling={key: 3000})
            self.assertNotIn("tool step think cap", log, key)

    def test_cap_never_raises_a_budget(self):
        engine, log = self.run_case(tool_step_think_cap=100000)
        self.assertIn("tool step think cap: 100000", log)   # no budget at all: the cap becomes one
        self.assertNotIn("thinking budget reached", log)


class LoopKnobs(unittest.TestCase):
    def history(self, n):
        msgs = [{"role": "user", "content": "go"}]
        for i in range(n):
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "run_ui_checks", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "5/5 UI checks pass"})
        return msgs

    def test_note_text(self):
        msgs = self.history(2)
        reps = toolguard.mark_loops(msgs, 3, "NEUTRAL NOTE")
        self.assertEqual(reps[0]["streak"], 1)
        self.assertIn("NEUTRAL NOTE", msgs[-1]["content"])
        self.assertNotIn("failing test", msgs[-1]["content"])

    def test_mask_streak(self):
        reps = toolguard.mark_loops(self.history(2), 3)
        self.assertEqual(toolguard.masked_names(reps, 1), ["run_ui_checks"])
        self.assertEqual(toolguard.masked_names(reps, 2), [])
        reps = toolguard.mark_loops(self.history(3), 3)
        self.assertEqual(toolguard.masked_names(reps, 2), ["run_ui_checks"])


if __name__ == "__main__":
    unittest.main()
