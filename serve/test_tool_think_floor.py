"""CPU-only test for the opt-in tool think floor (quality-max-1119): python -m unittest serve.test_tool_think_floor."""
import threading
import unittest
from serve.test_reasoning_loop_recovery import Tokenizer, RecordingEngine
from serve.server import Service


class ToolThinkFloor(unittest.TestCase):
    def run_case(self, floor, after_tool=True):
        tok = Tokenizer()
        tail = ('<|im_start|>user\n<tool_response>\nFAIL 0 passed\n</tool_response><|im_end|>\n' if after_tool
                else '<|im_start|>user\nfix it<|im_end|>\n')
        ids = tok.encode('<|im_start|>user\nfix it<|im_end|>\n' + tail + '<|im_start|>assistant\n<think>\n', parse_special=True)
        engine = RecordingEngine(tok, ['a' * 300 + '\n</think>\nok', 'more thinking.\n</think>\nanswer'])
        svc = Service(engine, tok, None)
        svc.answer_room = False
        svc.tool_think_floor = floor
        svc._think_close_id = 7   # the byte test tokenizer has no single </think> token
        events = list(svc.run(ids, True, [], 4000, {}, threading.Event()))
        close = 7
        return engine, events, close

    def test_floor_biases_then_releases(self):
        engine, _, close = self.run_case(100)
        self.assertEqual(engine.sampling_calls[0].get("logit_bias"), {close: -100.0})
        self.assertGreaterEqual(len(engine.calls), 2)
        self.assertNotIn("logit_bias", engine.sampling_calls[1])

    def test_off_and_not_after_tool(self):
        for floor, after in ((0, True), (100, False)):
            engine, _, _ = self.run_case(floor, after)
            self.assertNotIn("logit_bias", engine.sampling_calls[0])
            self.assertEqual(len(engine.calls), 1)


if __name__ == "__main__":
    unittest.main()
