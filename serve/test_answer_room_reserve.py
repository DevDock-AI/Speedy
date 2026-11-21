"""CPU-only tests for the answer-room reserve (llamAmpere v0.7.2 idea, Speedy 0.5.0; default on,
STRATA_ANSWER_ROOM=0 or "answer_room": false turns it off): python -m unittest serve.test_answer_room_reserve."""
import unittest

from serve.server import (ANSWER_RESERVE_MIN, NUDGE_TEXT, REASONING_WRAP_UP, answer_reserve,
                          answer_room_env_off)
from serve.test_server import ThinkingBudget, ThinkingEngine


class ReserveFormula(unittest.TestCase):
    def test_formula(self):
        for limit in (0, 1, 100, 300, 1000, 1024, 2048, 8192, 32768, 1 << 20):
            want = max(limit // 4, min(ANSWER_RESERVE_MIN, limit // 2))
            self.assertEqual(answer_reserve(limit), want)
            # same values as the #984 formula it replaces
            self.assertEqual(answer_reserve(limit), min(max(ANSWER_RESERVE_MIN, limit // 4), limit // 2))
        self.assertEqual(answer_reserve(8192), 2048)
        self.assertEqual(answer_reserve(1500), 512)
        self.assertEqual(answer_reserve(300), 150)

    def test_env_switch(self):
        self.assertFalse(answer_room_env_off({}))
        self.assertFalse(answer_room_env_off({"STRATA_ANSWER_ROOM": "1"}))
        for v in ("0", "false", "OFF", "no"):
            self.assertTrue(answer_room_env_off({"STRATA_ANSWER_ROOM": v}))


class AnswerRoomSmallCap(ThinkingBudget):
    def test_small_max_tokens_keeps_room_for_the_answer(self):
        self.assertTrue(self.svc.answer_room)
        code, b = self.openai(max_tokens=300)                    # thought ~184 tokens > 300 - 150
        self.assertEqual(code, 200, b)
        msg = b["choices"][0]["message"]
        self.assertIn(REASONING_WRAP_UP.split("</think>")[0], msg["reasoning_content"])
        self.assertTrue(msg["content"])
        self.assertEqual(len(self.engine.prompts), 2)

    def test_off_switch_leaves_it_as_before(self):
        self.svc.answer_room = False
        code, b = self.openai(max_tokens=300)
        self.assertEqual(code, 200, b)
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertNotIn(REASONING_WRAP_UP.split("</think>")[0], b["choices"][0]["message"]["reasoning_content"])

    def test_tiny_cap_untouched(self):
        code, b = self.openai(max_tokens=64)
        self.assertEqual(code, 200, b)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_large_cap_untouched_when_thinking_is_short(self):
        code, b = self.openai(max_tokens=4000)
        self.assertEqual(code, 200, b)
        self.assertEqual(b["choices"][0]["message"]["reasoning_content"], ThinkingEngine.THOUGHT)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_coexists_with_nudge(self):
        self.svc.reasoning_nudge = [0.1]
        code, b = self.openai(max_tokens=2000)                   # reserve 512, budget 1488: nudge at 148
        self.assertEqual(code, 200, b)
        r = b["choices"][0]["message"]["reasoning_content"]
        self.assertIn(NUDGE_TEXT[0], r)
        self.assertTrue(b["choices"][0]["message"]["content"])

    def test_no_nudge_on_a_short_cap_reserve(self):
        self.svc.reasoning_nudge = [0.2]
        code, b = self.openai(max_tokens=300)
        self.assertEqual(code, 200, b)
        self.assertNotIn(NUDGE_TEXT[0], b["choices"][0]["message"]["reasoning_content"])
        self.assertTrue(b["choices"][0]["message"]["content"])

    def test_coexists_with_budget_and_close_retry(self):
        self.svc.reasoning_close_retry = True
        code, b = self.openai(max_tokens=300, reasoning_budget_tokens=280)   # #984 cuts 280 to 150
        self.assertEqual(code, 200, b)
        self.assertTrue(b["choices"][0]["message"]["content"])
        self.assertEqual(len(self.engine.prompts), 2)


del ThinkingBudget

if __name__ == "__main__":
    unittest.main()
