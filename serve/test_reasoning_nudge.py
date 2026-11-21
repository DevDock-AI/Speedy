"""CPU-only tests for the opt-in staged reasoning nudge (rb-1118, config "reasoning_nudge" / STRATA_REASONING_NUDGE):
python -m unittest serve.test_reasoning_nudge."""
import unittest

from serve.server import NUDGE_TEXT, REASONING_WRAP_UP
from serve.test_server import ThinkingBudget, ThinkingEngine


class ReasoningNudge(ThinkingBudget):
    def test_off_by_default_is_the_plain_wrap_up(self):
        self.assertIsNone(getattr(self.svc, "reasoning_nudge", None))
        code, b = self.openai(reasoning_budget_tokens=20)
        self.assertEqual(code, 200, b)
        self.assertNotIn(NUDGE_TEXT[0].strip(), b["choices"][0]["message"]["reasoning_content"])
        self.assertEqual(len(self.engine.prompts), 2)

    def test_two_nudges_then_wrap_up(self):
        self.svc.reasoning_nudge = [0.2, 0.5]
        code, b = self.openai(reasoning_budget_tokens=100, max_tokens=4000)
        self.assertEqual(code, 200, b)
        msg = b["choices"][0]["message"]
        r = msg["reasoning_content"]
        i0, i1 = r.find(NUDGE_TEXT[0]), r.find(NUDGE_TEXT[1])
        self.assertTrue(0 <= i0 < i1, r)
        self.assertEqual(r[:i0], ThinkingEngine.THOUGHT[:20])                 # first nudge at 20% of 100
        self.assertIn(REASONING_WRAP_UP.split("</think>")[0], r[i1:])         # the budget still closes it
        self.assertEqual(msg["content"], ThinkingEngine.ANSWER)
        p = self.engine.prompts
        self.assertEqual(len(p), 4)                                           # nudge, nudge, wrap-up, answer
        self.assertEqual(p[1], p[0] + self.tok.encode(ThinkingEngine.THOUGHT[:20]) +
                         self.tok.encode(NUDGE_TEXT[0], parse_special=True))  # a prefix + more (cache kept)

    def test_no_budget_no_nudge(self):
        self.svc.reasoning_nudge = [0.6, 0.8]
        code, b = self.openai()
        self.assertEqual(code, 200, b)
        self.assertEqual(b["choices"][0]["message"]["reasoning_content"], ThinkingEngine.THOUGHT)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_short_thinking_is_untouched(self):
        self.svc.reasoning_nudge = [0.6, 0.8]
        code, b = self.openai(reasoning_budget_tokens=1000, max_tokens=4000)                # 184 tokens < 60% of 1000
        self.assertEqual(code, 200, b)
        self.assertEqual(b["choices"][0]["message"]["reasoning_content"], ThinkingEngine.THOUGHT)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_short_cap_skips_nudges_and_still_answers(self):
        # 0.5.0 needle bug: a config budget cut by #984 to fit a short max_tokens (60 -> 30 on the GPU) fired both
        # nudges, their text used up the cap and the reply ended "length" with content ''.
        self.svc.reasoning_nudge = [0.6, 0.8]
        code, b = self.openai(reasoning_budget_tokens=32768, max_tokens=300)     # cut to 150 of 300
        self.assertEqual(code, 200, b)
        msg = b["choices"][0]["message"]
        self.assertNotIn(NUDGE_TEXT[0].strip(), msg["reasoning_content"])
        self.assertIn(REASONING_WRAP_UP.split("</think>")[0], msg["reasoning_content"])
        self.assertEqual(msg["content"], ThinkingEngine.ANSWER)
        self.assertEqual(b["choices"][0]["finish_reason"], "stop")


del ThinkingBudget   # the base tests run once, in test_server

if __name__ == "__main__":
    unittest.main()
