"""CPU-only guard: a thinking budget limits the thinking, not the whole reply. Budget 800 + max_tokens 3000 must
leave room for an answer longer than 800 tokens, whether the budget closes the thinking (wrap-up, second pass) or
the model closes it early on its own: python -m unittest serve.test_budget_total_cap."""
import unittest

from serve.test_server import ThinkingBudget, ThinkingEngine


class LongEngine(ThinkingEngine):
    THOUGHT = "Thinking about the long answer now. " * 40     # 1440 reasoning tokens (one per byte) > budget 800
    ANSWER = "word " * 300                                    # 1500 answer tokens > budget 800


class ShortThoughtEngine(LongEngine):
    THOUGHT = "Short plan. "                                  # closes well before the budget


class _Base(ThinkingBudget):
    ENGINE = LongEngine

    def setUp(self):
        super().setUp()
        self.engine.__class__ = self.ENGINE

    def check(self, b, wrapped):
        self.assertEqual(b["choices"][0]["finish_reason"], "stop", b["usage"])
        msg = b["choices"][0]["message"]
        self.assertEqual(msg["content"].strip(), self.ENGINE.ANSWER.strip())
        self.assertGreater(b["usage"]["completion_tokens"], len(self.ENGINE.ANSWER))     # > 800: not capped at the budget
        self.assertEqual(len(self.engine.prompts), 2 if wrapped else 1)


class BudgetIsNotTheTotalCap(_Base):
    def test_wrapped_budget_leaves_the_answer_its_room(self):
        code, b = self.openai(reasoning_budget_tokens=800, max_tokens=3000)
        self.assertEqual(code, 200, b)
        self.check(b, wrapped=True)


class EarlyCloseIsNotCut(_Base):
    ENGINE = ShortThoughtEngine

    def test_early_close_is_not_cut_at_the_budget(self):
        code, b = self.openai(reasoning_budget_tokens=800, max_tokens=3000)
        self.assertEqual(code, 200, b)
        self.check(b, wrapped=False)


for _n in [n for n in dir(ThinkingBudget) if n.startswith("test_")]:   # the base tests run once, in test_server
    setattr(_Base, _n, None)
del ThinkingBudget, _Base


if __name__ == "__main__":
    unittest.main()
