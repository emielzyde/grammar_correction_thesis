from unittest import TestCase

from msq_grammar_correction import start


class StartTest(TestCase):
    def test_start(self):
        self.assertIsInstance(start.config, dict)
