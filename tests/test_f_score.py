from unittest import TestCase

import numpy as np

from msq_grammar_correction.f_score import precision, recall, f_score


class FScoreTest(TestCase):
    def test_precision(self):
        a = np.array([0,1,1,0,1,0,0,0])
        b = np.array([0,1,0,1,0,0,0,0])

        prec_ab = precision(a, b)

        self.assertAlmostEqual(0.333, prec_ab, 3)
