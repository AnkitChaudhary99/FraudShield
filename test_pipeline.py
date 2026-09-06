import unittest

from fusion import fuse_scores


class TestFusion(unittest.TestCase):
    def test_high_risk(self):
        score, verdict = fuse_scores(0.9, 0.8)
        self.assertAlmostEqual(score, 0.85)
        self.assertEqual(verdict, "Fraudulent Call Detected (High Risk)")

    def test_medium_risk(self):
        score, verdict = fuse_scores(0.5, 0.3)
        self.assertAlmostEqual(score, 0.4)
        self.assertEqual(verdict, "Suspicious Call (Medium Risk)")

    def test_low_risk(self):
        score, verdict = fuse_scores(0.1, 0.2)
        self.assertAlmostEqual(score, 0.15)
        self.assertEqual(verdict, "Genuine Call (Low Risk)")

    def test_invalid_score(self):
        with self.assertRaises(ValueError):
            fuse_scores(1.2, 0.2)


if __name__ == "__main__":
    unittest.main()
