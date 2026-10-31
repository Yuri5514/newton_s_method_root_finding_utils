import math
import unittest

from newton_s_method_root_finding_utils import (
    RootFindingError,
    newton,
    newton_derivative,
)


class TestNewtonDerivative(unittest.TestCase):
    def test_linear_root(self):
        root = newton_derivative(lambda x: x - 3.0, lambda x: 1.0, 0.0)
        self.assertAlmostEqual(root, 3.0, places=10)

    def test_quadratic_root(self):
        root = newton_derivative(
            lambda x: x * x - 2.0, lambda x: 2.0 * x, 1.0
        )
        self.assertAlmostEqual(root, math.sqrt(2.0), places=10)

    def test_immediate_root(self):
        root = newton_derivative(lambda x: x, lambda x: 1.0, 0.0)
        self.assertEqual(root, 0.0)

    def test_zero_derivative_raises(self):
        with self.assertRaises(RootFindingError):
            newton_derivative(lambda x: x * x + 1.0, lambda x: 2.0 * x, 0.0)

    def test_non_convergence_raises(self):
        with self.assertRaises(RootFindingError):
            newton_derivative(
                lambda x: math.exp(x) + 1.0,
                lambda x: math.exp(x),
                0.0,
                max_iter=5,
            )

    def test_bad_tolerance(self):
        with self.assertRaises(ValueError):
            newton_derivative(lambda x: x, lambda x: 1.0, 0.0, tol=0.0)

    def test_bad_max_iter(self):
        with self.assertRaises(ValueError):
            newton_derivative(lambda x: x, lambda x: 1.0, 0.0, max_iter=0)

    def test_nan_initial_guess(self):
        with self.assertRaises(ValueError):
            newton_derivative(lambda x: x, lambda x: 1.0, float("nan"))


class TestNewtonNumerical(unittest.TestCase):
    def test_linear_root(self):
        root = newton(lambda x: x - 3.0, 0.0)
        self.assertAlmostEqual(root, 3.0, places=7)

    def test_quadratic_root(self):
        root = newton(lambda x: x * x - 2.0, 1.0)
        self.assertAlmostEqual(root, math.sqrt(2.0), places=7)

    def test_transcendental_root(self):
        root = newton(lambda x: math.cos(x) - x, 1.0)
        self.assertAlmostEqual(root, 0.7390851332151607, places=7)

    def test_explicit_h(self):
        root = newton(lambda x: x * x - 4.0, 1.0, h=1e-6)
        self.assertAlmostEqual(root, 2.0, places=7)

    def test_zero_derivative_raises(self):
        with self.assertRaises(RootFindingError):
            newton(lambda x: x * x + 1.0, 0.0)

    def test_non_convergence_raises(self):
        with self.assertRaises(RootFindingError):
            newton(lambda x: math.exp(x) + 1.0, 0.0, max_iter=5)

    def test_bad_h(self):
        with self.assertRaises(ValueError):
            newton(lambda x: x, 0.0, h=0.0)

    def test_bad_tolerance(self):
        with self.assertRaises(ValueError):
            newton(lambda x: x, 0.0, tol=-1e-12)


if __name__ == "__main__":
    unittest.main()
