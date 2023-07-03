
import unittest
import numpy as np
import scipy.integrate
from solvers import FDDenseOutput, ForwardEuler


class TestFDDenseOutput(unittest.TestCase):
    def test_output(self):
        x = FDDenseOutput(1.0, 1.1, 1.0)


class TestForwardEuler(unittest.TestCase):
    def test_step_impl(self):
        fun = lambda t, y: y
        t0 = 0.0
        y0 = np.array([1.0])
        t_bound = 1.0
        step_size = 0.1
        solver = ForwardEuler(fun, t0, y0, t_bound, step_size)

        # Perform a step
        success, message = solver._step_impl()

        # Check the state
        self.assertTrue(success)
        self.assertIsNone(message)
        np.testing.assert_allclose(solver.y, np.array([1.1]))
        np.testing.assert_allclose(solver.t, 0.1)

    def test_dense_output_impl(self):
        fun = lambda t, y: y
        t0 = 0.0
        y0 = np.array([1.0])
        t_bound = 1.0
        step_size = 0.1
        solver = ForwardEuler(fun, t0, y0, t_bound, step_size)

        # Perform 2 steps
        solver.step()
        solver.step()

        # Get the dense output
        dense_output = solver._dense_output_impl()

        # Check the dense output
        self.assertIsInstance(dense_output, FDDenseOutput)
        np.testing.assert_allclose(dense_output.t_old, 0.1)
        np.testing.assert_allclose(dense_output.t, 0.2)
        np.testing.assert_allclose(dense_output.value, np.array([1.1]))
        np.testing.assert_allclose(dense_output(0.15), np.array([1.1]))


if __name__ == '__main__':
    unittest.main()
