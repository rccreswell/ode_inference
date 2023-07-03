
import unittest
import numpy as np
import scipy.integrate
from models import NumericalForwardModel, DampedOscillator


class TestNumericalForwardModel(unittest.TestCase):
    def test_set_step_size(self):
        y0 = np.array([0, 0])
        solver = "RK45"
        model = NumericalForwardModel(y0, solver)
        step_size = 0.01
        model.set_step_size(step_size)
        self.assertEqual(model.step_size, step_size)

    def test_set_tolerance(self):
        y0 = np.array([0, 0])
        solver = "RK45"
        model = NumericalForwardModel(y0, solver)
        tolerance = 1e-6
        model.set_tolerance(tolerance)
        self.assertEqual(model.tolerance, tolerance)


class TestDampedOscillator(unittest.TestCase):
    def test_n_parameters(self):
        stimulus = lambda t: 0
        y0 = np.array([0, 0])
        solver = "RK45"
        model = DampedOscillator(stimulus, y0, solver)
        self.assertEqual(model.n_parameters(), 3)

    def test_simulate(self):
        stimulus = lambda t: 1
        y0 = np.array([0, 0])
        solver = "RK45"
        model = DampedOscillator(stimulus, y0, solver)
        model.set_tolerance(1e-3)

        # Set up parameters and times
        parameters = np.array([1, 0.1, 0.5])
        times = np.linspace(0, 10, 100)

        # Simulate and check the output
        output = model.simulate(parameters, times)
        self.assertEqual(output.shape, times.shape)


if __name__ == '__main__':
    unittest.main()

