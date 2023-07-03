"""ODE models with unknown parameters for inference.

Extends from pints.
"""

import pints
import math
import scipy.integrate
import copy


class NumericalForwardModel(pints.ForwardModel):
    def __init__(self, y0, solver, step_size=None, tolerance=None):
        """
        Parameters
        ----------
        y0 : np.ndarray
            Initial condition
        solver : str or OdeSolver
            Method for solving ODE. Can be a str from the scipy choices or
            another OdeSolver.
        step_size : float, optional
            Step size used in the solver
        tolerance : float, optional
            Tolerance used in the solver
        """
        self.y_init = y0
        self.solver = solver
        self.step_size = step_size
        self.tolerance = tolerance

    def set_step_size(self, step_size):
        """Update the solver step size.

        Parameters
        ----------
        step_size : float
            New step size
        """
        self.step_size = step_size

    def set_tolerance(self, tolerance):
        """Update the solver tol.

        Parameters
        ----------
        tolerance : float
            New tolerance
        """
        self.tolerance = tolerance


class DampedOscillator(NumericalForwardModel):
    def __init__(self, stimulus, y0, solver, step_size=None, tolerance=None):
        r"""Model of a damped harmonic oscillator with forcing.

        The stimulus function is provided by the user. The model has three
        parameters: spring constant :math:`k`, damping coefficient :math:`c`,
        and mass :math:`m`.

        The differential equation is given by

        .. math::
            m \frac{d^2 x}{dt^2} + c \frac{dx}{dt} + k x = F(t)

        Parameters
        ----------
        stimulus : function
            Input stimulus as a function of time
        y0 : list or np.ndarry
            Initial condition. It should have two entries, the first is
            :math:`x(t=0)` and the second :math:`\dot{x}(t=0)`.
        solver : str or scipy.integrate.OdeSolver
            The ODE solver to use. Can be a string recognized by scipy or any
            other solver.
        step_size : float, optional (None)
            Step size to pass to solver
        tolerance : float, optional (None)
            Rtol to pass to solver
        """
        super(DampedOscillator, self).__init__(
            y0, solver, step_size=step_size, tolerance=tolerance)

        self.stimulus = stimulus
        self.rerun = False

    def n_parameters(self):
        return 3

    def simulate(self, parameters, times):
        k = parameters[0]
        c = parameters[1]
        m = parameters[2]
        w = math.sqrt(k/m)
        g = c / (2 * math.sqrt(m * k))

        def fun(t, y):
            return [y[1], self.stimulus(t) / m - 2*g*w*y[1] - w**2 * y[0]]

        t_range = (0, max(times))

        res = scipy.integrate.solve_ivp(
            fun,
            t_range,
            copy.copy(self.y_init),
            t_eval=times,
            method=self.solver,
            rtol=self.tolerance,
            atol=1e-9,
            step_size=self.step_size
        )
        y = res.y
        if y.ndim >= 2:
            y = res.y[0]

        if self.rerun:
            # rerun to get default solver points
            res = scipy.integrate.solve_ivp(
                fun,
                t_range,
                copy.copy(self.y_init),
                method=self.solver,
                rtol=self.tolerance,
                atol=1e-9,
                step_size=self.step_size
            )
            self.num_ts = len(res.t)

        return y
