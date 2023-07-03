"""Code for running simulations and generating the figures from the
paper.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pints
import pints.plot
import pickle
from scipy import stats
import pandas
import datetime

import models
import solvers

# Set Latex and fonts
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = \
    r'\usepackage{{amsmath}}\renewcommand{\sfdefault}{phv}'


def figure1():
    """Comparison of log-likelihood surfaces calculated using fixed step and
    adaptive step solvers."""
    np.random.seed(123456)

    pulse = 0.9

    def stimulus(t):
        return (1.0 * (t < 25)) + (pulse * (t >= 25))

    # Generate data
    y0 = np.array([0.0, 0.0])
    m = models.DampedOscillator(stimulus, y0, 'RK45')
    m.set_tolerance(1e-8)
    true_params = [1.0, 0.2, 1.0, 0.01]
    times = np.linspace(0, 50, 75)
    y = m.simulate(true_params[:-1], times)
    y += np.random.normal(0, true_params[-1], len(times))

    # Inference problem and true likelihood
    problem = pints.SingleOutputProblem(m, times, y)
    likelihood_true = pints.GaussianLogLikelihood(problem)
    ll_true = likelihood_true(true_params)

    # Find error in likelihood at ture params using an euler solver with step
    # 0.01
    m = models.DampedOscillator(stimulus, y0, solvers.ForwardEuler)
    problem = pints.SingleOutputProblem(m, times, y)
    likelihood_euler = pints.GaussianLogLikelihood(problem)
    m.set_step_size(0.01)
    ll_using_euler = likelihood_euler(true_params)
    error_using_euler = abs(ll_using_euler - ll_true)

    # Tune RK45 tolerance to have the same error in the likelihood
    m = models.DampedOscillator(stimulus, y0, 'RK45')
    problem = pints.SingleOutputProblem(m, times, y)
    likelihood = pints.GaussianLogLikelihood(problem)
    tol = 1e-2
    while True:
        m.set_tolerance(tol)
        error = abs(likelihood(true_params) - ll_true)
        if error < error_using_euler:
            break
        else:
            tol *= 0.999

    # Calculate likelihood surfaces for euler, RK45 (coarse), and accurate
    m_range = np.linspace(.9875, 1.0125, 100)
    lls = []
    lls_euler = []
    lls_true = []
    for mp in m_range:
        true_params[0] = mp
        lls.append(likelihood(true_params))
        lls_euler.append(likelihood_euler(true_params))
        lls_true.append(likelihood_true(true_params))

    fig = plt.figure(figsize=(4.25, 2.75))
    plt.plot(m_range, lls_euler, c='k', ls='--', label='Fixed grid')
    plt.plot(m_range, lls, c='k', ls=':', label='Adaptive grid')
    plt.plot(m_range, lls_true, c='k', ls='-', label='True')
    plt.legend(loc='lower left')
    ax = plt.gca()
    ax.axvline(1.0, color='k')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel('Log-likelihood')
    fig.set_tight_layout(True)
    plt.savefig('Figure1.pdf')
    plt.show()


def figure23(meth='Euler'):
    """Damped oscillator inference using numerical solvers.

    The likelihood surface is computed for a range of tolerances or step sizes
    and the results are plotted.

    Parameters
    ----------
    meth : {'Euler', 'rk'}
        Forward Euler or RK45 method
    """
    np.random.seed(123456)

    if meth == 'rk':
        def stimulus(t):
            return (1.0 * (t < 2.5)) \
                    + (-1.0 * ((t >= 2.5) & (t < 75))) \
                    + (1.0 * (t >= 75))
    else:
        def stimulus(t):
            return (1.0 * (t < 50)) \
                    + (-1.0 * ((t >= 50) & (t < 75))) \
                    + (1.0 * (t >= 75))

    # Generate data
    y0 = np.array([0.0, 0.0])
    m = models.DampedOscillator(stimulus, y0, solvers.ForwardEuler)
    m = models.DampedOscillator(stimulus, y0, "RK45")
    true_params = [1.0, 0.2, 1, 0.1]
    m.set_tolerance(1e-8)
    m.set_step_size(1e-4)
    sigma = 0.1
    times = np.linspace(0, 5, 25)
    y = m.simulate(true_params, times)
    y += np.random.normal(0, sigma, len(times))

    # Compute solution on dense grid for plotting
    times_t = np.linspace(0, 5, 1000)
    y_t = m.simulate(true_params, times_t)

    # Inference likelihood
    problem = pints.SingleOutputProblem(m, times, y)
    likelihood = pints.GaussianLogLikelihood(problem)
    m_range = np.linspace(0.95, 1.1, 100 if meth == 'Euler' else 1000)

    # calculate true likelihood surface
    true_likelihood = []
    for mv in m_range:
        true_likelihood.append(likelihood([mv, 0.2, 1.0, sigma]))

    solns = {}

    if meth == 'Euler':
        m = models.DampedOscillator(stimulus, y0, solvers.ForwardEuler)
        problem = pints.SingleOutputProblem(m, times, y)
        likelihood = pints.GaussianLogLikelihood(problem)
        steps = [0.1, 5e-2, 1e-2, 1e-3]
        lls = {}
        for step in steps:
            m.set_step_size(step)
            lls[step] = []
            for mv in m_range:
                lls[step].append(
                    likelihood([mv, true_params[1], true_params[2], sigma]))
            solns[step] = m.simulate(true_params, times_t)

    else:
        m = models.DampedOscillator(stimulus, y0, "RK45")
        problem = pints.SingleOutputProblem(m, times, y)
        likelihood = pints.GaussianLogLikelihood(problem)
        tols = [1e-2, 1e-3, 1e-4]
        lls = {}
        for tol in tols:
            m.set_tolerance(tol)
            lls[tol] = []
            for mv in m_range:
                lls[tol].append(
                    likelihood([mv, true_params[1], true_params[2], sigma]))
            solns[tol] = m.simulate(true_params, times_t)

    fig = plt.figure(figsize=(8, 2.5))

    ax = fig.add_subplot(1, 3, 1)
    ax.plot(times_t, y_t, color='k', zorder=-100, lw=1.25, label='Solution')
    ax.scatter(times, y, color='k', s=5.0, label='Data')
    ax.set_xlabel('Time')
    ax.legend()
    ax.set_ylabel(r'$x(t)$')

    ax = fig.add_subplot(1, 3, 2)
    if meth == 'Euler':
        styles = ['-', '-.', '--', ':'][::-1]
        for i, step in enumerate(steps):
            ax.plot(times_t,
                    solns[step],
                    label=r'$\Delta t={}$'.format(step),
                    linestyle=styles[i],
                    color='k')
    else:
        for i, tol in enumerate(tols):
            styles = ['-', '-.', '--'][::-1]
            ax.plot(times_t,
                    solns[tol],
                    label='tol={}'.format(tol),
                    linestyle=styles[i],
                    color='k')

    ax.set_xlabel('Time')
    ax.legend()
    ax.set_ylabel(r'$x(t)$')

    ax = fig.add_subplot(1, 3, 3)
    if meth == 'Euler':
        styles = ['-', '-.', '--', ':'][::-1]
        for i, step in enumerate(steps):
            ax.plot(m_range,
                    lls[step],
                    label=r'$\Delta t={}$'.format(step),
                    linestyle=styles[i],
                    color='k')
    else:
        styles = ['-', '-.', '--'][::-1]
        for i, tol in enumerate(tols):
            ax.plot(m_range,
                    lls[tol],
                    label='tol={}'.format(tol),
                    linestyle=styles[i],
                    color='k')

    ax.set_xlabel(r'$k$')
    ax.set_ylabel('Log-likelihood')
    ax.legend()

    fig.set_tight_layout(True)
    plt.savefig('Figure2.pdf' if meth == 'Euler' else 'Figure3.pdf')
    plt.show()


def figure4():
    """Likelihood surface as RHS is adjusted, using an adaptive solver.
    """
    np.random.seed(123456)

    fig = plt.figure(figsize=(8, 4))

    for i, pulse in enumerate([1, 0.5, 0, -1]):
        def stimulus(t):
            return (1.0 * (t < 25)) + (pulse * (t >= 25))

        # Generate data
        y0 = np.array([0.0, 0.0])
        m = models.DampedOscillator(stimulus, y0, 'RK45')
        true_params = [1.0, 0.2, 1.0, 0.01]
        m.set_tolerance(1e-8)
        times = np.linspace(0, 50, 75)
        y = m.simulate(true_params[:-1], times)
        y += np.random.normal(0, true_params[-1], len(times))

        problem = pints.SingleOutputProblem(m, times, y)
        likelihood = pints.GaussianLogLikelihood(problem)
        m.set_tolerance(1e-3)

        m_range = np.linspace(.975, 1.025, 100)
        lls = []
        for mp in m_range:
            true_params[0] = mp
            lls.append(likelihood(true_params))

        ax = fig.add_subplot(2, 4, i+1)
        ax.scatter(times, y, color='k', s=7.0, label='Data')
        ax.set_xlabel('Time')
        if i == 0:
            ax.set_ylabel(r'$x(t)$')
        ax.set_title(r'$f_1={}$'.format(pulse))
        dense_times = np.linspace(0, 50, 1000)
        m.set_tolerance(1e-8)
        true_params[0] = 1.0
        ax.plot(dense_times,
                m.simulate(true_params[:-1], dense_times),
                lw=1.25,
                color='k',
                label='Solution')
        ax = fig.add_subplot(2, 4, i+5)
        ax.plot(m_range, lls, color='k', lw=1.25)
        ax.set_xlabel(r'$k$')
        if i == 0:
            ax.set_ylabel('Log-likelihood')

    fig.set_tight_layout(True)
    plt.savefig('Figure4.pdf')
    plt.show()


def figure5():
    """Comparison of likelihood surface and number of solver grid points used
    by adaptive solver.
    """
    np.random.seed(123456)

    def stimulus(t):
        return (1.0 * (t < 5)) + (-5 * (t >= 5))

    # Generate data
    y0 = np.array([0.0, 0.0])
    m = models.DampedOscillator(stimulus, y0, 'RK45')
    true_params = [1.0, 0.2, 1.0, 0.01]
    m.set_tolerance(1e-8)
    times = np.linspace(0, 10, 50)
    y = m.simulate(true_params[:-1], times)
    y += np.random.normal(0, true_params[-1], len(times))

    problem = pints.SingleOutputProblem(m, times, y)
    likelihood = pints.GaussianLogLikelihood(problem)

    m.set_tolerance(1e-3)
    m.rerun = True

    m_range = np.linspace(.99, 1.01, 1000)
    lls = []
    num_ts = []
    for mp in m_range:
        true_params[0] = mp
        lls.append(likelihood(true_params))
        num_ts.append(m.num_ts)

    fig = plt.figure(figsize=(6, 2.75))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(m_range, lls, color='k')
    ax.set_ylabel('Log-likelihood')
    ax.set_xlabel(r'$k$')

    ax = ax.twinx()
    ax.plot(m_range, num_ts, color='royalblue', ls='--')
    ax.set_ylabel('Solver time points', color='royalblue')
    ax.set_yticks([19, 20, 21, 22])
    ax.set_xlabel(r'$m$')

    fig.set_tight_layout(True)
    plt.savefig('Figure5.pdf')
    plt.show()


def figure6():
    """Plotting MCMC chains with different tolerances.
    """
    np.random.seed(123456)

    fig2 = plt.figure(figsize=(9, 4.5))

    data_history = {}

    for ki, tol in enumerate([1e-3, 1e-8]):
        for j, noise in enumerate([0.01, 0.1]):
            for i, pulse in enumerate([1, 0.5, 0, -1]):

                def stimulus(t):
                    return (1.0 * (t < 25)) + (pulse * (t >= 25))

                if ki == 0:
                    # Generate data
                    y0 = np.array([0.0, 0.0])
                    m = models.DampedOscillator(stimulus, y0, 'RK45')
                    true_params = [1.0, 0.2, 1.0, noise]

                    m.set_tolerance(1e-8)
                    times = np.linspace(0, 50, 75)
                    y = m.simulate(true_params[:-1], times)
                    y += np.random.normal(0, true_params[-1], len(times))

                    if pulse == -1:
                        data_history[j] = y
                else:
                    try:
                        y = data_history[j]
                    except KeyError:
                        y = [0.1] * len(times)

                if pulse == -1:

                    class inf_model(models.DampedOscillator):
                        def simulate(self, parameters, times):
                            params = parameters
                            return super().simulate(params, times)

                        def n_parameters(self):
                            return 3

                    m = inf_model(stimulus, y0, 'RK45')
                    problem = pints.SingleOutputProblem(m, times, y)
                    likelihood = pints.GaussianLogLikelihood(problem)
                    m.set_tolerance(tol)

                    likelihood = pints.GaussianLogLikelihood(problem)

                    prior = pints.UniformLogPrior(
                        [0.1, 0.1, 0.1, 0],
                        [1.5, 1.5, 1.5, 1.0]
                    )
                    post = pints.LogPosterior(likelihood, prior)

                    if j == 0:
                        x0 = prior.sample(3)
                    mcmc = pints.MCMCController(post, 3, x0)
                    num_iter = 1500
                    mcmc.set_max_iterations(num_iter)
                    chains = mcmc.run()

                    ax = fig2.add_subplot(2, 2, 1+2*j+ki)
                    colors = [(0, 0, 0, 0.5), (0, 0, 0, 0.75), (0, 0, 0, 1)]
                    colors = ['darkgray', 'black', 'royalblue']
                    lss = ['-'] * 3
                    for k in range(3):
                        x = chains[k, :, 2]
                        ax.plot(x,
                                label='Chain {}'.format(k+1),
                                color=colors[k],
                                ls=lss[k])
                    ax.axhline(1.,
                               ls='--',
                               label='True value',
                               color='black',
                               zorder=-10)
                    ax.set_ylabel(r'$m$')
                    if j == 1:
                        ax.set_xlabel('MCMC iteration')
                    ax.set_title(r'$\sigma$={}, rtol={}'.format(noise, tol))
                    ax.set_ylim(.9, 1.1)
                    if j == 0 and ki == 1:
                        ax.legend()
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                    box = ax.get_position()
                    ax.set_position(
                        [box.x0, box.y0, box.width * 0.8, box.height])

    fig2.set_tight_layout(True)
    plt.show()


def figureS1():
    """Effect of smoothing approximation of RHS on likelihood.
    """
    np.random.seed(123456)

    def stimulus(t):
        return (1.0 * (t < 2.5)) \
                + (-1.0 * ((t >= 2.5) & (t < 75))) \
                + (1.0 * (t >= 75))

    def get_smooth_stimulus(smoothing):
        def stimulus_smooth(t):
            return -np.tanh(smoothing * (t-2.5))
        return stimulus_smooth

    # Generate data
    y0 = np.array([0.0, 0.0])
    m = models.DampedOscillator(stimulus, y0, "RK45")
    true_params = [1.0, 0.2, 1, 0.1]
    m.set_tolerance(1e-8)
    m.set_step_size(1e-4)
    sigma = 0.1
    times = np.linspace(0, 5, 25)
    y = m.simulate(true_params, times)
    y += np.random.normal(0, sigma, len(times))

    # Dense times and solution for plotting
    times_t = np.linspace(0, 5, 1000)
    y_t = m.simulate(true_params, times_t)

    problem = pints.SingleOutputProblem(m, times, y)
    likelihood = pints.GaussianLogLikelihood(problem)

    m_range = np.linspace(0.95, 1.1, 1000)

    # Calculate true likelihood curve
    true_likelihood = []
    for mv in m_range:
        true_likelihood.append(likelihood([mv, 0.2, 1.0, sigma]))

    solns = {}

    tols = [(1e-3, 0), (1e-3, 10), (1e-3, 5)]
    lls = {}
    for tol in tols:
        if tol[1] == 0:
            m = models.DampedOscillator(stimulus, y0, "RK45")
        else:
            m = models.DampedOscillator(
                lambda t: -np.tanh(tol[1] * (t-2.5)), y0, "RK45")

        problem = pints.SingleOutputProblem(m, times, y)
        likelihood = pints.GaussianLogLikelihood(problem)

        m.set_tolerance(tol[0])

        lls[tol] = []
        for mv in m_range:
            lls[tol].append(
                likelihood([mv, true_params[1], true_params[2], sigma]))
        solns[tol] = m.simulate(true_params, times_t)

    fig = plt.figure(figsize=(6, 5))

    ax = fig.add_subplot(2, 2, 1)
    ax.plot(times_t, y_t, color='k', zorder=-100, lw=1.25, label='Solution')
    ax.scatter(times, y, color='k', s=5.0, label='Data')
    ax.set_xlabel('Time')
    ax.legend()
    ax.set_ylabel(r'$x(t)$')

    styles = [':', '--', '-']
    ax = fig.add_subplot(2, 2, 3)
    for i, tol in enumerate(tols):
        ax.plot(times_t,
                solns[tol],
                label='a={}'.format(1 / tol[1] if tol[1] > 0 else 0),
                linestyle=styles[i],
                color='k')
    ax.set_xlabel('Time')
    ax.legend()
    ax.set_ylabel(r'$x(t)$')

    ax = fig.add_subplot(2, 2, 2)
    for i, tol in enumerate(tols):
        t_range = np.linspace(0, 5, 1000)
        if tol[1] == 0:
            ax.plot(t_range,
                    stimulus(t_range),
                    label='a={}'.format(1 / tol[1] if tol[1] > 0 else 0),
                    linestyle=styles[i],
                    color='k')
        else:
            ax.plot(t_range,
                    -np.tanh(tol[1] * (t_range-2.5)),
                    label='a={}'.format(1 / tol[1] if tol[1] > 0 else 0),
                    linestyle=styles[i],
                    color='k')
    ax.set_xlabel('Time')
    ax.legend()
    ax.set_ylabel(r'$F(t)$')

    ax = fig.add_subplot(2, 2, 4)
    for i, tol in enumerate(tols):
        ax.plot(m_range,
                lls[tol], label='a={}'.format(1 / tol[1] if tol[1] > 0 else 0),
                linestyle=styles[i],
                color='k')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel('Log-likelihood')
    ax.legend()

    fig.set_tight_layout(True)

    fig.text(.03, .95, 'a.', fontsize=15)
    fig.text(.5, .95, 'b.', fontsize=15)
    fig.text(.03, .48, 'c.', fontsize=15)
    fig.text(.5, .48, 'd.', fontsize=15)
    plt.show()


def figure9():
    """Streamflow model likelihood surfaces.
    """
    import pystreamflow
    data = pystreamflow.load_data('03451500')
    precip = data['precipitation'].to_numpy()[365:566]
    evap = data['evaporation'].to_numpy()[365:566]
    flow = data['streamflow'].to_numpy()[365:566]
    all_times = np.arange(len(precip))

    data_times = all_times[100:200]
    data_flow = flow[100:200]

    m = pystreamflow.RiverModel(
        all_times, precip, evap, solver='scipy', rtol=1e-3, atol=1e-3)

    m_accurate = pystreamflow.RiverModel(
        all_times, precip, evap, solver='scikit')

    problem = pints.SingleOutputProblem(m, data_times, data_flow)
    likelihood = pints.GaussianLogLikelihood(problem)

    problem_accurate = pints.SingleOutputProblem(
        m_accurate, data_times, data_flow)
    likelihood_accurate = pints.GaussianLogLikelihood(problem_accurate)

    params = [9.0, 900.0, 20.0, 8.0, 4.0, 20.0, 2.0, 0.2]
    names = [r'$I_\text{max}$',
             r'$S_\text{u,max}$',
             r'$Q_\text{s,max}$',
             r'$\alpha_e$',
             r'$\alpha_f$',
             r'$K_s$',
             r'$K_f$']

    fig = plt.figure()
    for i in range(7):
        fp = [x for x in params]
        p_range = np.linspace(fp[i]*.7, fp[i]*1.3, 100)
        lls = []
        lls_accurate = []
        for p in p_range:
            fp[i] = p
            lls.append(likelihood(fp))
            lls_accurate.append(likelihood_accurate(fp))

        ax = fig.add_subplot(2, 4, i+1)
        ax.plot(p_range, lls, color='k', ls='-', lw=1.1, label='tol=1e-3')
        ax.plot(p_range, lls_accurate, color='k', ls='-.', lw=1.1,
                label='tol=1e-7')
        if i == 0:
            ax.legend()
        ax.set_xlabel(names[i])
        if i == 0 or i == 4:
            ax.set_ylabel('Log-likelihood')

    fig.set_tight_layout(True)
    plt.show()


if __name__ == '__main__':
    # figure1()
    # figure23(meth='Euler')
    # figure23(meth='rk')
    # figure4()
    # figure5()
    figure6()
    # figure9()
    # figureS1()

