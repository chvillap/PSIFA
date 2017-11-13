from __future__ import print_function
from psifa.api import psifa
import numpy as np
import pprint


# Objective function.
def f(x):
    y = (1. / (27. * np.sqrt(3.))) * ((x[0] - 3.)**2 - 9.) * x[1]**3
    eval_success = True
    return y, eval_success


# Problem constraints.
A = np.array([
    [1. / np.sqrt(3.), -1.],
    [1., np.sqrt(3.)],
    [-1., -np.sqrt(3.)]
])
l = np.array([0., 0., -6.])
u = np.array([np.inf, np.inf, np.inf])
bl = np.array([0., 0.])
bu = np.array([np.inf, np.inf])
n = 2

# Initial guess for the solution.
x0 = np.array([1., 0.5])

# Find a global optimum using PSIFA.
x, y, iterations, alpha, evaluations, history, exit_cause = \
    psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

# Show the results.
print('------------------------------------------------------------')
print('                          Results                           ')
print('------------------------------------------------------------')
print('Solution: x = {}'.format(x))
print('Function value: f(x) = {}'.format(y))
print('Step length: alpha = {}'.format(alpha))
print('Number of iterations: {}'.format(iterations))
print('Function evaluations: {}'.format(evaluations))
print('Exit cause: {}'.format(exit_cause))
print('History:')
pp = pprint.PrettyPrinter(indent=2)
pp.pprint(history)
