import numpy as np
from psifa.utils import search_history


def quasinewton(f, W_stencil, y_stencil, x, y, yps, history, alpha,
                constrained, Tb, max_red, prec5):
    """Performs the quasi-Newton iteration of PSIFA, which obtains a new
    trial point `x`.

    Parameters
    ----------
    f : function
        Objective function.
    W_stencil : ndarray
        2D array of directions found in the pattern search step.
    y_stencil : ndarray
        1D array of function values obtained at the points reached using
        the directions found in the pattern search step.
    x : ndarray
        Current trial point as a 1D array.
    y : float
        Objective function value at `x`.
    yps : float
        Objective function value at `xps` (pattern search solution).
    history : dict
        Historical information of the optimization process.
    alpha : float
        Step length.
    constrained : dict
        Constraints and other problem info.
    Tb : ndarray
        Null space of the epsilon-active inequality constraints temporarily
        taken as equality constraints.
    max_red : float
        Maximum number of reductions in the line search step.
    prec5 : float
        Maximum distance to consider when checking whether a trial point was
        previously evaluated before.

    Return values
    -------------
    x_new : ndarray
        New trial point resulting from the quasi-Newton step.
    y_new : float
        Objective function value at `x_new`.
    history : dict
        Updated history.
    evals : int
        Number of internal evaluations of the objective function.
    x_new_status : int
        Flag that tells how the quasi-Newton solution compares to the current
        trial point and the pattern search solution (`xps`). Values can be:
            `0` : `x_new` is not better than any of them.
            `1` : `x_new` is a better solution than `x`.
            `2` : `x_new` is not better than `x`, but it is better than `xps`.
    """

    x_new_status = 0
    evals = 0
    x_new = x
    x_history = history['x']
    y_history = history['y']

    # List of points the objective function was evaluated at, and the values at
    # these points.
    evaluated_x = []
    evaluated_y = []

    # Compute the stencil gradient at x.
    # Depending on the gradient value, we either may or may not update our
    # definition of the previous value for x.
    Df, grad, zero_grad = stencil_gradient(
        W_stencil, y_stencil, y, alpha, history['grad'])
    x_old = x if zero_grad else history['x_old']

    # Compute the BFGS update to obtain a (possibly) feasible direction
    # from the stencil gradient.
    hessian = history['hessian'][-1]
    dk, hessian = bfgs_direction(hessian, x, x_old, Df, grad)

    # As the BFGS direction still may be unfeasible because of the constraints,
    # we perform the ratio test to obtain both a feasible (projected) direction
    # and the step length we may take along it.
    keep_trying = True
    attempt = 1
    while keep_trying:
        if attempt == 2:
            keep_trying = False

        dk, alpha = ratio_test(constrained, x, dk, Tb)

        # ---------------------------------------------------------------------
        # Line search

        # We have no guarantee that the direction provided is going downwards,
        # so in the line search step we check the opposite direction too.
        # The number of reductions in the step length is limited because we
        # don't want to evaluate the objective function too many times.
        step_reductions = 0  # j
        while alpha > 0 and step_reductions < max_red:
            # Get a new trial point and the corresponding function value at it.
            x_new = x + alpha * dk

            # Look for the new point in the history and evaluate the objective
            # function at such point if needed.
            unused = np.array([])
            found_x, _, position = search_history(
                x_history, unused, x_new, unused, prec5)
            if found_x:
                y_new = y_history[position]
            else:
                y_new, eval_success = f(x_new)
                evaluated_x.append(x_new)
                evaluated_y.append(y_new)
                evals += 1

            # If we get a lower value for the objective function at this new
            # point, mark the search as successful.
            if y_new < y and (found_x or eval_success):
                y = y_new
                x_old = x
                x = x_new
                alpha = 0
                keep_trying = False
                x_new_status = 1
            elif y_new < yps:
                xt = x_new
                yps = y_new
                x_new_status = 2

            # Reduce alpha and try again.
            alpha *= 0.5
            step_reductions += 1

        # Now try the opposite direction, if needed.
        dk = -dk
        attempt += 1

    # Set the outputs.
    y_new = y
    if x_new_status == 2:
        y_new = yps
        x_old = x
        x = xt
        x_new_status = 1
    x_new = x

    #  Update the history.
    history['x'] += evaluated_x
    history['y'] += evaluated_y
    history['x_old'] = x_old

    return x_new, y_new, history, evals, x_new_status

# -----------------------------------------------------------------------------


def stencil_gradient(W_stencil, y_stencil, yps, alpha, grad):
    """Computes the stencil gradient at `x` from a set of feasible directions
    and a particular step length.

    Parameters
    ----------
    W_stencil : ndarray
        2D array of directions found in the pattern search step.
    y_stencil : ndarray
        1D array of function values obtained at the points reached using
        the directions found in the pattern search step.
    yps : float
        Objective function value at `xps` (pattern search solution).
    alpha : float
        Step length.
    grad : float
        Previously known gradient approximation at `x`.

    Return values
    -------------
    Df : ndarray
        Stencil gradient for the given `x`, `alpha` and `W_stencil`.
    grad : ndarray
        Previous stencil gradient or the same as Df if `grad_computed == True`.
    grad_computed : bool
        `True` if `grad` had not been computed before and now it has; `False`
        otherwise.
    """

    grad_computed = False

    # The stencil gradient is computed using the pseudoinverse of W_stencil
    # and differences of previously computed function values.
    W_pinv = np.linalg.pinv(W_stencil)
    delta = y_stencil - yps
    Df = np.dot(delta, W_pinv) / alpha

    # If grad is zero, it means it hasn't been computed before.
    # So grad is actually computed only once (set as Df).
    if grad == 0.:
        grad = Df
        grad_computed = True

    return Df, grad, grad_computed

# -----------------------------------------------------------------------------


def bfgs_direction(hessian, x, x_old, Df, grad):
    """Obtains a (possibly) feasible quasi-Newton direction at `x` using the
    BFGS update.

    Parameters
    ----------
    hessian : ndarray
        Approximation for the Hessian matrix at `x`.
    x : ndarray
        Current trial point.
    x_old : ndarray
        Previous trial point.
    Df : ndarray
        Stencil gradient at `x`.
    grad : ndarray
        Gradient approximation at `x`.

    Return values
    -------------
    dk : ndarray
        Direction obtained by the BFGS algorithm.
    hessian : ndarray
        New approximation for the Hessian matrix.
    """

    # Update the Hessian matrix for the current trial point x.
    # OBS: do not update if any denominator is zero.
    s = np.atleast_2d(x - x_old).T
    gdiff = np.atleast_2d(Df - grad).T
    HS = np.dot(hessian, s)

    if 0 < np.linalg.norm(gdiff) < np.inf and \
       0 < np.linalg.norm(s) < np.inf and \
       0 < np.linalg.norm(HS) < np.inf and \
       np.dot(gdiff.T, s) != 0 and \
       np.dot(s.T, HS) != 0:
        hessian += np.dot(gdiff, gdiff.T) / \
            np.dot(gdiff.T, s) - np.dot(HS, HS.T) / np.dot(s.T, HS)

    # Check the condition number of the Hessian matrix in order to find out
    # whether it is ill-conditioned (the computation of its inverse, or
    # solution of a linear system of equations, is prone to large numerical
    # errors).
    if np.linalg.cond(hessian) > 1e6:
        hessian = np.eye(hessian.shape[0])

    # The BFGS direction is finally obtained from the solution of the
    # following linear system.
    dk = np.linalg.solve(hessian, -Df)

    return dk, hessian

# -----------------------------------------------------------------------------


def ratio_test(constrained, x, dk, Tb):
    """Performs the ratio test and provides both a feasible (projected)
    direction and the step length we may take along it.

    Parameters
    ----------
    constrained : dict
        Constraints and other problem info.
    x : ndarray
        Current trial point as a 1D array.
    dk : ndarray
        BFGS direction.
    Tb : ndarray
        Null space of the epsilon-active inequality constraints temporarily
        taken as equality constraints.

    Return values
    -------------
    dk : ndarray
        Projected direction.
    step : float
        Step length we may take along `dk`.
    """

    A = constrained['A']
    Aeq = constrained['Aeq']
    l = constrained['l']
    u = constrained['u']
    bl = constrained['bl']
    bu = constrained['bu']
    N = constrained['N']

    # Numerical tolerance.
    tolr = 2. * np.finfo(float).eps

    # Ratio test parameters.
    t1 = 1.
    t2 = 1.

    # Project the quasi-Newton direction onto the nullspace of the constraint
    # array, in order to obtain a feasible direction.
    if Aeq.size > 0:
        if Tb.size > 0:
            TbN = np.hstack((Tb, N))
            dk = np.dot(np.dot(TbN, TbN.T), dk)
        else:
            dk = np.dot(np.dot(N, N.T), dk)
    elif Tb.size > 0:
        dk = np.dot(np.dot(Tb, Tb.T), dk)

    # Perform the ratio test for inequality constraints.
    if A.size > 0:
        Ad = np.dot(A, dk)
        Ax = np.dot(A, x)

        t1_ind = Ad > 0
        t2_ind = Ad < 0
        t1 = np.inf
        t2 = np.inf
        if np.any(t1_ind):
            t1 = ((u[t1_ind] - tolr) - Ax[t1_ind]) / Ad[t1_ind]
        if np.any(t2_ind):
            t2 = ((l[t2_ind] + tolr) - Ax[t2_ind]) / Ad[t2_ind]

    # Perform the ratio test for box constraints.
    t3_ind = dk < 0
    t4_ind = dk > 0
    t3 = np.inf
    t4 = np.inf
    if np.any(t3_ind):
        t3 = np.min((x[t3_ind] - (bl[t3_ind] + tolr)) / -dk[t3_ind])
    if np.any(t4_ind):
        t4 = np.min((x[t4_ind] - (bu[t4_ind] - tolr)) / -dk[t4_ind])

    # Get the greatest step size that we can take along the quasi-Newton
    # direction (or its projection).
    step = np.min(np.hstack((t1, t2, t3, t4, 1.)))

    return dk, step
