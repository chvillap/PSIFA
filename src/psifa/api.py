from __future__ import print_function
import numpy as np
from psifa.patternsearch import classify_constraints
from psifa.patternsearch import check_feasibility
from psifa.patternsearch import find_epsilon_active_constraints
from psifa.patternsearch import build_pattern
from psifa.quasinewton import quasinewton
from psifa.utils import preprocess_constraints
from psifa.utils import search_history
from psifa.utils import null


def psifa(f, x0, A, l, u, n, bl, bu, **kwargs):
    """Runs PSIFA to minimize an objective function `f` in a linearly
    constrained problem.

    Parameters
    ----------
    f : function
        Objective function. It receives a `x` point as input and returns the
        function value at `x` (named `y`) and a boolean flag as outputs. The
        flag tells whether the evaluation at `x` was successful or not.
    x0 : ndarray
        Initial guess for the solution.
    A : ndarray
        2D array representing the constraints.
    l : ndarray
        1D array of constraint lower bounds.
    u : ndarray
        1D array of constraint upper bounds.
    n : int
        Number of variables.
    bl : ndarray
        1D array of lower bounds for box constraints.
    bu : ndarray
        1D array of upper bounds for box constraints.

    Optional parameters
    -------------------
    rho : float
        Step length update parameter. It is never lower than 1.
        (default = `2.0`)
    max_evals : int
        Maximum number of function evaluations before the algorithm stops.
        (default = `500`)
    prec1 : float
        Step length tolerance. The algorithm stops if `alpha` < `prec1`.
        (default = `1e-5`)
    prec2 : float
        Minimum variation of the objective function value at some iteration.
        (default = `1e-12`)
    prec3 : float
        Parameter to identify inequality constraints that are epsilon-active at
        both lower and upper bounds.
        (default = `1e-10`)
    prec4 : float
        Tolerance for violating constraints when checking point feasibility.
        (default = `1e-10`)
    prec5 : float
        Maximum distance to consider when checking if a trial point was
        previously evaluated before (default = `1e-12`).
    gamma : float
        Paramenter of the sufficient decrease condition.
        (default = `1e-5`)
    p : float
        Paramenter of the sufficient decrease condition.
        (default = `1.5`)
    alpha0 : float
        Initial step length.
        (default = `1.0`)
    grad : float
        Initial approximation for the gradient.
        (default = `1.0`)
    hessian : ndarray
        Initial approximation for the Hessian matrix at `x`.
        (default = `eye(n)`).
    max_red : int
        Maximum number of reductions in the line search step of the
        quasi-Newton iteration.
        (default = `3`)

    Return values
    -------------
    x : ndarray
        Final solution.
    y : ndarray
        Value of the objective function at `x`.
    iterations : int
        Total number of iterations.
    alpha : float
        Final step length.
    evaluations : int
        Total number of function evaluations.
    history: dict
        Historical information of the optimization process.
    exit_cause : list
        List containing the reasons the algorithm has stopped.
        `'TP'` : Step length lower than tolerance.
        `'AF'` : Maximum nunmber of function evaluations.
        `'VF'` : Function value variation lower than tolerance.
    """

    # Set values for the optional parameters.
    rho = kwargs['rho'] if 'rho' in kwargs else 2.
    max_evals = kwargs['max_evals'] if 'max_evals' in kwargs else 500
    prec1 = kwargs['prec1'] if 'prec1' in kwargs else 1e-5
    prec2 = kwargs['prec2'] if 'prec2' in kwargs else 1e-12
    prec3 = kwargs['prec3'] if 'prec3' in kwargs else 1e-10
    prec4 = kwargs['prec4'] if 'prec4' in kwargs else 1e-10
    prec5 = kwargs['prec5'] if 'prec5' in kwargs else 1e-12
    gamma = kwargs['gamma'] if 'gamma' in kwargs else 1e-5
    p = kwargs['p'] if 'p' in kwargs else 1.5
    alpha0 = kwargs['alpha0'] if 'alpha0' in kwargs else 1.
    grad = kwargs['grad'] if 'grad' in kwargs else 1.
    hessian = kwargs['hessian'] if 'hessian' in kwargs else np.eye(n)
    max_red = kwargs['max_red'] if 'max_red' in kwargs else 3

    # Preprocess the constraints.
    A, l, u = preprocess_constraints(A, l, u)

    # Identify the different kinds of constraints.
    Aeq, A, l, u, eq, most_general_kind = classify_constraints(A, l, u)

    # Test whether the initial point is feasible or not.
    is_feasible = check_feasibility(x0, A, Aeq, eq, l, u, bl, bu, prec4)

    # Terminate with error if the initial point is not feasible.
    if not is_feasible:
        error_msg = 'Initial point is not feasible'
        raise ValueError(error_msg)

    # Evaluate the objective function at the initial point, and also
    # initialize the evaluations counter.
    y0, eval_success = f(x0)
    evaluations = 1

    # Terminate with error if the objective function can't be evaluated at
    # the initial point.
    if not eval_success:
        error_msg = 'Evaluation failure at the initial point'
        raise ValueError(error_msg)

    # -------------------------------------------------------------------------
    # Initialize internal variables.

    # Step length.
    alpha = alpha0
    alpha_old = alpha

    # Minimum value for y0 and its corresponding x0.
    x_min = x0
    y_min = y0

    # Number of consecutive successes along the past iterations.
    # This value is used to increase the step length.
    beta = 1

    # Terms of the sufficient decrease condition.
    eta = gamma
    decrease = -min(alpha**p, 1.) * gamma + eta

    # Variation of the objective function value from the previous iteration
    # to the current one.
    y_var = 1.

    # Iterations counter.
    iterations = 0

    Tb = np.array([])

    # -------------------------------------------------------------------------
    # Build the pattern.

    # If any equality constraints exist, compute the null space of the
    # matrix that represents them.
    if Aeq.size > 0:
        N = null(Aeq, rational=True)
        # N = eqnull(Aeq, n)
    else:
        N = np.array([])

    if most_general_kind == 'equality':
        # In this case the pattern Pk will be the set of columns of N and -N
        # (after some normalization).
        Pk = np.hstack((N, -N))
        Pk = np.apply_along_axis(
            lambda z: z / np.linalg.norm(z, np.inf), 0, Pk)
    elif most_general_kind == 'box':
        # In this case the pattern Pk will be the set of columns of eye(n) and
        # -eye(n). Additional unit vectors are also included for the sake of
        # heuristics.
        V = np.eye(n)
        Pk = np.hstack((V, -V))
        if n > 1:
            ones = np.ones((n, 1))
            ones = np.apply_along_axis(
                lambda z: z / np.linalg.norm(z, np.inf), 0, ones)
            Pk = np.hstack((Pk, ones, -ones))
    elif most_general_kind == 'inequality':
        # In this case we find the epsilon-active inequality constraints and
        # then build the pattern using a special procedure.
        Vin, Veq, Vb, epsilon_active = find_epsilon_active_constraints(
            A, x0, l, u, N, alpha, prec3)
        Pk, Tb = build_pattern(Vin, Veq, Vb, Aeq, N, epsilon_active)

    # The history of the optimization process is represented as a dictionary of
    # lists. Each iteration will add a new element to (some of) these arrays.
    history = {
        'xps': [x0],           # x provided by the patern search iteration.
        'xqn': [np.nan],       # x provided by the quasi-Newton iteration.
        'x': [x0],             # Best x among xps and xqn.
        'yps': [y0],           # Value of the objective function at xps.
        'yqn': [np.nan],       # Value of the objective function at xqn.
        'y': [y0],             # Minimum y among yps and yqn.
        'alpha': [alpha0],     # Step length.
        'hessian': [hessian],  # Hessian approximation at x.
        'grad': [grad],        # Gradient approximation at x.
        'x_old': [0.],         # Old value for x.
    }

    # The set of constraints is also represented as a single dictionary so
    # we don't need to send a lot of parameters to some functions.
    constrained = {
        'A': A,
        'Aeq': Aeq,
        'eq': eq,
        'l': l,
        'u': u,
        'bl': bl,
        'bu': bu,
        'n': n,
        'N': N,
    }

    # -------------------------------------------------------------------------
    # Main iteration.

    # In the end, the final solution and its function value will be stored in
    # these two variables.
    xk = x0
    yk = y0

    # The loop continues until we reach the maximum number of evaluations
    # of the objective function, or the step length goes below the tolerance,
    # or the value of the objective function varies too little from some
    # iteration to the next one.
    while evaluations < max_evals and alpha > prec1 and y_var > prec2:
        iterations += 1

        y_old = yk

        # This flag decides whether the quasi-Newton iteration needs to be
        # computed. Its value can be True only if the pattern search step
        # was successful in finding feasible directions at the current x.
        quasinewton_needed = False

        # W_stencil is a 2D array whose columns are the directions found at the
        # pattern search step.
        # y_stencil is a 1D array of function values obtained at the x points
        # reached using the directions found by the pattern search step.
        # These arrays will be needed in the quasi-Newton iteration. From them
        # we'll be able to compute the stencil gradient that approximates the
        # derivative of the objective function at x.
        W_stencil = np.empty((Pk.shape[0], 0))
        y_stencil = []

        # ---------------------------------------------------------------------
        # Inner iteration (searches for the best direction in Pk).

        # Inner iterations counter.
        inner_iterations = 0

        # Flag that tells whether the search for a candidate x that decreases
        # the value of the objective function was successful in this iteration.
        search_success = False

        # The loop continues until the search is successful, or the same stop
        # conditions of the main loop are satisfied.
        while evaluations < max_evals and alpha > prec1 and not search_success:
            inner_iterations += 1

            # Change applied to the objective function's value.
            # (Not necessarily a decrease, though)
            decrease = -min(alpha**p, 1.) * gamma + eta

            # Iterate over all directions in Pk.
            for i in range(Pk.shape[1]):
                dk = Pk[:, i]

                # Use the i-th direction and the step length to calculate the
                # next candidate point, and then check its feasibility.
                x = xk + alpha * dk
                is_feasible = check_feasibility(
                    x, A, Aeq, eq, l, u, bl, bu, prec4)

                # Check the history of previously evaluated x points to avoid
                # evaluating the same point twice.
                if is_feasible:
                    found_x, found_dk, position = search_history(
                        history['x'], W_stencil, x, dk, prec5)

                    # If necessary, evaluate the objective function at the new
                    # trial point x.
                    if found_x and not found_dk:
                        y = history['y'][position]
                        y_stencil.append(y)
                        W_stencil = np.hstack((W_stencil, dk.reshape(-1, 1)))
                        eval_success = False
                    if not found_x:
                        y, eval_success = f(x)
                        evaluations += 1

                    # Update the history.
                    W_stencil = np.hstack((W_stencil, dk.reshape(-1, 1)))
                    y_stencil.append(y)
                    history['x'].append(x)
                    history['y'].append(y)

                    # Update x and y if the iteration was successful in
                    # finding a better trial point.
                    if y < (y_old + decrease) and \
                       eval_success and not search_success:
                        xp = x
                        yk = y
                        search_success = True
                    if y < yk and eval_success and search_success:
                        xp = x
                        yk = y

                    # Update the best x and y values if needed, and activate
                    # the quasi-Newton flag as well.
                    if y < y_min:
                        x_min = xk
                        y_min = yk
                        quasinewton_needed = True

            # If the search fails, we decrease the step length and reset the
            # successes counter (beta) to zero.
            if not search_success:
                if alpha > alpha_old:
                    alpha = alpha_old
                else:
                    alpha *= 0.5
                beta = 0
                W_stencil = np.empty((Pk.shape[0], 0))
                y_stencil = []

        # Update eta for the next iteration.
        eta = gamma / (iterations**3)

        # Update the history of pattern search solutions.
        history['xps'].append(xp if search_success else xk)
        history['yps'].append(yk)

        # print('xps = {}'.format(history['xps'][-1]))

        # ---------------------------------------------------------------------
        # Quasi-Newton iteration.

        # Compute the quasi-Newton iteration and save new points/values in
        # the histories.
        if quasinewton_needed:
            if alpha > np.sqrt(prec1):
                xq, yq, history, evals, x_new_status = quasinewton(
                    f, W_stencil, y_stencil, xk, y_old, yk, history,
                    alpha, constrained, Tb, max_red, prec5)

                evaluations += evals
                history['xqn'].append(xq)
                history['yqn'].append(yq)
            else:
                yq = yk + 1.
                history['xqn'].append(np.nan)
                history['yqn'].append(np.nan)
        else:
            yq = yk + 1.
            history['xqn'].append(np.nan)
            history['yqn'].append(np.nan)

        # print('xqn = {}'.format(history['xqn'][-1]))

        # Compare the objective function values at xps and xqn. Then select the
        # point whose function value is the lowest.
        if yq < yk and x_new_status == 1:
            yk = yq
            xk = xq
        else:
            xk = history['xps'][-1]

        # Update the minimums.
        if yk <= y_min:
            x_min = xk
            y_min = yk

        # If the step length needs to be reduced, we set beta to zero in order
        # to stop increasing alpha.
        if beta > 0 and yk > y_old:
            beta = 0

        # Update the variation in the objective function value.
        y_var = abs(yk - y_old)
        if alpha > 1e-2:
            y_var = 1.

        alpha_old = alpha
        if beta == 0 or alpha < 1.:
            beta = 1
        else:
            # Increase alpha so we may take larger steps in the next
            # iterations.
            alpha = min(rho**beta * alpha, 10.)
            beta += 1

        # print('xk = {}'.format(xk))

        # Update the pattern and working set.
        if most_general_kind == 'inequality':
            Vin, Veq, Vb, epsilon_active = find_epsilon_active_constraints(
                A, xk, l, u, N, min(alpha, 1.), prec3)

            Pk, Tb = build_pattern(Vin, Veq, Vb, Aeq, N, epsilon_active)

    # Look for any of the reasons that may cause the optimization process to
    # stop. These are: (1) reaching the maximum number of evaluations of the
    # objective function; (2) Reaching a very low value for the step length;
    # or (3) Having no significant variation in the objective function value
    # from one iteration to another.
    exit_cause = []
    if alpha < prec1:
        exit_cause.append('TP')
    if y_var < prec2:
        exit_cause.append('VF')
    if evaluations >= max_evals:
        exit_cause.append('AF')

    # Update values for the next iteration.
    x = x_min
    y = y_min

    return x, y, iterations, alpha, evaluations, history, exit_cause
