import numpy as np
from psifa.utils import null


def classify_constraints(A, l, u):
    """Identifies the different kinds of constraints in A and return them in
    separate arrays.

    Parameters
    ----------
    A : ndarray
        2D array representing the constraints.
    l : ndarray
        1D array of constraint lower bounds.
    u : ndarray
        1D array of constraint upper bounds.

    Return values
    -------------
    Aeq : ndarray
        2D array of equality constraints.
    A : ndarray
        2D array of inequality constraints.
    l : ndarray
        1D array of inequality constraint lower bounds.
    u : ndarray
        1D array of inequality constraint upper bounds.
    eq : ndarray
        1D array of equality constraint values.
    most_general_kind : str
        Name the most general kind of constraint originally contained in `A`.
        Values can be:
            `'box'`: Box constraints;
            `'equality'`: Equality constraints;
            `'inequality'` Inequality constraints.
    """

    Aeq = np.array([])
    eq = np.array([])

    # Only box constraints exist if A is empty.
    if A.size == 0:
        most_general_kind = 'box'
    else:
        # Locate equality and inequality constraints.
        ind_eq = l == u
        ind_in = l != u

        # Leave only equality constraints for Aeq and eq.
        if np.any(ind_eq):
            Aeq = A[ind_eq, :]
            eq = u[ind_eq]

        # Leave only inequality constraints for A, l and u.
        A = A[ind_in, :]
        l = l[ind_in]
        u = u[ind_in]

        # If A is empty at this point, we have no inequality constraints.
        if A.size == 0:
            most_general_kind = 'equality'
        else:
            most_general_kind = 'inequality'

    return Aeq, A, l, u, eq, most_general_kind

# -----------------------------------------------------------------------------


def check_feasibility(x, A, Aeq, eq, l, u, bl, bu, prec4):
    """Checks whether a trial point `x` is feasible or not according to the
    problem's constraints.

    Parameters
    ----------
    x : ndarray
        Current trial point as a 1D array.
    A : ndarray
        2D array of inequality constraints.
    Aeq : ndarray
        2D array of equality constraints.
    eq : ndarray
        1D array of equality constraint values.
    l : ndarray
        1D array of inequality constraint lower bounds.
    u : ndarray
        1D array of inequality constraint upper bounds.
    bl : ndarray
        1D array of box constraint lower bounds.
    bu : ndarray
        1D array of box constraint upper bounds.
.   prec4 : float
        Tolerance for violating constraints.

    Return value
    ------------
    is_feasible : bool
        `True` if `x` is a feasible point; `False` otherwise.
    """

    # Check whether any inequality constraint is being violated at x.
    # Some tolerance is allowed to compensate eventual numerical
    # roundoff errors.
    if A.size > 0:
        b = np.dot(A, x)
        l_ok = l <= b + prec4
        u_ok = u >= b + prec4
    else:
        l_ok = np.array([True])
        u_ok = np.array([True])

    # Analogously, check the equality constraints too.
    eq_ok = True
    if Aeq.size > 0:
        b = np.dot(Aeq, x)
        eq_ok = np.linalg.norm(b - eq, np.inf) < prec4

    # Finally, check the box constraints.
    bl_ok = x >= bl - prec4
    bu_ok = x <= bu + prec4

    # The trial point is feasible if it passes all tests.
    is_feasible = np.all(l_ok) and np.all(u_ok) and \
        np.all(bl_ok) and np.all(bu_ok) and eq_ok

    return is_feasible

# -----------------------------------------------------------------------------


def find_epsilon_active_constraints(A, x, l, u, Z, tol, prec3):
    """Identifies all epsilon-active constraints at the current trial point `x`
    return them in separate arrays according to their kinds.

    Parameters
    ----------
    A : ndarray
        2D array of inequality constraints.
    x : ndarray
        Current trial point as a 1D array.
    l : ndarray
        1D array of inequality constraint lower bounds.
    u : ndarray
        1D array of inequality constraint upper bounds.
    Z : ndarray
        Null space of the array of equality constraints.
    tol : float
        Step length tolerance for the epsilon-active inequality constraints.
    prec3 : float
        Numerical tolerance that decides whether a constraint can be
        temporarily taken as an equality constraint.

    Return values
    -------------
    Vin : ndarray
        2D array of epsilon-active inequality constraints.
    Veq : ndarray
        2D array of epsilon-active equality constraints.
    Vb : ndarray
        2D array of epsilon-active inequality constraints that were temporarily
        taken as equality constraints.
    epsilon_active: bool
        `True` if any constraint is epsilon-active at `x`; `False` otherwise.
    """

    Vin = np.array([])
    Veq = np.array([])
    Vb = np.array([])

    # Inner products.
    b = np.dot(A, x)
    c = np.dot(Z.T, A) if Z.size > 0 else 1.

    # Find the epsilon-active inequality constraints.
    # OBS: np.where() returns a 1-sized tuple with the result inside it.
    il = np.where(np.abs((b - l) / c) <= tol)[0]
    iu = np.where(np.abs((b - u) / c) <= tol)[0]

    # Find the inequality constraints that are active both at the lower bound
    # and at the upper bound (thus being temporarily taken as equality
    # constraints in order to avoid making the working set degenerate).
    eq = np.intersect1d(il, iu)

    # Remove the constraints temporarily taken as equality constraints from
    # the lists of inequality constraint indices.
    il = np.setdiff1d(il, eq, assume_unique=True)
    iu = np.setdiff1d(iu, eq, assume_unique=True)
    # for j in range(eq.size):
    #     il = il[il != eq[j]]
    #     iu = iu[iu != eq[j]]

    # If at least one inequality constraint has reached one of its bounds, we
    # add to the pattern the directions that are in the null space of such
    # constraints. This way we allow the trial point to move over the bound's
    # face.
    le = np.where(np.abs((b - l) / c) <= prec3)[0]
    ue = np.where(np.abs((b - u) / c) <= prec3)[0]

    # Now that the epsilon-active constraints are known, use this information
    # to compute the matrices of generators that later we use to calculate the
    # feasible directions.
    if eq.size == 0:
        Vin = np.hstack((-A[il, :].T, A[iu, :].T))
    if eq.size > 0 or le.size > 0 or ue.size > 0:
        Vin = np.hstack((-A[il, :].T, A[iu, :].T))
        if np.any(eq):
            Veq = A[eq, :].T
        Vb = np.hstack((A[le, :].T, A[ue, :].T))

    # If all matrices are empty (no epsilon-active constraint at x), set Vin
    # as the identity matrix. The others remain empty.
    if Vin.size == 0 and Veq.size == 0 and Vb.size == 0:
        Vin = np.eye(A.shape[1])
        epsilon_active = False
    else:
        epsilon_active = True

    return Vin, Veq, Vb, epsilon_active

# -----------------------------------------------------------------------------


def build_pattern(Vin, Veq, Vb, Aeq, Z, epsilon_active):
    """Builds a pattern of feasible directions from the current trial point,
    given the current working set.

    Parameters
    ----------
    Vin : ndarray
        2D array of epsilon-active inequality constraints.
    Veq : ndarray
        2D array of epsilon-active equality constraints.
    Vb : ndarray
        2D array of epsilon-active inequality constraints that were temporarily
        taken as equality constraints.
    Aeq : ndarray
        2D array of equality constraints.
    Z : ndarray
        Null space of the array of equality constraints.
    epsilon_active : bool
        `True` if any constraint is epsilon-active at `x`; `False` otherwise.

    Return values
    -------------
    Pk : ndarray
        2D array whose columns represent the feasible directions from `x`.
    Tb : ndarray
        Null space of `Vb`.
    """

    m = Vin.shape[0]
    Tb = np.array([])

    # If no epsilon-active constraint exists, build a pattern that contains
    # only eye(n) and -eye(n), since we can move to any direction from the
    # current point.
    # OBS: notice that Vin == eye(n) when epsilon_active == False.
    if not epsilon_active:
        ones = np.ones((m, 1))
        Pk = np.hstack((Vin, -Vin, ones, -ones))
    else:
        # Build a basis for the null space of each matrix of epsilon-active
        # constraints.
        if Veq.size > 0:
            Aeq = np.vstack((Aeq, Veq.T))
            Teq = null(Aeq)
        else:
            Teq = Z
        Tb = null(Vb.T)
        Tin = null(Vin.T)
        F = np.dot(Vin, np.linalg.pinv(np.dot(Vin.T, Vin)))

        # Build a pattern composed by the generated bases.
        # OBS: empty matrices can't be stacked as in MATLAB, so these
        # conditionals below are necessary.
        Pk = np.empty((m, 0))
        if Teq.size > 0:
            Pk = np.hstack((Pk, -Teq, Teq))
        if Tin.size > 0:
            Pk = np.hstack((Pk, Tin, -Tin))
        if F.size > 0:
            Pk = np.hstack((Pk, F, -F))
        if Tb.size > 0:
            Pk = np.hstack((Pk, Tb, -Tb))

    if Pk.size > 0:
        # Normalize the feasible directions (columns of Pk).
        Pk = np.apply_along_axis(
            lambda z: z / np.linalg.norm(z), 0, Pk)

    return Pk, Tb
