import numpy as np
from scipy.linalg import qr
from fractions import Fraction


def preprocess_constraints(A, l, u):
    """Normalizes constraints and their lower/upper bounds using a common factor.

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
    A : ndarray
        Normalized constraints.
    l : ndarray
        Normalized constraint lower bounds.
    u : ndarray
        Normalized constraint upper bounds.
    """

    # Normalizes each row independently.
    for i in range(A.shape[0]):
        norm = np.linalg.norm(A[i, :], np.inf)
        A[i, :] /= norm
        l[i] /= norm
        u[i] /= norm

    return A, l, u

# -----------------------------------------------------------------------------


def search_history(x_history, W_stencil, x, dk, prec5):
    """Searches the history in order to find out whether the objective function
    was already evaluated at a given trial point `x`.

    Parameters
    ----------
    x_history : list
        List of arrays representing the `x` points the objective function was
        already evaluated at.
    W_stencil: ndarray
        2D array of directions already evaluated in the current iteration of
        the main loop.
    x : ndarray
        Current trial point as a 1D array.
    dk : ndarray
        Current candidate direction as a 1D array.
    prec5 : float
        Maximum distance used to assess equality between `x` and the points in
        the history.

    Return values
    -------------
    found_x : bool
        `True` if `x` was found in the history; `False` otherwise.
    found_dk : bool
        `True` if `dk` was already evaluated in the current iteration of the
        main loop; `False` otherwise.
    position : int
        Index of `x` in the history, or -1 if it was never evaluated before.
    """

    found_x = False
    found_dk = False
    position = -1

    # First check if dk was already evaluated in the current iteration of the
    # main loop. If it was, there is no need to search x_history.
    if W_stencil.size > 0:
        for i in range(W_stencil.shape[1]):
            diff = W_stencil[:, i] - dk
            if np.linalg.norm(diff, np.inf) < prec5:
                found_x = True
                found_dk = True
                break

    # If dk was not found, we look for x in the history.
    if not found_dk:
        for i in range(len(x_history)):
            diff = x_history[i] - x
            if np.linalg.norm(diff, np.inf) < prec5:
                found_x = True
                position = i
                break

    return found_x, found_dk, position

# -----------------------------------------------------------------------------


def null(A, rational=False):
    """Computes an orthonormal basis for the null space of `A`.

    Parameters
    ----------
    A : ndarray
        Input array.
    r : bool
        If `True`, returns a "rational" basis obtained from the reduced
        row echelon form.

    Return value
    ------------
    Z : ndarray
        Orthonormal basis for the null space of A.
    """

    Z = np.array([])

    if A.size > 0:
        A = np.atleast_2d(A)
        m, n = A.shape

        if rational:
            # Rational basis.
            R, pivcol = rref(A)
            r = pivcol.size
            nopiv = np.delete(np.arange(n), pivcol)
            Z = np.zeros((n, n - r), dtype=A.dtype)
            if n > r:
                Z[nopiv, :] = np.eye(n - r, dtype=A.dtype)
                if r > 0:
                    Z[pivcol, :] = -R[:r, nopiv]
        else:
            # Orthonormal basis.
            _, S, V = np.linalg.svd(A, full_matrices=m <= n)
            if m > 1:
                s = np.diag(S)
            elif m == 1:
                s = S[0]
            else:
                s = 0.
            tol = max(m, n) * np.max(s) * np.finfo(A.dtype).eps
            r = np.sum(s > tol)
            Z = V[:, r:n]

    return Z

# -----------------------------------------------------------------------------


def rref(A, tol=None):
    """Produces the reduced row echelon form of `A`.

    Parameters
    ----------
    A : ndarray
        Input array.
    tol: float
        Tolerance to be used in the rank tests.

    Return value
    ------------
    R : ndarray
        Reduced row echelon form of `A`.
    jb : ndarray
        A vector so that:
            r = length(jb) is this algorithm's idea of the rank of A;
            x[jb] are the bound variables in a linear system, Ax = b;
            A[:,jb] is a basis for the range of A;
            R[1:r,jb] is the r-by-r identity matrix.
    """

    R = A.copy()
    m, n = R.shape

    # Vectorized version of the rat() function.
    vec_rat = np.vectorize(rat)

    # Does it appear that elements of A are ratios of small integers?
    num, den = vec_rat(R)
    rats = np.allclose(R, num / den)

    # Compute the default tolerance if none was provided.
    if tol is None:
        tol = max(m, n) * np.finfo(R.dtype).eps * np.linalg.norm(R, np.inf)

    # Loop over the entire matrix.
    i = j = 0
    jb = []
    while i < m and j < n:
        # Find value and index of largest element in the remainder of column j.
        p = np.max(np.abs(R[i:m, j]))
        k = np.argmax(np.abs(R[i:m, j])) + i

        if p <= tol:
            # The column is negligible, zero it out.
            R[i:m, j] = np.zeros((m - i,))
            j += 1
        else:
            # Remember the column index.
            jb.append(j)
            # Swap i-th and k-th rows.
            R[[i, k], j:n] = R[[k, i], j:n]
            # Divide the pivot row by the pivot element.
            R[i, j:n] = R[i, j:n] / R[i, j]
            # Subtract multiples of the pivot row from all the other rows.
            for k in np.append(np.arange(i), np.arange(i + 1, m)):
                R[k, j:n] -= np.dot(R[k, j], R[i, j:n])
            i += 1
            j += 1

    # Return "rational" numbers if appropriate.
    if rats:
        num, den = vec_rat(R)
        R = num / den

    return R, np.array(jb)

# -----------------------------------------------------------------------------


def rat(x):
    """Produces a rational fraction approximation for `x`.

    Parameters
    ----------
    x : float
        Input floating point number.

    Return value
    ------------
    num : int
        Fraction numerator.
    den : int
        Fraction denominator.
    """

    f = Fraction(x).limit_denominator()
    num, den = f.numerator, f.denominator

    return num, den

# -----------------------------------------------------------------------------


def eqnull(Aeq, ndims):
    """Computes the null space of a given 2D array of equality constraints.

    Parameters
    ----------
    Aeq : ndarray
        2D array of equality constraints.
    ndims : int
        Dimensionality of the optimization space.

    Return value
    ------------
    Anull : ndarray
        Null space of `Aeq`.
    """

    Anull = np.eye(ndims)

    if Aeq.size > 0:
        Q, _ = qr(Aeq.T, mode='economic')
        if Aeq.shape[0] <= ndims:
            Anull = np.eye(ndims) - np.dot(Q, Q.T)
        else:
            raise ValueError('Infeasible')

    return Anull
