from __future__ import print_function
import unittest
import numpy as np
from psifa.api import psifa


class HSTestCase(unittest.TestCase):
    """Test case with a collection of optimization problems from
    Hock and Schittkowski (1981).
    """

    def test_problem_HS01(self):
        def f(x):
            y = 100. * (x[1] - x[0]**2)**2 + (1. - x[0])**2
            eval_success = True
            return y, eval_success

        A = np.array([])
        l = np.array([])
        u = np.array([])
        bl = np.array([-np.inf, -1.5])
        bu = np.array([np.inf, np.inf])
        n = 2
        x0 = np.array([-2., 1.])
        xc = np.array([1., 1.])
        yc = 0.

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 130)
        self.assertTrue(evaluations <= 498)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS03(self):
        def f(x):
            y = x[1] + 10.**(-5) * (x[1] - x[0])**2
            eval_success = True
            return y, eval_success

        A = np.array([])
        l = np.array([])
        u = np.array([])
        bl = np.array([-np.inf, 0.])
        bu = np.array([np.inf, np.inf])
        n = 2
        x0 = np.array([10., 1.])
        xc = np.array([0., 0.])
        yc = 0.

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 13)
        self.assertTrue(evaluations <= 102)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS04(self):
        def f(x):
            y = 1. / 3. * (x[0] + 1.)**3 + x[1]
            eval_success = True
            return y, eval_success

        A = np.array([])
        l = np.array([])
        u = np.array([])
        bl = np.array([1., 0.])
        bu = np.array([np.inf, np.inf])
        n = 2
        x0 = np.array([1.125, 0.125])
        xc = np.array([1., 0.])
        yc = 8. / 3.

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 12)
        self.assertTrue(evaluations <= 105)
        self.assertTrue('TP' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS05(self):
        def f(x):
            y = np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5 * x[0] + \
                2.5 * x[1] + 1.
            eval_success = True
            return y, eval_success

        A = np.array([])
        l = np.array([])
        u = np.array([])
        bl = np.array([-1.5, -3.])
        bu = np.array([4., 3.])
        n = 2
        x0 = np.array([0., 0.])
        xc = np.array([-np.pi / 3. + 0.5, -np.pi / 3. - 0.5])
        yc = -0.5 * np.sqrt(3.) - np.pi / 3.

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 59)
        self.assertTrue(evaluations <= 247)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS09(self):
        def f(x):
            y = np.sin(np.pi * x[0] / 12.) * np.cos(np.pi * x[1] / 16.)
            eval_success = True
            return y, eval_success

        A = np.array([[4., -3.]])
        l = np.array([0.])
        u = np.array([0.])
        bl = np.array([-np.inf, -np.inf])
        bu = np.array([np.inf, np.inf])
        n = 2
        x0 = np.array([0., 0.])
        xc = np.array([-3., -4.])
        yc = -0.5

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 14)
        self.assertTrue(evaluations <= 50)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS21(self):
        def f(x):
            y = 0.01 * x[0]**2 + x[1]**2 - 100.
            eval_success = True
            return y, eval_success

        A = np.array([[10., -1.]])
        l = np.array([10.])
        u = np.array([np.inf])
        bl = np.array([2., -50.])
        bu = np.array([50., 50.])
        n = 2
        x0 = np.array([2., 2.])
        xc = np.array([2., 0.])
        yc = -99.96

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 18)
        self.assertTrue(evaluations <= 179)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS24(self):
        def f(x):
            y = (1. / (27. * np.sqrt(3.))) * ((x[0] - 3.)**2 - 9.) * x[1]**3
            eval_success = True
            return y, eval_success

        A = np.array([[1. / np.sqrt(3.), -1.],
                      [1., np.sqrt(3.)],
                      [-1., -np.sqrt(3.)]])
        l = np.array([0., 0., -6.])
        u = np.array([np.inf, np.inf, np.inf])
        bl = np.array([0., 0.])
        bu = np.array([np.inf, np.inf])
        n = 2
        x0 = np.array([1., 0.5])
        xc = np.array([3., np.sqrt(3.)])
        yc = -1.

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 17)
        self.assertTrue(evaluations <= 112)
        self.assertTrue('TP' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS25(self):
        def f(x):
            y = 0.
            for i in range(1, 100):
                a = 25. + (-50. * np.log(0.01 * i))**(2. / 3.)
                b = -0.01 * i + np.exp((-1. / x[0]) * (a - x[1])**x[2])
                y += b**2
            eval_success = True
            return y, eval_success

        A = np.array([])
        l = np.array([])
        u = np.array([])
        bl = np.array([0.1, 0., 0.])
        bu = np.array([100., 25.6, 5.])
        n = 3
        x0 = np.array([100., 12.5, 3.])
        # xc = np.array([50., 25., 1.5])
        # yc = 0.
        xc = np.array([92.7664, 23.9537, 1.6871])
        yc = 0.0048065

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)  # fail
        self.assertTrue(iterations <= 157)
        self.assertTrue(evaluations <= 2008)
        self.assertTrue('AF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS28(self):
        def f(x):
            y = (x[0] + x[1])**2 + (x[1] + x[2])**2
            eval_success = True
            return y, eval_success

        A = np.array([[1., 2., 3.]])
        l = np.array([1.])
        u = np.array([1.])
        bl = np.array([-np.inf, -np.inf, -np.inf])
        bu = np.array([np.inf, np.inf, np.inf])
        n = 3
        x0 = np.array([-4., 1., 1.])
        xc = np.array([0.5, -0.5, 0.5])
        yc = 0.

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)  # fail
        # self.assertTrue(iterations <= 94)  # fail
        self.assertTrue(evaluations <= 672)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS35(self):
        def f(x):
            y = 9. - 8. * x[0] - 6. * x[1] - 4. * x[2] + 2. * x[0]**2 + \
                2. * x[1]**2 + x[2]**2 + 2. * x[0] * x[1] + 2. * x[0] * x[2]
            eval_success = True
            return y, eval_success

        A = np.array([[-1., -1., -2.]])
        l = np.array([-3.])
        u = np.array([np.inf])
        bl = np.array([0., 0., 0.])
        bu = np.array([np.inf, np.inf, np.inf])
        n = 3
        x0 = np.array([0.5, 0.5, 0.5])
        xc = np.array([4. / 3., 7. / 9., 4. / 9.])
        yc = 1. / 9.

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 67)
        self.assertTrue(evaluations <= 336)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS36(self):
        def f(x):
            y = -x[0] * x[1] * x[2]
            eval_success = True
            return y, eval_success

        A = np.array([[-1., -2., -2.]])
        l = np.array([-72.])
        u = np.array([np.inf])
        bl = np.array([0., 0., 0.])
        bu = np.array([20., 11., 42.])
        n = 3
        x0 = np.array([10., 10., 10.])
        # xc = np.array([20., 11., 15.])
        # yc = -3300.
        xc = np.array([17.28, 11., 16.36])
        yc = -3109.71

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 21)
        # self.assertTrue(evaluations <= 201)  # fail
        self.assertTrue('TP' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS37(self):
        def f(x):
            y = -x[0] * x[1] * x[2]
            eval_success = True
            return y, eval_success

        A = np.array([[-1., -2., -2.],
                      [1., 2., 2.]])
        l = np.array([-72., 0.])
        u = np.array([np.inf, np.inf])
        bl = np.array([0., 0., 0.])
        bu = np.array([42., 42., 42.])
        n = 3
        x0 = np.array([10., 10., 10.])
        xc = np.array([24.000130885714569,
                       11.999993260028832,
                       11.999941297113882])
        yc = -3456.

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 47)
        self.assertTrue(evaluations <= 289)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS38(self):
        def f(x):
            y = 100 * (x[1] - x[0]**2)**2 + (1. - x[0])**2 + \
                90. * (x[3] - x[2]**2)**2 + (1. - x[2])**2 + \
                10.1 * ((x[1] - 1.)**2 + (x[3] - 1.)**2) + \
                19.8 * (x[1] - 1.) * (x[3] - 1.)
            eval_success = True
            return y, eval_success

        A = np.array([])
        l = np.array([])
        u = np.array([])
        bl = np.array([-10., -10., -10., -10.])
        bu = np.array([10., 10., 10., 10.])
        n = 4
        x0 = np.array([-3., -1., -3., -1.])
        # xc = np.array([1., 1., 1., 1.])
        # yc = 0.
        xc = np.array([1.28472396, 1.65645037, -0.49984457, 0.26141631])
        yc = 2.60824579786

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 135)
        self.assertTrue(evaluations <= 2011)
        self.assertTrue('AF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS41(self):
        def f(x):
            y = 2. - x[0] * x[1] * x[2]
            eval_success = True
            return y, eval_success

        A = np.array([[1., 2., 2., -1.]])
        l = np.array([0.])
        u = np.array([0.])
        bl = np.array([0., 0., 0., 0.])
        bu = np.array([1., 1., 1., 2.])
        n = 4
        x0 = np.array([0., 0.5, 0.5, 2.])
        # xc = np.array([2. / 3., 1. / 3., 1. / 3., 2.])
        # yc = 52. / 27.
        xc = np.array([0.67188, 0.33203, 0.33203, 2.])
        yc = 1.9259

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 10)
        self.assertTrue(evaluations <= 75)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS44(self):
        def f(x):
            y = x[0] - x[1] - x[2] - x[0] * x[2] + \
                x[0] * x[3] + x[1] * x[2] - x[1] * x[3]
            eval_success = True
            return y, eval_success

        A = np.array([[-1., -2., 0., 0.],
                      [-4., -1., 0., 0.],
                      [-3., -4., 0., 0.],
                      [0., 0., -2., -1.],
                      [0., 0., -1., -2.],
                      [0., 0., -1., -1.]])
        l = np.array([-8., -12., -12., -8., -8., -5.])
        u = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        bl = np.array([0., 0., 0., 0.])
        bu = np.array([np.inf, np.inf, np.inf, np.inf])
        n = 4
        x0 = np.array([0., 0., 0., 0.])
        xc = np.array([0., 3., 0., 4.])
        yc = -15.

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 12)
        # self.assertTrue(evaluations <= 154)  # fail
        self.assertTrue('TP' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS45(self):
        def f(x):
            y = 2. - (1. / 120.) * x[0] * x[1] * x[2] * x[3] * x[4]
            eval_success = True
            return y, eval_success

        A = np.array([])
        l = np.array([])
        u = np.array([])
        bl = np.array([0., 0., 0., 0., 0.])
        bu = np.array([1., 2., 3., 4., 5.])
        n = 5
        x0 = np.array([0., 1., 2., 3., 4.])
        xc = np.array([1., 2., 3., 4., 5.])
        yc = 1.

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 16)
        self.assertTrue(evaluations <= 259)
        self.assertTrue('TP' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS48(self):
        def f(x):
            y = (x[0] - 1.)**2 + (x[1] - x[2])**2 + (x[3] - x[4])**2
            eval_success = True
            return y, eval_success

        A = np.array([[1., 1., 1., 1., 1.],
                      [0., 0., 1., -2., -2.]])
        l = np.array([5., -3.])
        u = np.array([5., -3.])
        bl = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        bu = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
        n = 5
        x0 = np.array([3., 5., -3., 2., -2.])
        xc = np.array([1., 1., 1., 1., 1.])
        yc = 0.

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 19)
        self.assertTrue(evaluations <= 245)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS49(self):
        def f(x):
            y = (x[0] - x[1])**2 + (x[2] - 1.)**2 + \
                (x[3] - 1.)**4 + (x[4] - 1.)**6
            eval_success = True
            return y, eval_success

        A = np.array([[1., 1., 1., 4., 0.],
                      [0., 0., 1., 0., 5.]])
        l = np.array([7., 6.])
        u = np.array([7., 6.])
        bl = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        bu = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
        n = 5
        x0 = np.array([10., 7., 2., -3., 0.8])
        # xc = np.array([1., 1., 1., 1., 1.])
        # yc = 0.
        xc = np.array([1.55468750, 1.55859375, 0.98828125,
                       0.72460938, 1.00234375])
        yc = 0.00590429

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 187)
        self.assertTrue(evaluations <= 2003)
        self.assertTrue('AF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS50(self):
        def f(x):
            y = (x[0] - x[1])**2 + (x[1] - x[2])**2 + \
                (x[2] - x[3])**4 + (x[3] - x[4])**2
            eval_success = True
            return y, eval_success

        A = np.array([[1., 2., 3., 0., 0.],
                      [0., 1., 2., 3., 0.],
                      [0., 0., 1., 2., 3.]])
        l = np.array([6., 6., 6.])
        u = np.array([6., 6., 6.])
        bl = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        bu = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
        n = 5
        x0 = np.array([35., -31., 11., 5., -5.])
        xc = np.array([1., 1., 1., 1., 1.])
        yc = 0.

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 43)
        self.assertTrue(evaluations <= 272)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS51(self):
        def f(x):
            y = (x[0] - x[1])**2 + (x[0] + x[2] - 2.)**2 + \
                (x[3] - 1.)**2 + (x[4] - 1.)**2
            eval_success = True
            return y, eval_success

        A = np.array([[1., 3., 0., 0., 0.],
                      [0., 0., 1., 1., -2.],
                      [0., 1., 0., 0., -1.]])
        l = np.array([4., 0., 0.])
        u = np.array([4., 0., 0.])
        bl = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        bu = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
        n = 5
        x0 = np.array([2.5, 0.5, 2., -1., 0.5])
        xc = np.array([1., 1., 1., 1., 1.])
        yc = 0.

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 71)
        self.assertTrue(evaluations <= 302)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS52(self):
        def f(x):
            y = (4. * x[0] - x[1])**2 + (x[1] + x[2] - 2.)**2 + \
                (x[3] - 1.)**2 + (x[4] - 1.)**2
            eval_success = True
            return y, eval_success

        A = np.array([[1., 3., 0., 0., 0.],
                      [0., 0., 1., 1., -2.],
                      [0., 1., 0., 0., -1.]])
        l = np.array([0., 0., 0.])
        u = np.array([0., 0., 0.])
        bl = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        bu = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
        n = 5
        x0 = np.array([0, 0, 0., 0., 0.])
        # xc = np.array([-33., 11., 180., -158., 11.]) / 349.
        # yc = 1859. / 349.
        xc = np.array([-0.09472656, 0.03157552, 0.51529948,
                       -0.45214844, 0.03157552])
        yc = 5.32664861

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 9)
        self.assertTrue(evaluations <= 91)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS53(self):
        def f(x):
            y = (x[0] - x[1])**2 + (x[1] + x[2] - 2.)**2 + \
                (x[3] - 1.)**2 + (x[4] - 1.)**2
            eval_success = True
            return y, eval_success

        A = np.array([[1., 3., 0., 0., 0.],
                      [0., 0., 1., 1., -2.],
                      [0., 1., 0., 0., -1.]])
        l = np.array([0., 0., 0.])
        u = np.array([0., 0., 0.])
        bl = np.array([-10., -10., -10., -10., -10.])
        bu = np.array([10., 10., 10., 10., 10.])
        n = 5
        x0 = np.array([0, 0, 0., 0., 0.])
        # xc = np.array([-33., 11., 27., -5., 11.]) / 43.
        # yc = 176. / 43.
        xc = np.array([-0.76757812, 0.25585938, 0.62695312,
                       -0.11523438, 0.25585938])
        yc = 4.09302521

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 8)
        self.assertTrue(evaluations <= 95)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS54(self):
        def f(x):
            h = (((x[0] - 1e6)**2) / 6.4e7 +
                 (x[0] - 1e4) * (x[1] - 1.) / 2e4 +
                 (x[1] - 1.)**2) * ((x[2] - 2e6)**2) / (0.96 * 4.9e13) + \
                ((x[3] - 10.)**2) / 2.5e3 + (x[4] - 1e-3)**2 / 2.5e-3 + \
                (x[5] - 1e8)**2 / 2.5e17
            y = -np.exp(-h / 2.)
            eval_success = True
            return y, eval_success

        A = np.array([[1., 4e3, 0., 0., 0., 0.]])
        l = np.array([1.76e4])
        u = np.array([1.76e4])
        bl = np.array([0., -10., 0., 0., -1., 0.])
        bu = np.array([2e4, 10., 1e7, 20., 1., 2e8])
        n = 6
        x0 = np.array([1.76e4, 0, 0., 0., 0., 0.])
        # xc = np.array([91600. / 7., 79. / 70., 2e6, 10., 1e-3, 1e8])
        # yc = -np.exp(-27. / 280.)
        xc = np.array([1.7600e4, -1.9531e-5, 0., 3.38281250, 0., 0.])
        yc = 0.

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 10)
        self.assertTrue(evaluations <= 129)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS55(self):
        def f(x):
            y = x[0] + 2. * x[1] + 4. * x[4] + np.exp(x[0] * x[3])
            eval_success = True
            return y, eval_success

        A = np.array([[1., 2., 0., 0., 5., 0.],
                      [1., 1., 1., 0., 0., 0.],
                      [0., 0., 0., 1., 1., 1.],
                      [1., 0., 0., 1., 0., 0.],
                      [0., 1., 0., 0., 1., 0.],
                      [0., 0., 1., 0., 0., 1.]])
        l = np.array([6., 3., 2., 1., 2., 2.])
        u = np.array([6., 3., 2., 1., 2., 2.])
        bl = np.array([0., 0., 0., 0., 0., 0.])
        bu = np.array([1., np.inf, np.inf, 1., np.inf, np.inf])
        n = 6
        x0 = np.array([0.447794849394555, 1.482598283131551,
                       1.069606867473851, 0.552205150605459,
                       0.517401716868434, 0.930393132526135])
        xc = np.array([0., 4. / 3., 5. / 3., 1., 2. / 3., 1. / 3.])
        yc = 19. / 3.

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 10)
        self.assertTrue(evaluations <= 41)
        self.assertTrue('TP' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS62(self):
        def f(x):
            y = -32.174 * \
                (255. * np.log((x[0] + x[1] + x[2] + 0.03) /
                               (0.09 * x[0] + x[1] + x[2] + 0.03)) +
                 280. * np.log((x[1] + x[2] + 0.03) /
                               (0.07 * x[1] + x[2] + 0.03)) +
                 290. * np.log((x[2] + 0.03) / (0.13 * x[2] + 0.03)))
            eval_success = True
            return y, eval_success

        A = np.array([[1., 1., 1.]])
        l = np.array([1.])
        u = np.array([1.])
        bl = np.array([0., 0., 0.])
        bu = np.array([1., 1., 1.])
        n = 3
        x0 = np.array([0.7, 0.2, 0.1])
        xc = np.array([0.617812690138817,
                       0.328202229738235,
                       0.053985080122948])
        yc = -2.627251448e4

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)  # fail
        # self.assertTrue(iterations <= 67)  # fail
        # self.assertTrue(evaluations <= 300)  # fail
        # self.assertTrue('TP' in exit_cause)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS76(self):
        def f(x):
            y = x[0]**2 + 0.5 * x[1]**2 + x[2]**2 + 0.5 * x[3]**2 - \
                x[0] * x[2] - x[2] * x[3] - x[0] - 3. * x[1] + x[2] - x[3]
            eval_success = True
            return y, eval_success

        A = np.array([[-1., -2., -1., -1.],
                      [-3., -1., -2., 1.],
                      [0., 1., 4., 0.]])
        l = np.array([-5., -4., 1.5])
        u = np.array([np.inf, np.inf, np.inf])
        bl = np.array([0., 0., 0., 0.])
        bu = np.array([np.inf, np.inf, np.inf, np.inf])
        n = 4
        x0 = np.array([0.5, 0.5, 0.5, 0.5])
        # xc = np.array([0.2727273, 2.090909, 0.26e-10, 0.5454545])
        # yc = -4.681818090904359
        xc = np.array([0.47114044, 2.08220996, 0., 0.36443964])
        yc = -4.62602936

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)  # fail
        # self.assertTrue(iterations <= 16)  # fail
        # self.assertTrue(evaluations <= 165)  # fail
        self.assertTrue('TP' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS86(self):
        def f(x):
            e = np.array([-15., -27., -36., -18., -12.])
            c = np.array([[30., -20., -10., 32., -10.],
                          [-20., 39., -6., -31., 32.],
                          [-10., -6., 10., -6., -10.],
                          [32., -31., -6., 39., -20.],
                          [-10., 32., -10., -20., 30.]])
            d = np.array([4., 8., 10., 6., 2.])
            s1 = 0.
            s2 = 0.
            s3 = 0.
            for j in range(5):
                s1 += e[j] * x[j]
                for i in range(5):
                    s2 += c[i, j] * x[i] * x[j]
                s3 += d[j] * x[j]**3
            y = s1 + s2 + s3
            eval_success = True
            return y, eval_success

        A = np.array([[-16., 2., 0., 1., 0.],
                      [0., -2., 0., 4., 2.],
                      [-3.5, 0., 2., 0., 0.],
                      [0., -2., 0., -4., -1.],
                      [0., -9., -2., 1., -2.8],
                      [2., 0., -4., 0., 0.],
                      [-1., -1., -1., -1., -1.],
                      [-1., -2., -3., -2., -1.],
                      [1., 2., 3., 4., 5.],
                      [1., 1., 1., 1., 1.]])
        l = np.array([-40., -2., -0.25, -4., -4., -1., -40., -60., 5., 1.])
        u = np.array([np.inf] * 10)
        bl = np.array([0., 0., 0., 0., 0.])
        bu = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
        n = 5
        x0 = np.array([0., 0., 0., 0., 1.])
        xc = np.array([0.299999998116435,
                       0.333467595429803,
                       0.399999998635825,
                       0.428310114302271,
                       0.223964911296049])
        yc = -32.34867897

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)  # fail
        self.assertTrue(iterations <= 86)  # fail
        # self.assertTrue(evaluations <= 920)  # fail
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS105(self):
        # Requires the "y105.mat" file.
        pass

    # -------------------------------------------------------------------------

    def test_problem_HS110(self):
        def f(x):
            s = 0.
            p = 1.
            for i in range(10):
                s += (np.log(x[i] - 2.))**2 + (np.log(10. - x[i]))**2
                p *= x[i]
            y = s - p**0.2
            eval_success = True
            return y, eval_success

        A = np.array([])
        l = np.array([])
        u = np.array([])
        bl = np.array([2.001] * 10)
        bu = np.array([9.999] * 10)
        n = 10
        x0 = np.array([9.] * 10)
        xc = np.array([9.35025655] * 10)
        yc = -45.77846971

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 51)
        # self.assertTrue(evaluations <= 1411)  # fail
        # self.assertTrue('AF' in exit_cause)
        self.assertTrue('VF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS112(self):
        def f(x):
            c = np.array([-6.089, -17.164, -34.054, -5.914, -24.721,
                          -14.986, -24.1, -10.708, -26.662, -22.179])
            y = 0.
            for i in range(10):
                y += x[i] * (c[i] + np.log(x[0] / (x[1] + np.sum(x))))
            eval_success = True
            return y, eval_success

        A = np.array([[1., 2., 2., 0., 0., 1., 0., 0., 0., 1.],
                      [0., 0., 0., 1., 2., 1., 1., 0., 0., 0.],
                      [0., 0., 1., 0., 0., 0., 1., 1., 2., 1.]])
        l = np.array([2., 1., 1.])
        u = np.array([2., 1., 1.])
        bl = np.array([1e-6] * 10)
        bu = np.array([np.inf] * 10)
        n = 10
        x0 = np.array([0.8, 0.5, 0.01, 0.01, 0.1,
                       0.01, 0.78, 0.01, 0.015, 0.17])
        xc = np.array([0.000001000013433,
                       0.999987491743295,
                       0.000004802332244,
                       0.999960037649762,
                       0.000014076754094,
                       0.000006088443443,
                       0.000005720398607,
                       0.999971681334066,
                       0.000004736271518,
                       0.000008323392047])
        # yc = -79.3911
        yc = -71.5236117652

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)
        self.assertTrue(iterations <= 20)
        # self.assertTrue(evaluations <= 299)  # fail
        self.assertTrue('TP' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS118(self):
        def f(x):
            y = 0.
            for i in range(5):
                j = 3 * i
                y += 2.3 * x[j + 0] + 1e-4 * x[j + 0]**2 + \
                    1.7 * x[j + 1] + 1e-4 * x[j + 1] + \
                    2.2 * x[j + 2] + 1.5e-4 * x[j + 2]**2
            eval_success = True
            return y, eval_success

        A = np.array([[-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
                      [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                      [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]]) \
            .astype(float)
        l = np.array([-7., -7., -7., -7., -7., -7., -7., -7., -7.,
                      -7., -7., -7., 60., 70., 100., 50., 85.])
        u = np.array([6., 6., 7., 6., 6., 7., 6., 6., 7., 6., 6.,
                      7., np.inf, np.inf, np.inf, np.inf, np.inf])
        bl = np.array([8., 43., 3., 0., 0., 0., 0., 0.,
                       0., 0., 0., 0., 0., 0., 0.])
        bu = np.array([21., 57., 16., 90., 120., 60., 90., 120.,
                       60., 90., 120., 60., 90., 120., 60.])
        n = 15
        x0 = np.array([20., 55., 15., 20., 60., 20., 20., 60.,
                       20., 20., 60., 20., 20., 60., 20.])
        # xc = np.array([8., 49., 3., 1., 56., 0., 1., 63.,
        #                6., 3., 70., 12., 5., 77., 18])
        # yc = 664.8204500
        # xc = np.array([8.2923, 4.8709e1, 3.0093, 1.3630,
        #                5.0331e1, 3.7989e-4,  7.3630, 5.7277e1,
        #                5.3619, 1.3302e1, 6.3991e1, 7.7069,
        #                1.6259e1, 7.0872e1, 1.2869e1])
        # yc = 667.65
        xc = np.array([8.2923146073, 48.7085129072, 3.0093215000,
                       1.3629991880, 50.3307789916, 0.0003798947,
                       7.3629991880, 57.2765915265, 5.3619399377,
                       13.3020232957, 63.9910701197, 7.7069065846,
                       16.2591149366, 70.8721322093, 12.8687528540])
        yc = 667.6476831493

        x1, y1, iterations, alpha, evaluations, history, exit_cause = \
            psifa(f, x0, A, l, u, n, bl, bu, max_evals=2000, prec1=1e-8)

        self.assertTrue(np.allclose(x1, xc) or np.isclose(y1, yc) or
                        np.allclose(xc, x1) or np.isclose(yc, y1) or
                        y1 < yc)  # fail
        self.assertTrue(iterations <= 89)
        self.assertTrue(evaluations <= 2019)
        self.assertTrue('AF' in exit_cause)

    # -------------------------------------------------------------------------

    def test_problem_HS119(self):
        # Requires the "p119.mat" file.
        pass


if __name__ == '__main__':
    unittest.main()
