import numpy as np

def lsqderiv(drdp, drdy, drdpdp=None, drdpdpdp=None, residuals=None, compute_hessian=False):
    """
    Compute the derivatives of a least squares estimate w.r.t. data. The least
    squares estimate is defined as
    
        p(y) = argmin_{p'} sum_i r_i(p', y)^2
    
    where r(p, y) is a "residuals" vector that depends on a "parameters" vector
    p and a "data" vector y.
    
    Parameters
    ----------
    drdp : array (n, k)
        Jacobian of residuals w.r.t. parameters.
    drdy : array (n, n)
        Jacobian of residuals w.r.t. data.
    drdpdp : array (n, k, k) or None
        Hessian of residuals w.r.t. parameters. It is not needed if you are
        computing only the first derivatives and `residuals` is zero.
    drdpdpdp : array (n, k, k, k) or None
        Tensor of third derivatives of residuals w.r.t parameters. Needed only
        if `compute_hessian` is True and `residuals` is nonzero.
    residuals : array (n,) or None
        If None, the derivatives are not computed exactly, but instead with
        residuals set to 0. If you got p(y) minimizing ||y - f(p)||^2, this
        means computing the derivatives at f(p) instead of y. This gives an
        estimate of the "true" derivatives. If specified, the exact derivatives
        at the given residuals are computed.
    compute_hessian : bool (default False)
        If False, compute and return only the jacobian.
    
    Returns
    -------
    A : array (k, k)
        Hessian of the cost function 1/2 * sum(residuals ** 2).
    dpdy : array (k, n)
        Jacobian of least squares estimate of parameters w.r.t. data.
    dpdydy : array (k, n, n)
        Hessian of least squares estimate of parameters w.r.t. data. Returned
        only if `compute_hessian` is True.
    """
    # TODO (eventually):
    # diagonal hessian when drdpdp.shape == (n, k)
    # broadcasting (mainly for drdy which is often np.eye(n))
    # vectorization
    # replace various einsums with direct op
    # remove sign changes and zero term
    # decompose A only once
    # avoid duplicate off-diagonal in computing dpdydy
    # do I need an SVD cut on drdp?
    # allow asymmetric inputs by implicitly taking the symmetric part
    # support sparse matrices
    
    # Check input
    drdp = np.asarray_chkfinite(drdp)
    assert len(drdp.shape) == 2
    n, k = drdp.shape
    
    drdy = np.asarray_chkfinite(drdy)
    assert drdy.shape == (n, n)
    
    zero_residuals = residuals is None
    if not zero_residuals:
        residuals = np.asarray_chkfinite(residuals)
        assert residuals.shape == (n,)
    
    compute_hessian = bool(compute_hessian)
    
    if drdpdp is None:
        assert zero_residuals and not compute_hessian
    else:
        drdpdp = np.asarray_chkfinite(drdpdp)
        assert drdpdp.shape == (n, k, k)
        assert np.allclose(drdpdp, np.swapaxes(drdpdp, 1, 2))
    
    if drdpdpdp is None:
        assert zero_residuals or not compute_hessian
    else:
        drdpdpdp = np.asarray_chkfinite(drdpdpdp)
        assert drdpdpdp.shape == (n, k, k, k)
        assert np.allclose(drdpdpdp, np.swapaxes(drdpdpdp, 1, 2))
        assert np.allclose(drdpdpdp, np.swapaxes(drdpdpdp, 1, 3))
    
    # g: gradient of the cost function 1/2 * r.T @ r
    dgdp = np.einsum('ia,ib->ab', drdp, drdp)
    if not zero_residuals:
        dgdp += np.einsum('i,iab->ab', residuals, drdpdp)
    dgdy = np.einsum('ia,ik->ak', drdp, drdy)
    if compute_hessian:
        dgdydy = 0
        dgdpdy = np.einsum('iab,ik->abk', drdpdp, drdy)
        dr3 = np.einsum('ia,ibg->abg', drdp, drdpdp)
        dgdpdp = dr3 + np.einsum('bga', dr3) + np.einsum('gab', dr3)
        if not zero_residuals:
            dgdpdp += np.einsum('i,iabg->abg', residuals, drdpdpdp)

    A = dgdp
    B = -dgdy
    dpdy = linalg.solve(A, B, assume_a='pos')

    if compute_hessian:
        C = -dgdydy
        dg2 = np.einsum('abk,bq->akq', dgdpdy, dpdy)
        C -= dg2 + np.einsum('aqk', dg2)
        C -= np.einsum('abg,bk,gq->akq', dgdpdp, dpdy, dpdy)
        assert np.allclose(C, np.swapaxes(C, 1, 2))
        C = C.reshape(k, n * n)
        dpdydy = linalg.solve(A, C, assume_a='pos')
        dpdydy = dpdydy.reshape(k, n, n)
    
        return A, dpdy, dpdydy
    
    return A, dpdy

if __name__ == '__main__':
    import unittest
    from autograd import numpy as np
    import autograd
    from scipy import optimize, linalg
    import numdifftools
    
    class TestDerivatives(unittest.TestCase):
        
        def fit(self, f, true_p, y=None, deriv3=False):
            def r(p, y):
                return y - f(p)
            jac = autograd.jacobian(r, 0)
            jacdata = autograd.jacobian(r, 1)
            hess = autograd.hessian(r, 0)

            if y is None:
                true_y = f(true_p)
                y = true_y + np.random.randn(*true_y.shape)

            result = optimize.least_squares(r, true_p, jac=jac, args=(y,))
            assert result.success
            
            drdp = jac(result.x, y)
            assert np.allclose(drdp, result.jac)
            
            drdy = jacdata(result.x, y)
            assert np.allclose(drdy, np.eye(len(y)))
            
            drdpdp = hess(result.x, y)
            
            if deriv3:
                jhess = autograd.jacobian(hess, 0)
                drdpdpdp = jhess(result.x, y)
                return result.x, drdp, drdy, drdpdp, drdpdpdp, result.fun
            
            return result.x, drdp, drdy, drdpdp
        
        def fit_numjac(self, f, p0, y=None):
            def r(p, y):
                return y - f(p)
            jac = autograd.jacobian(r, 0)

            def fun(y):
                re = optimize.least_squares(
                    r, p0, jac=jac, args=(y,), method='lm',
                    ftol=1e-10, xtol=1e-10, gtol=1e-10
                )
                return re.x
            
            if y is None:
                y = f(p0)
            return numdifftools.Jacobian(fun)(y)
                
        def fit_numhess(self, f, p0, y=None):
            def r(p, y):
                return y - f(p)
            jac = autograd.jacobian(r, 0)
            jacdata = autograd.jacobian(r, 1)
            hess = autograd.hessian(r, 0)

            def fun(y):
                result = optimize.least_squares(
                    r, p0, jac=jac, args=(y,), method='lm',
                    ftol=1e-10, xtol=1e-10, gtol=1e-10
                )
                drdp = result.jac
                drdy = jacdata(result.x, y)
                drdpdp = hess(result.x, y)
                _, dpdy = lsqderiv(drdp, drdy, drdpdp, residuals=result.fun, compute_hessian=False)
                return dpdy
            
            if y is None:
                y = f(p0)
            dpdydy = numdifftools.Jacobian(fun)(y)
            return np.swapaxes(dpdydy, 1, 2)
            
        def common_checks(self, A, dpdy, dpdydy, n, k, dtype=np.float64):
            # Check dtypes
            self.assertEqual(A.dtype, dtype)
            self.assertEqual(dpdy.dtype, dtype)
            self.assertEqual(dpdydy.dtype, dtype)
            
            # Check shapes
            self.assertEqual(A.shape, (k, k))
            self.assertEqual(dpdy.shape, (k, n))
            self.assertEqual(dpdydy.shape, (k, n, n))
            
            # Check finiteness
            self.assertTrue(np.all(np.isfinite(A)))
            self.assertTrue(np.all(np.isfinite(dpdy)))
            self.assertTrue(np.all(np.isfinite(dpdydy)))
            
            # Check symmetries
            self.assertTrue(np.allclose(A, A.T))
            self.assertTrue(np.allclose(dpdydy, np.swapaxes(dpdydy, 1, 2)))
        
        def test_linear(self, dtype=np.float64):
            p = np.random.randn(10)
            H = np.random.randn(100, len(p))
            f = lambda p: H @ p
           
            p, drdp, drdy, drdpdp = self.fit(f, p)
            assert np.all(drdpdp == 0)
            assert np.all(drdp == H) or np.all(drdp == -H)
            
            drdp = np.array(drdp, dtype=dtype)
            drdy = np.array(drdy, dtype=dtype)
            drdpdp = np.array(drdpdp, dtype=dtype)
            
            A, dpdy, dpdydy = lsqderiv(drdp, drdy, drdpdp, compute_hessian=True)
            self.common_checks(A, dpdy, dpdydy, *H.shape, dtype)
            
            # Check values
            self.assertTrue(np.allclose(A, drdp.T @ drdp))
            self.assertTrue(np.allclose(dpdy, linalg.solve(A, H.T)))
            self.assertTrue(np.all(dpdydy == 0))
        
        def test_dtype(self):
            self.test_linear(np.float32)
        
        def test_stochastic(self):
            p = 1/5 * np.random.randn(2)
            H = 1/5 * np.random.randn(3, len(p))
            f = lambda p: np.cos(1/2 + 1/3 * (1 + H) @ (1 + p))
            
            y = f(p) + 1/5 * np.random.randn(H.shape[0])
            p, drdp, drdy, drdpdp = self.fit(f, p, y)
            A, dpdy, dpdydy = lsqderiv(drdp, drdy, drdpdp, compute_hessian=True)
            self.common_checks(A, dpdy, dpdydy, *H.shape)
            
            dpdy_num = self.fit_numjac(f, p)
            dpdydy_num = self.fit_numhess(f, p)
            
            jclose = np.allclose(dpdy, dpdy_num)
            if not jclose:
                print(dpdy)
                print(dpdy_num)
            hclose = np.allclose(dpdydy, dpdydy_num)
            if not hclose:
                print(dpdydy)
                print(dpdydy_num)
            self.assertTrue(jclose)
            self.assertTrue(hclose)
    
        def test_exact(self):
            p = 1/5 * np.random.randn(2)
            H = 1/5 * np.random.randn(3, len(p))
            f = lambda p: np.cos(1/2 + 1/3 * (1 + H) @ (1 + p))
            
            y = f(p) + 1/5 * np.random.randn(H.shape[0])
            p, drdp, drdy, drdpdp, drdpdpdp, r = self.fit(f, p, y, True)
            A, dpdy, dpdydy = lsqderiv(drdp, drdy, drdpdp, drdpdpdp, r, True)
            self.common_checks(A, dpdy, dpdydy, *H.shape)
            
            dpdy_num = self.fit_numjac(f, p, y)
            dpdydy_num = self.fit_numhess(f, p, y)
            
            jclose = np.allclose(dpdy, dpdy_num, rtol=1e-4)
            if not jclose:
                print(dpdy)
                print(dpdy_num)
            hclose = np.allclose(dpdydy, dpdydy_num, rtol=1e-3)
            if not hclose:
                print(dpdydy)
                print(dpdydy_num)
            self.assertTrue(jclose)
            self.assertTrue(hclose)

    class TestInput(unittest.TestCase):
        
        def setUp(self):
            p = np.random.randn(3)
            # len(p) >= 3 so we do not trigger broadcasting by popping
            H = np.random.randn(4, len(p))
            f = lambda p: np.cos(H @ p)
            y = f(p) + np.random.randn(H.shape[0])
            
            def r(p, y):
                return y - f(p)
            jac = autograd.jacobian(r, 0)
            jacdata = autograd.jacobian(r, 1)
            hess = autograd.hessian(r, 0)
            deriv3 = autograd.jacobian(hess, 0)
            
            result = optimize.least_squares(r, p, jac, args=(y,))
            
            self.r = result.fun
            self.drdp = result.jac
            self.drdy = jacdata(result.x, y)
            self.drdpdp = hess(result.x, y)
            self.drdpdpdp = deriv3(result.x, y)
        
        def test_optional_derivatives(self):
            with self.assertRaises(AssertionError):
                lsqderiv(self.drdp, self.drdy, residuals=self.r)
            with self.assertRaises(AssertionError):
                lsqderiv(self.drdp, self.drdy, compute_hessian=True)
            with self.assertRaises(AssertionError):
                lsqderiv(self.drdp, self.drdy, self.drdpdp, residuals=self.r, compute_hessian=True)
        
        def test_shapes(self):
            with self.assertRaises(AssertionError):
                drdp = self.drdp[:, :-1]
                lsqderiv(drdp, self.drdy, self.drdpdp, self.drdpdpdp, self.r)
            with self.assertRaises(AssertionError):
                drdy = self.drdy[:, :-1]
                lsqderiv(self.drdp, drdy, self.drdpdp, self.drdpdpdp, self.r)
            with self.assertRaises(AssertionError):
                drdpdp = self.drdpdp[:, :-1, :]
                lsqderiv(self.drdp, self.drdy, drdpdp, self.drdpdpdp, self.r)
            with self.assertRaises(AssertionError):
                drdpdpdp = self.drdpdpdp[:, :-1, :, :]
                lsqderiv(self.drdp, self.drdy, self.drdpdp, drdpdpdp, self.r)
            with self.assertRaises(AssertionError):
                r = self.r[:-1]
                lsqderiv(self.drdp, self.drdy, self.drdpdp, self.drdpdpdp, r)
        
        def test_asymmetry(self):
            with self.assertRaises(AssertionError):
                drdpdp = np.array(self.drdpdp)
                drdpdp[:, 0, 1] += 1
                lsqderiv(self.drdp, self.drdy, drdpdp, self.drdpdpdp, self.r)
            with self.assertRaises(AssertionError):
                drdpdpdp = np.array(self.drdpdpdp)
                drdpdpdp[:, 0, 1, 2] += 1
                lsqderiv(self.drdp, self.drdy, self.drdpdp, drdpdpdp, self.r)
            with self.assertRaises(AssertionError):
                drdpdpdp = np.array(self.drdpdpdp)
                drdpdpdp[:, 0, 0, 1] += 1
                lsqderiv(self.drdp, self.drdy, self.drdpdp, drdpdpdp, self.r)
            with self.assertRaises(AssertionError):
                drdpdpdp = np.array(self.drdpdpdp)
                drdpdpdp[:, 0, 1, 0] += 1
                lsqderiv(self.drdp, self.drdy, self.drdpdp, drdpdpdp, self.r)
            with self.assertRaises(AssertionError):
                drdpdpdp = np.array(self.drdpdpdp)
                drdpdpdp[:, 1, 0, 0] += 1
                lsqderiv(self.drdp, self.drdy, self.drdpdp, drdpdpdp, self.r)
            with self.assertRaises(AssertionError):
                drdpdpdp = np.array(self.drdpdpdp)
                drdpdpdp[:, 1, 0, 0] += 1
                drdpdpdp[:, 0, 1, 0] += 1
                lsqderiv(self.drdp, self.drdy, self.drdpdp, drdpdpdp, self.r)
            with self.assertRaises(AssertionError):
                drdpdpdp = np.array(self.drdpdpdp)
                drdpdpdp[:, 1, 0, 0] += 1
                drdpdpdp[:, 0, 0, 1] += 1
                lsqderiv(self.drdp, self.drdy, self.drdpdp, drdpdpdp, self.r)
            with self.assertRaises(AssertionError):
                drdpdpdp = np.array(self.drdpdpdp)
                drdpdpdp[:, 0, 1, 0] += 1
                drdpdpdp[:, 0, 0, 1] += 1
                lsqderiv(self.drdp, self.drdy, self.drdpdp, drdpdpdp, self.r)
        
        def test_nonfinite(self):
            def dirty(a, x):
                b = np.array(a)
                b[tuple(np.random.randint(l) for l in a.shape)] = x
                return b
            
            for x in (np.inf, np.nan):
                with self.assertRaises(ValueError):
                    drdp = dirty(self.drdp, x)
                    lsqderiv(drdp, self.drdy, self.drdpdp, self.drdpdpdp, self.r)
                with self.assertRaises(ValueError):
                    drdy = dirty(self.drdy, x)
                    lsqderiv(self.drdp, drdy, self.drdpdp, self.drdpdpdp, self.r)
                with self.assertRaises(ValueError):
                    drdpdp = dirty(self.drdpdp, x)
                    lsqderiv(self.drdp, self.drdy, drdpdp, self.drdpdpdp, self.r)
                with self.assertRaises(ValueError):
                    drdpdpdp = dirty(self.drdpdpdp, x)
                    lsqderiv(self.drdp, self.drdy, self.drdpdp, drdpdpdp, self.r)
                with self.assertRaises(ValueError):
                    r = dirty(self.r, x)
                    lsqderiv(self.drdp, self.drdy, self.drdpdp, self.drdpdpdp, r)
    
    unittest.main()
