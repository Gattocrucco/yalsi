import numpy as np

def lsqderiv(drdp, drdy, drdpdp):
    """
    Compute the derivatives of a least squares estimate w.r.t. data. The least
    squares estimate is defined as
    
        p = argmin_{p'} sum_i r_i(p', y)^2
    
    where r(p, y) is a "residuals" vector that depends on a "parameters" vector
    p and a "data" vector y.
    
    Parameters
    ----------
    drdp : array (n, k)
        Jacobian of residuals w.r.t. parameters.
    drdy : array (n, n)
        Jacobian of residuals w.r.t. data.
    drdpdp : array (n, k, k)
        Hessian of residuals w.r.t. parameters.
    
    Return
    ------
    A : array (k, k)
        drdp.T @ drdp
    dpdy : array (k, n)
        Jacobian of least squares estimate of parameters w.r.t. data.
    dpdydy : array (k, n, n)
        Hessian of least squares estimate of parameters w.r.t. data.
    """
    # TODO:
    # diagonal hessian with drdpdp.shape == (n, k)
    # compute gradient only with drdpdp=None by default
    # broadcasting
    # vectorization
    # replace various einsums with direct op
    # remove sign changes and zero term
    # decompose A only once
    # avoid duplicate off-diagonal in computing dpdydy
    # do I need an SVD cut on drdp?
    
    # Check input
    drdp = np.asarray_chkfinite(drdp)
    assert len(drdp.shape) == 2
    n, k = drdp.shape
    
    drdy = np.asarray_chkfinite(drdy)
    assert drdy.shape == (n, n)
    
    drdpdp = np.asarray_chkfinite(drdpdp)
    assert drdpdp.shape == (n, k, k)
    
    # g: gradient of the cost function 1/2 * r.T @ r
    dgdp = np.einsum('ia,ib->ab', drdp, drdp)
    # dgdp += np.einsum('i,iab->ab', r, drdpdp)
    dgdy = np.einsum('ia,ik->ak', drdp, drdy)
    dgdydy = 0
    dgdpdy = np.einsum('iab,ik->abk', drdpdp, drdy)
    dr3 = np.einsum('ia,ibg->abg', drdp, drdpdp)
    dgdpdp = dr3 + np.einsum('bga', dr3) + np.einsum('gab', dr3)
    # dgdpdp += np.einsum('i,iabg->abg', r, dgdpdpdp)

    A = dgdp
    B = -dgdy
    dpdy = linalg.solve(A, B, assume_a='pos')

    C = -dgdydy
    dg2 = np.einsum('abk,bq->akq', dgdpdy, dpdy)
    C -= dg2 + np.einsum('aqk', dg2)
    C -= np.einsum('abg,bk,gq->akq', dgdpdp, dpdy, dpdy)
    assert np.allclose(C, np.swapaxes(C, 1, 2))
    C = C.reshape(k, n * n)
    dpdydy = linalg.solve(A, C, assume_a='pos')
    dpdydy = dpdydy.reshape(k, n, n)
    
    return A, dpdy, dpdydy

if __name__ == '__main__':
    import unittest
    from autograd import numpy as np
    import autograd
    from scipy import optimize, linalg
    import numdiff
    
    class TestLSQDeriv(unittest.TestCase):
        
        def fit(self, f, true_p, y=None):
            def r(p, y):
                return y - f(p)
            jac = autograd.jacobian(r, 0)
            if y is None:
                true_y = f(true_p)
                y = true_y + np.random.randn(*true_y.shape)
            result = optimize.least_squares(r, true_p, jac=jac, args=(y,), ftol=1e-12, gtol=1e-12, xtol=1e-10)
            assert result.success
            
            drdp = jac(result.x, y)
            assert np.allclose(drdp, result.jac)
            jacdata = autograd.jacobian(r, 1)
            hess = autograd.hessian(r, 0)
            drdy = jacdata(result.x, y)
            drdpdp = hess(result.x, y)
            
            assert np.allclose(drdy, np.eye(len(y)))
            assert np.allclose(drdpdp, np.swapaxes(drdpdp, 1, 2))
            return result.x, drdp, drdy, drdpdp
        
        def fit_dd(self, f, p0, y, step=None):
            def r(p, y):
                return y - f(p)
            jac = autograd.jacobian(r, 0)
            result = optimize.least_squares(r, p0, jac=jac, args=(y,), ftol=1e-12, gtol=1e-12, xtol=1e-10)
            assert result.success

            def fun(y):
                re = optimize.least_squares(
                    r, result.x, jac=jac, args=(y,), method='lm',
                    ftol=1e-12, xtol=1e-12, gtol=1e-12
                )
                assert re.success
                return re.x
            dpdy, dpdydy = numdiff.numdiff(fun, y, step=step)
                            
            return result.x, dpdy, dpdydy
                
        def common_checks(self, A, dpdy, dpdydy, n, k, dtype=np.float64):
            # Check dtypes
            self.assertEqual(A.dtype, dtype)
            self.assertEqual(dpdy.dtype, dtype)
            self.assertEqual(dpdydy.dtype, dtype)
            
            # Check shapes
            self.assertEqual(A.shape, (k, k))
            self.assertEqual(dpdy.shape, (k, n))
            self.assertEqual(dpdydy.shape, (k, n, n))
            
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
            
            A, dpdy, dpdydy = lsqderiv(drdp, drdy, drdpdp)
            self.common_checks(A, dpdy, dpdydy, *H.shape, dtype)
            
            # Check values
            self.assertTrue(np.allclose(A, drdp.T @ drdp))
            self.assertTrue(np.allclose(dpdy, linalg.solve(A, H.T)))
            self.assertTrue(np.all(dpdydy == 0))
        
        def test_dtype(self):
            self.test_linear(np.float32)
        
        def test_generic(self):
            p = np.random.randn(3)
            H = np.random.randn(5, len(p))
            f = lambda p: H @ p + (H @ p) ** 3
            
            y = f(p) + np.random.randn(H.shape[0])
            p, drdp, drdy, drdpdp = self.fit(f, p, y)
            A, dpdy, dpdydy = lsqderiv(drdp, drdy, drdpdp)
            self.common_checks(A, dpdy, dpdydy, *H.shape)
            
            p2, dpdy_num, dpdydy_num = self.fit_dd(f, p, y)
            assert np.allclose(p, p2)
            
            self.assertTrue(np.allclose(dpdy, dpdy_num))
            self.assertTrue(np.allclose(dpdydy, dpdydy_num, atol=1e-6))
    
    unittest.main()
