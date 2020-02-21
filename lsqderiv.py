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
    C -= 2 * np.einsum('abk,bq->akq', dgdpdy, dpdy)
    C -= np.einsum('abg,bk,gq->akq', dgdpdp, dpdy, dpdy)
    C = C.reshape(k, n * n)
    dpdydy = linalg.solve(A, C, assume_a='pos')
    dpdydy = dpdydy.reshape(k, n, n)
    
    return A, dpdy, dpdydy

if __name__ == '__main__':
    import unittest
    from autograd import numpy as np
    import autograd
    from scipy import optimize, linalg
    
    class TestLSQDeriv(unittest.TestCase):
        
        def fit(self, f, true_p, compute_derivs=True):
            def r(p, y):
                return y - f(p)
            jac = autograd.jacobian(r, 0)
            true_y = f(true_p)
            y = true_y + np.random.randn(*true_y.shape)
            result = optimize.least_squares(r, true_p, jac=jac, args=(y,))
            assert result.success
            
            if not compute_derivs:
                return result.x
            
            drdp = result.jac
            jacdata = autograd.jacobian(r, 1)
            hess = autograd.hessian(r, 0)
            drdy = jacdata(result.x, y)
            assert np.all(drdy == np.eye(len(y)))
            drdpdp = hess(result.x, y)
            return result.x, drdp, drdy, drdpdp
        
        def test_linear(self):
            p = np.random.randn(10)
            H = np.random.randn(100, len(p))
            f = lambda p: H @ p
           
            p, drdp, drdy, drdpdp = self.fit(f, p)
            assert np.all(drdpdp == 0)
            assert np.all(drdp == H) or np.all(drdp == -H)
            
            A, dpdy, dpdydy = lsqderiv(drdp, drdy, drdpdp)
            
            # # Check dtypes
            # self.assertEqual(A.dtype, dtype)
            # self.assertEqual(dpdy.dtype, dtype)
            # self.assertEqual(dpdydy.dtype, dtype)
            
            # Check shapes
            self.assertEqual(A.shape, (len(p), len(p)))
            self.assertEqual(dpdy.shape, (len(p), H.shape[0]))
            self.assertEqual(dpdydy.shape, (len(p), H.shape[0], H.shape[0]))
            
            # Check symmetries
            self.assertTrue(np.allclose(A, A.T))
            self.assertTrue(np.allclose(dpdydy, np.swapaxes(dpdydy, 1, 2)))
            
            # Check values
            self.assertTrue(np.allclose(A, drdp.T @ drdp))
            self.assertTrue(np.allclose(dpdy, linalg.solve(A, H.T)))
            self.assertTrue(np.all(dpdydy == 0))
    
    unittest.main()
