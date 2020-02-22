import numpy as np

def numdiff(f, x, step=None):
    """
    Compute jacobian and hessian with second order finite central differences.
    
    Parameters
    ----------
    f : function R^n -> R^m 
    x : array (n,)
    step : float or None
        If None, a default value is used which is reasonable if the function
        and its derivatives are O(1) and all the digits in the result are
        accurate.
    
    Return
    ------
    jacobian : array (m, n)
    hessian : array (m, n, n)
    """
    x = np.asarray_chkfinite(x)
    assert len(x.shape) == 1
    
    if step is None:
        step = np.finfo(x.dtype).eps ** (1/4)
    else:
        assert np.isfinite(step)
        assert step > 0
        
    fcc = f(x)
    
    jac = np.empty((len(fcc), len(x)))
    hess = np.empty((len(fcc), len(x), len(x)))
    
    fcc2 = 2 * fcc
    step2 = 2 * step
    stepsq = step ** 2
    stepsq2 = 2 * stepsq
    
    a = np.array(x)
    for i in range(len(x)):
        a[i] = x[i] + step
        frc = f(a)
        a[i] = x[i] - step
        flc = f(a)
        a[i] = x[i]
        
        jac[:, i] = (frc - flc) / step2
        
        frclc = frc + flc
        hess[:, i, i] = (frclc - fcc2) / stepsq
        
        for j in range(i + 1, len(x)):
            a[i] = x[i] + step
            a[j] = x[j] + step
            frr = f(a)
            a[i] = x[i]
            fcr = f(a)
            a[j] = x[j] - step
            fcl = f(a)
            a[i] = x[i] - step
            fll = f(a)
            a[i] = x[i]
            a[j] = x[j]
            
            fcrcl = fcr + fcl
            hess[:, i, j] = (frr + fll - frclc - fcrcl + fcc2) / stepsq2
            hess[:, j, i] = hess[:, i, j]
    
    return jac, hess

if __name__ == '__main__':
    from autograd import numpy as np
    import autograd
    import unittest
    
    class TestNumdiff(unittest.TestCase):
        
        def checkfun(self, f, x):
            jac = autograd.jacobian(f)(x)
            hess = autograd.hessian(f)(x)
            jac_num, hess_num = numdiff(f, x)
            self.assertEqual(jac.shape, jac_num.shape)
            self.assertEqual(hess.shape, hess_num.shape)
            self.assertTrue(np.all(hess_num == np.swapaxes(hess_num, 1, 2)))
            self.assertTrue(np.allclose(jac, jac_num))
            self.assertTrue(np.allclose(hess, hess_num, atol=1e-6))
        
        def test_linear(self):
            x = np.random.randn(10)
            H = np.random.randn(5, 10)
            f = lambda x: H @ x
            self.checkfun(f, x)
        
        def test_quadratic(self):
            x = np.random.randn(10)
            Q = np.random.randn(5, 10, 10)
            f = lambda x: np.einsum('ijk,j,k', Q, x, x)
            self.checkfun(f, x)
        
        def test_generic(self):
            x = np.random.randn(10)
            H = np.random.randn(5, 10)
            f = lambda x: np.cos(H @ x)
            self.checkfun(f, x)
    
    unittest.main()
