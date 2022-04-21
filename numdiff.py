import numpy as np

def numdiff(f, x, estjac=None, esthess=None):
    """
    Compute jacobian and hessian with second order finite central differences.
    
    Parameters
    ----------
    f : function R^n -> R^m 
    x : array (n,)
    
    Return
    ------
    jacobian : array (m, n)
    hessian : array (m, n, n)
    """
    # TODO:
    # broadcast (scalars too)
    
    x = np.asarray_chkfinite(x)
    assert len(x.shape) == 1
    assert np.issubdtype(x.dtype, np.floating)
    
    eps = np.finfo(x.dtype).eps
    
    fcc = f(x)
    
    jac = np.empty((len(fcc), len(x)))
    hess = np.empty((len(fcc), len(x), len(x)))
    
    if not (estjac is None):
        estjac = np.ones(jac.shape) * estjac # TODO use broadcast_to
        assert np.all(np.isfinite(estjac))
    if not (esthess is None):
        esthess = np.ones(hess.shape) * esthess
        assert np.all(np.isfinite(esthess))
        
    a = np.array(x)
    for k in range(len(fcc)):
        
        for i in range(len(x)):
            sj = (eps * (1 if estjac is None else np.abs(fcc[k] / estjac[k, i]))) ** (1/3)
        
            a[i] = x[i] + sj
            fr = f(a)[k]
            a[i] = x[i] - sj
            fl = f(a)[k]
            a[i] = x[i]
        
            jac[k, i] = (fr - fl) / (2 * sj)
        
            sh = (eps * (1 if esthess is None else np.abs(fcc[k] / esthess[k, i, i]))) ** (1/4)
        
            a[i] = x[i] + sh
            fr = f(a)[k]
            a[i] = x[i] - sh
            fl = f(a)[k]
            a[i] = x[i]
            hess[k, i, i] = (fr + fl - 2 * fcc[k]) / sh ** 2
        
            for j in range(i + 1, len(x)):
                sh = (eps * (1 if esthess is None else np.abs(fcc[k] / esthess[k, i, j]))) ** (1/4)
            
                a[i] = x[i] + sh
                frc = f(a)[k]
                a[j] = x[j] + sh
                frr = f(a)[k]
                a[i] = x[i]
                fcr = f(a)[k]
                a[j] = x[j] - sh
                fcl = f(a)[k]
                a[i] = x[i] - sh
                fll = f(a)[k]
                a[j] = x[j]
                flc = f(a)[k]
                a[i] = x[i]
            
                frclc = frc + flc
                fcrcl = fcr + fcl
                hess[k, i, j] = (frr + fll - frclc - fcrcl + 2 * fcc[k]) / (2 * sh ** 2)
                hess[k, j, i] = hess[k, i, j]
    
    return jac, hess

if __name__ == '__main__':
    from autograd import numpy as np
    import autograd
    import unittest
    from matplotlib import pyplot as plt
    
    f = lambda x: 2 + np.sin(x) - np.cos(x)
    steps = np.logspace(-15, 0, 100)
    result = np.empty((2, len(steps)))
    for i in range(len(steps)):
        s = steps[i]
        result[0, i] = (f(s) - f(-s)) / (2 * s)
        result[1, i] = (f(s) + f(-s) - 2 * f(0)) / s ** 2
    
    fig = plt.figure('numdiff')
    fig.clf()
    axs = fig.subplots(2, 1)
    for i, ax in enumerate(axs):
        ax.plot(steps, np.abs(result[i] - 1))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.axvline(np.finfo(float).eps ** (1/(3 + i)))
        ax.set_title(f'derivative {i + 1}')
    fig.tight_layout()
    fig.show()
    
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
