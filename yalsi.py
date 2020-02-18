import autograd
from autograd import numpy as np
from autograd.scipy import linalg
from scipy import optimize

class fit:

    def __init__(
            self,
            function, data, p0,
            sdev=None, variance=None, covariance=None,
            stdmoments=None,
            args=(), kwargs={}
        ):
        """
        Parameters
        ----------
        function : a function
            The signature must be function(parameters array, *args, **kwargs).
            It must return an array with shape (N,).
        data : 1D array shape (N,)
        p0 : 1D array of shape (M,)
            Initial estimate of the parameters to start minimization.
        sdev : None or 1D array shape (N,) or scalar
            Standard deviation of data. If a scalar, it is broadcasted to the
            shape of `data`.
        variance : None or 1D array shape (N,) or scalar
            Variance of data. If a scalar, it is broadcasted to the shape of
            `data`.
        covariance : None or 2D array shape (N, N)
            Covariance matrix of data. Only one of `sdev`, `variance` or
            `covariance` can be specified.
        stdmoments : None or 2D array shape (9, N) or (9, 1) or (9,).
            The standardized moments of data from 0 to 8. If not specified,
            normality is assumed.
        args : tuple
            Additional arguments passed when calling `function`.
        kwargs : dict
            Additional keyword arguments passed when calling `function`.
        """
    
        ###### Check arguments ######
    
        assert sum(map(lambda x: x is None, [sdev, variance, covariance])) == 2
        
        data = np.asarray_chkfinite(data, dtype=float)
        assert len(data.shape) == 1
    
        p0 = np.asarray_chkfinite(p0, dtype=float)
        assert len(p0.shape) == 1
        assert len(p0) <= len(data)
    
        if not (sdev is None):
            sdev = np.asarray_chkfinite(sdev, dtype=float)
            assert sdev.shape in {data.shape, (1,), ()}
            assert np.all(sdev > 0)
    
        if not (variance is None):
            variance = np.asarray_chkfinite(variance, dtype=float)
            assert variance.shape in {data.shape, (1,), ()}
            assert np.all(variance > 0)
            sdev = np.sqrt(variance)
    
        if not (covariance is None):
            covariance = np.asarray_chkfinite(covariance, dtype=float)
            assert covariance.shape == 2 * data.shape
            assert np.all(np.diag(covariance) > 0)
            assert np.allclose(covariance.T, covariance)
    
        if stdmoments is None:
            stdmoments = [
                1,
                0, 1,
                0, 3,
                0, 3 * 5,
                0, 3 * 5 * 7
            ]
    
        stdmoments = np.asarray_chkfinite(stdmoments, dtype=float)
        assert stdmoments.shape in {(9, len(data)), (9, 1), (9,)}
        if stdmoments.shape == (9,):
            stdmoments = stdmoments.reshape(9, 1)
        assert np.allclose(stdmoments[0], 1)
        assert np.allclose(stdmoments[1], 0)
        assert np.allclose(stdmoments[2], 1)
    
        indices = np.arange(5).reshape(1, -1) + np.arange(5).reshape(-1, 1)
        hankel = stdmoments.T[:, indices]
        eigvals = np.linalg.eigvalsh(hankel)
        assert np.all(eigvals >= 0)
    
        ###### Do minimization ######
    
        if not (sdev is None):
            def fun(parameters, data, *args, **kw):
                return (data - function(parameters, *args, **kw)) / sdev
    
        else:
            L = linalg.cholesky(covariance)
            def fun(parameters, data, *args, **kw):
                residuals = data - function(parameters, *args, **kw)
                return linalg.solve_triangular(L, residuals)
    
        jac = autograd.jacobian(fun, 0)
        result = optimize.least_squares(fun, p0, jac=jac, args=(data,) + args, kwargs=kwargs)
        assert result.success
    
        ###### Compute output ######
    
        par = result.x
        hessian = result.jac.T @ result.jac
        cov = linalg.solve(hessian, np.eye(len(p0)), assume_a='pos')
    
        hess = autograd.jacobian(jac, 0)
        jacdata = autograd.jacobian(fun, 1)
    
        # r: residuals, p: parameters, y: data
        drdp = result.jac
        drdy = jacdata(result.x, data, *args, **kwargs)
        drdpdp = hess(result.x, data, *args, **kwargs)
    
        # g: gradient of the cost function 1/2 * r.T @ r
        dgdp = np.einsum('ia,ib->ab', drdp, drdp)
        # dgdp += np.einsum('i,iab->ab', result.fun, drdpdp)
        dgdy = np.einsum('ia,ik->ak', drdp, drdy)
        dgdydy = 0
        dgdpdy = np.einsum('iab,ik->abk', drdpdp, drdy)
        dr3 = np.einsum('ia,ibg->abg', drdp, drdpdp)
        dgdpdp = dr3 + np.einsum('bga', dr3) + np.einsum('gab', dr3)
        # dgdpdp += np.einsum('i,iabg->abg', result.fun, dgdpdpdp)
    
        A = dgdp
        B = -dgdy
        dpdy = linalg.solve(A, B, assume_a='pos')
    
        C = -dgdydy
        C -= 2 * np.einsum('abk,bq->akq', dgdpdy, dpdy)
        C -= np.einsum('abg,bk,gq->akq', dgdpdp, dpdy, dpdy)
        C = C.reshape(len(p0), len(data) ** 2)
        dpdydy = linalg.solve(A, C, assume_a='pos')
        dpdydy = dpdydy.reshape(len(p0), len(data), len(data))
    
        bias = 1/2 * np.einsum('akq,kq->a', dpdydy, covariance)
        
        ###### Save variables in instance ######
        
        self.p = par
        self.pcov = cov
        self.pbias = bias
