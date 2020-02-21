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
    # diagonal hessian with drdpdp.shape = (n, k)
    # compute gradient only with drdpdp=None by default
    # broadcasting
    # vectorization
    # replace various einsums with direct op
    # remove sign changes and zero term
    # decompose A only once
    # avoid duplicate off-diagonal in computing dpdydy
    
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
    C = C.reshape(len(p0), len(data) ** 2)
    dpdydy = linalg.solve(A, C, assume_a='pos')
    dpdydy = dpdydy.reshape(len(p0), len(data), len(data))
    
    return A, dpdy, dpdydy
    