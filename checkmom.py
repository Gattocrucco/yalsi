import numpy as np
from scipy import special

def checkmom(moments, threshold=0, standardize=True):
    """
    Check that the provided array is a valid sequence of moments for a
    probability distribution. The computational complexity is O(n^3), where
    `n = moments.shape[-1]`. The computation is vectorized along the leading
    dimensions of `moments`.
    
    Parameters
    ----------
    moments : array (..., n)
        The array of moments. `moments[..., k]` is the kth moment, i.e. E[x^k].
    threshold : scalar or array (default 0)
        Due to finite numerical precision, moments that are close to the
        boundary of the allowed moments will give imprecise results. Pass a
        positive threshold to be sure that moments are ok, and a negative one
        to be sure that ok moments are not rejected. The threshold should be
        a small number. If `threshold` is an array, it is broadcasted against
        the leading axes of `moments`.
    standardize : bool (default True)
        Standardize the moments to E[1] = 1, E[x] = 0, E[x^2] = 1 before
        computation for numerical stability. The result does not change up to
        numerical precision, pass False only if the moments are already
        standard or near-standard to save computational time.
    
    Returns
    -------
    momentsok : array (...,) of bool
        Boolean saying if the moments are realizable, vectorizing along the
        leading axes of `moments`.
    """
    # Check input
    moments = np.asarray_chkfinite(moments)
    assert not np.isscalar(moments)
    threshold = np.asarray_chkfinite(threshold)
    np.broadcast(threshold, moments[..., 0])
    standardize = bool(standardize)
    
    n = moments.shape[-1]
    r = np.arange(n)
    
    # Standardize
    if standardize:
        # Normalization
        moments /= moments[..., [0]]
        
        # Zero mean
        shiftpow = (-moments[..., [1]]) ** r
        moments[..., 1] = 0
        for k in range(2, n):
            binpow = special.binom(k, r[:k + 1]) * moments[..., :k + 1] * shiftpow[..., k::-1]
            moments[..., k] = np.sum(binpow, axis=-1)
        
        # Unitary variance
        sdev_pow = moments[..., [2]] ** (r / 2)
        moments /= sdev_pow
    
    # Check moments
    indices = r[None, ...] + r[..., None]
    hankel = moments[..., indices]
    eigvals = np.linalg.eigvalsh(hankel)
    cond = np.min(eigvals, axis=-1) / np.max(eigvals, axis=-1)
    
    return cond >= threshold

if __name__ == '__main__':
    import unittest
    from scipy import stats
    
    class TestCheckMom(unittest.TestCase):
        pass
    
    unittest.main()
