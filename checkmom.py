import numpy as np
from scipy import special

def checkmom(moments, threshold=0):
    """
    Check that the provided array is a valid sequence of moments for a
    probability distribution. The computational complexity is O(n^3), where
    `n = moments.shape[-1]`. The computation is vectorized along the leading
    dimensions of `moments`.
    
    For numerical stability, you should standardize the moments to E[1] = 1,
    E[x] = 0 and E[x^2] = 1.
    
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
    
    # Check moments
    r = np.arange(moments.shape[-1])
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
