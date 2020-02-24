import numpy as np
from scipy import special

def stdmom(moments, compact_output=False, inplace=False):
    """
    Standardize moments of a distribution, i.e. transform them to have E[1] = 1,
    E[x] = 0, E[x^2] = 1. The computation is vectorized along the leading axes
    of `moments`. The computational complexity is O(n^2) where
    `n = moments.shape[-1]`.
    
    Parameters
    ----------
    moments : array (..., n)
        The array of moments. `moments[..., k]` is the kth moment, i.e. E[x^k].
    compact_output : bool (default False)
        If True, return an array with the same shape as `moments` containing
        normalization, mean and variance and standardized moments from 3
        onward, otherwise return separately an array with standardized moments
        from 0 to n-1 and three arrays for normalization, mean and variance.
    inplace : bool (default False)
        If True, the results are directly written in the input array. An error
        is raised if the input is not a numpy array.
    
    Returns
    -------
    stdmoments : array (..., n)
        The standardized moments. If `compact_output` is True,
        `stdmoments[..., [0, 1, 2]]` contains the normalization, the mean, and
        the variance, otherwise they are 1, 0, and 1. If `inplace` is True,
        this array is actually the modified input array.
    
    The following are returned only if `compact_output` is False (default):
    
    norm : scalar or array (...,)
        The normalization, i.e. `moments[..., 0]`.
    mean : scalar or array (...,)
        The mean, i.e. `moments[..., 1]` if `moments[..., 0] == 1`.
    var : scalar or array (...,)
        The variance, i.e. `moments[..., 2]` if `moments[..., 1] == 0` and
        `moments[..., 0] == 0`.
    """
    # Check input
    assert not np.isscalar(moments)
    original_moments = moments
    moments = np.asarray(moments)
    compact_output = bool(compact_output)
    inplace = bool(inplace)
    assert not inplace or moments is original_moments
    
    # Copy for out-of-place operations
    if not inplace and moments is original_moments:
        moments = np.copy(moments)
    
    # Normalization
    moments[..., 1:] /= moments[..., [0]]
    if not compact_output:
        norm = np.copy(moments[..., 0])
        moments[..., 0] = 1
    
    # Zero mean
    n = moments.shape[-1]
    r = np.arange(n)
    shiftpow = (-moments[..., [1]]) ** r
    for k in range(2, n):
        binpow = special.binom(k, r[:k + 1]) * moments[..., :k + 1] * shiftpow[..., k::-1]
        moments[..., k] = np.sum(binpow, axis=-1)
    if not compact_output:
        mean = np.copy(moments[..., 1])
        moments[..., 1] = 0
    
    # Unitary variance
    sdev_pow = moments[..., [2]] ** (r[3:] / 2)
    moments[..., 3:] /= sdev_pow
    if not compact_output:
        var = np.copy(moments[..., 2])
        moments[..., 2] = 1
    
    # Return
    if compact_output:
        return moments
    else:
        return moments, norm, mean, var
    
if __name__ == '__main__':
    import unittest
    
    def normal_moments(n):
        a = np.zeros(n)
        a[0] = 1
        a[2::2] = np.cumprod(2 * np.arange((n-1) // 2) + 1)
        return a
    
    def rescale(moments, scale):
        moments *= scale ** np.arange(moments.shape[-1])
    
    class TestStdMom(unittest.TestCase):
        
        def test_normal(self):
            mom = normal_moments(10)
            s, n, m, v = stdmom(mom)
            self.assertTrue(not (s is mom))
            self.assertTrue(np.all(s == mom))
            self.assertEqual(n, 1)
            self.assertEqual(m, 0)
            self.assertEqual(v, 1)
        
        def test_compact(self):
            mom = normal_moments(10)
            s = stdmom(mom, compact_output=True)
            self.assertTrue(not (s is mom))
            self.assertTrue(np.all(s[2:] == mom[2:]))
            self.assertEqual(s[0], 1)
            self.assertEqual(s[1], 0)
            self.assertEqual(s[2], 1)

        def test_vectorize(self):
            mom = np.empty((100, 10))
            mom[...] = normal_moments(10)
            scale = 1 + np.arange(100)
            rescale(mom, scale[:, None])
            s, n, m, v = stdmom(mom)
            self.assertTrue(np.allclose(s, mom[0]))
            self.assertTrue(np.all(n == 1))
            self.assertTrue(np.all(m == 0))
            self.assertTrue(np.allclose(v, scale ** 2))
        
        def test_inplace(self):
            mom = normal_moments(10)
            rescale(mom, np.pi)
            s, _, _, _ = stdmom(mom, inplace=True)
            self.assertTrue(s is mom)
            self.assertTrue(np.allclose(mom, normal_moments(len(mom))))
        
        def test_scalar(self):
            with self.assertRaises(AssertionError):
                stdmom(1)
    
    unittest.main()
