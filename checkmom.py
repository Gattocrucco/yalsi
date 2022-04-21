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
        to be sure that ok moments are not rejected. The threshold should be a
        small number. If `threshold` is an array, it is broadcasted against the
        leading axes of `moments`.
    
    Returns
    -------
    momentsok : array (...,) of bool
        Boolean saying if the moments are realizable, vectorizing along the
        leading axes of `moments`.
    """
    # TODO: is there a specific better way of checking if a hankel matrix is
    # positive definite?
    # Problem: in general positive definite hankel matrices are ill-conditioned.
    # What are the implications for moment checking? That with many (where
    # "many" < 100) moments the checking is inaccurate?
    # Possible idea: standardize in a way that makes the moments more uniform
    # instead of E[x^2] = 1. At some point it would stop working because for
    # example the normal moments go like k! while the scaling is s^k.
    # See https://doi.org/10.1016/0377-0427(95)00108-5, who claims a
    # preconditioning that obtains linear condition in n. I don't have clear
    # how I check positive definiteness through a conditioning, think it over.
    
    # Check input
    assert not np.isscalar(moments)
    moments = np.asarray_chkfinite(moments)
    threshold = np.asarray_chkfinite(threshold)
    np.broadcast(threshold, moments[..., 0])
    
    # Check moments
    r = np.arange((1 + moments.shape[-1]) // 2)
    indices = r[None, ...] + r[..., None]
    hankel = moments[..., indices]
    eigvals = np.linalg.eigvalsh(hankel)
    cond = np.min(eigvals, axis=-1) / np.max(eigvals, axis=-1)
    # TODO handle the case max(eigvals) < 0
    
    return cond >= threshold

if __name__ == '__main__':
    import unittest
    import tqdm
    from scipy.stats import *
    
    distributions = [
        anglit,
        arcsine,
        (argus, 1),
        (beta, 1, 2),
        (bradford, 1),
        (chi, 4),
        (chi2, 10),
        cosine,
        (dgamma, 2),
        (dweibull, 2),
        expon,
        (exponnorm, 1.5),
        (exponweib, 2, 3),
        (exponpow, 2),
        (fatiguelife, 1),
        (foldnorm, 3),
        (genlogistic, 2),
        (gennorm, 3.4),
        (genexpon, 1, 2, 3),
        (gamma, 3),
        (gengamma, 2, 3),
        gilbrat,
        (gompertz, 2),
        gumbel_r,
        gumbel_l,
        halfnorm,
        (halfgennorm, 3),
        hypsecant,
        (invgauss, 1),
        laplace,
        logistic,
        (loggamma, 1.2),
        maxwell,
        moyal,
        (nakagami, 0.9),
        norm,
        (norminvgauss, 2, 1),
        (pearson3, 1),
        (reciprocal, 1, 2),
        rayleigh,
        (rice, 1.2),
        semicircular,
        (skewnorm, -0.5),
        (t, 10),
        (trapz, 0.2, 0.7),
        (triang, 0.3),
        (truncexpon, 2.9),
        (truncnorm, -2, 1.5),
        uniform,
        wald,
        (weibull_min, 2.2),
        (weibull_max, 1.3),
        (wrapcauchy, 0.4)
    ]
    
    def normal_moments(n):
        a = np.zeros(n)
        a[0] = 1
        a[2::2] = np.cumprod(2 * np.arange((n-1) // 2) + 1)
        return a
    
    def rescale(moments, scale):
        moments *= scale ** np.arange(moments.shape[-1])
    
    class TestCheckMom(unittest.TestCase):
        
        def test_parity(self):
            checkmom([1., 0, 1, 0, 3])
            checkmom([1., 0, 1, 0, 3, 0])
        
        def test_nonfinite(self):
            with self.assertRaises(ValueError):
                checkmom([np.nan, 0, 1])
            with self.assertRaises(ValueError):
                checkmom([1, 0, np.inf])
            with self.assertRaises(ValueError):
                checkmom([1, 0, 1], np.nan)
        
        def test_scalar(self):
            with self.assertRaises(AssertionError):
                checkmom(1)
        
        def test_broadcast(self):
            mom = np.empty((100, 11))
            mom[...] = normal_moments(11)
            threshold = np.zeros(100)
            threshold[50:] = 1
            ok = checkmom(mom, threshold)
            self.assertTrue(np.all(ok[:50]))
            self.assertFalse(np.any(ok[50:]))
        
        def test_delta(self):
            self.assertTrue(checkmom([1, 0, 0], -1e-12))
            self.assertFalse(checkmom([1, 0, 0], 1e-12))
            self.assertTrue(checkmom([1, 0, 1, 0, 1], -1e-12))
            self.assertFalse(checkmom([1, 0, 1, 0, 1], 1e-12))
        
        def test_scipy_dists(self):
            all_ok = True
            for distr in tqdm.tqdm(distributions):
                if isinstance(distr, tuple):
                    d = distr[0](*distr[1:])
                    name = distr[0].name
                else:
                    d = distr
                    name = distr.name
                
                mu = d.mean()
                s = d.std()
                sm = [1, 0, 1]
                for k in range(3, 6 + 1):
                    central_moment = d.expect(lambda x: (x - mu) ** k)
                    std_moment = central_moment / s ** k
                    sm.append(std_moment)
                
                ok = checkmom(sm)
                all_ok = all_ok and ok
                if not ok:
                    print(f'{name} not ok')
            
            self.assertTrue(all_ok)
    
    unittest.main()
