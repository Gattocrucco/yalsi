from autograd import numpy as np
from numpy.lib import format as nplf
import tqdm
import sys

sys.path.insert(0, '..')
import yalsi

M = int(sys.argv[1]) # number of monte carlo
N = 2 # number of parameters
def f(x, p):
    return p[0] * np.cos(x / p[1])
true_x = np.linspace(0, 30, 10)
true_par = np.array([10, 4])

##################################

true_y = f(true_x, true_par)

np.savez(
    'test1-info.npz',
    true_x=true_x,
    true_par=true_par,
    true_y=true_y
)

table = nplf.open_memmap('test1.npy', mode='w+', shape=(M,), dtype=[
    ('success', bool),
    ('estimate', float, N),
    ('bias', float, N),
    ('cov', float, (N, N)),
    ('data_y', float, len(true_x)),
    ('data_x', float, len(true_x)),
    ('complete_estimate', float, N + len(true_x)),
    ('complete_bias', float, N + len(true_x)),
    ('complete_cov', float, (N + len(true_x), N + len(true_x))),
    ('grad', float, (N + len(true_x), 2 * len(true_x))),
    ('hessian', float, (N + len(true_x), 2 * len(true_x), 2 * len(true_x))),
])
table['success'] = False

def mu(params):
    p = params[:N]
    x = params[N:]
    return np.concatenate([x, f(x, p)])

for i in tqdm.tqdm(range(M)):
    data_x = true_x + np.random.randn(len(true_x))
    data_y = true_y + np.random.randn(len(true_x))
    data = np.concatenate([data_x, data_y])
    
    p0 = np.concatenate([true_par, true_x])
    fit = yalsi.fit(mu, data, p0, sdev=1)
        
    table[i]['estimate'] = fit.p[:N]
    table[i]['bias'] = fit.pbias[:N]
    table[i]['cov'] = fit.pcov[:N, :N]
    table[i]['data_y'] = data_y
    table[i]['data_x'] = data_x
    table[i]['complete_estimate'] = fit.p
    table[i]['complete_bias'] = fit.pbias
    table[i]['complete_cov'] = fit.pcov
    table[i]['grad'] = fit.pgrad
    table[i]['hessian'] = fit.phessian
    table[i]['success'] = True

    table.flush()

del table
