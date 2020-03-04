from sympy.combinatorics.partitions import IntegerPartition
from sympy.combinatorics import Permutation
import copy
import numpy as np
from fractions import Fraction
from collections import Counter, OrderedDict, defaultdict
import itertools

__doc__ = """
Usage examples:
>>> import gencorr
>>> gencorr.Corr('','').corr().p # second order variance
>>> gencorr.Corr('a','b').corr().p # s.o. covariance
>>> gencorr.Corr('','','','').corr().p # s.o. 4th moment
The output from first line (variance) is:
    G_i G_i V_ii + 
    2 G_i H_ii V_iii + 
    H_ii H_ii V_iiii + 
    2 H_ij H_ij V_iijj + 
    H_ii H_jj V_iijj
G stands for gradient, H for half hessian, V for moment. The indices are summed
over, with the constraint that different indices can not yield the same value
(but we are not summing over unique sets of indices, each index runs from
start to finish).
"""

indices = ('i', 'j', 'k', 'l', 'm', 'n', 'o', 'p')

def isnum(x):
    return isinstance(x, int) or isinstance(x, Fraction)

class Expression:
    
    def recursive(self, method, *args):
        """
        Recursively apply
            obj = obj.method(*args)
        on all the objects in the expression, starting from the leaves.
        """
        # print(f'recursive {method} on {self}')
        if hasattr(self, '_list'):
            for i in range(len(self._list)):
                if hasattr(self._list[i], 'recursive'):
                    self._list[i] = self._list[i].recursive(method, *args)
                elif hasattr(self._list[i], method):
                    # print(f'    {method} on {self._list[i]}')
                    self._list[i] = getattr(self._list[i], method)(*args)
        if hasattr(self, method):
            # print(f'    {method} on {self}')
            self = getattr(self, method)(*args)
        return self

class Tensor(Expression):
    
    def __init__(self, rank, indices=()):
        """
        rank = number of indices
        indices = tuple of integers. can be empty, can have repetitions
        """
        assert isinstance(rank, int) and rank >= 0
        self._rank = rank
        self._indices = tuple(indices)
        assert not self._indices or len(self._indices) == self._rank
        assert all(isinstance(x, int) and 0 <= x < 8 for x in self._indices)
    
    def sort_indices(self):
        self._indices = tuple(sorted(self._indices))
        return self
    
    def clear_indices(self):
        self._indices = ()
        return self
    
    def __lt__(self, d):
        if isinstance(d, Tensor):
            return self._rank < d._rank or self._rank == d._rank and self._indices < d._indices
        elif isnum(d):
            return False
        else:
            return NotImplemented
    
    def __gt__(self, d):
        if isinstance(d, Tensor):
            return self._rank > d._rank or self._rank == d._rank and self._indices > d._indices
        elif isnum(d):
            return True
        else:
            return NotImplemented

    def __eq__(self, obj):
        return isinstance(obj, type(self)) and self._rank == obj._rank and self._indices == obj._indices
    
    def __repr__(self, prefix='T'):
        if self._indices:
            rankstr = ''
            indstr = ''.join(map(lambda i: indices[i], self._indices))
        else:
            indstr = ''
            rankstr = f'{self._rank}'
        return f'{prefix}{rankstr}{indstr}'
    
class V(Tensor):
    """
    Represent an expectation over a product of independent zero-mean variables.
    """
    
    def __repr__(self):
        return super().__repr__('V')
    
    def index_V(self):
        """
        Split into a summation over products of moments of the unique variables
        in the product.
        """
        ilist = gen_ordered_groupings(self._rank, no_loners=True)
        # We skip loners because E[x_i] = 0. We need at least two equal
        # variables in the product to have a nonzero moment.
        return Sum(*[V(self._rank, indices) for indices in ilist])
    
    def separate_V(self):
        """
        Separate into a product of V for different indices, assuming unitary
        variance.
        """
        assert self._indices
        c = Counter(self._indices)
        return Mult(*[
            0 if count == 1
            else 1 if count == 2
            else V(count, (idx,) * count)
            for idx, count in c.items()
        ])
    
    def varname(self):
        return f'v{self._rank}'
    
    def python(self):
        return f'V[{self._rank}]'

def gen_ordered_groupings(rank, no_loners=True):
    p = IntegerPartition([rank])
    end = IntegerPartition([1] * rank)
    ilist = []
    while True:
        if not no_loners or p.as_dict().get(1, 0) == 0:
            indices_bags = dict()
            base_index = 0
            for group_size, group_count in p.as_dict().items():
                bags = list(() for _ in range(group_size))
                bags[-1] = tuple(base_index + j for j in range(group_count))
                base_index += group_count
                indices_bags[group_size] = bags
            gen_index_perm(ilist, rank, (), indices_bags)
        if p == end:
            break
        p = p.prev_lex()
    return ilist

def gen_index_perm(ilist, rank, indices, indices_bags):
    # ind = ' ' * 4 * len(indices)
    # print(f'{ind}gen_index_perm({len(ilist)}, {rank}, {indices}, {indices_bags})')
    if len(indices) == rank:
        ilist.append(indices)
        return
    
    for group_size, bags in indices_bags.items():
        # print(f'{ind} {group_size}: {bags}')
        assert len(bags) == group_size
        for i, bag in enumerate(bags):
            # print(f'{ind}  {i}: {bag}')
            for j in range(len(bag)):
                # print(f'{ind}   {j}: {bag[j]}')
                new_indices_bags = copy.deepcopy(indices_bags)
                new_indices_bags[group_size][i] = bag[:j] + bag[j + 1:]
                if i > 0:
                    new_indices_bags[group_size][i - 1] = prev_bag + bag[j:j + 1]
                gen_index_perm(ilist, rank, indices + bag[j:j + 1], new_indices_bags)
                if i == len(bags) - 1:
                    break
            prev_bag = bag

class D(Tensor):
    """
    Represents a derivatives tensor.
    """
    
    def __init__(self, rank, var, indices=()):
        """
        rank = order of derivation, e.g. 1 = gradient, 2 = hessian
        var = arbitrary object used as label to identify what is being derived
        indices = tuple of indices for the tensor, default empty
        """
        super().__init__(rank, indices)
        assert self._rank <= 2
        self._var = var
    
    def __repr__(self):
        letter = 'G' if self._rank == 1 else 'H'
        if self._indices:
            indstr = ''.join(map(lambda i: indices[i], self._indices))
        else:
            indstr = ''
        if str(self._var) == '':
            varstr = ''
        elif self._indices:
            varstr = f'{self._var}_'
        else:
            varstr = f'{self._var}'
        return f'{letter}{varstr}{indstr}'
    
    def __lt__(self, d):
        if isinstance(d, D):
            return self._var < d._var or self._var == d._var and super().__lt__(d)
        else:
            return super().__lt__(d)
    
    def __eq__(self, obj):
        return super().__eq__(obj) and self._var == obj._var
    
    def assume_diagonal(self):
        """
        Return self if all the indices are equal, otherwise 0.
        """
        return self if len(set(self._indices)) <= 1 else 0
    
    def varname(self):
        return {1: 'g', 2: 'h'}[self._rank] + str(self._var)
    
    def python(self):
        return {1: 'G', 2: 'H'}[self._rank] + str(self._var)
    
    def gather_vars(self, vars):
        vars.add(self._var)
        return self
    
    def as_J(self, index=None):
        """
        Decompose into J tensors with specified summation index.
        """
        assert self._rank == 2
        bracket = (index,) if index else ()
        a = J(0, self._var, (),                bracket)
        b = J(1, self._var, self._indices[:1], bracket)
        c = J(1, self._var, self._indices[1:], bracket)
        return Mult(a, b, c)

class J(D):
    """
    Represent a part of the decomposition of a D tensor. Example with D(2):
    
    Hij = J[k]i D[k] J[k]j
    
    The "bracket" index is separated from the normal index and is not
    considered an index of the tensor.
    """
    
    def __init__(self, rank, var, indices=(), bracket=()):
        """
        rank
        var = like D's var
        indices = normal tensor indices (as much as rank)
        bracket = "internal" summed over indices, must be negative
        """
        super().__init__(rank, var, indices)
        assert self._rank <= 1
        self._bracket = tuple(bracket)
        assert all(isinstance(x, int) and x < 0 for x in self._bracket)
    
    def __repr__(self):
        letter = {0: 'D', 1: 'J'}[self._rank]
        
        if str(self._var) == '':
            varstr = ''
        elif self._indices and not self._bracket:
            varstr = f'{self._var}_'
        else:
            varstr = f'{self._var}'
        
        brastr = ''.join(indices[i] for i in self._bracket)
        if brastr:
            brastr = f'[{brastr}]'
        
        indstr = ''.join(indices[i] for i in self._indices)
        
        return f'{letter}{varstr}{brastr}{indstr}'
    
    def __lt__(self, d):
        if isinstance(d, J):
            return self._var < d._var or self._var == d._var and self._bracket < d._bracket or self._bracket == d._bracket and super().__lt__(d)
        else:
            return super().__lt__(d)
    
    def __eq__(self, obj):
        return super().__eq__(obj) and self._bracket == obj._bracket
    
    def varname(self):
        return {0: 'd', 1: 'j'}[self._rank] + str(self._var)
    
    def python(self):
        return {0: 'D', 1: 'J'}[self._rank] + str(self._var)
    
class Reductor(Expression):
    """
    Represent an associative operation.
    """
    
    def __init__(self, *args):
        self._list = list(args)

    def concat(self):
        """
        If there are reductors of the same kind in the operands, concatenate
        them.
        """
        l = []
        for obj in self._list:
            if isinstance(obj, self.__class__):
                l += obj._list
            else:
                l.append(obj)
        self._list = l
        return self
    
    def sort(self):
        try:
            self._list = sorted(self._list)
        except TypeError:
            pass
        return self
    
    def __lt__(self, obj):
        return isinstance(obj, Reductor) and self._list[::-1] < obj._list[::-1]
    
    def __gt__(self, obj):
        return not isinstance(obj, Reductor) or self._list[::-1] > obj._list[::-1]

    def __eq__(self, obj):
        return isinstance(obj, self.__class__) and len(self._list) == len(obj._list) and all(x == y for x, y in zip(self._list, obj._list))
    
    def __repr__(self):
        return ' '.join('(' + x.__repr__() + ')' if isinstance(x, Reductor) else x.__repr__() for x in self._list)
    
class Mult(Reductor):
    """
    Represent a product.
    """
    
    def reduce(self):
        """
        Multiply numbers, leaving alone other factors. If they multiply to
        zero, return zero. If they multiply to one, the factor is stripped. If
        the resulting product has only one element, return the element. If it
        is empty, return 1. Otherwise, return self.
        """
        numbers = []
        l = []
        for obj in self._list:
            if isnum(obj):
                numbers.append(obj)
            else:
                l.append(obj)
        p = 1
        for n in numbers:
            p *= n
        if int(p) == p:
            p = int(p)
        if p == 1:
            self._list = l
        elif p == 0:
            return 0
        else:
            self._list = [p] + l
        if len(self._list) > 1:
            return self
        elif len(self._list) == 1:
            return self._list[0]
        else:
            return 1
    
    def gen_V(self):
        """
        Add a V object to the product with rank equal to the sum of the ranks
        of the D objects.
        """
        rank = sum(map(lambda x: x._rank, filter(lambda x: isinstance(x, D), self._list)))
        if rank:
            self._list.append(V(rank))
        return self
    
    def expand(self):
        """
        Distribute all the sums in the product.
        """
        for i in range(len(self._list)):
            obj = self._list[i]
            if isinstance(obj, Sum):
                terms = []
                for t in obj._list:
                    l_left = copy.deepcopy(self._list[:i])
                    l_right = copy.deepcopy(self._list[i + 1:])
                    terms.append(Mult(*l_left, t, *l_right).expand())
                return Sum(*terms)
        return self
    
    def index_D_from_V(self):
        """
        Take indices from V objects, in order, and give them to D objects, in
        order.
        """
        indices = ()
        for obj in self._list:
            if isinstance(obj, V):
                indices += obj._indices
        
        i = 0
        for obj in self._list:
            if isinstance(obj, D):
                obj._indices = indices[i:i + obj._rank]
                i += obj._rank
        assert i == len(indices)
        
        return self
    
    def split_D(self):
        """
        Replace D objects with a sum over all permutations of indices of the
        D objects. Makes sense because the V object represents a symmetric
        tensor. NOOOOO! After pruning they are not totally symmetric!
        
        Currently not used anywhere.
        """
        Dobjs = []
        other = []
        for obj in self._list:
            if isinstance(obj, D):
                Dobjs.append(obj)
            else:
                other.append(obj)
        if not Dobjs:
            return self
        indices = ()
        for obj in Dobjs:
            indices += obj._indices
        pfirst = Permutation(list(range(len(indices))))
        p = pfirst
        terms = []
        while True:
            l = copy.deepcopy(Dobjs)
            i = 0
            for d in l:
                d._indices = tuple(indices[(i + j)^p] for j in range(len(d._indices)))
                i += len(d._indices)
            terms.append(Mult(*l))
            p += 1
            if p == pfirst:
                break
        self._list = other + [Fraction(1, len(terms)), Sum(*terms)]
        return self
    
    def normalize_D(self):
        """
        Canonicalize the usage of mute indices.
        
        BUUUUG: I'm not catching this symmetry:
        24 Hij Hil Hjk Hkl Viijjkkll + 
        24 Hij Hik Hjl Hkl Viijjkkll
        swapping k and l they are the same.
        Also this:
        H(a)ij G(b)j G(c)i Viijj + 
        H(a)ij G(b)i G(c)j Viijj
        """
        Dobjs = []
        for obj in self._list:
            if isinstance(obj, D):
                Dobjs.append(obj)
        if not Dobjs:
            return self
        indices = ()
        r2s_indices = ()
        for d in Dobjs:
            indices += obj._indices
            if d._rank == 2 and d._indices and d._indices[0] == d._indices[1]:
                r2s_indices += d._indices[:1]
        if not r2s_indices:
            return self
        # print(f'normalize_D on {self}')
        count = Counter(indices)
        count_count = Counter(sorted(count.values()))
        p = list(range(len(set(indices))))
        assert(set(indices) == set(range(len(set(indices)))))
        for c, cc in filter(lambda ccc: ccc[1] > 1, count_count.items()):
            swappable_indices = set(filter(lambda i: count[i] == c, indices))
            # print(f'    can swap {swappable_indices}')
            assert(len(swappable_indices) > 1)
            first_indices = tuple(filter(lambda i: i in swappable_indices, OrderedDict(zip(r2s_indices, (None,) *len(r2s_indices)))))
            second_indices = tuple(filter(lambda i: i in swappable_indices and not i in r2s_indices, OrderedDict(zip(indices, (None,) *len(indices)))))
            old_indices = first_indices + second_indices
            new_indices = sorted(swappable_indices)
            for i, j in zip(old_indices, new_indices):
                # print(f'        send {i} -> {j}')
                p[i] = j
        # print(f'    final perm is {p}')
        assert(set(p) == set(indices))
        for d in Dobjs:
            d._indices = tuple(p[i] for i in d._indices)
        return self
    
    def as_summation(self):
        """
        If indices are separated by tensor, transform to a Summation object.
        """
        if any(isinstance(x, Tensor) and len(set(x._indices)) > 1 for x in self._list):
            return self
        
        tensors_by_index = defaultdict(list)
        nonindexed = []
        for obj in self._list:
            if isinstance(obj, Tensor) and obj._rank >= 1:
                tensors_by_index[obj._indices[0]].append(obj)
            else:
                nonindexed.append(obj)
        return Mult(Summation(*[Mult(*l) for l in tensors_by_index.values()]), *nonindexed)
    
    def decompose_hessians(self):
        """
        Decompose D(2) tensors.
        """
        idx = -1
        for i, obj in enumerate(self._list):
            if isinstance(obj, D) and obj._rank == 2:
                self._list[i] = obj.as_J(idx)
                idx -= 1
        return self
    
    def varname(self):
        return ''.join(map(lambda x: x.varname(), self._list))
    
    def python(self):
        return ' * '.join(map(lambda x: str(x) if isnum(x) else x.python(), self._list))
    
def stripfactor(x):
    if isinstance(x, Mult) and x._list and isnum(x._list[0]):
        return Mult(*x._list[1:]) if len(x._list) >= 3 else x._list[1]
    else:
        return x

def getfactor(x):
    if isinstance(x, Mult) and x._list and isnum(x._list[0]):
        return x._list[0]
    else:
        return 1

class Sum(Reductor):
    
    def __repr__(self, sep=''):
        return f' + {sep}'.join('(' + x.__repr__() + ')' if isinstance(x, Sum) else x.__repr__() for x in self._list)
    
    @property
    def p(self):
        """
        Quick printer with newlines between terms
        """
        print(self.__repr__('\n'))
        
    def reduce(self):
        numbers = []
        l = []
        for obj in self._list:
            if isnum(obj):
                numbers.append(obj)
            else:
                l.append(obj)
        s = sum(numbers)
        if int(s) == s:
            s = int(s)
        if s == 0:
            self._list = l
        else:
            self._list = [s] + l
        if len(self._list) > 1:
            return self
        elif len(self._list) == 1:
            return self._list[0]
        else:
            return 0
    
    def harvest(self):
        """
        The name of this function should be `collect`. Just an Italian joke.
        """
        terms = []
        counts = []
        while self._list:
            obj = stripfactor(self._list[-1])
            count = getfactor(self._list[-1])
            for i in range(len(terms)):
                if terms[i] == obj:
                    counts[i] += count
                    break
            else:
                terms.append(obj)
                counts.append(count)
            self._list.pop()
        self._list = []
        # print(grouped_terms)
        for i in reversed(range(len(terms))):
            self._list.append(Mult(counts[i], terms[i]) if counts[i] != 1 else terms[i])
        return self

class Summation(Reductor):
    """
    Represent a summation where the summand is a product with indices separated
    and unique by factor.
    
    Example:
    gencorr.Summation(*'abcde').explode().recursive('expand')
    .recursive('concat').recursive('reduce').recursive('sort').harvest()
    .recursive('reduce')
    """
    
    def __repr__(self):
        return '∑ ' + super().__repr__()
    
    def explode(self):
        """
        Transform from a summation where items with overlapping indices are
        skipped to a sum of summations over full ranges.
        """
        if len(self._list) <= 1:
            return self
        
        ilist = gen_ordered_groupings(len(self._list), no_loners=False)
        
        terms = []
        for indices in ilist[:-1]:
            # [:-1] because the last element is all indices different
            objs = [[] for _ in range(max(indices) + 1)]
            for i, idx in enumerate(indices):
                objs[idx].append(copy.deepcopy(self._list[i]))
            groups = [Mult(*group) for group in objs]
            terms.append(Mult(-1, Summation(*groups).explode()))
        
        return Sum(self, *terms)
    
    def index_summation(self):
        """
        Remove indices from tensors in the summation since they are implicit.
        """
        for i, obj in enumerate(self._list):
            if isinstance(obj, Expression):
                self._list[i] = obj.recursive('clear_indices')
        return self
    
    def gather_summation_terms(self, bag):
        bag += self._list
        return self
    
    def python(self):
        return ' * '.join(map(lambda x: x.varname(), self._list))

class Corr:
    """
    Represent the correlation function for a list of quadratic functions.
    """
    
    def __init__(self, *vars):
        """
        *vars = list of arbitrary labels for the functions
        """
        assert 1 <= len(vars) <= 4
        self._vars = vars
        self._expr = None
    
    def corr(self):
        """
        Return the full correlation formula.
        """
        if self._expr:
            return copy.deepcopy(self._expr)
        
        e = Mult(*[Sum(D(1, v), D(2, v)) for v in self._vars])
    
        # do multiplication
        e = e.recursive('expand')
        e = e.recursive('concat')
    
        # group equal terms in case there are repeated variables
        e = e.recursive('sort')
        e = e.recursive('harvest')
    
        # put V tensors
        e = e.recursive('gen_V')
    
        # split summations into diagonal and off-diagonal
        e = e.recursive('index_V')
        e = e.recursive('expand')
        e = e.recursive('concat')
    
        # put indices on D tensors
        e = e.recursive('index_D_from_V')
        e = e.recursive('sort_indices')
        e = e.recursive('sort')
        e = e.recursive('harvest')
        e = e.recursive('concat')
        e = e.recursive('reduce')
    
        # normalize usage of mute indices to spot equal terms
        e = e.recursive('normalize_D')
        e = e.recursive('sort_indices')
        e = e.recursive('sort')
        e = e.recursive('harvest')
        e = e.recursive('concat')

        self._expr = e
        return copy.deepcopy(self._expr)

    def diag(self):
        """
        Return simplified formula assuming hessians are diagonal.
        """
        if len(self._vars) == 1:
            return Sum(Summation(D(2, self._vars[0])))
        
        e = self.corr()
        
        # remove off-diagonal hessians
        e = e.recursive('assume_diagonal').recursive('reduce')

        # separate moments by variable, e.g. Viijj = Vii Vjj
        e = e.recursive('separate_V').recursive('concat').recursive('reduce')

        # turn implicit summations into Summation objects
        e = e.recursive('as_summation').recursive('index_summation').recursive('reduce')

        # convert from summations over non overlapping indices to full range
        e = e.recursive('explode').recursive('expand').recursive('concat').recursive('reduce')

        # final simplification
        e = e.recursive('sort').recursive('harvest')
       
        return e

    def diagcode(self):
        """
        Return code for a function computing the formula given by diag().
        """    
        # collect all summation terms from the expression
        expr = self.diag()
        terms = []
        expr.recursive('gather_summation_terms', terms)
    
        # remove duplicate terms
        terms = [term for term, _ in itertools.groupby(sorted(terms))]
    
        # write function head
        fname = f'corrdiag{len(self._vars)}' + ''.join(sorted(map(str, self._vars)))
        fparams = ', '.join(f'G{v}, H{v}' for v in sorted(set(map(str, self._vars))))
        fhead = f'def {fname}(V, {fparams}):\n'
    
        # write code
        terms_code = ''.join(f'    {t.varname()} = np.sum({t.python()})\n' for t in terms)
        corr_code = ''.join(f'    c += {m.python()}\n' for m in expr._list)
    
        return fhead + terms_code + '\n    c = 0\n' + corr_code + '\n    return c\n'

if __name__ == '__main__':
    import unittest
    from scipy import special
    import checkmom
    import numpy as np
    
    np.random.seed(20200301)
    
    moments = np.array([
        [ 0.000000000000000e+00,  2.193750230045810e+00,  0.000000000000000e+00,  6.307464518351152e+00,  0.000000000000000e+00,  2.089842143605476e+01],
        [-1.374677761397093e-15,  1.500000000000648e+00, -2.855240539539878e-15,  2.500000000006676e+00, -5.710481079079755e-15,  4.375000000040987e+00],
        [-4.203979006661760e-01,  2.306106586517433e+00, -2.401538376980557e+00,  7.714387209802071e+00, -1.227409781769712e+01,  3.230234798950539e+01],
        [ 5.656854249492382e-01,  2.400000000000000e+00,  3.232488142567076e+00,  8.857142857142861e+00,  1.697056274847715e+01,  4.160000000000002e+01],
        [ 2.393417858997477e-01,  1.862753541523876e+00,  1.043117800515642e+00,  4.312283112177668e+00,  3.770708129670418e+00,  1.130784371593390e+01],
        [ 4.056950772626715e-01,  3.059295089399554e+00,  3.909491946824366e+00,  1.739515608079365e+01,  4.052581111168719e+01,  1.569803094595993e+02],
        [ 8.944271909999155e-01,  4.199999999999998e+00,  1.109089716839895e+01,  4.579999999999998e+01,  1.894396790537821e+02,  9.136399999999995e+02],
        [ 0.000000000000000e+00,  2.406237124401720e+00,  0.000000000000000e+00,  8.033107300728211e+00,  0.000000000000000e+00,  3.210895347595000e+01],
        [ 0.000000000000000e+00,  3.333333333333335e+00,  0.000000000000000e+00,  2.333333333333335e+01,  0.000000000000000e+00,  2.800000000000002e+02],
        [ 0.000000000000000e+00,  2.000000000000000e+00,  0.000000000000000e+00,  6.000000000000000e+00,  0.000000000000000e+00,  2.400000000000000e+01],
        [ 2.000000000000000e+00,  9.000000000000002e+00,  4.400000000000001e+01,  2.650000000000001e+02,  1.854000000000000e+03,  1.483300000000000e+04],
        [ 1.152069638313938e+00,  5.875739644970417e+00,  2.109173645528594e+01,  1.112266727355033e+02,  6.366991251413222e+02,  4.260154511396660e+03],
        [ 1.443018742885296e-01,  2.916620697907702e+00,  1.421872037810624e+00,  1.399773065059918e+01,  1.420477187613419e+01,  9.446377344841353e+01],
        [-7.996816282072437e-02,  2.355798704472346e+00, -3.353097872836812e-01,  7.639635809949493e+00, -1.081109795606021e+00,  2.965452075728869e+01],
        [ 2.518518518518572e+00,  1.285185185185180e+01,  7.713580246913584e+01,  5.702400548696919e+02,  4.964241426611791e+03,  4.980798308184728e+04],
        [ 1.879105754584661e-02,  2.940610025577012e+00,  3.302465491425197e-01,  1.390203748897559e+01,  4.842768251638423e+00,  8.848519792614236e+01],
        [ 5.771840025973655e-01,  4.332675511075347e+00,  8.796559191783279e+00,  4.866244906904266e+01,  1.906728110497943e+02,  1.112419348006789e+03],
        [ 0.000000000000000e+00,  2.305197417170408e+00,  0.000000000000000e+00,  7.532301401602690e+00,  0.000000000000000e+00,  3.066472285630520e+01],
        [ 1.558209836032660e+00,  6.780542855264994e+00,  2.825765501191453e+01,  1.502538636221089e+02,  9.177347339633483e+02,  6.419732973880627e+03],
        [ 1.154700538379252e+00,  5.000000000000003e+00,  1.616580753730953e+01,  7.166666666666670e+01,  3.452554609753964e+02,  1.896999999999746e+03],
        [ 5.978271350466868e-02,  2.857225515906674e+00,  6.392418424521026e-01,  1.304324619278369e+01,  6.650380648400196e+00,  8.072851306852142e+01],
        [ 6.184877138634218e+00,  1.139363921776051e+02,  5.215977014368461e+03,  6.194384868955223e+05,  1.949406911671746e+08,  1.647082722285542e+11],
        [ 1.010125734156103e+00,  3.753230654956891e+00,  9.260001561187531e+00,  3.066070273614897e+01,  1.021833487962241e+02,  3.730641831155965e+02],
        [ 1.139547099404653e+00,  5.399999999999991e+00,  1.856661598538669e+01,  9.141424734618681e+01,  4.931498916956295e+02,  3.091022944253579e+03],
        [-1.139547099404653e+00,  5.399999999999991e+00, -1.856661598538669e+01,  9.141424734618681e+01, -4.931498916956295e+02,  3.091022944253579e+03],
        [ 9.952717464311586e-01,  3.869177303606004e+00,  9.896966346585272e+00,  3.476177935541727e+01,  1.245213878107983e+02,  4.953978799440135e+02],
        [ 6.327758511185098e-01,  2.841515913822089e+00,  4.870427468322759e+00,  1.527386968129090e+01,  3.963720036171044e+01,  1.234311073294849e+02],
        [ 0.000000000000000e+00,  5.000000000000003e+00,  0.000000000000000e+00,  6.100000000000002e+01,  0.000000000000000e+00,  1.385000000000001e+03],
        [ 3.000000000000104e+00,  1.799999999999990e+01,  1.350000000000001e+02,  1.274999999999999e+03,  1.449000000000001e+04,  1.928850000000000e+05],
        [ 0.000000000000000e+00,  5.999999999999999e+00,  0.000000000000000e+00,  8.999999999999997e+01,  0.000000000000000e+00,  2.519999999999999e+03],
        [ 3.761250603026065e-16,  4.200000000000001e+00,  5.071868925236719e-15,  3.985714285714286e+01,  1.051209903294108e-13,  6.858000000000001e+02],
        [-1.035919976644915e+00,  5.020236578764479e+00, -1.600684299570175e+01,  7.636658267260194e+01, -3.896678157964968e+02,  2.327567997706484e+03],
        [ 4.856928280495909e-01,  3.108163842816296e+00,  4.642979865519745e+00,  1.866866492994786e+01,  4.855583865547229e+01,  1.810911663140202e+02],
        [ 1.535141590722907e+00,  7.000000000000003e+00,  2.961232004341132e+01,  1.625665970356616e+02,  1.020973368362676e+03,  7.358845112401205e+03],
        [ 6.762848427722848e-01,  3.300696203799757e+00,  6.433293472225520e+00,  2.298191741858393e+01,  7.051140073252003e+01,  2.609528751822280e+02],
        [ 0.000000000000000e+00,  3.000000000000005e+00,  0.000000000000000e+00,  1.500000000000000e+01,  0.000000000000000e+00,  1.050000000000000e+02],
        [ 1.139753528477388e+00,  6.464101615137752e+00,  2.455827541429880e+01,  1.474519052838328e+02,  9.530968562856007e+02,  7.437902700278833e+03],
        [ 1.000000000302479e+00,  4.499999999394173e+00,  1.300000000121339e+01,  5.499999999756978e+01,  2.430000000726666e+02,  1.235499999855199e+03],
        [ 2.393417858997484e-01,  1.862753541523877e+00,  1.043117800515645e+00,  4.312283112177676e+00,  3.770708129670429e+00,  1.130784371593392e+01],
        [ 6.311106578190736e-01,  3.245089300687638e+00,  5.997969288502222e+00,  2.179105809171146e+01,  6.481657048417622e+01,  2.388265888542778e+02],
        [ 4.544693815370049e-01,  2.930596525635642e+00,  4.095598493513753e+00,  1.596415545492245e+01,  3.890097958030331e+01,  1.374664823718727e+02],
        [ 0.000000000000000e+00,  1.999999999995189e+00,  0.000000000000000e+00,  4.999999999999769e+00,  0.000000000000000e+00,  1.399999999999761e+01],
        [-2.391933082665460e-02,  3.006028161013639e+00, -2.390817249766078e-01,  1.509430615515463e+01, -2.512718210851153e+00,  1.063756010820516e+02],
        [ 0.000000000000000e+00,  4.000000000000000e+00,  0.000000000000000e+00,  3.999999999999999e+01,  0.000000000000000e+00,  1.119999999999999e+03],
        [ 6.384207794190290e-02,  2.022320860210693e+00,  3.779208079596722e-01,  5.300426892174850e+00,  1.806110727912262e+00,  1.608401975423991e+01],
        [ 3.560874150046195e-01,  2.399999971389194e+00,  2.034785361431682e+00,  8.167134781472743e+00,  1.068256736701236e+01,  3.387196350736040e+01],
        [ 9.634681056331822e-01,  3.132297995454179e+00,  6.261534965445733e+00,  1.642029342804775e+01,  4.018104274985134e+01,  1.049048422930964e+02],
        [-1.556661789761165e-01,  2.275002446880146e+00, -9.693451338586150e-01,  6.900292847631023e+00, -5.091315648191496e+00,  2.446010477214697e+01],
        [ 5.481456321918090e-17,  1.800000000000000e+00,  1.117370096036793e-16,  3.857142857142859e+00,  3.718939564401180e-16,  9.000000000000005e+00],
        [ 3.000000000000104e+00,  1.799999999999990e+01,  1.350000000000001e+02,  1.274999999999999e+03,  1.449000000000001e+04,  1.928850000000000e+05],
        [ 5.086956691981936e-01,  3.040664799148528e+00,  4.668289436691043e+00,  1.776138292185108e+01,  4.630433038382748e+01,  1.655949969677136e+02],
        [-1.345932950297855e+00,  5.432257378437289e+00, -1.875638173411538e+01,  8.258976108327212e+01, -4.009375927192170e+02,  2.179189762992663e+03],
        [ 2.339377560439094e-16,  1.361490812261402e+00,  5.587228929520455e-16,  2.042878242450591e+00,  1.150867896773673e-15,  3.230744266235025e+00],
    ])
    moments = np.concatenate([
        np.ones((len(moments), 1)),
        np.zeros((len(moments), 1)),
        np.ones((len(moments), 1)),
        moments
    ], axis=-1)

    # y = gx + hx^2
    # E[y^n] = E[(x(g + hx))^n]
    #        = E[x^n (g + hx)^n]
    #        = E[x^n sum_k (n k) g^(n-k) h^k x^k]
    #        = sum_k (n k) g^(n-k) h^k E[x^(n+k)]
    def direct_diag_comp(Ex, g, h):
        assert g.shape == h.shape == Ex.shape[1:]
        assert Ex.shape[0] == 9
        assert np.all(Ex[0] == 1) and np.all(Ex[1] == 0) and np.all(Ex[2] == 1)
        assert np.all(checkmom.checkmom(Ex.T))
        
        Ey = np.empty((5, len(g)))
        for n in range(len(Ey)):
            r = np.arange(n + 1)[:, None]
            Ey[n] = np.sum(special.binom(n, r) * g ** r[::-1] * h ** r * Ex[n:2*n+1], axis=0)
        
        Esumy = np.zeros(5)
        Esumy[0] = 1
        for i in range(len(g)):
            Esumy = acc_mom(Esumy, Ey[:, i])
        
        return Esumy
    
    # E[(x + y)^k] = sum_n (k n) E[x^n] E[y^(k-n)]
    def acc_mom(m1, m2):
        nsup = len(m1)
        m = np.empty(nsup)
        for n in range(nsup):
            m[n] = np.sum(special.binom(n, np.arange(n + 1)) * m1[:n + 1] * m2[n::-1])
        return m
    
    c1 = Corr('')
    c2 = Corr('','')
    c3 = Corr('','','')
    c4 = Corr('','','','')
    
    c2ab = Corr('a','b')
    c3abc = Corr('a','b','c')
    c4abcd = Corr('a','b','c','d')
    
    c3aab = Corr('a','a','b')
    c4aabb = Corr('a','a','b','b')
    c4aabc = Corr('a','a','b','c')
    c4aaab = Corr('a','a','a','b')
    
    class TestDiagDirect(unittest.TestCase):
        
        def setUp(self):
            n = np.random.randint(5, 15)
            self.V = moments[np.random.randint(len(moments), size=n)].T
            self.G = np.random.randn(n)
            self.H = np.random.randn(n)
            self.ddc = direct_diag_comp(self.V, self.G, self.H)
        
        def testc1(self):
            exec(c1.diagcode(), globals())
            c = corrdiag1(self.V, self.G, self.H)
            self.assertTrue(np.allclose(c, self.ddc[1]))
        
        def testc2(self):
            exec(c2.diagcode(), globals())
            c = corrdiag2(self.V, self.G, self.H)
            self.assertTrue(np.allclose(c, self.ddc[2]))

        def testc3(self):
            exec(c3.diagcode(), globals())
            c = corrdiag3(self.V, self.G, self.H)
            self.assertTrue(np.allclose(c, self.ddc[3]))
        
        def testc4(self):
            exec(c4.diagcode(), globals())
            c = corrdiag4(self.V, self.G, self.H)
            self.assertTrue(np.allclose(c, self.ddc[4]))
        
        def testc2ab(self):
            exec(c2ab.diagcode(), globals())
            c = corrdiag2ab(self.V, self.G, self.H, self.G, self.H)
            self.assertTrue(np.allclose(c, self.ddc[2]))
        
        def testc3abc(self):
            exec(c3abc.diagcode(), globals())
            c = corrdiag3abc(self.V, self.G, self.H, self.G, self.H, self.G, self.H)
            self.assertTrue(np.allclose(c, self.ddc[3]))
        
        def testc4abcd(self):
            exec(c4abcd.diagcode(), globals())
            c = corrdiag4abcd(self.V, self.G, self.H, self.G, self.H, self.G, self.H, self.G, self.H)
            self.assertTrue(np.allclose(c, self.ddc[4]))
    
        def testc3aab(self):
            exec(c3aab.diagcode(), globals())
            c = corrdiag3aab(self.V, self.G, self.H, self.G, self.H)
            self.assertTrue(np.allclose(c, self.ddc[3]))
        
        def testc4aabb(self):
            exec(c4aabb.diagcode(), globals())
            c = corrdiag4aabb(self.V, self.G, self.H, self.G, self.H)
            self.assertTrue(np.allclose(c, self.ddc[4]))
        
        def testc4aabc(self):
            exec(c4aabc.diagcode(), globals())
            c = corrdiag4aabc(self.V, self.G, self.H, self.G, self.H, self.G, self.H)
            self.assertTrue(np.allclose(c, self.ddc[4]))

        def testc4aaab(self):
            exec(c4aaab.diagcode(), globals())
            c = corrdiag4aaab(self.V, self.G, self.H, self.G, self.H)
            self.assertTrue(np.allclose(c, self.ddc[4]))
        
    class TestDiagAutoDegeneracy(unittest.TestCase):
        
        def setUp(self):
            n = np.random.randint(5, 15)
            self.V = moments[np.random.randint(len(moments), size=n)].T
            self.G = np.random.randn(3, n)
            self.H = np.random.randn(3, n)
        
        def testc3abc_c3aab(self):
            exec(c3abc.diagcode(), globals())
            exec(c3aab.diagcode(), globals())
            c1 = corrdiag3abc(self.V, self.G[0], self.H[0], self.G[0], self.H[0], self.G[1], self.H[1])
            c2 = corrdiag3aab(self.V, self.G[0], self.H[0], self.G[1], self.H[1])
            self.assertTrue(np.allclose(c1, c2))

        def testc4abcd_c4aaab(self):
            exec(c4abcd.diagcode(), globals())
            exec(c4aaab.diagcode(), globals())
            c1 = corrdiag4abcd(self.V, self.G[0], self.H[0], self.G[0], self.H[0], self.G[0], self.H[0], self.G[1], self.H[1])
            c2 = corrdiag4aaab(self.V, self.G[0], self.H[0], self.G[1], self.H[1])
            self.assertTrue(np.allclose(c1, c2))

        def testc4abcd_c4aabb(self):
            exec(c4abcd.diagcode(), globals())
            exec(c4aabb.diagcode(), globals())
            c1 = corrdiag4abcd(self.V, self.G[0], self.H[0], self.G[0], self.H[0], self.G[1], self.H[1], self.G[1], self.H[1])
            c2 = corrdiag4aabb(self.V, self.G[0], self.H[0], self.G[1], self.H[1])
            self.assertTrue(np.allclose(c1, c2))

        def testc4abcd_c4aabc(self):
            exec(c4abcd.diagcode(), globals())
            exec(c4aabc.diagcode(), globals())
            c1 = corrdiag4abcd(self.V, self.G[0], self.H[0], self.G[0], self.H[0], self.G[1], self.H[1], self.G[2], self.H[2])
            c2 = corrdiag4aabc(self.V, self.G[0], self.H[0], self.G[1], self.H[1], self.G[2], self.H[2])
            self.assertTrue(np.allclose(c1, c2))

    class TestDiagCrossDegeneracy(unittest.TestCase):
        
        def setUp(self):
            m = 4
            n = 5
            self.V = moments[np.random.randint(len(moments), size=m * n)].T
            self.GH = np.zeros((2, m, m, n))
            for i in range(m):
                self.GH[:, i, i] = np.random.randn(2, n)
            self.GH = self.GH.reshape(2, m, m * n)
        
        def testc2ab_c1_c1(self):
            exec(c2ab.diagcode(), globals())
            exec(c1.diagcode(), globals())
            co1 = corrdiag2ab(self.V, *self.GH[:, 0], *self.GH[:, 1])
            co2a = corrdiag1(self.V, *self.GH[:, 0])
            co2b = corrdiag1(self.V, *self.GH[:, 1])
            self.assertTrue(np.allclose(co1, co2a * co2b))
        
        def testc3abc_c1_c1_c1(self):
            exec(c3abc.diagcode(), globals())
            exec(c1.diagcode(), globals())
            co1 = corrdiag3abc(self.V, *self.GH[:, 0], *self.GH[:, 1], *self.GH[:, 2])
            co2a = corrdiag1(self.V, *self.GH[:, 0])
            co2b = corrdiag1(self.V, *self.GH[:, 1])
            co2c = corrdiag1(self.V, *self.GH[:, 2])
            self.assertTrue(np.allclose(co1, co2a * co2b * co2c))
        
        def testc3aab_c2_c1(self):
            exec(c3aab.diagcode(), globals())
            exec(c2.diagcode(), globals())
            exec(c1.diagcode(), globals())
            co1 = corrdiag3aab(self.V, *self.GH[:, 0], *self.GH[:, 1])
            co2a = corrdiag2(self.V, *self.GH[:, 0])
            co2b = corrdiag1(self.V, *self.GH[:, 1])
            self.assertTrue(np.allclose(co1, co2a * co2b))
        
        def testc4abcd_c1_c1_c1_c1(self):
            exec(c4abcd.diagcode(), globals())
            exec(c1.diagcode(), globals())
            co1 = corrdiag4abcd(self.V, *self.GH[:, 0], *self.GH[:, 1], *self.GH[:, 2], *self.GH[:, 3])
            co2a = corrdiag1(self.V, *self.GH[:, 0])
            co2b = corrdiag1(self.V, *self.GH[:, 1])
            co2c = corrdiag1(self.V, *self.GH[:, 2])
            co2d = corrdiag1(self.V, *self.GH[:, 3])
            self.assertTrue(np.allclose(co1, co2a * co2b * co2c * co2d))
        
        def testc4aabc_c2_c1_c1(self):
            exec(c4aabc.diagcode(), globals())
            exec(c2.diagcode(), globals())
            exec(c1.diagcode(), globals())
            co1 = corrdiag4aabc(self.V, *self.GH[:, 0], *self.GH[:, 1], *self.GH[:, 2])
            co2a = corrdiag2(self.V, *self.GH[:, 0])
            co2b = corrdiag1(self.V, *self.GH[:, 1])
            co2c = corrdiag1(self.V, *self.GH[:, 2])
            self.assertTrue(np.allclose(co1, co2a * co2b * co2c))
        
        def testc4aabb_c2_c2(self):
            exec(c4aabb.diagcode(), globals())
            exec(c2.diagcode(), globals())
            co1 = corrdiag4aabb(self.V, *self.GH[:, 0], *self.GH[:, 1])
            co2a = corrdiag2(self.V, *self.GH[:, 0])
            co2b = corrdiag2(self.V, *self.GH[:, 1])
            self.assertTrue(np.allclose(co1, co2a * co2b))
        
        def testc4aaab_c3_c1(self):
            exec(c4aaab.diagcode(), globals())
            exec(c3.diagcode(), globals())
            exec(c1.diagcode(), globals())
            co1 = corrdiag4aaab(self.V, *self.GH[:, 0], *self.GH[:, 1])
            co2a = corrdiag3(self.V, *self.GH[:, 0])
            co2b = corrdiag1(self.V, *self.GH[:, 1])
            self.assertTrue(np.allclose(co1, co2a * co2b))
        
    unittest.main()
