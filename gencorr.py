from sympy.combinatorics.partitions import IntegerPartition
from sympy.combinatorics import Permutation
import copy
import numpy as np
from fractions import Fraction
from collections import Counter, OrderedDict, defaultdict

__doc__ = """
Usage examples:
>>> import gencorr
>>> print(gencorr.gen_corr('','').__repr__('\n')) # second order variance
>>> print(gencorr.gen_corr('a','b').__repr__('\n')) # s.o. covariance
>>> print(gencorr.gen_corr('','','','').__repr__('\n')) # s.o. 4th moment
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
        self._rank = rank
        self._indices = tuple(indices)
        assert not self._indices or len(self._indices) == self._rank
    
    def sort_indices(self):
        self._indices = tuple(sorted(self._indices))
        return self
    
    def apply_index(self, idx):
        """
        Replace all indices with idx.
        """
        self._indices = (idx,) * len(self._indices)
        return self
    
    def __lt__(self, d):
        if isinstance(d, Tensor):
            return self._rank < d._rank or self._rank == d._rank and self._indices < d._indices
        elif isnum(d):
            return False
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
    
    def __lt__(self, d):
        if isinstance(d, D):
            return False
        else:
            return super().__lt__(d)
    
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
        self._var = var
    
    def __repr__(self):
        letter = 'G' if self._rank == 1 else 'H'
        if self._indices:
            indstr = ''.join(map(lambda i: indices[i], self._indices))
        else:
            indstr = ''
        if str(self._var) == '':
            varstr = ''
        else:
            varstr = f'({self._var})'
        return f'{letter}{varstr}{indstr}'
    
    def __lt__(self, d):
        if isinstance(d, D):
            return self._var < d._var or self._var == d._var and super().__lt__(d)
        elif isinstance(d, V):
            return True
        else:
            return super().__lt__(d)
    
    def __eq__(self, obj):
        return super().__eq__(obj) and self._var == obj._var
    
    def assume_diagonal(self):
        """
        Return self if all the indices are equal, otherwise 0.
        """
        if not self._indices:
            return self
        i0 = self._indices[0]
        return self if all(i == i0 for i in self._indices) else 0

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
        return isinstance(obj, self.__class__) and self._list[::-1] < obj._list[::-1]
    
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
    
    def index_D(self):
        """
        If there is a V object, take its indices and distribute them in order
        over D objects.
        """
        for Vobj in self._list:
            if isinstance(Vobj, V):
                break
        else:
            return self
        Dobjs = []
        for obj in self._list:
            if isinstance(obj, D):
                Dobjs.append(obj)
        i = 0
        for d in Dobjs:
            d._indices = Vobj._indices[i:i + d._rank]
            i += d._rank
        assert(i == Vobj._rank)
        return self
    
    def split_D(self):
        """
        Replace D objects with a sum over all permutations of indices of the
        D objects. Makes sense because the V object represents a symmetric
        tensor.
        
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
        if any(map(lambda x: isinstance(x, Tensor) and len(set(x._indices)) != 1, self._list)):
            return self
        
        factors_by_index = defaultdict(list)
        for obj in self._list:
            if isinstance(obj, Tensor):
                key = obj._indices[0]
            else:
                key = None
            factors_by_index[key].append(obj)
        return Mult(Summation(*[Mult(*l) for l in factors_by_index.values()]), *factors_by_index[None])
    
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
        Put indices on all tensors according to the summation term they are in.
        """
        for i, obj in enumerate(self._list):
            if isinstance(obj, Expression):
                self._list[i] = obj.recursive('apply_index', i)
        return self

# def gen_terms(terms, vars, l, n_2, n_1):
#     if n_2 == n_1 == 0:
#         terms.append(Mult(*(D(l[i], vars[i]) for i in range(len(l)))))
#     else:
#         if n_1 > 0: gen_terms(terms, vars, l + (1,), n_2, n_1 - 1)
#         if n_2 > 0: gen_terms(terms, vars, l + (2,), n_2 - 1, n_1)
#
# def gen_corr_base_expr(*vars):
#     terms = []
#     for v_rank in range(len(vars), 1 + 2 * len(vars)):
#         v_terms = []
#         for n_2 in range(1 + v_rank // 2):
#             n_1 = len(vars) - n_2
#             if n_1 + 2 * n_2 == v_rank:
#                 gen_terms(v_terms, vars, (), n_2, n_1)
#         if v_terms:
#             terms.append(Mult(Sum(*v_terms), V(v_rank)))
#     return Sum(*terms)

def gen_corr(*vars, what='full'):
    # e = gen_corr_base_expr(*vars)
    
    e = Mult(*[Sum(D(1, v), D(2, v)) for v in vars])
    
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
    e = e.recursive('index_D')
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

    # complete formula
    if what == 'full':
        return e
    
    # assume diagonal hessian
    elif what == 'diag':
        e = e.recursive('assume_diagonal') # replace off diagonal D with 0
        e = e.recursive('reduce') # remove 0 terms from the sum
        
        return e
        
    else:
        raise KeyError(what)
