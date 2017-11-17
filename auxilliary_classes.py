import collections, functools
import numpy as np
from nutils import *
from scipy.linalg import block_diag
import scipy.sparse


########################################

## MAKE ALL THIS STUFF ADAPTIVE TO nD

########################################


Pointset = collections.namedtuple('Pointset', ['verts', 'geom'])
gauss = 'gauss{:d}'.format
preproc_info_list = ['cons_lib', 'dirichlet_lib', 'cons_lib_lib', 'error_lib']
planar_sides = ['left', 'right', 'bottom', 'top']
side_dict = {'left':1, 'right':1, 'bottom':0, 'top':0}
opposite_side = {'left': 'right', 'right': 'left', 'bottom': 'top', 'top': 'bottom'}
dim_boundaries = {0: ['left', 'right'], 1: ['bottom', 'top']}
corners = lambda n,m: {'left':[0, m-1], 'bottom':[0, n-1], 'top':[0, n-1], 'right':[0, m-1]}


def side_indices(fromside, side, *args):  ## returns side_indices for 2D and 1D
    ## 3D ain't work yet, the 'side' argument is passed just in case, for 1D we need it
    if len(args) == 2:
        n,m = args
        return {'bottom': [i*m for i in range(n)], 'top': [i*m + m - 1 for i in range(n)], 'left': list(range(m)), 'right': list(range((n - 1)*m, n*m))}[side]
    elif len(args) == 1:
        assert side in planar_sides
        if fromside in ['left', 'right']:
            return {'bottom':0, 'top': -1}[side]
        else:
            return {'left':0, 'right': -1}[side]
        
        
class tensor_index:  ## for now only planar, make more efficient
    'returns indices of sides and corners'
    
    _p = None 
    _side = None  
    _l = 1  ## godfather length
    _n = 0 ## amount of dims
    
    @classmethod
    def from_go(cls, go, *args, **kwargs):
        ret = cls(go.ndims, repeat = go.repeat, side = go._side)
        ret._n, ret._l = len(go.ndims), np.prod(go.ndims)
        ret._indices = np.asarray([int(i) for i in range(ret._l)], dtype=np.int)
        return ret
    
    @classmethod
    def from_parent(cls,parent,side):
        #######  FUGLY, GET RID OF THIS
        assert side in planar_sides
        if parent._side in ['left', 'right']:
            assert side in ['bottom', 'top']
        elif parent._side in ['bottom', 'top']:
            assert side in ['left', 'right']
        #######  FUGLY, GET RID OF THIS
        ndims = [parent._ndims[side_dict[side]]] if len(parent._ndims) == 2 else [1]  ## select dimension corresponding to side
        ret = cls(ndims, repeat = parent._repeat, side = side, fromside = parent._side)  ## instantiate
        ret._p = parent  ## set parent
        ret._indices = parent._indices[side_indices(parent._side, side, *parent._ndims)]
        ret._l, ret._n = parent._l, parent._n - 1
        return ret
    
    def __init__(self, ndims, repeat = 1, side = None, fromside = None):  ## adapt to dims of any size
        assert len(ndims) < 3, 'Not yet implemented'
        self._ndims, self._repeat = ndims, repeat
        self._side = side
        
        
    @property
    def p(self):
        return self._p if self._p is not None else self
    
    def c(self, side):
        return tensor_index.from_parent(self,side) if self._n != 0 else self
    
    def __getitem__(self,side_):
        return self.c(side_)
    
    @property
    def indices(self):
        return np.concatenate([self._indices + i*self._l*np.ones(np.prod(self._ndims), dtype=np.int) for i in range(self._repeat)])


###############################################################


## Prolongation / restriction matrix

## Make prolongation object-oriented maybe
        
        
def prolongation_matrix(*args):  ## MAKE THIS SPARSE
    ## args = [kv_new, kv_old] is [tensor_kv]*2, if len(kv_new) < len(kv_old) return restriction
    #assert all([len(args) == 2] + [len(k) == 1 for k in args])
    ## make sure we've got 2 tensor_kvs with dimension 1
    assert_params = [args[0] <= args[1], args[1] <= args[0]]
    assert any(assert_params) and args[0]._degree == args[1]._degree, 'The kvs must be nested'  ## check for nestedness
    if all(assert_params):
        return np.eye(args[0].dim)
    p = args[0]._degree
    kv_new, kv_old = [k.extend_knots() for k in args] ## repeat first and last knots
    if assert_params[0]:  ## kv_new <= kv_old, reverse order 
        kv_new, kv_old = list(reversed([kv_new, kv_old]))
    n = len(kv_new) - 1
    m = len(kv_old) - 1
    T = numpy.zeros([n, m])
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if kv_new[i] >= kv_old[j] and kv_new[i] < kv_old[j+1]:
                T[i,j] = 1
    for q in range(p):
        q = q+1
        T_new = numpy.zeros([n - q, m - q])
        for i in range(T_new.shape[0]):
            for j in range(T_new.shape[1]):
                fac1 = (kv_new[i + q] - kv_old[j])/(kv_old[j+q] - kv_old[j]) if kv_old[j+q] != kv_old[j] else 0
                fac2 = (kv_old[j + 1 + q] - kv_new[i + q])/(kv_old[j + q + 1] - kv_old[j + 1]) if kv_old[j + q + 1] != kv_old[j + 1] else 0
                T_new[i,j] = fac1*T[i,j] + fac2*T[i,j + 1]
        T = T_new
    if args[0].periodic:  ## some additional tweaking in the periodic case
        T_ = T
        T = T[:n-2*p,:m-2*p]
        T[:,0:p] += T_[:n-2*p,m-2*p: m-2*p+p]
    ## return T if kv_new >= kv_old else the restriction
    return T if not assert_params[0] else np.linalg.inv(T.T.dot(T)).dot(T.T)

def _prolongation_matrix(*args):  ## A bit more efficient but doesn't always work
    assert_params = [args[0] <= args[1], args[1] <= args[0]]
    assert any(assert_params) and args[0]._degree == args[1]._degree, 'The kvs must be nested'  ## check for nestedness
    if all(assert_params):
        return np.eye(args[0].dim)
    p = args[0]._degree
    kv_new, kv_old = [k.extend_knots() for k in args] ## repeat first and last knots
    if assert_params[0]:  ## kv_new <= kv_old, reverse order 
        kv_new, kv_old = list(reversed([kv_new, kv_old]))
    n = len(kv_new) - 1
    m = len(kv_old) - 1
    T = numpy.zeros([n, m])
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if kv_new[i] >= kv_old[j] and kv_new[i] < kv_old[j+1]:
                T[i,j] = 1
    for q in range(p):
        q = q+1
        fac1 = np.array([[(kv_new[i + q] - kv_old[j])/(kv_old[j+q] - kv_old[j]) if kv_old[j+q] != kv_old[j] else 0 for j in range(m-q)] for i in range(n-q)])
        fac2 = np.array([[(kv_old[j + 1 + q] - kv_new[i + q])/(kv_old[j + q + 1] - kv_old[j + 1]) if kv_old[j + q + 1] != kv_old[j + 1] else 0 for j in range(m-q)] for i in range(n-q)])
        T = T[:-1,:][:,:-1]*fac1 + T[:-1,1:]*fac2
    if args[0].periodic:  ## some additional tweaking in the periodic case
        T_ = T
        T = T[:n-2*p,:m-2*p]
        T[:,0:p] += T_[:n-2*p,m-2*p: m-2*p+p]
    return T if not assert_params[0] else np.linalg.inv(T.T.dot(T)).dot(T.T)

### go.cons prolongation / restriction

def prolong_bc_go(fromgo, togo, *args, return_type = 'nan'):  ## args = [T_n, T_m , ...]
    to_shape = np.prod(togo.ndims)
    repeat = togo.repeat
    if return_type == 'nan':
        ret = util.NanVec(repeat*to_shape)  ## create empty NanVec of appropriate size
    else:
        ret = np.zeros(repeat*to_shape)
    if len(args) == 2:  ## more than one dimension
        for side in togo._sides:
            T = block_diag(*[args[side_dict[side]]]*repeat)
            vecs = fromgo.get_side(side)
            ret[togo._indices[side].indices] = T.dot(vecs[1])
        return ret
    elif len(args) == 1:
        for side in togo._sides:
            vecs = fromgo.get_side(side)
            ret[togo._indices[side].indices] = vecs[1]
        return ret
    else:
        raise NotImplementedError
        
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    return [l[i:i + n] for i in range(0, len(l), n)]