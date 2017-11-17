import numpy as np
import scipy as sp
import utilities as ut
from nutils import *
from auxilliary_classes import *

class reparam(object):

    @staticmethod
    def reparam(method, dc_indicator, info):
        if method == 'standard':
            return std_reparam(dc_indicator, *info)
        elif method == 'length':
            return {'discrete': discrete_length_param, 'continuous': minimize_action}[dc_indicator](*info)
        elif method == 'angle':
            if dc_indicator == 'continuous':
                return minimize_angle(*info)
            else:
                raise NotImplementedError
        elif method == 'Joost':
            if dc_indicator == 'discrete':
                return discrete_reparam_joost(*info)
            else:
                raise NotImplementedError
        else:
            raise ValueError('unknown reparameterization method: {}'.format(method))


def std_reparam(dc_indicator, arg):
    if dc_indicator == 'discrete':
        return np.linspace(0,1, len(arg))
    elif dc_indicator == 'continuous':
        return lambda x: x
    else:
        raise NotImplementedError

def minimize(func, initial, method):
    return sp.optimize.minimize(func, initial, method = method)

def distance(arr1, arr2):
    assert all([arr.shape[0] in [2,3] for arr in [arr1,arr2]])
    return ((arr1.T[:,None,:]-arr2.T[None,:,:])**2).sum(2)**0.5

def discrete_length(points): ## Return a vector of distances
    distances = ((points[:,1:] - points[:,:-1])**2).sum(0)**0.5
    return distances

def discrete_length_param(points): ## return points in the domain that correspond to a discrete length-parameterization
    distances = discrete_length(points)
    verts = np.concatenate([[0], distances]).cumsum()
    verts /= verts[-1]
    return verts

def match_indices_sequential(arr1, arr2, tol): ## A[0,0] and A[-1,-1] are automatically matched
    A = distance(arr1, arr2)
    indices =[[0,0]]
    maxindex = 1
    a = np.min(A)
    ## match indices such that i,j satisfy A[i,j] <= tol*a
    for i in range(1,A.shape[0] -1):
        j = np.argmin(A[i,maxindex:-1]) + maxindex
        if A[i,j] <= tol*a:
            indices.append([i,j])
            maxindex = j+1
    indices.append([A.shape[0]-1,A.shape[1]-1])
    return indices

def make_indices(A):  ## generate a matrix with the indices of A[i,j], i.e. a[i,j,0] = i and a[i,j,1] = j
    a = np.zeros(A.shape + (2,))
    a[:,:,0] = np.array(range(a.shape[0]))[:,None]*np.ones(a.shape[1])[None,:]
    a[:,:,1] = np.array(range(a.shape[1]))[None,:]*np.ones(a.shape[0])[:,None]
    return a.astype(int)

def match(A, Indices, tol = 0):
    assert A.shape == Indices.shape[:-1]
    minimum = np.min(A)
    if minimum <= tol: 
        return list(Indices[np.unravel_index(np.argmin(A), A.shape)])
    else:
        raise ValueError

def match_indices_minimum(A, Indices, tol = 1):  ## A[i,j] = ||p_i - p_j||
    ## Iteratively match the currently closest points
    matches = []
    try:
        matches.append(match(A, Indices, tol = tol))
        i,j = np.unravel_index(np.argmin(A), A.shape)
        matches += match_indices_minimum(A[:i, :j], Indices[:i, :j], tol = tol) ## try to match to the left
        matches += match_indices_minimum(A[i+1:, j+1:], Indices[i+1:, j+1:], tol = tol) ## try to match to the right
    except:
        pass
    return matches

def match_minimum(leader, follower, fac = 4):
    ## Iteratively match the currently closest points with initialization
    A = distance(leader, follower)
    Indices = make_indices(A)
    matches = [[0,0], [A.shape[0] - 1, A.shape[1] - 1]]    
    A = A[1:-1,1:-1]
    Indices = Indices[1:-1, 1:-1]
    tol = fac*np.min(A)
    matches += match_indices_minimum(A, Indices, tol = tol)
    return list(sorted(matches, key = lambda x: x[0]))

def discrete_reparam_joost(leader, follower): ## leader is fixed by discrete length param and follower is adjusted accordingly
    if not (isinstance(leader, np.ndarray) and isinstance(follower, np.ndarray)):
        raise ValueError('Joost can only be used when input-data is a Pointset')
    leader_distances, follower_distances = discrete_length(leader), discrete_length(follower)
    leader_verts = discrete_length_param(leader)
    i, j = np.unravel_index(np.argmin(((leader.T[:,None,:]-follower.T[None,:,:])**2).sum(2)), [leader.shape[1], follower.shape[1]])
    # Scale `bot_distances` such that `bot_verts[j]` is close to `top_verts[i]`.
    follower_distances[:j] *= leader_distances[:i].sum() / follower_distances[:j].sum()
    follower_distances[j:] *= leader_distances[i:].sum() / follower_distances[j:].sum()
    follower_verts = np.concatenate([[0], follower_distances]).cumsum()
    follower_verts /= follower_verts[-1]
    return leader_verts, follower_verts

def constrained_arc_length(leader, follower, fac = 16):
    ## length param such that  p_i, p_j are matched when ||p_i - p_j|| <= tol * min(||p_i - p_j||)
    indices = match_minimum(leader, follower, fac = fac)
    assert all([m in indices for m in [[0,0], [leader.shape[1] - 1, follower.shape[1] -1]]])
    verts1, verts2 = discrete_length_param(leader), [0]
    for i in range(len(indices) - 1):
        start, stop = verts1[indices[i][0]], verts1[indices[i+1][0]]
        distances = discrete_length(follower[:,indices[i][1]:indices[i+1][1] + 1])
        csum = np.concatenate([[0],distances]).cumsum() 
        verts2 +=  list(start + csum[1:]*(stop - start)/csum[-1])
    return verts1, np.array(verts2) 
    

def grid_func(func_,side):  ## make local function at the boundary global
    k = side_dict[side]
    if k == 0:
        return function.stack([func_, 0 if side == 'bottom' else 1])
    else:
        return function.stack([0 if side == 'left' else 1, func_])
    
    
def expr(go,c, side, basis = None):  ## return either pol(c) or basis(c)
    if basis is None:
        pol_ = np.polynomial.polynomial.Polynomial(np.concatenate(([0],c)))
        return ut.nutils_function(pol_, derivative = pol_.deriv())(go.geom[side_dict[side]])
    else:
        return basis.dot(np.concatenate(([0], c, [1])))
    
def norm(func):
    return function.sqrt(sum([(func[i])**2 for i in range(len(func))]))


def right_angle_func(leader, follower, fixed, geom, k):
    def func(leader, follower, reparam):
        leader = leader(reparam)
        follower = follower(fixed)
        diff = (leader - follower)/norm(leader - follower)
        scale = 1/norm(leader - follower)  ## scale with the inverse of the distance to make small gaps more important
        #scale = 1
        ## in the log run make sure only the part orthogonal to the curve plays a role
        return sum([scale*((leader+follower).grad(geom)[i,k]*diff[i])**2  for i in range(2)])
    return lambda x: func(leader, follower, x)

def action_func(func, geom, side):
    k = side_dict[side]
    return sum([(func.grad(geom)[i,k])**2 for i in range(2)])


def return_relevant(x):  ## return the entries of elem_eval vector without duplications
    mask = np.concatenate((2*np.linspace(0, len(x) - 2, len(x) // 2), len(x) - 1))
    return x[mask]

def make_jac_fd(func, k, eps = 1e-5): ## k = amount of dependencies
    unit_vector = np.eye(k)
    def jac(c):
        now = func(c)
        return function.stack([(func(c + eps*unit_vector[:,i]) - now)/eps for i in range(k)])
    return jac

def minimization_info(go, method = 'bspline', degree = 2):  ## info = {method: 'method', degree: 'p'}, go = ordinary_go[side]
    domain_, geom, k, ischeme = go.domain, go.geom, side_dict[go._side], go.ischeme
    if method == 'bspline':
        rbasis = domain_.basis('bspline', degree = degree, knotvalues = go.knots)
        initial = domain_.project(geom[k], onto = rbasis, geometry = geom, ischeme = gauss(ischeme))[1:-1]
        if degree != 2:
            print('Warning B-spline reparameterization with p != 2 is unconstrained')
            cons = ()
        else:
            rbasis_ = domain_.basis('bspline', degree = 1, knotvalues = go.knots, periodic = go._knots.periodic)
            mass_1, two_on_one = domain_.integrate([function.outer(rbasis_), function.outer(rbasis_, rbasis.grad(geom)[:,k])], geometry = geom, ischeme = 'gauss4')
            two_on_one = two_on_one.toscipy()
            #cons = {'type': 'ineq', 'fun': lambda x: domain_.project(rbasis.dot(np.concatenate(([0],x,[1]))).grad(geom)[k], onto = rbasis_, geometry = geom, ischeme = gauss(ischeme)) - 0.001*np.ones(len(rbasis_))}
            cons = {'type': 'ineq', 'fun': lambda x: mass_1.solve(two_on_one.dot(np.concatenate(([0],x,[1])))) - 0.001*np.ones(len(rbasis_))}
    elif method == 'pol':
        raise NotImplementedError
        #initial = unit_vector(info['order'] - 1,0)
        #cons = {'type': 'eq', 'fun': lambda x:  1 - sum(x)}
        #rbasis = None
    return rbasis, initial, cons    


def minimize_action(go, func, side, info = {'method': 'spline', 'order': 2}): ##minimize the action with B-splines - works great
    geom, ischeme, p, basis  =go.domain, go.geom, go.ischeme, go.degree, go.basis
    domain_ = go.domain.boundary[side]
    k = side_dict[side]
    rbasis, initial, cons = minimization_info(go[side], info)
    def main_func(c, integrate = False):
        arg = func(grid_func(expr(go,c, side, rbasis),side))
        action = action_func(arg, go.geom, side)
        if integrate:
            ret = domain_.integrate(action, geometry = geom, ischeme = gauss(ischeme))
            print(ret)
            return ret
        else:
            return action
    jac = lambda c: domain_.integrate(make_jac_fd(main_func,len(initial))(c), geometry = geom, ischeme = gauss(ischeme))
    opt = sp.optimize.minimize(lambda x: main_func(x, integrate = True), initial, method = 'SLSQP', jac = jac)
    return grid_func(expr(go, opt.x, side, rbasis), side)

def minimize_angle(go, goal_boundaries, side, method = 'bspline', degree  = 2):  ## UNFINISHED
    domain, geom, ischeme, p, basis  = go.domain, go.geom, go.ischeme, go.degree, go.basis
    domain_ = domain.boundary[side]
    k = side_dict[side]
    opposite = opposite_side[side]
    pull_dict = {'left':[0,1], 'right':[0,-1], 'bottom':[-1,0], 'top':[1,0]}
    rbasis, initial, cons = minimization_info(go[side], method = method, degree = degree)
    #cons = ()
    def main_func(c, integrate = False):
        arg = grid_func(expr(go,c, side, rbasis), side)
        #val = right_angle_func(goal_boundaries[side], goal_boundaries[opposite], grid_func(geom[k], opposite) - pull_dict[opposite], geom, k )(arg)
        val = right_angle_func(goal_boundaries[side], goal_boundaries[opposite], geom, geom, k )(arg)
        val += 1*action_func(arg, go.geom, side)
        if integrate:
            ret = domain_.elem_eval(val, ischeme = 'bezier1')
            print(sum(ret))
            return sum(ret)
        else:
            return val
    jac = lambda c: domain_.integrate(make_jac_fd(main_func,len(initial))(c), geometry = geom, ischeme = gauss(ischeme))
    opt = sp.optimize.minimize(lambda x: main_func(x, integrate = True), initial, method = 'SLSQP', jac = jac, constraints = cons, options = {'maxiter':5})
    print(opt, 'opt')
    return grid_func(expr(go, opt.x, side, rbasis), side)


def bnd_vec_to_global(go, vec, side):
    domain, basis = go.domain, go.basis
    if not side in planar_sides:
        raise ValueError('side-keyword must be a valid side')
    else:
        k = len(basis) // len(vec)
        vec_ = np.repeat(vec,k) if side in ['bottom', 'top'] else np.tile(vec, k)
    return vec_
