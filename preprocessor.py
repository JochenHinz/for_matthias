import numpy as np
import scipy as sp
import scipy.interpolate
from nutils import *
import inspect
import collections
import itertools
import utilities as ut
from auxilliary_classes import *


def vec_union(vec1, vec2):  ## return the union of two refine indices
    return np.asarray(sorted(list(set(list(vec1) + list(vec2)))))
    
    
class preproc_dict:
    
    def __init__(self, dictionary, go):
        self._dict = dictionary
        self._go = go
        
    @property
    def go(self):
        return self._go
        
        
    def instantiate(self, rep_dict):
        if rep_dict is None:
            return self.from_geom()
        ret = self._dict.copy()
        for side in self._dict:
            if isinstance(self._dict[side], Pointset):
                ## Pointsets should be reparameterized from the problem library
                pass
            elif isinstance(self._dict[side], ut.base_grid_object):
                pass
                ##Forthcoming
            else:
                ret[side] = ret[side](self._go.geom if rep_dict[side] is None else rep_dict[side])
        return ret
        
    def from_geom(self):
        rep_dict = {'left': self.go.geom, 'right': self.go.geom, 'bottom': self.go.geom, 'top': self.go.geom}
        return self.instantiate(rep_dict)
    
    def items(self):
        return self._dict.items()
    
    def plot(self, go, name, ref = 1):
        d = self.from_geom(go.geom)
        for side in d.keys():
            points = go.domain.boundary[side].refine(ref).elem_eval( d[side], ischeme='vtk', separate=True)
            with plot.VTKFile(name+'_'+side) as vtu:
                vtu.unstructuredgrid( points, npars=2 )
            
            
def log_iter_sorted_dict_items(title, d):
    for k, v in sorted(d.items()):
        with log.context(title + ' ' + k):
            yield k, v

            
            
def generate_cons(go, boundary_funcs, corners = None, btol = 1e-2):
    domain, geom, basis, degree, ischeme, basis_type, knots = go.domain, go.geom, go.basis.vector(2), go.degree, go.ischeme, go.basis_type, go.knots 
    cons = util.NanVec(len(go.basis)*go.repeat)
    ## constrain the corners
    if corners:
        for (i, j), v in log.iter('corners', corners.items()):
            domain_ = (domain.levels[-1] if isinstance(domain, topology.HierarchicalTopology) else domain).boundary[{0: 'bottom', 1: 'top'}[j]].boundary[{0: 'left', 1: 'right'}[i]]
            cons = domain_.project(v, onto=basis, constrain=cons, geometry=geom, ischeme='vertex')
    # Project all boundaries onto `gbasis` and collect all elements where
    # the projection error is larger than `btol` in `refine_elems`.
    cons_library = {'left':0, 'right':0, 'top':0, 'bottom':0}
    #refine_elems = set()
    for side, goal in log_iter_sorted_dict_items('boundary', boundary_funcs):
        dim = side_dict[side]
        if isinstance(goal, Pointset):
            domain_ = domain.boundary[side].locate(geom[dim], goal.verts)
            ischeme_ = 'vertex'
            goal = function.elemwise(
                dict(zip((elem.transform for elem in domain_), goal.geom)),
                [domain.ndims])
        elif isinstance(goal, ut.base_grid_object):
            temp = goal + go  ## take the union
            domain_ = temp[side].domain  ## restrict to boundary
            goal = temp.basis.vector(2).dot(temp.cons | 0)  ## create mapping
        else:  ## differentiable curve
            domain_ = domain.boundary[side]
            ischeme_ = gauss(ischeme*2)
        cons_library[side] = domain_.refine(2).project(goal, onto=basis, geometry=geom, ischeme=ischeme_, constrain=cons)
        cons |= cons_library[side]
    return cons


def constrained_boundary_projection(go, goal_boundaries_, corners, btol = 1e-2, rep_dict = None, ref = 0):  #Needs some fixing
    degree, ischeme, basis_type = go.degree, go.ischeme, go.basis_type
    goal_boundaries = preproc_dict(goal_boundaries_, go)
    goal_boundaries = goal_boundaries.from_geom() if not rep_dict else goal_boundaries.instantiate(rep_dict)
    if basis_type == 'bspline':  ## the basis type is b_spline = > we need to refine on knots
        assert go._knots is not None
    cons = go.cons
    refine_elems = set()
    error_dict = {'left':0, 'right':0, 'top':0, 'bottom':0}
    for side, goal in log_iter_sorted_dict_items('boundary', goal_boundaries):
        dim = side_dict[side]
        goal_ = goal if not isinstance(goal, ut.base_grid_object) else go.bc()  
        ## for now, if goal is a grid_object assume that go already satisfies the b.c.. Change this in the long run !
        error = ((goal_ - go.bc())**2).sum(0)**0.5
        ## replace goal by goal.function() or something once it becomes an object
        go_ = go[side]
        if basis_type == 'spline':   ## basis is spline so operate on the elements
            error_ = go_.domain.project(error, ischeme= 2*ischeme)
            refine_elems.update(
                elem.transform.promote(domain.ndims)[0]
                for elem in go_.domain.supp(basis_, numpy.where(error_ > btol)[0]))
        elif basis_type == 'bspline':  ## basis is b_spline just compute error per element, refinement comes later
            basis0 = go_.domain.basis_bspline(degree = 0, periodic = tuple(go.periodic))
            error_ = np.divide(*go_.integrate([basis0*error, basis0]))
            print(numpy.max(error_), side)
            error_dict[side] = error_
    if basis_type == 'spline':
        if len(refine_elems) == 0:  ## no refinement necessary = > finished
            return None
        else:   ## refinement necessary = > return new go with refined grid
            domain = domain.refined_by(refine_elems)
            new_go = go.update_from_domain(domain, ref_basis = True)   ## return hierarchically refined grid object
            new_go.set_cons(goal_boundaries_, corners)
            return new_go
    else:
        union_dict = {0: ['bottom', 'top'], 1: ['left', 'right']}
        ## take union of refinement indices
        ref_indices = [vec_union(*[numpy.where(error_dict[i] > btol)[0] for i in union_dict[j]]) for j in range(2)]
        print(ref_indices, 'ref_indices')
        if len(ref_indices[0]) == 0 and len(ref_indices[1]) == 0:  ## no refinement
            return None
        else: ## return refined go
            ## refine according to union
            ## create new topology
            new_go = go.ref_by(ref_indices, prolong_mapping = False, prolong_constraint = False)
            #cons_funcs = goal_boundaries_.from_geom(new_go.geom) if not rep_dict else goal_boundaries_.instantiate(rep_dict)
            new_go.set_cons(goal_boundaries_, corners)
            return new_go



def boundary_projection(go, goal_boundaries, corners = None, btol = 1e-2, rep_dict = None, maxref = 10):
    basis_type = go.basis_type
    go.set_cons(goal_boundaries,corners, rep_dict = rep_dict)
    #go.quick_plot_boundary()
    go_list = [go]
    for bndrefine in log.count('boundary projection'):
        proj = constrained_boundary_projection(go_list[-1], goal_boundaries, corners, btol = btol, rep_dict = rep_dict)
        if proj is None or bndrefine > maxref:
            break
        else:
            go_list.append(proj)
            go_list[-1].quick_plot_boundary()
            #go_list[-1].quick_plot_boundary()
    return ut.multigrid_object(go_list)