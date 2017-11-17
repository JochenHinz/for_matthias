from nutils import *
import utilities as ut
import numpy as np
import scipy as sp
from problem_library import Pointset
from auxilliary_classes import *
from preprocessor import preproc_dict

class Solver(object):
    
    
    
    sides = ['bottom', 'right', 'top', 'left']
    topo_dict = {'bottom':[1,0], 'right': [0,1], 'top':[1,0], 'left':[0,1]}
    dual_basis = 0
    mass = None
    mass_hom = None 
        
  
    def __init__(   
                    self, 
                    grid_object,             # [geom, domain, basis, ischeme,...]
                    cons,           # must be compatible with domain
                    dirichlet_ext_vec = None,# If None, dirichlet_ext is projected onto basis
                    initial_guess = None,    # initial guess such that the initial geometry = \
                                             # basis_hom.vector(2).dot(initial_guess) + dirichlet_ext
                    bad_initial_guess = False,
                    maxiter = 12,
                    mass_hom = None,
                    mass = None,
                    curves = None
                                                 ):
        
        self.go = grid_object
        self.cons = cons     
                
        if mass_hom is not None:
            self.mass_hom = mass_hom
            
            
            
        if mass is not None:
            self.mass = mass
            
            
    def update_all(self, additional_rhs = None):
        go = self.go
        rhs = [go.basis.vector(2).dot(self.cons | 0)]
        if additional_rhs is not None:
            rhs.append(additional_rhs)
        print(rhs, 'rhs')
        self.go = self.go.update_from_domain(self.go.domain, ref_basis = True)
        rhs = [rhs[i]*go.basis for i in range(len(rhs))]
        rhs.append(function.outer(go.basis))        
        rhs = self.std_int(rhs)
        print(rhs, 'rhs')
        mass = rhs[-1]
        rhs = rhs[:-1]
        lhs = mass.solve(rhs)
        self.cons = lhs[0]
        return lhs[1:] if len(lhs) > 0 else 0
            
            
    def one_d_laplace(self, direction, ltol = 1e-7):  ## 0:xi, 1:eta
        go = self.go
        gbasis = go.basis.vector(2)
        target = function.DerivativeTarget([len(go.basis.vector(2))])
        res = model.Integral(gbasis['ik,' + str(direction)]*gbasis.dot(target)['k,' + str(direction)], domain=go.domain, geometry=go.geom, degree=go.degree*3)
        lhs = model.newton(target, res, lhs0=self.cons | 0, freezedofs=self.cons.where).solve(ltol)
        return lhs
    
    
    def linear_spring(self):
        go = self.go
        mat = sp.sparse.csr_matrix(sp.sparse.block_diag([sp.sparse.diags([-1, -1, 4, -1, -1], [-go.ndims[1], -1, 0, 1, go.ndims[1]], shape=[np.prod(go.ndims)]*2)]*2))
        mat = matrix.ScipyMatrix(mat)
        return mat.solve(constrain = go.cons)
    
    
    def transfinite_interpolation(self, curves_library_, corners = None, rep_dict = None):    ## NEEDS FIXING
        go = self.go
        geom = go.geom
        curves_library = preproc_dict(curves_library_, go).instantiate(rep_dict)
        for item in curves_library:
            if isinstance(curves_library[item], Pointset):
                pnts = curves_library[item]
                curves_library[item] = ut.interpolated_univariate_spline(pnts.verts, pnts.geom, {'left': geom[1], 'right': geom[1], 'bottom': geom[0], 'top':geom[0]}[item])
        basis = go.basis.vector(2)
        expression = 0
        expression += (1 - geom[1])*curves_library['bottom'] + geom[1]*curves_library['top'] if 'top' in curves_library else 0
        expression += (1 - geom[0])*curves_library['left'] + geom[0]*curves_library['right']if 'left' in curves_library else 0
        if corners is not None:
            expression += -(1 - geom[0])*(1 - geom[1])*np.array(corners[(0,0)]) - geom[0]*geom[1]*np.array(corners[(1,1)])
            expression += -geom[0]*(1 - geom[1])*np.array(corners[(1,0)]) - (1 - geom[0])*geom[1]*np.array(corners[(0,1)])
        return go.domain.project(expression, onto=basis, geometry=geom, ischeme=gauss(go.ischeme), constrain = go.cons)
            
              
    def Elliptic(self,c, russian = True):
        go = self.go
        g11, g12, g22 = self.fff(c)
        x_xi, x_eta, y_xi, y_eta = self.func_derivs(c)
        vec1 = go.basis*(g22*x_xi.grad(go.geom,ndims = 2)[0] - 2*g12*x_xi.grad(go.geom,ndims = 2)[1] + g11*x_eta.grad(go.geom,ndims = 2)[1])
        vec2 = go.basis*(g22*y_xi.grad(go.geom,ndims = 2)[0] - 2*g12*y_xi.grad(go.geom,ndims = 2)[1] + g11*y_eta.grad(go.geom,ndims = 2)[1])
        if russian:
            return -function.concatenate((vec1,vec2))/(2*g11 + 2*g22)
        else:
            return -function.concatenate((vec1,vec2))
        
        
    def Elliptic_DG(self,c):
        go = self.go
        g11, g12, g22 = self.fff(c)
        x_xi, x_eta, y_xi, y_eta = self.func_derivs(c)
        b = go.basis
        b_xi, b_eta = [b.grad(go.geom)[:,i] for i in range(2)]
        x = go.basis.vector(go.repeat).dot(c)
        #beta = function.repeat(x[_], go.ndims, axis = 0)
        J = function.determinant(x.grad(go.geom))
        vec1 = x_xi*((b*g22).grad(go.geom)[:,0] - (b*g12).grad(go.geom)[:,1]) + x_eta*(-(b*g12).grad(go.geom)[:,0] + (b*g11).grad(go.geom)[:,1])
        vec2 = y_xi*((b*g22).grad(go.geom)[:,0] - (b*g12).grad(go.geom)[:,1]) + y_eta*(-(b*g12).grad(go.geom)[:,0] + (b*g11).grad(go.geom)[:,1])
        A = function.stack([(x_xi*function.stack([g22, -g12]) + x_eta*function.stack([-g12, g11])).dotnorm(go.geom), (y_xi*function.stack([g22, -g12]) + y_eta*function.stack([-g12, g11])).dotnorm(go.geom)])
        #A = function.stack([(function.stack([x_xi*g22, x_eta*g11])).dotnorm(go.geom), (function.stack([y_xi*g22, y_eta*g11])).dotnorm(go.geom)]) ## set g_12 to 0
        alpha = 1000000
        vec_d_1 = b*(function.mean(A[0])) + alpha*b*function.jump(x_xi) + alpha*b*function.jump(x_eta)
        vec_d_2 = b*(function.mean(A[1])) + alpha*b*function.jump(y_xi) + alpha*b*function.jump(y_eta)
        #vec_d_1 -= alpha*b*function.jump((x.grad(go.geom)[:,0]).dotnorm(go.geom))
        #vec_d_2 -= alpha*b*function.jump((x.grad(go.geom)[:,1]).dotnorm(go.geom))
        return -function.concatenate((vec1,vec2)), function.concatenate((vec_d_1,vec_d_2))
    
    def Elliptic_partial_bnd_orth(self,c):
        pass
        
        
    def Liao(self,c):
        g11, g12, g22 = self.fff(c)
        return g11**2 + g22**2 + 2*g12**2
    
    def AO(self,c):
        g11, g12, g22 = self.fff(c)
        return g11*g22
    
    def Winslow(self,c):
        go = self.go
        g11, g12, g22 = self.fff(c)
        det =function.determinant(go.basis.vector(go.repeat).dot(c).grad(go.geom))
        return (g11+g22)/det
    
    def func_derivs(self,c):
        go = self.go
        s = go.basis.vector(2).dot(c)
        x_g, y_g = [s[i].grad(go.geom,ndims = 2) for i in range(2)]
        x_xi, x_eta = [x_g[i] for i in range(2)]
        y_xi, y_eta = [y_g[i] for i in range(2)]
        return x_xi, x_eta, y_xi, y_eta
    
    def elliptic_conformal(self,c, alpha_1 = 25, alpha_2 = 2):
        g11, g12, g22 = self.fff(c)
        return alpha_1*(g11- g22)**2 + alpha_2*g12**2
        
        
    def fff(self, c):  ## first fundamental form
        x_xi, x_eta, y_xi, y_eta = self.func_derivs(c)
        g11, g12, g22 = x_xi**2 + y_xi**2, x_xi*x_eta + y_xi*y_eta, x_eta**2 + y_eta**2
        return g11, g12, g22
    
    
    def solve(self, init = None, method = 'Elliptic', solutionmethod = 'Newton', t0 = None, cons = None):
        go = self.go
        basis = go.basis
        ltol = 1e-1/float(len(basis))
        if cons is None:
            cons = self.cons
        if init is None:
            init = self.cons|0
        target = function.DerivativeTarget([len(go.basis.vector(2))])
        if method == 'Elliptic':
            res = model.Integral(self.Elliptic(target), domain=go.domain, geometry=go.geom, degree=go.ischeme*3)
        elif method == 'Liao':
            res = model.Integral(self.Liao(target), domain=go.domain, geometry=go.geom, degree=go.ischeme*3).derivative(target)
        elif method == 'AO':
            res = model.Integral(self.AO(target), domain=go.domain, geometry=go.geom, degree=go.ischeme*3).derivative(target)
        elif method == 'Winslow':
            res = model.Integral(self.Winslow(target), domain=go.domain, geometry=go.geom, degree=go.ischeme*3).derivative(target)
        elif method == 'Elliptic_conformal':
            res = model.Integral(self.elliptic_conformal(target), domain=go.domain, geometry=go.geom, degree=go.ischeme*3).derivative(target)
        elif method == 'Elliptic_partial':     
            res = model.Integral(go.basis.vector(go.repeat)['ik,l']*go.geom['k,l'], domain=go.domain, geometry=go.basis.vector(go.repeat).dot(target), degree=go.ischeme*3)
        elif method == 'Elliptic_DG':
            G, DG = self.Elliptic_DG(target)
            res = model.Integral(G, domain=go.domain, geometry=go.geom, degree=go.ischeme*3)
            res += model.Integral(DG, domain = go.domain.interfaces, geometry=go.geom, degree=go.ischeme*3)
        else:
            raise ValueError('unknown method: ' + method)
        if solutionmethod == 'Newton':
            lhs = model.newton(target, res, lhs0=init, freezedofs=cons.where).solve(ltol)
        elif solutionmethod == 'Pseudotime':
            if t0 is None:
                t0 = 0.0001
            term = basis.vector(2).dot(target)
            inert = model.Integral(function.concatenate((basis*term[0], basis*term[1])), domain=go.domain, geometry=go.geom, degree=go.ischeme*3)
            lhs = model.pseudotime(target, res, inert, t0, lhs0=init, freezedofs=cons.where).solve(ltol)
        else:
            raise ValueError('unknown solution method: ' + solutionmethod)
        return lhs
    
    
    def solve_with_repair(self, s):  ## repair defects using 'strategy' while keeping the dirichlet condition fixed
        go = self.go
        if not isinstance(s, np.ndarray):
            raise ValueError('Parameter s needs to be an np.ndarray')
        s = self.solve(init = s, cons = self.cons)
        jac = function.determinant(go.basis.vector(2).dot(s).grad(go.geom))
        basis_indices = ut.defect_check(self.go, jac)
        for irefine in log.count('Defect correction '):
            self.go.domain = self.go.domain.refined_by(elem for elem in self.go.domain.supp(self.go.basis, basis_indices))
            s = self.update_all(go.basis.vector(2).dot(s))
            s = self.solve(init = s, cons = self.cons)
            jac = function.determinant(go.basis.vector(2).dot(s).grad(go.geom))
            basis_indices = ut.defect_check(self.go, jac)
            if len(basis_indices) == 0:
                break
        return s
            
            
