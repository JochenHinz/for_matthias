from nutils import *
import utilities as ut
import numpy as np
import scipy as sp
import problem_library as pl
import Solver
import preprocessor as prep
from auxilliary_classes import *
numpy.set_printoptions(threshold=numpy.nan)


def grid_convergence(go, load = None):
    domain, geom, basis, s, p, ischeme = go.domain, go.geom, go.basis, go.s, go.degree, go.ischeme
    if not load:
        load = numpy.ones(len(basis))
    mapping = basis.vector(2).dot(s)
    funcs = []
    for i in range(3):
        if i != 0:
            domain = domain.refine(1)
            basis = domain.basis('spline', degree = p)
        mat = domain.integrate(function.outer( basis.grad(mapping,ndims=2), basis.grad(mapping,ndims=2) ).sum(2), geometry = mapping, ischeme = gauss(ischeme))
        rhs = domain.integrate(basis,geometry = geom, ischeme = gauss(ischeme))
        cons = domain.boundary.project(0, onto = basis, geometry = geom, ischeme = gauss(ischeme))
        lhs = mat.solve(rhs, constrain=cons, symmetric = True)
        funcs.append(basis.dot(lhs))
    diff1, diff2 = np.sqrt(domain.integrate([(funcs[1] - funcs[0])**2, (funcs[2] - funcs[1])**2], geometry = geom, ischeme = gauss(ischeme)))
    return np.log(diff1/diff2)/np.log(2)


def flatten_weights_to_gismo(go):
    s, knots, p = go.s, go.knots, go.degree
    l = len(s) // 2
    n,m = go.ndims
    s1, s2 = [vec.reshape(n,m).T.flatten()[:,None] for vec in [s[:l], s[l:]]]
    return str(np.hstack([s1,s2])).replace('[','').replace(']','')
    


def export_to_gismo(go):
    p, knots = go.degree, go._knots
    knots_xi, knots_eta = [(str(kn.extend_knots())[1:-1]).replace(',','') for kn in knots]
    string = '<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n' \
             ' <xml> \n' \
             '   <Geometry type=\"TensorBSpline2\" id=\"0\"> \n' \
             '   <Basis type=\"TensorBSplineBasis2\"> \n' \
            '     <Basis type=\"BSplineBasis\" index=\"0\"> \n' \
            '<KnotVector degree=\"' + str(p[0]) + '\"> ' + knots_xi +' </KnotVector> \n' \
            '     </Basis> \n' \
            '     <Basis type=\"BSplineBasis\" index=\"1\"> \n' \
            '<KnotVector degree=\"' + str(p[1]) + '\"> ' + knots_eta +' </KnotVector> \n' \
            '     </Basis> \n' \
            '    </Basis> \n' \
            '    <coefs geoDim=\"2\"> \n'\
            +flatten_weights_to_gismo(go)+'\n' \
            '    </coefs> \n' \
            '   </Geometry> \n' \
            '</xml>'
    #return np.asarray([string])
    return string


def export_to_gismo_1D(go):
    p, knots = go.degree, go.knots
    l = len(go.s) // 2
    retvec = numpy.vstack([go.s[:l], go.s[l:]]).T
    assert isinstance(knots, ut.knot_object)
    knots_xi = (str(go.knots.extend_knots(go.degree))[1:-1]).replace(',','')
    string = '<?xml version="1.0" encoding="UTF-8"?> \n' \
             '    <xml> \n' \
             '    <Geometry type="BSpline"> \n' \
             '    <Basis type="BSplineBasis"> \n' \
            '    <KnotVector degree=\"' + str(p) + '\"> ' + knots_xi +' </KnotVector> \n' \
             '    </Basis> \n' \
            '   <coefs geoDim=\"2\"> \n' \
            + str(retvec).replace('[','').replace(']','') + '\n' \
            '</coefs> \n' \
            '</Geometry> \n' \
            '</xml>'
    return np.asarray([string])