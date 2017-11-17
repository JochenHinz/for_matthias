#! /usr/bin/env python3

from nutils import *
from auxilliary_classes import *
import numpy


@log.title
def makeplots( domain, geom, stress ):

  points, colors = domain.elem_eval( [ geom, stress[0,1] ], ischeme='bezier3', separate=True )
  with plot.PyPlot( 'stress', ndigits=0 ) as plt:
    plt.mesh( points, colors, tight=False )
    plt.colorbar()


def main(from_go_, to_go_, lmbda = 1., mu = 1., alpha = 1):
    assert from_go_.periodic == to_go_.periodic
    assert all([len(from_go_.ndims) == 2, len(to_go_.ndims) == 2])
    
    ## prolong everything to unified grid

    from_go = from_go_ + to_go_
    
    to_go = to_go_ + from_go_
    
    final_go = from_go_ + to_go_
    
    #final_go.quick_plot_boundary()
    
    ##
    
    dbasis = to_go.basis.vector(2)
    domain = to_go.domain

  # construct matrix
    stress = library.Hooke( lmbda=lmbda, mu=mu )
    elasticity = function.outer( dbasis.grad(from_go.mapping()), stress(dbasis.symgrad(from_go.mapping())) ).sum([2,3])
    matrix = domain.integrate( elasticity, geometry=from_go.mapping(), ischeme = gauss(from_go.ischeme) )

  # construct dirichlet boundary constraints
    cons = alpha*(to_go.cons - from_go.cons)

  # solve system
    lhs = matrix.solve( constrain=cons, tol=1e-10, symmetric=True, precon='diag' )

  # construct solution function
    final_go.cons = final_go.cons + cons
    final_go.s = final_go.s + lhs

    return final_go
