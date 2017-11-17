from nutils import *
import numpy as np
import scipy as sp
import shapely.geometry as sh
import problem_library as pl
from matplotlib import pyplot as plt
import reparam as rep

def rotate_all(x,y,angle):
    x_, y_ = [pl.rotate(x, angle), pl.rotate(y, -4/6.0*angle, center = np.array([56.52,0]))]
    i,j = [np.argmax(pl.norm(pc, center = cusp_center)) for pc in [x_,y_]]
    x_ = np.roll(x_, -i, axis = 1)
    y_ = np.roll(y_, -j, axis = 1)
    return x_, y_

def False_True(l):  # return indices where list switches from false to true or vice versa, assume it starts on false
    ret = []
    indicator = True
    for i in range(len(l)):
        if l[i] == indicator:
            ret.append(i)
            indicator = not indicator
    ret = [[ret[i], ret[i+1]] for i in range(0,len(ret),2)]
    return ret

distance = rep.distance

def separator(angle_):
    
    c_1, c_0 = pl.cusp(36.1,36, 56.52)
    theta_0, theta_1 = pl.theta(c_0[:,None])[0], pl.theta(c_1[:,None])[0]   ## CUSP angles from [0,0]
    theta_2, theta_3 = pl.theta(c_0[:,None],center = np.array([56.52,0]))[0], pl.theta(c_1[:,None], center = np.array([56.52,0]))[0]


    male, female = pl.twin_screw()
    rotate_all = lambda x,y,angle: [pl.rotate(x, angle), pl.rotate(y, -4/6.0*angle, center = np.array([56.52,0]))]
    ## circle.T[250].T = c_0; circle.T[750].T = c_1
    cusp_center = (c_1 + c_0)/2.0
    circle = pl.circle_point(np.linalg.norm((c_1-c_0)/2),  - np.pi + np.linspace(0,2*np.pi,1001)) + cusp_center[:,None]
    cuspolygon = sh.Polygon([p for p in circle.T])
    
    def cut_paste(x_, y_):
        lists = [[sh.Point(P).within(cuspolygon) for P in z.T] for z in [x_,y_]]  ## list of indices inside and outside of CUSP
        lists = [False_True(l) for l in lists]
        list_0, list_1 = [lists[i][np.argmax([np.diff(j) for j in lists[i]])] for i in range(2)]
        i_0, i_1 = [np.argmin(distance(circle,j[:,None])) for j in [x_.T[list_0[0]].T,y_.T[list_1[0]].T]]
        i_3, i_2 = [np.argmin(distance(circle,j[:,None])) for j in [x_.T[list_0[1]].T,y_.T[list_1[1]].T]]
        return circle.T[i_0:i_1+1].T, circle.T[i_2:i_3+1].T, x_.T[list_0[0]:list_0[1]+1].T, y_.T[list_1[0]:list_1[1]+1].T
    
    male_, female_ = rotate_all(male, female, angle_)
    bottom, top, left, right = cut_paste(male_,female_)
    top = top.T[::-1].T
    left = np.concatenate([bottom[:,0][:,None], left.T[1:-1].T, top[:,0][:,None]], axis = 1)
    right = np.concatenate([bottom[:,-1][:,None], right.T[1:-1].T, top[:,-1][:,None]], axis = 1)
    
    return bottom, top, left, right