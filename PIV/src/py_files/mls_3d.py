import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
import corrLib as cl
import os
import sys
import time
import pdb

"""
Smoothing a 3D matrix using moving least square algorithm.
"""
# determine neighbors
def determine_neighbors(pivData, coords, dm):
    dist = ((pivData.x - coords[0])**2 + (pivData.y-coords[1])**2)**.5
    neighbors = pivData[dist<=dm]
    return neighbors
    
# weight function
def weight_function(coords1, coords2, dm):
    # here we still introduce dm, in case the neighbor list is not correct
    distsq = (coords1[0] - coords2[0]) ** 2 + (coords1[1] - coords2[1]) ** 2
    if distsq <= dm**2:
        f = (np.exp(1-distsq/dm**2) - 1) / (np.e - 1)
    else:
        f = 0
    return f

# calculate matrix A
def calculate_matrix_A(coords, neighbors, dm):
    # assume polynomial basis for trial
    k = 0
    for num, i in neighbors.iterrows():
        c = [i.x, i.y]
        P = cubic_basis(c)
        if k == 0:
            A = weight_function(coords, c, dm) * np.outer(P, P)
        else:
            A += weight_function(coords, c, dm) * np.outer(P, P)
        k += 1
    return A  
    
# calculate matrix B
def calculate_matrix_B(coords, neighbors, dm):
    # assume polynomial basis for trial
    B = []
    k = 0
    for num, i in neighbors.iterrows():
        c = [i.x, i.y]
        P = cubic_basis(c)
        B.append(weight_function(coords, c, dm) * P)
    return np.array(B)
    
# get exp data
def get_exp_data(pivData, neighbors):
    w = []
    for num, i in neighbors.iterrows():
        c = [i.x, i.y]
        subdata = pivData.loc[(pivData.x==c[0])&(pivData.y==c[1])]
        w.append([subdata.u.values[0], subdata.v.values[0]])
    return np.array(w).transpose()
    
# calculate cubic basis
def cubic_basis(coords):
    x = coords[0]
    y = coords[1]
    P = np.array([1, x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2])
    return P

# Take derivatives of the cubic fitting
def cubic_derivatives(coef):
#     coef = np.array(coef)
    dudxL = []
    dudyL = []
    dvdxL = []
    dvdyL = []
    for num, i in coef.iterrows():
        x = i['x']
        y = i['y']
        dudx = i['u_x'] + i['u_x2']*2*x + i['u_xy']*y + i['u_x3']*3*x**2 + i['u_x2y']*2*x*y + i['u_y2x']*y**2
        dudy = i['u_y'] + i['u_y2']*2*y + i['u_xy']*x + i['u_y3']*3*y**2 + i['u_y2x']*2*x*y + i['u_x2y']*x**2
        dvdx = i['v_x'] + i['v_x2']*2*x + i['v_xy']*y + i['v_x3']*3*x**2 + i['v_x2y']*2*x*y + i['v_y2x']*y**2
        dvdy = i['v_y'] + i['v_y2']*2*y + i['v_xy']*x + i['v_y3']*3*y**2 + i['v_y2x']*2*x*y + i['v_x2y']*x**2
        dudxL.append(dudx)
        dudyL.append(dudy)
        dvdxL.append(dvdx)
        dvdyL.append(dvdy)
    deriv = pd.DataFrame().assign(x=coef['x'], y=coef['y'], dudx=dudxL, dudy=dudyL, dvdx=dvdxL, dvdy=dvdyL)
    return deriv
    
# 1 frame process using MLS
def mls_smoothing_1(pivData, dm):
    uv = []
    coefL = []
    for num, i in pivData.iterrows():
        coords = [i.x, i.y]
        neighbors = determine_neighbors(pivData, coords, dm)
        A = calculate_matrix_A(coords, neighbors, dm)
        B = calculate_matrix_B(coords, neighbors, dm)
        w = get_exp_data(pivData, neighbors)
        coef1 = np.inner(np.inner(np.linalg.inv(A), B), w).transpose()
        coefL.append(coef1.flatten())
        smoothed_velocity = np.inner(np.inner(cubic_basis(coords), np.inner(np.linalg.inv(A), B).transpose()), w)
        uv.append(smoothed_velocity)
    uv = np.array(uv)
    coefL = np.array(coefL)
    smoothed_piv = pd.DataFrame().assign(x=pivData.x, y=pivData.y, u=uv[:, 0], v=uv[:, 1])
    coef = pd.DataFrame().assign(x=pivData.x, y=pivData.y, 
                                 u_1=coefL[:, 0], u_x=coefL[:, 1], u_y=coefL[:, 2], u_x2=coefL[:, 3], u_y2=coefL[:, 4],
                                 u_xy=coefL[:, 5], u_x3=coefL[:, 6], u_y3=coefL[:, 7], u_x2y=coefL[:, 8], u_y2x=coefL[:, 9],
                                 v_1=coefL[:, 10], v_x=coefL[:, 11], v_y=coefL[:, 12], v_x2=coefL[:, 13], v_y2=coefL[:, 14],
                                 v_xy=coefL[:, 15], v_x3=coefL[:, 16], v_y3=coefL[:, 17], v_x2y=coefL[:, 18], v_y2x=coefL[:, 19])
    return smoothed_piv, coef

# multi-frame process using MLS
folder = r'E:\moreData\02042020\piv_result_50\80-1'
l = cl.readdata(folder)
pivData = pd.read_csv(l.Dir[0])
dm = 100
smoothed_piv, coef = mls_smoothing_1(pivData, dm)
pdb.set_trace()
