import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corrLib
from skimage import io
import os
import time
from myImageLib import dirrec
from scipy.signal import savgol_filter
import cv2
import sys


"""
compute divergence, vorticity, convection and div(cn), save data in .npy binary files.
All quantities are saved in subfolders under 'flow_folder'

Here, divcn is computed by assuming the relation between intensity and concentration to be the calibration curve:
c = 186 - 0.19I

function vorticity, divergence divcn and convection should be defined
"""

def divcn(pivData, img, winsize, step=None, shape=None):
    """
    Compute the div(cn) term from PIV and image.
    
    Args:
    pivData -- (X, Y, U, V)
    img -- 2d matrix
    winsize -- window size of PIV, used to scale down image
    step -- step of PIV, used to scale down image
    
    Returns:
    div_cn -- 2d matrix
    """
    
    x = pivData.sort_values(by=['x']).x.drop_duplicates()
    if step == None:
        # Need to infer the step size from pivData
        step = x.iat[1] - x.iat[0]
    
    if shape == None:
        # Need to infer shape from pivData
        y = pivData.y.drop_duplicates()
        shape = (len(y), len(x))
        
    X, Y, I = corrLib.divide_windows(img, windowsize=[winsize, winsize], step=step)
    c = 186 - 0.19 * I
    
    assert(I.shape == shape)
    X = np.array(pivData.x).reshape(shape)
    Y = np.array(pivData.y).reshape(shape)
    U = np.array(pivData.u).reshape(shape)
    V = np.array(pivData.v).reshape(shape)
    
    cu = c * U
    cv = c * V
    
    dudx = np.gradient(cu, step, axis=1)
    dvdy = np.gradient(cv, step, axis=0)
    div_cn = dudx + dvdy
    
    return div_cn

piv_folder = sys.argv[1]
img_folder = sys.argv[2] # for computing divcn
flow_folder = sys.argv[3]

winsize = int(sys.argv[4])
step = int(sys.argv[5])

flow_derivs_list = ['divergence', 'vorticity', 'convection', 'divcn']

sub_folders = {}
if os.path.exists(flow_folder) == False:
    os.makedirs(flow_folder)
# also create subfolders for divv, vort, conv and divcn
for kw in flow_derivs_list:
    sub_folders[kw] = os.path.join(flow_folder, kw)
    if os.path.exists(sub_folders[kw]) == False:
        os.makedirs(sub_folders[kw])
with open(os.path.join(flow_folder, 'log.txt'), 'w') as f:
    f.write('piv_folder: ' + piv_folder + '\n')
    f.write('img_folder: ' + img_folder + '\n')
    f.write('flow_folder: ' + flow_folder + '\n')
    f.write('winsize: ' + str(winsize) + '\n')
    f.write('step: ' + str(step) + '\n')
    
l = corrLib.readdata(piv_folder)
data = {}

for num, i in l.iterrows():
    img_num = int(i.Name.split('-')[0])
    img = io.imread(os.path.join(img_folder, '{:04d}.tif'.format(img_num)))
    pivData = pd.read_csv(i.Dir)
    divv = corrLib.divergence(pivData)
    vort = corrLib.vorticity(pivData)
    conv = corrLib.convection(pivData, img, winsize)
    div_cn = divcn(pivData, img, winsize)    
    np.save(os.path.join(sub_folders['divergence'], i.Name+'.npy'), divv)
    np.save(os.path.join(sub_folders['vorticity'], i.Name+'.npy'), vort)
    np.save(os.path.join(sub_folders['convection'], i.Name+'.npy'), conv)
    np.save(os.path.join(sub_folders['divcn'], i.Name+'.npy'), div_cn)
    with open(os.path.join(flow_folder, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // ' + i.Name + ' done\n')

""" SYNTAX
python flow-derivs.py piv_folder img_folder flow_folder winsize step

piv_folder: PIV data folder 
img_folder: .tif image sequence folder 
flow_folder: output folder, with subfolders  
winsize: PIV window size 
step: PIV step size
"""

""" TEST PARAMS
piv_folder: E:\Github\Python\Correlation\test_images\dc+adv\piv
img_folder: E:\Github\Python\Correlation\test_images\dc+adv\img
flow_folder: E:\Github\Python\Correlation\test_images\dc+adv\flow_derivs
winsize: 50
step: 25
"""

""" LOG 
piv_folder: E:\Github\Python\Correlation\test_images\dc+adv\piv
img_folder: E:\Github\Python\Correlation\test_images\dc+adv\img
flow_folder: E:\Github\Python\Correlation\test_images\dc+adv\flow_derivs
winsize: 50
step: 25
Tue Sep  1 15:35:07 2020\0000-0001 done
Tue Sep  1 15:35:07 2020\0002-0003 done
"""