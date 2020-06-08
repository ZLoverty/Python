import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
import os
import corrLib as cl
import matplotlib
import myImageLib as mil
import sys
import time
import pdb
"""
Batch calculation of dc+advection.
fix windowsize=50 and step=25
"""
# folder_img = r'E:\Github\Python\Correlation\test_images\dc+adv\img'
# folder_piv = r'E:\Github\Python\Correlation\test_images\dc+adv\piv'
# folder_out = r'E:\Github\Python\Correlation\test_images\dc+adv\out'
# interval = 1
# fps = 30
# step = 25

folder_img = sys.argv[1]
folder_piv = sys.argv[2]
folder_out = sys.argv[3]
interval = int(sys.argv[4])
fps = int(sys.argv[5])
step = int(sys.argv[6])

if os.path.exists(folder_out) == False:
    os.makedirs(folder_out)
with open(os.path.join(folder_out, 'log.txt'), 'w') as f:
    f.write('adv = dc/interval + ux/fps*dcx/step + uy/fps*dcy/step\n')
    f.write('interval = {:d}\n'.format(interval))

limg = cl.readseq(folder_img)
I0 = io.imread(limg.Dir[0])
X, Y, I0s = cl.divide_windows(I0, windowsize=[50, 50], step=25)

# generate shift matrices
shiftx = np.zeros(I0s.shape)
shifty = np.zeros(I0s.shape)
for x in range(0, I0s.shape[0]):
    for y in range(0, I0s.shape[1]):
        if x == y - 1:
            shiftx[x, y] = 1
for x in range(0, I0s.shape[0]):
    for y in range(0, I0s.shape[1]):
        if x == y + 1:
            shifty[x, y] = 1
shiftx[0, 0] = 1
shifty[0, 0] = 1

# load piv and corresponding images
l = cl.readdata(folder_piv)
for num, i in l.iterrows():
    name = i.Name
    n0 = int(name.split('-')[0])
    n1 = n0 + interval
    if n1 > len(limg) -1:
        continue
    I0 = io.imread(os.path.join(folder_img, '{:04d}.tif'.format(n0)))
    I1 = io.imread(os.path.join(folder_img, '{:04d}.tif'.format(n1)))
    X, Y, I0s = cl.divide_windows(I0, windowsize=[50, 50], step=25)
    X, Y, I1s = cl.divide_windows(I1, windowsize=[50, 50], step=25)
    pivData = pd.read_csv(i.Dir)
    ux = np.array(pivData.u).reshape(I0s.shape)
    uy = np.array(pivData.v).reshape(I0s.shape)
    dcx = I0s - np.matmul(I0s, shiftx)
    dcy = I0s - np.matmul(shifty, I0s)
    dc = I1s - I0s
    adv = dc/interval + ux/fps*dcx/step + uy/fps*dcy/step
    data = pd.DataFrame().assign(x=pivData.x, y=pivData.y, dcx=dcx.flatten(), dcy=dcy.flatten(), dc=dc.flatten(), adv=adv.flatten())
    data.to_csv(os.path.join(folder_out, '{:04d}-{:04d}.csv'.format(n0, n1)), index=False)
    with open(os.path.join(folder_out, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // {:04d}-{:04d} calculated\n'.format(n0, n1))


""" TEST COMMAND
python dc_adv.py folder_img folder_piv folder_out interval fps step
"""
        
"""  TEST PARAMS
folder_img = r'E:\Github\Python\Correlation\test_images\dc+adv\img'
folder_piv = r'E:\Github\Python\Correlation\test_images\dc+adv\piv'
folder_out = r'E:\Github\Python\Correlation\test_images\dc+adv\out'
interval = 1
fps = 30
step = 25
"""

""" LOG
adv = dc/interval + ux/fps*dcx/step + uy/fps*dcy/step
interval = 1
Mon Jun  8 16:51:36 2020 // 0000-0001 calculated
Mon Jun  8 16:51:36 2020 // 0002-0003 calculated
""" 

