import pandas as pd
import numpy as np
import corrLib as cl
import os
import sys
import time

# pivData = pd.read_csv(r'E:\moreData\02042020\piv_result_50\80-1\0000-0001.csv')
# mpp = 0.33
def inner(Ax, Ay, Bx, By):
    """
    define inner product between two matrices
    """
    return (Ax*Bx + Ay*By) / (Ax**2+Ay**2)**0.5 / (Bx**2+By**2)**0.5
    
folder_piv = sys.argv[1]
folder_out = sys.argv[2]
mpp = float(sys.argv[3])
fps = int(sys.argv[4])

if os.path.exists(folder_out) == False:
    os.makedirs(folder_out)
with open(os.path.join(folder_out, 'log.txt'), 'w') as f:
    pass

t = []
E = []
OP = []

l = cl.readdata(folder_piv)

# load a sample PIV data
if l.empty == False:
    pivData_sample = pd.read_csv(l.Dir[0])
else:
    raise ValueError('The folder given does not contain data')

# measure the dimensions of PIV data  
col = len(pivData_sample.x.drop_duplicates())
row = len(pivData_sample.y.drop_duplicates())

# create shift matrix
shift = np.zeros((row, col))
for i in range(0, min(row, col)):
    shift[i, (i+1)%min(row, col)] = 1
    
for num, i in l.iterrows():
    t_frame = int(i.Name.split('-')[0])
    t_second = t_frame / fps
    t.append(t_second)
    pivData = pd.read_csv(i.Dir)
    # calculate energy
    E.append(((pivData.u * mpp)**2 + (pivData.v * mpp)**2).mean())
    # calculate order
    u = np.array(pivData.u).reshape((row, col))
    v = np.array(pivData.v).reshape((row, col))
    # calculate shifted matrices
    # u1 = np.matmul(shift, u) # up
    # v1 = np.matmul(shift, v) # up
    # u2 = np.matmul(shift.transpose(), u) # down
    # v2 = np.matmul(shift.transpose(), v) # down
    # u3 = np.matmul(u, shift) # right
    # v3 = np.matmul(v, shift) # right
    # u4 = np.matmul(u, shift.transpose()) # left
    # v4 = np.matmul(v, shift.transpose()) # left
    
    u1 = np.roll(u, -1, axis=0) # up
    v1 = np.roll(v, -1, axis=0) # up
    u2 = np.roll(u, 1, axis=0) # down
    v2 = np.roll(v, 1, axis=0) # down
    u3 = np.roll(u, 1, axis=1) # right
    v3 = np.roll(v, 1, axis=1) # right
    u4 = np.roll(u, -1, axis=1) # left
    v4 = np.roll(v, -1, axis=1) # left
    
    # do inner products with original matrix
    I1 = inner(u, v, u1, v1)
    I2 = inner(u, v, u2, v2)
    I3 = inner(u, v, u3, v3)
    I4 = inner(u, v, u4, v4)
    # average the products
    I = (I1 + I2 + I3 + I4) / 4
    # exclude the outer row and column, to get a region of interest
    I_roi = I[1:row-1, 1:col-1]
    # threshold setting and OP calculation
    I_roi[I_roi < .9] = 0
    OP.append(np.count_nonzero(I_roi) / col / row)
    with open(os.path.join(folder_out, 'log.txt'), 'a') as f: 
        f.write(time.asctime() + ' // t = {0:.2f} s calculated\n'.format(t_second))
data = pd.DataFrame().assign(t=t, E=E, OP=OP)
data.to_csv(os.path.join(folder_out, 'energy_order.csv'), index=False)

""" SYNTAX
python energy_order.py folder_piv folder_out mpp fps
"""

""" TEST PARAMS
folder_piv = E:\Github\Python\Correlation\test_images\energy_order\piv_data
folder_out = E:\Github\Python\Correlation\test_images\energy_order\out
mpp = 0.33
fps = 10
"""

""" LOG
Thu Jul  2 17:02:10 2020 // t = 90.00 s calculated
Thu Jul  2 17:02:10 2020 // t = 90.20 s calculated
"""