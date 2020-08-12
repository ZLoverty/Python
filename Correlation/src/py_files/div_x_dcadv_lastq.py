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

"""
Similar to div_x_dcadv.py, but only deal with the last 1/10 of data.
"""

folder_div = sys.argv[1]
folder_dcadv = sys.argv[2]
folder_out = sys.argv[3]

if os.path.exists(folder_out) == 0:
    os.makedirs(folder_out)
with open(os.path.join(folder_out, 'log.txt'), 'w') as f:
    pass

folder_list_dt = next(os.walk(folder_dcadv))[1]

entries = ['adv_divv', 'adv_divcv', 'dc_divv', 'dc_divcv', 'vdc_divv', 'vdc_divcv']

corr_lists = {}
corr_values = {}

for entry in entries:
    corr_lists[entry] = []    

dtL = [] # Delta t
for s in folder_list_dt:
    dt = int(s.split('=')[1])
    folder = os.path.join(folder_dcadv, s)
    l = cl.readdata(folder)
    l = l.loc[l.index > len(l)*0.9]
    count = 0

    for entry in entries:
        corr_values[entry] = 0
    for num, i in l.iterrows():
        f, file = os.path.split(i.Dir)
        name = file.split('-')[0]
        n0 = int(name)
        n1 = n0 + 1
        divDir = os.path.join(folder_div, '{0:04d}-{1:04d}.csv'.format(n0, n1))
        divData = pd.read_csv(divDir)
        advData = pd.read_csv(i.Dir)

        for entry in entries:
            key1, key2 = entry.split('_')
            mat1, mat2 = advData[key1], divData[key2]
            corr_values[entry] += ((mat1 - mat1.mean())*(mat2 - mat2.mean())).mean() / mat1.std() / mat2.std()
        
        count += 1
    for entry in entries:
        corr_lists[entry].append(corr_values[entry] / count)
    print(corr_lists)
    dtL.append(dt)
    with open(os.path.join(folder_out, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // ' + 'dt={0:d} calculated\n'.format(dt))

# Save data
data = pd.DataFrame.from_dict(corr_lists).assign(dt=dtL)
data.to_csv(os.path.join(folder_out, 'divvXdcadv.csv'), index=False)

""" SYNTAX
python div_x_dcadv.py folder_div folder_dcadv folder_out

Note that folder_dcadv should contain subfolders named "dt=N" (N is integer) to indicate the choice of \Delta t, even if there is only one dt. Otherwise, error may occur.
"""

""" TEST PARAMETERS
folder_div = r'E:\Github\Python\Correlation\test_images\divvXdcadv\div'
folder_dcadv = r'E:\Github\Python\Correlation\test_images\divvXdcadv\dcadv'
folder_out = r'E:\Github\Python\Correlation\test_images\divvXdcadv\result'
"""

""" LOG
Mon Jun 22 11:36:25 2020 // dt=3 calculated
Mon Jun 22 11:36:25 2020 // dt=7 calculated
"""