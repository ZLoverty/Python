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

folder_div = sys.argv[1]
folder_dcadv = sys.argv[2]
folder_out = sys.argv[3]

if os.path.exists(folder_out) == 0:
    os.makedirs(folder_out)
with open(os.path.join(folder_out, 'log.txt'), 'w') as f:
    pass

folder_list_dt = next(os.walk(folder_dcadv))[1]
adv_divvL = [] # dcadv X divv
adv_divcvL = [] # dcadv X divcv
dc_divvL = [] # dc X divv
dc_divcvL = [] # dc X divcv
dtL = [] # Delta t
for s in folder_list_dt:
    dt = int(s.split('=')[1])
    folder = os.path.join(folder_dcadv, s)
    l = cl.readdata(folder)
    count = 0
    adv_divv = 0
    adv_divcv = 0
    dc_divv = 0
    dc_divcv = 0
    for num, i in l.iterrows():
        f, file = os.path.split(i.Dir)
        name = file.split('-')[0]
        n0 = int(name)
        n1 = n0 + 1
        divDir = os.path.join(folder_div, '{0:04d}-{1:04d}.csv'.format(n0, n1))
        divData = pd.read_csv(divDir)
        divv = divData['divv']
        divcv = divData['divcv']
        advData = pd.read_csv(i.Dir)
        adv = advData['adv']
        dc = advData['dc']
        adv_divv += ((divv - divv.mean())*(adv - adv.mean())).mean() / divv.std() / adv.std()
        adv_divcv += ((divcv - divv.mean())*(adv - adv.mean())).mean() / divcv.std() / adv.std()
        dc_divv += ((divv - divv.mean())*(dc - dc.mean())).mean() / divv.std() / dc.std()
        dc_divcv += ((divv - divcv.mean())*(dc - dc.mean())).mean() / divcv.std() / dc.std()
        count += 1
    adv_divv /= count
    adv_divcv /= count
    dc_divv /= count
    dc_divcv /= count
    adv_divvL.append(adv_divv)
    adv_divcvL.append(adv_divcv)
    dc_divvL.append(dc_divv)
    dc_divcvL.append(dc_divcv)
    dtL.append(dt)
    with open(os.path.join(folder_out, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // ' + 'dt={0:d} calculated\n'.format(dt))

# Save data
data = pd.DataFrame().assign(dt=dtL, adv_divv=adv_divvL, adv_divcv=adv_divcvL, dc_divv=dc_divvL, dc_divcv=dc_divcvL)
data.to_csv(os.path.join(folder_out, 'divvXdcadv.csv'), index=False)

""" SYNTAX
python div_x_dcadv.py folder_div folder_dcadv folder_out

Note that folder_dcadv should contain subfolders named "dt=N" (N is integer) to indicate the choice of \Delta t, even if there is only one dt. Otherwise, the code will not execute properly.
"""

""" TEST PARAMETERS
folder_div = r'I:\Github\Python\Correlation\test_images\divvXdcadv\div'
folder_dcadv = r'I:\Github\Python\Correlation\test_images\divvXdcadv\dcadv'
folder_out = r'I:\Github\Python\Correlation\test_images\divvXdcadv\result'
"""

""" LOG
Mon Jun 22 11:36:25 2020 // dt=3 calculated
Mon Jun 22 11:36:25 2020 // dt=7 calculated
"""