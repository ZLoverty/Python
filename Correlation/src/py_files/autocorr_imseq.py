import pandas as pd
from corrLib import readseq, match_hist
from myImageLib import bpass
from skimage import io
import numpy as np
import sys
import os
import time

input_folder = sys.argv[1]
output_folder = sys.argv[2]

if os.path.exists(output_folder) == 0:
    os.makedirs(output_folder)

with open(os.path.join(output_folder, 'log.txt'), 'w') as f:
    pass
    
with open(os.path.join(output_folder, 'autocorr_data.csv'), 'w') as f:
    f.write('Name,ac\n')
    
center = [500, 500]
box = 40
xcoor = range(center[1]-int(box/2), center[1]+int(box/2))
ycoor = range(center[0]-int(box/2), center[0]+int(box/2))

Iseq = []
l = readseq(input_folder)
for num, i in l.iterrows():
    img = io.imread(i.Dir)
    bp = bpass(img, 3, 100)
    mh = match_hist(bp, img)
    subbox = mh[xcoor, ycoor]
    I = subbox.mean()
    Iseq.append(I)
    with open(os.path.join(output_folder, 'autocorr_data.csv'), 'a') as f:
        f.write(i.Name + ',' + str(I) + '\n')    
    with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // ' + i.Name + ' calculated\n')
        
data = l.assign(ac=Iseq)[['Name', 'ac']]

data.to_csv(os.path.join(output_folder, 'autocorr.csv'), index=False)

""" TEST COMMAND
python autocorr_imseq.py input_folder output_folder wsize step
"""
        
"""  TEST PARAMS
input_folder = I:\Github\Python\Correlation\test_images\cl
output_folder = I:\Github\Python\Correlation\test_images\cl\ac_result
wsize = 20
step = 20
"""

""" LOG
Mon Jan 13 11:21:44 2020 // 100-2 calculated
Mon Jan 13 11:22:05 2020 // 40-2 calculated
Mon Jan 13 11:22:27 2020 // 60-2 calculated
Mon Jan 13 11:22:50 2020 // 80-2 calculated
"""