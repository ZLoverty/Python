from corrLib import corrS, corrI, divide_windows, distance_corr, corrIseq, readseq, density_fluctuation
from myImageLib import bpass
from scipy.signal import savgol_filter
from skimage import io
import pandas as pd
import os
import time
import sys

input_folder = sys.argv[1]
output_folder = sys.argv[2]
wsize = int(sys.argv[3])
step = int(sys.argv[4])

# check output dir existence
if os.path.exists(output_folder) == 0:
    os.makedirs(output_folder)
with open(os.path.join(output_folder, 'log.txt'), 'w') as f:
    pass
l = readseq(input_folder)
for num, i in l.iterrows():
    print('Frame ' + i.Name)
    img = io.imread(i.Dir)
    bp = bpass(img, 3, 100)
    X, Y, I = divide_windows(bp, windowsize=[wsize, wsize], step=step)
    CI = corrI(X, Y, I)
    dc = distance_corr(X, Y, CI)
    # Save data
    dc.to_csv(os.path.join(output_folder, i.Name+'.csv'), index=False)
    # Write log
    with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // ' + i.Name + ' calculated\n')


""" TEST COMMAND
python corr_imseq.py input_folder output_folder wsize step
"""
        
"""  TEST PARAMS
input_folder = I:\Github\Python\Correlation\test_images\cl
output_folder = I:\Github\Python\Correlation\test_images\cl\result_test
wsize = 20
step = 20
"""

""" LOG
Mon Jan 13 11:21:44 2020 // 100-2 calculated
Mon Jan 13 11:22:05 2020 // 40-2 calculated
Mon Jan 13 11:22:27 2020 // 60-2 calculated
Mon Jan 13 11:22:50 2020 // 80-2 calculated
"""

""" SPEED PER FRAME
# min_size=20, step=min_size, t=1min 35s
# min_size=20, step=boxsize, t=3.1s
# min_size=20, step=5*min_size, t=5.14 s
"""
    
    
    
    