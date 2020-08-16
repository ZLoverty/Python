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
    f.write('Input folder: ' + str(input_folder) + '\n')
    f.write('Output folder: ' + str(output_folder) + '\n')
    f.write('wsize: ' + str(wsize) + '\n')
    f.write('step: ' + str(step) + '\n')
    
l = readseq(input_folder)
num_frames = len(l)
num_sample = 100 # can modify in the future
if num_sample <= num_frames:
    for num, i in l.iterrows():
        if num % int(num_frames / num_sample):
            img = io.imread(i.Dir)
            # bp = bpass(img, 3, 100)
            X, Y, I = divide_windows(img, windowsize=[wsize, wsize], step=step)
            XI, YI, CI = corrI(X, Y, I)
            dc = distance_corr(XI, YI, CI)
            # Save data
            dc.to_csv(os.path.join(output_folder, i.Name+'.csv'), index=False)
            # Write log
            with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
                f.write(time.asctime() + ' // ' + i.Name + ' calculated\n')
else:
    for num, i in l.iterrows():
        img = io.imread(i.Dir)
        X, Y, I = divide_windows(img, windowsize=[wsize, wsize], step=step)
        XI, YI, CI = corrI(X, Y, I)
        dc = distance_corr(XI, YI, CI)
        dc.to_csv(os.path.join(output_folder, i.Name+'.csv'), index=False)
        with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
            f.write(time.asctime() + ' // ' + i.Name + ' calculated\n')

""" Edit
08162020 - remove bpass step
           change corrI return value according to the change of corrI(), to speed up the code
           write parameters in log
           down sampling: instead of computing correlations for all frames, now only take 100 frames
                          if the video is shorter than 100 frames, do the whole video
"""

""" TEST COMMAND
python corr_imseq.py input_folder output_folder wsize step
"""
        
"""  TEST PARAMS
input_folder = E:\Github\Python\Correlation\test_images\cl
output_folder = E:\Github\Python\Correlation\test_images\cl\result_test
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
    
    
    
    