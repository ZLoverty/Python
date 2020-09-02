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

""" DESCRIPTION
Compute local concentration fluctuations, and save data in .npy binary files.
"""

img_folder = sys.argv[1]
save_folder = sys.argv[2]
seg_length = int(sys.argv[3])
winsize = int(sys.argv[4])
step = int(sys.argv[5])

# local_df(img_folder, seg_length=50, winsize=50, step=25)
if os.path.exists(save_folder) == False:
    os.makedirs(save_folder)
with open(os.path.join(save_folder, 'log.txt'), 'w') as f:
    f.write('img_folder: ' + img_folder + '\n')
    f.write('save_folder: ' + save_folder + '\n')
    f.write('seg_length: ' + str(seg_length) + '\n')
    f.write('winsize: ' + str(winsize) + '\n')
    f.write('step: ' + str(step) + '\n')
    f.write(time.asctime() + ' // computation starts!\n')
    
df = corrLib.local_df(img_folder, seg_length=seg_length, winsize=winsize, step=step)

for t, std in zip(df['t'], df['std']):
    np.save(os.path.join(save_folder, '{:04d}.npy'.format(t)), std)
    
with open(os.path.join(save_folder, 'log.txt'), 'a') as f:
    f.write(time.asctime() + ' // computation ends!\n')
    
""" SYNTAX
python local_df.py img_folder save_folder seg_length winsize step
 
img_folder: .tif image sequence folder 
save_folder: output folder
seg_length: length of each segment 
winsize: PIV window size 
step: PIV step size
"""

""" TEST PARAMS
img_folder: E:\Github\Python\Correlation\test_images\df2_kinetics\img
save_folder: E:\Github\Python\Correlation\test_images\df2_kinetics\local_df
seg_length: 50
winsize: 50
step: 25
"""

""" LOG 
img_folder: E:\Github\Python\Correlation\test_images\df2_kinetics\img
save_folder: E:\Github\Python\Correlation\test_images\df2_kinetics\local_df
seg_length: 2
winsize: 50
step: 25
Tue Sep  1 15:59:22 2020 // computation starts!
Tue Sep  1 15:59:23 2020 // computation ends!
"""