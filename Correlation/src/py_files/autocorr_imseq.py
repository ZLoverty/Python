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
boxsize = int(sys.argv[3])

if os.path.exists(output_folder) == False:
    os.makedirs(output_folder)
with open(os.path.join(output_folder, 'log.txt'), 'w') as f:
    pass

s = boxsize / 2
l = readseq(input_folder)
for num, i in l.iterrows():
    img = io.imread(i.Dir)
    row, col = img.shape
    break
    
xc1 = range(int(col/4-s), int(col/4+s))
yc1 = range(int(row/4-s), int(row/4+s))
xc2 = range(int(col*3/4-s), int(col*3/4+s))
yc2 = range(int(row/4-s), int(row/4+s))
xc3 = range(int(col/2-s), int(col/2+s))
yc3 = range(int(row/2-s), int(row/2+s))
xc4 = range(int(col/4-s), int(col/4+s))
yc4 = range(int(row*3/4-s), int(row*3/4+s))
xc5 = range(int(col*3/4-s), int(col*3/4+s))
yc5 = range(int(row*3/4-s), int(row*3/4+s))
xc = [xc1, xc2, xc3, xc4, xc5]
yc = [yc1, yc2, yc3, yc4, yc5]

data = pd.DataFrame()
for num, i in l.iterrows():
    img = io.imread(i.Dir)
    k = 0
    Iframe = []
    for x, y in zip(xc, yc):
        I = img[x, y].mean()
        Iframe.append(I)
        k += 1
    Iframe_array = np.array(Iframe)
    frameData = pd.DataFrame(data=[Iframe_array],
                columns=['spot1', 'spot2', 'spot3', 'spot4', 'spot5'])
    frameData = frameData.assign(t=int(i.Name))
    data = data.append(frameData)
data.to_csv(os.path.join(output_folder, 'time_series.csv'), index=False)
with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
    f.write(time.asctime() + ' // time series is saved!\n')
    
corrMax = int(len(data)/2)
# maximum autocorr lag, lag will be from 0 to corrMax
# int(len(data)/2) is experimental and is subject to change
ac_data = pd.DataFrame()
for spot in data:
    if spot == 't':
        continue
    ac_list = []
    for lag in range(0, corrMax):
        ac_list.append(data[spot].autocorr(lag=lag))
    ac_data = ac_data.assign(**{spot: ac_list})
ac_data.to_csv(os.path.join(output_folder, 'ac_result.csv'), index=False)
with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
    f.write(time.asctime() + ' // autocorrelation is saved!\n')
    

""" TEST COMMAND
python autocorr_imseq.py input_folder output_folder boxsize
"""
        
"""  TEST PARAMS
input_folder = I:\Github\Python\Correlation\test_images\autocorr\80
output_folder = I:\Github\Python\Correlation\test_images\autocorr\80\autocorr
boxsize = 40
"""

""" LOG
Tue Mar  3 16:08:28 2020 // time series is saved!
Tue Mar  3 16:08:28 2020 // autocorrelation is saved!
"""