import numpy as np
import matplotlib.pyplot as plt
from myImageLib import dirrec, bestcolor, bpass, wowcolor
from skimage import io, measure
import pandas as pd
from scipy.signal import savgol_filter, medfilt
import os
import corrLib as cl
from scipy.signal import savgol_filter
import matplotlib as mpl
from numpy.polynomial.polynomial import polyvander
from scipy.optimize import curve_fit
from miscLib import label_slope
from scipy import signal
import matplotlib as mpl
import sys
import time
import pdb

"""
Using method II to (temporal variance -> spatial average) to calculate the kinetics of GNF during the onset of active turbulence.
"""

# necessary
folder = sys.argv[1]
folder_out = sys.argv[2]
seg_length = int(sys.argv[3])
# optional
normalize = 0 # default to no normalization
if len(sys.argv) > 4:
    normalize = int(sys.argv[4])

if os.path.exists(folder_out) == False:
    os.makedirs(folder_out)
with open(os.path.join(folder_out, 'log.txt'), 'w') as f:
    pass

l = cl.readseq(folder)
length = len(l)
seg = range(0, length, seg_length)

img = io.imread(l.Dir.loc[0])
size_min = 5
L = min(img.shape)
boxsize = np.unique(np.floor(np.logspace(np.log10(size_min), np.log10((L-size_min)/2),50)))

data = pd.DataFrame()
for idx in range(1, len(seg)):
    subdata = l.loc[(l.index>=seg[idx-1])&(l.index<seg[idx])]
    df = pd.DataFrame()
    for num, i in subdata.iterrows():
        img = io.imread(i.Dir)
        framedf = pd.DataFrame()
        for bs in boxsize: 
            X, Y, I = cl.divide_windows(img, windowsize=[bs, bs], step=50*size_min)
            if normalize == 1:
                mean = I.mean()
                I = I / mean
            tempdf = pd.DataFrame().assign(I=I.flatten(), t=int(i.Name), size=bs, 
                           number=range(0, len(I.flatten())))
            framedf = framedf.append(tempdf)
        df = df.append(framedf)
    with open(os.path.join(folder_out, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // ' + 'Segment {0:d}: frame {1:04d}-{2:04d}, generate sequence\n'.format(idx, seg[idx-1], seg[idx]))
    
    df_out = pd.DataFrame()
    for number in df.number.drop_duplicates():
        subdata1 = df.loc[df.number==number]
        for s in subdata1['size'].drop_duplicates():
            subdata = subdata1.loc[subdata1['size']==s] 
            d = s**2 * np.log(np.array(subdata.I)).std()
            n = s**2 
            tempdf = pd.DataFrame().assign(n=[n], d=d, size=s, number=number)
            df_out = df_out.append(tempdf)
    with open(os.path.join(folder_out, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // ' + 'Segment {0:d}: frame {1:04d}-{2:04d}, calculate GNF for each box size and box number\n'.format(idx, seg[idx-1], seg[idx]))
    
    average = pd.DataFrame()
    for s in df_out['size'].drop_duplicates():
        subdata = df_out.loc[df_out['size']==s]
        avg = subdata.drop(columns=['size', 'number']).mean().to_frame().T
        average = average.append(avg)
    data = data.append(average.assign(segment=idx))
    with open(os.path.join(folder_out, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // ' + 'Segment {0:d}: frame {1:04d}-{2:04d}, take spatial average\n'.format(idx, seg[idx-1], seg[idx]))
        
data.to_csv(os.path.join(folder_out, 'kinetics_data.csv'), index=False)
with open(os.path.join(folder_out, 'log.txt'), 'a') as f:
    f.write(time.asctime() + ' // ' + 'Output data')    
                  

""" SYNTAX
python df2_kinetics.py folder folder_out seg_length 
"""

""" TEST PARAMETERS
folder = I:\Github\Python\Correlation\test_images\df2_kinetics\img
folder_out = I:\Github\Python\Correlation\test_images\df2_kinetics\out
seg_length = 2
"""

""" LOG
Tue Jun  9 12:16:44 2020 // Segment 1: frame 0000-0002, generate sequence
Tue Jun  9 12:16:45 2020 // Segment 1: frame 0000-0002, calculate GNF for each box size and box number
Tue Jun  9 12:16:45 2020 // Segment 1: frame 0000-0002, take spatial average
Tue Jun  9 12:16:45 2020 // Segment 2: frame 0002-0004, generate sequence
Tue Jun  9 12:16:46 2020 // Segment 2: frame 0002-0004, calculate GNF for each box size and box number
Tue Jun  9 12:16:46 2020 // Segment 2: frame 0002-0004, take spatial average
Tue Jun  9 12:16:47 2020 // Segment 3: frame 0004-0006, generate sequence
Tue Jun  9 12:16:48 2020 // Segment 3: frame 0004-0006, calculate GNF for each box size and box number
Tue Jun  9 12:16:48 2020 // Segment 3: frame 0004-0006, take spatial average
Tue Jun  9 12:16:48 2020 // Segment 4: frame 0006-0008, generate sequence
Tue Jun  9 12:16:49 2020 // Segment 4: frame 0006-0008, calculate GNF for each box size and box number
Tue Jun  9 12:16:50 2020 // Segment 4: frame 0006-0008, take spatial average
Tue Jun  9 12:16:50 2020 // Output data
"""


