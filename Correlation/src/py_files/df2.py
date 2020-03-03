import numpy as np
import matplotlib.pyplot as plt
from myImageLib import dirrec, bestcolor, bpass, wowcolor
from skimage import io, measure
import pandas as pd
from scipy.signal import savgol_filter, medfilt
import os
from corrLib import corrS, corrI, divide_windows, distance_corr, corrIseq, readseq, match_hist
from scipy.signal import savgol_filter
import matplotlib as mpl
from numpy.polynomial.polynomial import polyvander
from scipy.optimize import curve_fit
from miscLib import label_slope
from corrLib import density_fluctuation
from scipy import signal
from scipy.interpolate import griddata
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.scalebar import SI_LENGTH
import matplotlib as mpl
import sys
import time

folder = sys.argv[1]
output_folder = sys.argv[2]

if os.path.exists(output_folder) == False:
    os.makedirs(output_folder)
with open(os.path.join(output_folder, 'log.txt'), 'w') as f:
    pass

l = readseq(folder)
img = io.imread(l.Dir.loc[0])
size_min = 20
L = min(img.shape)
boxsize = np.unique(np.floor(np.logspace(np.log10(size_min),
                    np.log10((L-size_min)/2),100)))
df = pd.DataFrame()
for num, i in l.iterrows():
    with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // ' + i.Name + ' calculated\n')
    img = io.imread(i.Dir)
    bp = bpass(img, 3, 100)
    bp_mh = match_hist(bp, img)
    framedf = pd.DataFrame()
    for bs in boxsize: 
        X, Y, I = divide_windows(bp_mh, windowsize=[bs, bs], step=50*size_min)
        tempdf = pd.DataFrame().assign(I=I.flatten(), t=int(i.Name), size=bs, 
                       number=range(0, len(I.flatten())))
        framedf = framedf.append(tempdf)
    df = df.append(framedf)
    
df_out = pd.DataFrame()
for number in df.number.drop_duplicates():
    subdata1 = df.loc[df.number==number]
    for s in subdata1['size'].drop_duplicates():
        subdata = subdata1.loc[subdata1['size']==s]
        d = s**2 * np.log(np.array(subdata.I)).std()
        n = s**2 
        tempdf = pd.DataFrame().assign(n=[n], d=d, size=s, number=number)
        df_out = df_out.append(tempdf)
        
average = pd.DataFrame()
for s in df_out['size'].drop_duplicates():
    subdata = df_out.loc[df_out['size']==s]
    avg = subdata.drop(columns=['size', 'number']).mean().to_frame().T
    average = average.append(avg)
    
average.to_csv(os.path.join(output_folder, 'df_average.csv'), index=False)

""" ABOUT
An alternative method to quantify density fluctuation: time variance -> spatial average
"""

""" TEST COMMAND
python df2.py input_folder output_folder
"""

""" PARAMETERS
input_folder = I:\Github\Python\Correlation\test_images\GNF\alternative\lowc 
output_folder = I:\Github\Python\Correlation\test_images\GNF\alternative\lowc\result
"""

""" LOG
Tue Mar  3 13:14:42 2020 // 922 calculated
Tue Mar  3 13:14:44 2020 // 923 calculated
Tue Mar  3 13:14:45 2020 // 924 calculated
"""