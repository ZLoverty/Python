import numpy as np
import matplotlib.pyplot as plt
from myImageLib import dirrec, bestcolor, bpass
from skimage import io, measure
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.signal import savgol_filter, medfilt
import os
from corrLib import corrS, corrI, divide_windows, distance_corr, corrIseq, readseq, match_hist
from scipy.signal import savgol_filter
from corrLib import boxsize_effect_spatial
import matplotlib as mpl
from numpy.polynomial.polynomial import polyvander
from scipy.optimize import curve_fit
from miscLib import label_slope
from corrLib import density_fluctuation
from scipy import signal
import pdb

def readdata(folder):
    dataDirs = dirrec(folder, '*.csv')
    nameList = []
    dirList = []
    for dataDir in dataDirs:
        path, file = os.path.split(dataDir)
        name, ext = os.path.splitext(file)
        nameList.append(name)
        dirList.append(dataDir)
    fileList = pd.DataFrame()
    fileList = fileList.assign(Name=nameList, Dir=dirList)
    fileList.Name = fileList.Name.astype('int32')
    fileList = fileList.sort_values(by=['Name'])
    return fileList

def corrlength(corrData, fitting_range=1000):
    xx = np.array(corrData.R)
    yy = np.array(corrData.C)
    x = xx[xx<fitting_range]
    y = yy[xx<fitting_range]
    p = np.polyfit(x, y, 8)
    xsolve = np.array(range(0, fitting_range))
    yfit = np.dot(polyvander(xsolve, 8), np.flip(p).transpose())
    try:
        corrlen = xsolve[yfit>1/np.e].max()
    except:
        corrlen = 0

    return corrlen
	
folders = [r'D:\Wei\transient\cl_result\020',
          r'D:\Wei\transient\cl_result\030',
          r'D:\Wei\transient\cl_result\040',
          r'D:\Wei\transient\cl_result\050',
          r'D:\Wei\transient\060\result',
          r'D:\Wei\transient\cl_result\070',
          r'D:\Wei\transient\080\result',
          r'D:\Wei\transient\cl_result\090',
          r'D:\Wei\transient\100\result']
names = ['20', '30', '40', '50', '60', '70', '80', '90', '100']
count1 = 0
rec = pd.DataFrame()
for name, folder in zip(names, folders):
    corrL = []
    frame = []
    l = readdata(folder)
    count = 0
    interval = 100
    for num, i in l.iterrows():
        if num % interval != 0:
            count += 1
            continue
        data = pd.read_csv(i.Dir)
        corrlen = corrlength(data, fitting_range=400)
        frame.append(int(i.Name))
        corrL.append(corrlen)
        subrec = pd.DataFrame().assign(frame=frame, corrL=corrL, concentration=name)
        count += 1
    rec = rec.append(subrec)
    plt.plot(frame, corrL, label=name, ls='', marker='o',
            mfc=(0,0,0,0), mec=bestcolor(count1))
    pdb.set_trace()
    
    count1 += 1