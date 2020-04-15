import numpy as np
import matplotlib.pyplot as plt
import myImageLib as mil
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
from scipy.interpolate import griddata
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.scalebar import SI_LENGTH
import matplotlib
import pandas as pd
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.figsize'] = 10, 6
matplotlib.rcParams['font.family'] = 'serif'


folder = r'E:\Google Drive\data_share\Dynamics_raw\80_fillhole'
folder_piv = r'E:\Google Drive\data_share\Dynamics_raw\piv_result_10\80'
folder_out = r'E:\Google Drive\data_share\Dynamics_raw\subtraction_div_overlay\80\tau=0'
if os.path.exists(folder_out) == False:
    os.makedirs(folder_out)
tau = 0
l = cl.readseq(folder)
for num, i in l.iterrows():
    name1 = str(int(i.Name)+2)
    if int(name1) > 999:
        break
    I0 = io.imread(os.path.join(folder, i.Name+'.tif'))
    I1 = io.imread(os.path.join(folder, name1+'.tif'))
    X, Y, I0s = cl.divide_windows(I0, windowsize=[10, 10], step=10)
    X, Y, I1s = cl.divide_windows(I1, windowsize=[10, 10], step=10)
    I0s = I0s.astype('int16')
    I1s = I1s.astype('int16')
    pivframe = int(i.Name)+tau
    name2 = str(pivframe) + '-' + str(pivframe+1) + '.csv'
    if os.path.exists(os.path.join(folder_piv, name2)) == False:
        continue
    pivData = pd.read_csv(os.path.join(folder_piv, name2))
    c, v, divcn, divcv, divv = cl.div_field(I0, pivData, 10, 10)
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(divv, cmap='cool', vmax=2, vmin=-2)
    plt.colorbar()
    plt.imshow(-I1s+I0s, cmap='seismic', alpha=0.5, interpolation='spline16')
    plt.savefig(os.path.join(folder_out, 'tau=' + str(tau) + '_' + i.Name + '.png'), format='png')
    plt.clf