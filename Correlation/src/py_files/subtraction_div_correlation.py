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

# time difference tau
folder_dif = r'E:\Google Drive\data_share\Dynamics_raw\80_fillhole_subtraction'
folder_div = r'E:\Google Drive\data_share\Dynamics_raw\div_result_10\80'
folder_out = r'E:\Google Drive\data_share\Dynamics_raw\subtraction_div_correlation\80'
if os.path.exists(folder_out) == False:
	os.makedirs(folder_out)
ldiv = cl.readdata(folder_div)
tauL = range(0, 25, 1)
CLL = []
for tau in tauL:
    print('Calculating tau = {:d}'.format(tau))
    CL = []
    t = []
    for num, i in ldiv.iterrows():
        print(i.Name)
        div = pd.read_csv(i.Dir)
        name = i.Name.split('-')[0]
        img_name = str(int(name) - tau)
        if int(img_name) < 900 or int(img_name) > 997:
            continue
        img = np.loadtxt(os.path.join(folder_dif, img_name+'-'+str(int(img_name)+2)+'.txt'))
#         img = 1 - abs(img/abs(img).max())
        xlen = len(div.x.drop_duplicates())
        ylen = len(div.y.drop_duplicates())
        divfield = np.array(div['div']).reshape(ylen, xlen)
        divfield_0mean = divfield - divfield.mean()
        X, Y, I = cl.divide_windows(img, windowsize=[10, 10], step=10)
        I_0mean = I - I.mean()
        C = (divfield_0mean * I_0mean).mean() / abs(divfield_0mean*I_0mean).mean()
        t.append(int(name))
        CL.append(C)
    CLL.append(CL)

np.savetxt(os.path.join(folder_out, 'tau.dat'), np.array(tauL))
np.savetxt(os.path.join(folder_out, 'C.dat'), np.array(CLL))

# plt.figure(figsize=(10, 8))
# for tau, CL in zip(tauL, CLL):
    # plt.plot(CL, label=('$\\tau=$ ' + str(tau)))
# plt.legend()
# plt.xlabel('frame number')
# plt.ylabel('correlation')

# Cavg = []
# for CL in CLL:
    # Cavg.append(np.array(CL).mean())
# plt.plot(tauL, Cavg)
# plt.xlabel('$\\tau$ [frame]')
# plt.ylabel('average correlation')