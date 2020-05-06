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
from skimage import filters
import time
import sys


""" DESCRIPTION
Calculate the autocorrelation between a velocity divergence field and an image intensity field.
"""

""" INPUT PARAMETERS
folder_den: images
folder_div: divergence field data (DataFrame)
folder_ixdiv: output folder, saving autocorrelation data (density/tau=xx/data.csv)
options: "default", "raw"
tauL: range of tau
"""

folder_den = sys.argv[1]
folder_div = sys.argv[2]
folder_ixdiv = sys.argv[3]
if len(sys.argv) > 4:
    options = sys.argv[4]
else:
    options = 'default'

tauL = range(-200, 200, 10)

# folder_den = r'E:\Google Drive\data_share\Dynamics_raw\processed_image\60_bp'
# folder_div = r'E:\Google Drive\data_share\Dynamics_raw\concentration_velocity_field\div_result_50\60'
# folder_ixdiv = r'E:\Github\Python\Correlation\test_images\div\ixdiv_test\60'
# tauL = range(-90, 90, 3)
if os.path.exists(folder_ixdiv) == False:
    os.makedirs(folder_ixdiv)
with open(os.path.join(folder_ixdiv, 'log.txt'), 'w') as f:
    pass


lden = cl.readseq(folder_den)
ldiv = cl.readdata(folder_div)
CLL = []
for tau in tauL:
    CL = []
    tL = []
    for num, i in ldiv.iterrows():        
        div = pd.read_csv(i.Dir)
        name = i.Name.split('-')[0]
        # img_name = str(int(name) - tau)
        img_name = str('{:04d}'.format(int(name) - tau))
        if os.path.exists(os.path.join(folder_den, img_name+'.tif')) == False:
            print('no match image')
            continue
        img = io.imread(os.path.join(folder_den, img_name + '.tif'))
        if options == 'raw':
            print('performing bpass on ' + img_name)
            img = cl.match_hist(mil.bpass(img, 3, 500), img)
        print('tau={:d}'.format(tau) + ', div-' + name + ' x img-' + img_name)
        xlen = len(div.x.drop_duplicates())
        ylen = len(div.y.drop_duplicates())
        divfield = np.array(div['div']).reshape(ylen, xlen)
        divfield = divfield - divfield.mean()
        divfield_crop = divfield#[20:36, 8:24]
        X, Y, I = cl.divide_windows(img, windowsize=[50, 50], step=25)
        I_norm = I / abs(I).max()
        I_norm = I_norm - I_norm.mean()
        I_crop = I_norm#[20:36, 8:24]
        tL.append(int(img_name))
        C = (divfield_crop * I_crop).mean() / divfield_crop.std() / I_crop.std()
        CL.append(C)
        data = pd.DataFrame().assign(t=tL, autocorr=CL)
        saveDir = os.path.join(folder_ixdiv, 'tau={:d}'.format(tau))
        if os.path.exists(saveDir) == False:
            os.makedirs(saveDir)
        data.to_csv(os.path.join(saveDir, 'data.csv'), index=False)
    with open(os.path.join(folder_ixdiv, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // tau=' + str(tau) + ' calculated\n')
    CLL.append(CL)

""" SYNTAX
python i_div_autocorr.py folder_den folder_div folder_ixdiv
"""

""" TEST PARAMS
folder_den = r'E:\Google Drive\data_share\Dynamics_raw\processed_image\60_bp'
folder_div = r'E:\Google Drive\data_share\Dynamics_raw\concentration_velocity_field\div_result_50\60'
folder_ixdiv = r'E:\Github\Python\Correlation\test_images\div\ixdiv_test\60'
"""

""" LOG
Tue Apr 21 21:20:15 2020 // tau=-90 calculated
Tue Apr 21 21:20:16 2020 // tau=-87 calculated
Tue Apr 21 21:20:18 2020 // tau=-84 calculated
Tue Apr 21 21:20:20 2020 // tau=-81 calculated
Tue Apr 21 21:20:22 2020 // tau=-78 calculated
Tue Apr 21 21:20:24 2020 // tau=-75 calculated
Tue Apr 21 21:20:27 2020 // tau=-72 calculated
Tue Apr 21 21:20:30 2020 // tau=-69 calculated
Tue Apr 21 21:20:33 2020 // tau=-66 calculated
Tue Apr 21 21:20:36 2020 // tau=-63 calculated
Tue Apr 21 21:20:39 2020 // tau=-60 calculated
Tue Apr 21 21:20:43 2020 // tau=-57 calculated
Tue Apr 21 21:20:47 2020 // tau=-54 calculated
Tue Apr 21 21:20:51 2020 // tau=-51 calculated
Tue Apr 21 21:20:56 2020 // tau=-48 calculated
Tue Apr 21 21:21:00 2020 // tau=-45 calculated
"""