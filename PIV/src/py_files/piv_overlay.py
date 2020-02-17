import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import numpy as np
from myImageLib import bpass, dirrec
from skimage import io
import time
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.scalebar import SI_LENGTH

pivDataFolder = sys.argv[1]
imgFolder = sys.argv[2]
output_folder = sys.argv[3]
if len(sys.argv) == 5:
    sparcity = int(sys.argv[4])
else:
    sparcity = 1
    
pivDataDir = dirrec(pivDataFolder, '*.csv')
for pivDir in pivDataDir:
    # PIV data
    folder, pivname = os.path.split(pivDir)    
    pivData = pd.read_csv(pivDir)
    col = len(pivData.x.drop_duplicates())
    row = len(pivData.y.drop_duplicates())
    x = np.array(pivData.x).reshape(row, col)
    y = np.array(pivData.y).reshape(row, col)
    u = np.array(pivData.u).reshape(row, col)
    v = np.array(pivData.v).reshape(row, col)
    xs = x[0:row:sparcity, 0:col:sparcity]
    ys = y[0:row:sparcity, 0:col:sparcity]
    us = u[0:row:sparcity, 0:col:sparcity]
    vs = v[0:row:sparcity, 0:col:sparcity]
    # overlay image
    imgname = pivname[0: pivname.find('-')]
    imgDir = os.path.join(folder.replace(pivDataFolder, imgFolder), imgname + '.tif')
    img = io.imread(imgDir)
    bp = bpass(img, 3, 100)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(bp, cmap='gray')
    ax.quiver(xs, ys, us, vs, color='yellow', width=0.003)
    plt.axis('off')
    scalebar = ScaleBar(0.33, 'um', SI_LENGTH, frameon=False, color='white', font_properties={'size': 20}, pad=0, location='lower right', fixed_value=100) # 1 pixel = 0.2 1/cm
    ax.add_artist(scalebar)
    outfolder = folder.replace(pivDataFolder, output_folder)
    if os.path.exists(outfolder) == False:
        os.makedirs(outfolder)
    fig.savefig(os.path.join(outfolder, imgname + '.png'), dpi=150)
    with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // ' + imgname + ' calculated\n')

""" TEST COMMAND
python piv_overlay.py pivDataFolder imgFolder output_folder sparcity
"""
        
"""  TEST PARAMS
pivDataFolder = I:\Github\Python\PIV\test_images\piv_overlay\pivdata
imgFolder = I:\Github\Python\PIV\test_images\piv_overlay\img
output_folder = I:\Github\Python\PIV\test_images\piv_overlay\output
sparcity = 2
"""

""" LOG 1 s/frame
Mon Feb 17 15:34:54 2020 // 900 calculated
Mon Feb 17 15:34:55 2020 // 902 calculated
"""
