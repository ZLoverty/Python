import pandas as pd
import os
# import matplotlib.pyplot as plt
import sys
import numpy as np
from myImageLib import bpass, dirrec
from skimage import io
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

pivDataFolder = sys.argv[1]
imgFolder = sys.argv[2]
output_folder = sys.argv[3]
if len(sys.argv) == 5:
    sparcity = int(sys.argv[4])
else:
    sparcity = 1
if os.path.exists(output_folder) == False:
    os.makedirs(output_folder)
with open(os.path.join(output_folder, 'log.txt'), 'w') as f: 
    f.write('piv_folder: ' + pivDataFolder + '\n')
    f.write('img_folder: ' + imgFolder + '\n')
    f.write('output_folder: ' + output_folder + '\n')
    f.write('sparcity: ' + str(sparcity) + '\n')
    
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
    bp = bpass(img, 2, 100)
    # fig = plt.figure(figsize=(3, 3*row/col))
    fig = Figure(figsize=(3, 3*row/col))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(bp, cmap='gray')
    ax.quiver(xs, ys, us, vs, color='yellow', width=0.003)
    ax.axis('off')
    outfolder = folder.replace(pivDataFolder, output_folder)
    if os.path.exists(outfolder) == False:
        os.makedirs(outfolder)
    fig.savefig(os.path.join(outfolder, imgname + '.jpg'), dpi=120)
    with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // ' + imgname + ' calculated\n')

""" TEST COMMAND
python piv_overlay.py pivDataFolder imgFolder output_folder sparcity
"""
        
"""  TEST PARAMS
pivDataFolder = E:\Github\Python\PIV\test_images\piv_overlay\pivdata
imgFolder = E:\Github\Python\PIV\test_images\piv_overlay\img
output_folder = E:\Github\Python\PIV\test_images\piv_overlay\output
sparcity = 2
"""

""" LOG 1 s/frame
Mon Feb 17 15:34:54 2020 // 900 calculated
Mon Feb 17 15:34:55 2020 // 902 calculated
"""
