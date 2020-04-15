import corrLib
import sys
import os
from skimage import io
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import time

# piv_folder = sys.argv[1]
# img_folder = sys.argv[2]
# output_folder = sys.argv[3]
# winsize = int(sys.argv[4])
# step = int(sys.argv[5])

piv_folder = r'E:\Google Drive\data_share\Dynamics_raw\piv_result_10\80'
img_folder = r'E:\Google Drive\data_share\Dynamics_raw\80'
output_folder = r'E:\Google Drive\data_share\Dynamics_raw\fields\divv-bound'
winsize = 10
step = 10

if os.path.exists(output_folder) == 0:
    os.makedirs(output_folder)
with open(os.path.join(output_folder, 'log.txt'), 'w') as f:
    pass

count = 0
ld = corrLib.readdata(piv_folder)
for num, i in ld.iterrows():
    pivData = pd.read_csv(os.path.join(piv_folder, i.Dir))
    folder, file = os.path.split(i.Dir)
    name_ind = file.find('-')
    name = file[0: name_ind]
    imgDir = os.path.join(img_folder, name + '.tif')
    img = io.imread(imgDir)
    c, v, divcn, divcv, divv = corrLib.div_field(img, pivData, winsize, step)
    # Plot
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), dpi=100)
    ax[0, 0].imshow(c, cmap='seismic')
    ax[0, 0].set_title('$c$ field')
    ax[0, 1].imshow(divv, cmap='seismic')
    ax[0, 1].set_title('$\\nabla \cdot v$ field')
    ax[1, 0].imshow(divcn, cmap='seismic')
    ax[1, 0].set_title('$\\nabla\cdot(cn)$ field')
    ax[1, 1].imshow(divcv, cmap='seismic')
    ax[1, 1].set_title('$\\nabla\cdot(cv)$ field')

    normc = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    normv = mpl.colors.Normalize(vmin=-3, vmax=3)
    normcv = mpl.colors.Normalize(vmin=divcv.min(), vmax=divcv.max())
    normcn = mpl.colors.Normalize(vmin=divcn.min(), vmax=divcn.max())

    plt.colorbar(mpl.cm.ScalarMappable(norm=normc, cmap='seismic'), ax=ax[0, 0], shrink=0.8, drawedges=False)
    plt.colorbar(mpl.cm.ScalarMappable(norm=normv, cmap='seismic'), ax=ax[0, 1], shrink=0.8, drawedges=False)
    plt.colorbar(mpl.cm.ScalarMappable(norm=normcn, cmap='seismic'), ax=ax[1, 0], shrink=0.8, drawedges=False)
    plt.colorbar(mpl.cm.ScalarMappable(norm=normcv, cmap='seismic'), ax=ax[1, 1], shrink=0.8, drawedges=False)
    # save the figure
    outputDir = os.path.join(output_folder, name + '.png')
    plt.savefig(outputDir, dpi=72)
    # log 
    with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // ' + name + ' calculated\n')

""" SYNTAX
python div.py piv_folder img_folder output_folder winsize step
"""
        
""" TEST PARAMS
piv_folder = I:\Github\Python\Correlation\test_images\div
img_folder = I:\Github\Python\Correlation\test_images\div
output_folder = I:\Github\Python\Correlation\test_images\div
winsize = 10
step = 10
"""

""" LOG 3 sec / frame
Wed Feb 26 16:16:24 2020 // 900 calculated
Wed Feb 26 16:16:27 2020 // 916 calculated
"""