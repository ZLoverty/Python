import corrLib
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import io
import numpy as np

piv_folder = r'D:\Wei\Dynamics_raw\piv_result_10\80'
img_folder = r'D:\Wei\Dynamics_raw\100'
output_folder = r'D:\Wei\Dynamics_raw\div_dc\80'
dc_folder = r'D:\Wei\Dynamics_raw\80_diff'
winsize = 10
step = 10
if os.path.exists(output_folder) == 0:
    os.makedirs(output_folder)
with open(os.path.join(output_folder, 'log.txt'), 'w') as f:
    pass
count = 0
ld = corrLib.readdata(piv_folder)
for num, i in ld.iterrows():
    print('Drawing ' + i.Name)
    pivData = pd.read_csv(os.path.join(piv_folder, i.Dir))
    folder, file = os.path.split(i.Dir)
    name_ind = file.find('-')
    name = file[0: name_ind]
    imgDir = os.path.join(img_folder, name + '.tif')
    img = io.imread(imgDir)
    c, v, divcn, divcv, divv = corrLib.div_field_2(img, pivData, 10, 10)
    dc = np.load(os.path.join(dc_folder, name + '.npy'))
    # Plot
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), dpi=200)
    ax[0, 0].imshow(dc, cmap='seismic')
    ax[0, 0].set_title('$dc$ field')
    ax[0, 1].imshow(divv, cmap='seismic')
    ax[0, 1].set_title('$\\nabla\cdot(v)$ field')
    ax[1, 0].imshow(divcn, cmap='seismic')
    ax[1, 0].set_title('$\\nabla\cdot(cn)$ field')
    ax[1, 1].imshow(divcv, cmap='seismic')
    ax[1, 1].set_title('$\\nabla\cdot(cv)$ field')

    normdc = mpl.colors.Normalize(vmin=dc.min(), vmax=dc.max())
    normv = mpl.colors.Normalize(vmin=divv.min(), vmax=divv.max())
    normcv = mpl.colors.Normalize(vmin=divcv.min(), vmax=divcv.max())
    normcn = mpl.colors.Normalize(vmin=divcn.min(), vmax=divcn.max())

    plt.colorbar(mpl.cm.ScalarMappable(norm=normdc, cmap='seismic'), ax=ax[0, 0], shrink=0.8, drawedges=False)
    plt.colorbar(mpl.cm.ScalarMappable(norm=normv, cmap='seismic'), ax=ax[0, 1], shrink=0.8, drawedges=False)
    plt.colorbar(mpl.cm.ScalarMappable(norm=normcn, cmap='seismic'), ax=ax[1, 0], shrink=0.8, drawedges=False)
    plt.colorbar(mpl.cm.ScalarMappable(norm=normcv, cmap='seismic'), ax=ax[1, 1], shrink=0.8, drawedges=False)
    # save the figure
    outputDir = os.path.join(output_folder, name + '.png')
    plt.savefig(outputDir, dpi=200)