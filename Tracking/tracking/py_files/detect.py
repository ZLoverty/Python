from skimage import io, util, filters, measure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import os
from myImageLib import dirrec, to8bit, bpass
import trackpy as tp
from scipy import signal, ndimage, misc
import sys
from corrLib import readseq
import time


img_folder = sys.argv[1]
output_folder = sys.argv[2]
area_min = int(sys.argv[3])
header = bool(int(sys.argv[4]))
filter = 'yen'
if len(sys.argv) > 5:
    filter = sys.argv[5]
    
if os.path.exists(output_folder) == False:
    os.makedirs(output_folder)
with open(os.path.join(output_folder, 'log.txt'), 'w') as f: 
    f.write('img_folder: ' + img_folder + '\n')
    f.write('output_folder: ' + output_folder + '\n')
    f.write('area_min: ' + str(area_min) + '\n')
    f.write('filter: ' + str(filter) + '\n')


l = readseq(img_folder)
for num, i in l.iterrows():
    img = io.imread(i.Dir)
    if filter == 'yen':
        filt = filters.threshold_yen(img)
    filtered_img = img < filt
    label_image = measure.label(filtered_img)
    
    # temp data in this frame, save in DataFrame
    Area_temp = []
    X_temp = []
    Y_temp = []
    Major_temp = []
    Minor_temp = []
    Angle_temp = []
    Slice_temp = []
    for region in measure.regionprops(label_image):
        if region.area > area_min:
            Area_temp.append(region.area)
            X_temp.append(region.centroid[0])
            Y_temp.append(region.centroid[1])
            Major_temp.append(region.major_axis_length)
            Minor_temp.append(region.minor_axis_length)
            Angle_temp.append(region.orientation)
            Slice_temp.append(num)
    traj_temp = pd.DataFrame().assign(Area=Area_temp, X=X_temp, Y=Y_temp, \
            Major=Major_temp, Minor=Minor_temp, Angle=Angle_temp, Slice=Slice_temp)
    if num == 0:
        traj = traj_temp
    else:
        traj = traj.append(traj_temp)
        
    with open(os.path.join(output_folder, 'log.txt'), 'a') as f: 
        f.write(time.asctime() + ' // Frame ' + i.Name + ' detected\n')
    
    if header == False:
        if num == 0:
            traj_temp.to_csv(os.path.join(output_folder, 'detect.csv'), index=False, header=header)
        else:
            traj_temp.to_csv(os.path.join(output_folder, 'detect.csv'), index=False, header=header, mode='a')
            
if header == True:
    traj.to_csv(os.path.join(output_folder, 'detect.csv'), index=False, header=header)
    
    
""" EDIT
11172020 - initial commit
11182020 - fix saving bugs. Use mode 'a' for writing when header is turned off.
           Change log open to mode 'a'
"""
        
""" DESCRIPTION
Particle tracking routine: the detect part
"""
        
""" SYNTAX
python detect.py img_folder out_folder area_min [filter]

img_folder -- folder containing .tif images
out_folder -- folder to save particle detection data (detect.csv, columns: Area, X, Y, Major, Minor, Angle, Slice)
area_min -- min area to filter false positive results, only area greater than this value will be recorded.
            In the future, more optional filter params can be added.
[filter] -- the filter used to binarize the input images, default to 'yen', choose with care
"""

""" TEST PARAMS
img_folder -- E:\Github\Python\Tracking\tracking\test_images
out_folder -- E:\Github\Python\Tracking\tracking\test_images
area_min -- 1500
header -- 0
[filter] -- [leave blank]

"""

""" LOG

"""