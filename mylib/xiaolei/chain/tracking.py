import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util, filters, measure
import os
from scipy import ndimage
from myImageLib import dirrec, to8bit, bpass, track_spheres_dt
from corrLib import readseq

def get_chain_mask(img, feature_size=7000, feature_number=1):
    maxfilt = ndimage.maximum_filter(img, size=15)
    maxfilt_thres = maxfilt > filters.threshold_isodata(maxfilt)
    label_image = measure.label(maxfilt_thres, connectivity=1)
    num = 0
    coordsL = []
    for region in measure.regionprops(label_image):
        if region.area < feature_size:
            continue
        coordsL.append(region.coords)
        num += 1
        if num > feature_number:
            break
    mask = np.zeros(img.shape)
    for coords in coordsL:
        mask[coords[:, 0], coords[:, 1]] = 1
    return mask

def dt_track_1(img, target_number, feature_size=7000, feature_number=1):
    mask = get_chain_mask(img, feature_size, feature_number)
    isod = img > filters.threshold_isodata(img)
    masked_isod = mask * isod
    despeck = ndimage.median_filter(masked_isod, size=10)
    dt = ndimage.distance_transform_edt(despeck)
    max_coor, pk_value = track_spheres_dt(dt, target_number)
    return max_coor

def dt_track(folder, target_number, feature_size=7000, feature_number=1):
    traj = pd.DataFrame()
    l = readseq(folder)
    for num, i in l.iterrows():
        print('Processing frame ' + i.Name + ' ...')
        img = io.imread(i.Dir)
        try:
            cent = dt_track_1(img, target_number, feature_size, feature_number)
        except:
            print('Frame {:s} tracking failed, use dt_track_1(img) to find out the cause'.format(i.Name))
            continue
        subtraj = pd.DataFrame(data=cent.transpose(), columns=['y', 'x']).assign(Name=i.Name)
        traj = traj.append(subtraj)
        traj = traj[['x', 'y', 'Name']]
    return traj
    
if __name__ == '__main__':
    # dt_track test code
    traj = dt_track(r'R:\Dip\DNA_chain\fluorescent\problem_image', 15)

    
    # avg_cos test code 
    # traj = pd.read_csv(r'E:\Github\Python\ForFun\Misc\cos\data.csv')
    # order_pNo = [0, 13, 12, 11, 9, 8, 6, 7, 1, 2, 4, 3, 5, 14, 10]
    # df = avg_cos(traj, order_pNo)
    # plt.plot(df.frame, df.cos)
    # plt.show()
    