import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util, filters, measure
import os
from scipy import ndimage
from myImageLib import dirrec, to8bit, bpass, FastPeakFind
from corrLib import readseq


def avg_cos_1(traj, order_pNo):
    # average cos's of adjacent particles, for single frame
    cos = 0
    count = 0
    k = 0
    for i in order_pNo:
        if k == 0:
            try:
                x1 = traj.x.loc[traj.particle==i].values[0]
                y1 = traj.y.loc[traj.particle==i].values[0]
                k += 1
                continue
            except:
                continue
        if k == 1:
            try:
                x2 = traj.x.loc[traj.particle==i].values[0]
                y2 = traj.y.loc[traj.particle==i].values[0]
                k += 1
                continue
            except:
                continue

        v1x = x2 - x1
        v1y = y2 - y1

        xt = x2
        yt = y2
        try:
            x2t = traj.x.loc[traj.particle==i].values[0]
            y2t = traj.y.loc[traj.particle==i].values[0]
        except:
            continue
        v2x = x2t - xt
        v2y = y2t - yt

        x1 = xt
        y1 = yt
        x2 = x2t
        y2 = y2t
        cos = cos + (v1x*v2x+v1y*v2y)/((v1x**2+v1y**2)**0.5*(v2x**2+v2y**2)**0.5)
        count += 1
    cos = cos / count
    return cos

def avg_cos(traj, order_pNo):
    # average cos's of adjacent particles, for whole video
    t = []
    cosL = []
    for frame in traj.frame.drop_duplicates():
        subtraj = traj.loc[traj.frame==frame]
        cos = avg_cos_1(subtraj, order_pNo)
        t.append(frame)
        cosL.append(cos)
    df = pd.DataFrame().assign(frame=t, cos=cosL)
    return df

def get_chain_mask(img, feature_size=7000, feature_number=1):
    maxfilt = ndimage.maximum_filter(img, size=15)
    maxfilt_thres = maxfilt > filters.threshold_minimum(maxfilt)
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

def dt_track_1(img, feature_size=7000, feature_number=1):
    mask = get_chain_mask(img, feature_size, feature_number)
    isod = img > filters.threshold_isodata(img)
    masked_isod = mask * isod
    despeck = ndimage.median_filter(masked_isod, size=3)
    dt = ndimage.distance_transform_edt(despeck)
    cent = FastPeakFind(dt)
    return cent

def dt_track(folder, feature_size=7000, feature_number=1):
    traj = pd.DataFrame()
    l = readseq(folder)
    for num, i in l.iterrows():
        img = io.imread(i.Dir)
        try:
            cent = dt_track_1(img, feature_size, feature_number)
        except:
            ValueError('Frame {:05d} tracking failed, use dt_track_1(img) to find out the cause')
        subtraj = pd.DataFrame(data=cent.transpose(), columns=['y', 'x']).assign(Name=i.Name)
        traj = traj.append(subtraj)
        traj = traj[['x', 'y', 'Name']]
    return traj
    
if __name__ == '__main__':
    pass
    # dt_track test code
    # traj = dt_track(r'I:\Github\Python\mylib\xiaolei\chain\test_files')
    
    # avg_cos test code 
    # traj = pd.read_csv(r'E:\Github\Python\ForFun\Misc\cos\data.csv')
    # order_pNo = [0, 13, 12, 11, 9, 8, 6, 7, 1, 2, 4, 3, 5, 14, 10]
    # df = avg_cos(traj, order_pNo)
    # plt.plot(df.frame, df.cos)
    # plt.show()
    