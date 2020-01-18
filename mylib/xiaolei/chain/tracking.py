import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util, filters, measure
import os
from scipy import ndimage, optimize, exp
from myImageLib import dirrec, to8bit, bpass, maxk, FastPeakFind, track_spheres_dt
from corrLib import readseq
import pdb

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

def dt_track(folder, target_number, min_dist=20, feature_size=7000, feature_number=1):
    traj = pd.DataFrame()
    l = readseq(folder)
    for num, i in l.iterrows():
        print('Processing frame ' + i.Name + ' ...')
        img = io.imread(i.Dir)
        try:
            cent = dt_track_1(img, target_number, min_dist=min_dist, feature_size=feature_size, feature_number=feature_number)
        except:
            print('Frame {:s} tracking failed, use dt_track_1(img) to find out the cause'.format(i.Name))
            continue
        subtraj = pd.DataFrame(data=cent.transpose(), columns=['y', 'x']).assign(Name=i.Name)
        traj = traj.append(subtraj)
        traj = traj[['x', 'y', 'Name']]
    return traj

def track_spheres_dt(img, num_particles, min_dist=20):
    def gauss1(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2)) 
    cent = FastPeakFind(img)
    num_particles = min(num_particles, cent.shape[1])
    peaks = img[cent[0], cent[1]]
    ind = maxk(peaks, cent.shape[1])
    max_coor_tmp = []
    pk_value = []
    count = 0
    for num, i in enumerate(ind):
        distance_check = True
        if num == 0:
            x1 = cent[0, i]
            y1 = cent[1, i]
            max_coor_tmp.append([x1, y1])
            pk_value.append(peaks[i])
            count += 1
            continue
        x2 = cent[0, i]
        y2 = cent[1, i]
        for xy in max_coor_tmp:
            x1 = xy[0]
            y1 = xy[1]
            dist = ((x1-x2)**2 + (y1-y2)**2)**.5
            if dist < min_dist:
                distance_check = False
                break
        if distance_check == True:
            max_coor_tmp.append([x2, y2])
            pk_value.append(peaks[i])
            count += 1
        if count >= num_particles:
            break
    max_coor_tmp = np.array(max_coor_tmp).transpose()
    max_coor = max_coor_tmp.astype('float32')
    pk_value = np.array(pk_value)
    for num in range(0, max_coor_tmp.shape[1]):
        try:
            x = max_coor_tmp[0, num]
            y = max_coor_tmp[1, num]
            fitx1 = np.asarray(range(x-7, x+8))
            fity1 = np.asarray(img[fitx1, y])
            popt,pcov = optimize.curve_fit(gauss1, fitx1, fity1, p0=[1, x, 3])
            max_coor[0, num] = popt[1]
            fitx2 = np.asarray(range(y-7, y+8))
            fity2 = np.asarray(img[x, fitx2])
            popt,pcov = optimize.curve_fit(gauss1, fitx2, fity2, p0=[1, y, 3])
            max_coor[1, num] = popt[1]
        except:
            print('Fitting failed')
            max_coor[:, num] = max_coor_tmp[:, num]
            continue
    return max_coor, pk_value
    
def dt_track_1(img, target_number, min_dist=20, feature_size=7000, feature_number=1):
    mask = get_chain_mask(img, feature_size, feature_number)
    isod = img > filters.threshold_isodata(img)
    masked_isod = mask * isod
    despeck = ndimage.median_filter(masked_isod, size=10)
    dt = ndimage.distance_transform_edt(despeck)
    max_coor, pk_value = track_spheres_dt(dt, target_number, min_dist=min_dist)
    return max_coor
    
if __name__ == '__main__':
    pass
    # min dist test code
    # img = io.imread(r'E:\Github\Python\mylib\xiaolei\chain\test_files\problem_image\0035.tif')
    # new_track = dt_track_1(img, 15, min_dist=20)
    # plt.imshow(img, cmap='gray')
    # plt.plot(new_track[1, :], new_track[0, :], marker='o', markersize=12, ls='', mec='red', mfc=(0,0,0,0))
    # plt.show()
    
    # dt_track test code
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    folder = r'E:\Github\Python\mylib\xiaolei\chain\test_files\problem_image'
    traj = dt_track(folder, 15, min_dist=20)
    l = readseq(folder)
    # plt.ion()    
    for num, i in l.iterrows():        
        img = io.imread(i.Dir)
        subtraj = traj.loc[traj.Name==i.Name]
        ax.imshow(img, cmap='gray')
        plt.axis('off')
        ax.plot(subtraj.x, subtraj.y, marker='o', markersize=12, ls='', mec='red', mfc=(0,0,0,0))
        plt.pause(.1)
        
        plt.savefig(os.path.join(folder, r'mindist_20', i.Name + '.png'), format='png')
        plt.cla()
        
    
    # avg_cos test code 
    # traj = pd.read_csv(r'E:\Github\Python\ForFun\Misc\cos\data.csv')
    # order_pNo = [0, 13, 12, 11, 9, 8, 6, 7, 1, 2, 4, 3, 5, 14, 10]
    # df = avg_cos(traj, order_pNo)
    # plt.plot(df.frame, df.cos)
    # plt.show()
    