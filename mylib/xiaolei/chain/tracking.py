import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util, filters, measure
import os
from scipy import ndimage, optimize, exp
from myImageLib import dirrec, to8bit, bpass, maxk, FastPeakFind
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

def gauss1(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2)) 
        
def track_spheres_dt(img, num_particles, min_dist=20):
    # implement distance check
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
    
def preprocessing_dt(img, feature_size=7000, feature_number=1, despeckle_size=15):
    mask = get_chain_mask(img, feature_size, feature_number)
    isod = img > filters.threshold_isodata(img)
    masked_isod = mask * isod
    despeck = ndimage.median_filter(masked_isod, size=15)
    dt = ndimage.distance_transform_edt(despeck)
    return dt

def prelim_tracking_dt(dt):
    # Find peaks on distance transform map, return integer coords of peaks (pd.DataFrame)
    cent = FastPeakFind(dt)
    coords = pd.DataFrame(data=cent.transpose(), columns=['y', 'x'])
    return coords

def sort_prelim(coords, img, radius):
    # Sort prelim tracking according to peak score, for now peak score is defined as total mass.
    # I have code in chain.ipynb for comparing bandwidth and quality of fitting. They are not included in this version because:
    #   - They incur more computation
    #   - The correlation between good tracking and fitting quality is not clear.
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    mass = []
    for num, coord in coords.iterrows():
        x = coord.x
        y = coord.y
        dist = ((X - x)**2 + (Y-y)**2)**.5
        mass.append(img[dist<radius].sum())
    mass = np.array(mass)
    coords_rank = coords.assign(mass=mass).sort_values(by=['mass'], ascending=False)
    return coords_rank[['x', 'y']]

def refine(coords, target_number, min_dist=20):
    # Many more infomation could be used here to refine the preliminary tracking result.
    # coords is already sorted and the most possible coordinates are at top in coords
    # We use two here:
    #   - Distance check: a minimal distance min_dist can be set to avoid redundant tracking
    #   - Total number check: Since the total number of particles is known, the function looks for that number of particles and skip those low possibility features.
    count = 0
    coords_tmp = pd.DataFrame()
    for num, coord in coords.iterrows():
        distance_check = True
        if num == 0:
            coords_tmp = coords_tmp.append(coord)
            count += 1
            continue
        for num1, coord1 in coords_tmp.iterrows():
            dist = ((coord.x-coord1.x)**2 + (coord.y-coord1.y)**2)**.5
            # min distance check
            if dist < min_dist:
                distance_check = False
                break
        if distance_check == True:
            coords_tmp = coords_tmp.append(coord)
            count += 1
        # Total number check
        if count >= target_number:
            break
    return coords_tmp

def subpixel_res(coords, dt, fitting_range):    
    # Use gaussian fitting to get subpixel resolution
    half_range = int(np.floor(fitting_range/2))
    
    for num, coord in coords.iterrows():
        
        try:
            x = int(coord.x)
            y = int(coord.y)
            fitx1 = np.asarray(range(x-half_range, x+half_range-1))
            fity1 = np.asarray(dt[y, fitx1])
            popt, pcov = optimize.curve_fit(gauss1, fitx1, fity1, p0=[80, x, 10])
            coords.x.loc[num] = popt[1]
            fitx2 = np.asarray(range(y-half_range, y+half_range-1))
            fity2 = np.asarray(dt[fitx2, x])
            popt, pcov = optimize.curve_fit(gauss1, fitx2, fity2, p0=[80, y, 10])
            coords.y.loc[num] = popt[1]
        except:
            print('Fitting failed')
            continue
    return coords
    
def dt_track_1(img, target_number, min_dist=20, radius=15, fitting_range=40, feature_size=7000, feature_number=1):
    # Preprocessing
    dt = preprocessing_dt(img, feature_size=feature_size, feature_number=feature_number, despeckle_size=15)
    # Prelim tracking on dt    
    # prelim_result = prelim_tracking_dt(dt)
    coords_pre = prelim_tracking_dt(dt)
    # Sorting
    #   - Peak score
    coords_sort = sort_prelim(coords_pre, img, radius)
    # Refine result
    #   - Distance check
    #   - Total target number    
    coords_refine = refine(coords_sort, target_number, min_dist)
    #   - Gaussian fitting to get subpixel resolution
    coords_sr = subpixel_res(coords_refine, dt, fitting_range)
    return coords_sr
    
def dt_track(folder, target_number, min_dist=20, feature_size=7000, feature_number=1):
    traj = pd.DataFrame()
    l = readseq(folder)
    for num, i in l.iterrows():
        print('Processing frame ' + i.Name + ' ...')
        img = io.imread(i.Dir)
        try:
            coords = dt_track_1(img, 15, min_dist=15)
        except:
            print('Frame {:s} tracking failed, use dt_track_1(img) to find out the cause'.format(i.Name))
            continue
        coords = coords.assign(Name=i.Name)
        traj = traj.append(coords)
    return traj   

    
if __name__ == '__main__':
    pass    
    # peack score (dt_track_1) test code
    # img = io.imread(r'E:\Github\Python\mylib\xiaolei\chain\test_files\problem_image\0035.tif')  
    # coords = dt_track_1(img, 15, min_dist=15)
    # plt.imshow(img, cmap='gray')
    # plt.plot(coords.x, coords.y, marker='o', markersize=12, ls='', mec='red', mfc=(0,0,0,0))
    # for num, coord in coords.iterrows():
        # plt.text(coord.x, coord.y, str(num), color='yellow')
    # plt.show()
    
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
   
    for num, i in l.iterrows():
        plt.cla()
        img = io.imread(i.Dir)
        subtraj = traj.loc[traj.Name==i.Name]
        ax.imshow(img, cmap='gray')
        plt.axis('off')
        ax.plot(subtraj.x, subtraj.y, marker='o', markersize=12, ls='', mec='red', mfc=(0,0,0,0))
        plt.pause(.1)        
        # plt.savefig(os.path.join(folder, r'mindist_20', i.Name + '.png'), format='png')