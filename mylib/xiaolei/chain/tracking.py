import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util, filters, measure
import os
from scipy import ndimage
from corrLib import readseq
import pdb

def get_chain_mask(img, feature_size=7000, feature_number=1):
    # pdb.set_trace()
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
def FastPeakFind(data):
    """
    Find all local maxima (peaks) in given 2D array
    """
    def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
    if str(data.dtype) != 'float32':
        data = data.astype('float32')
    mf = medfilt2d(data, kernel_size=3)
    mf = mf.astype('float32')
    thres = max(min(np.amax(mf,axis=0)), min(np.amax(mf,axis=1)))    
    filt = matlab_style_gauss2D()
    conv = convolve2d(mf, filt, mode='same')
    w_idx = conv > thres
    bw = conv.copy()
    bw[w_idx] = 1
    bw[~w_idx] = 0
    thresholded = np.multiply(bw, conv)
    edg = 3
    shape = data.shape
    idx = np.nonzero(thresholded[edg-1: shape[0]-edg-1, edg-1: shape[1]-edg-1])
    idx = np.transpose(idx)
    cent = []
    for xy in idx:
        x = xy[0]
        y = xy[1]
        if thresholded[x, y] >= thresholded[x-1, y-1] and \
            thresholded[x, y] > thresholded[x-1, y] and \
            thresholded[x, y] >= thresholded[x-1, y+1] and \
            thresholded[x, y] > thresholded[x, y-1] and \
            thresholded[x, y] > thresholded[x, y+1] and \
            thresholded[x, y] >= thresholded[x+1, y-1] and \
            thresholded[x, y] > thresholded[x+1, y] and \
            thresholded[x, y] >= thresholded[x+1, y+1]:
            cent.append(xy)
    cent = np.asarray(cent).transpose()
    return cent

def find_maxima_num(img, num_particles):
    """
    Find given number of local maxima in a 2D array. If the actual number of peaks is less than given number (num_particles), the code will find all the existing peaks instead.
    """
    def gauss1(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2)) 
    def maxk(array, num_max):
        array = np.asarray(array)
        length = array.size
        array = array.reshape((1, length))
        idx = np.argsort(array)
        idx2 = np.flip(idx)
        return idx2[0, 0: num_max]
    cent = FastPeakFind(img)
    num_particles = min(num_particles, cent.shape[1])
    peaks = img[cent[0], cent[1]]
    ind = maxk(peaks, num_particles)
    max_coor_tmp = cent[:, ind]
    max_coor = max_coor_tmp.astype('float32')
    pk_value = peaks[ind]    
    for num in range(0, num_particles):
        x = max_coor_tmp[0, num]
        y = max_coor_tmp[1, num]
        try:
            fitx1 = np.asarray(range(x-7, x+8))
            fity1 = np.asarray(img[range(x-7, x+8), y])        
            popt1,pcov1 = curve_fit(gauss1, fitx1, fity1, p0=[1, x, 3])
            max_coor[0, num] = popt1[1]      
            fitx2 = np.asarray(range(y-7, y+8))
            fity2 = np.asarray(img[x, range(y-7, y+8)])
            popt2,pcov2 = curve_fit(gauss1, fitx2, fity2, p0=[1, y, 3])
            max_coor[1, num] = popt2[1]
        except:
            print('Fitting error')
            continue
    return max_coor, pk_value

def dt_track_1(img, target_number, feature_size=7000, feature_number=1):
    mask = get_chain_mask(img, feature_size, feature_number)
    isod = img > filters.threshold_isodata(img)
    masked_isod = mask * isod
    despeck = ndimage.median_filter(masked_isod, size=10)
    dt = ndimage.distance_transform_edt(despeck)
    max_coor, pk_value = find_maxima_num(dt, target_number)
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
    traj = dt_track_1(r'I:\Github\Python\mylib\xiaolei\chain\test_files\problem_image\0008.tif', 15)

    
    # avg_cos test code 
    # traj = pd.read_csv(r'E:\Github\Python\ForFun\Misc\cos\data.csv')
    # order_pNo = [0, 13, 12, 11, 9, 8, 6, 7, 1, 2, 4, 3, 5, 14, 10]
    # df = avg_cos(traj, order_pNo)
    # plt.plot(df.frame, df.cos)
    # plt.show()
    