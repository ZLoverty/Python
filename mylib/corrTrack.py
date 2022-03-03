import numpy as np
from skimage import io
from scipy.optimize import curve_fit
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from xcorr_funcs import normxcorr2, FastPeakFind, maxk, gauss1
from scipy import exp
import time
import pdb

def track_spheres(img, mask, num_particles, subpixel=True):
    def gauss1(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
    corr = normxcorr2(mask, img, mode='same')
    cent = FastPeakFind(corr)
    peaks = corr[cent[0], cent[1]]
    ind = maxk(peaks, num_particles)
    max_coor_tmp = cent[:, ind]
    max_coor = max_coor_tmp.astype('float32')
    pk_value = peaks[ind]
    if subpixel == True:
        for num in range(0, num_particles):
            x = max_coor_tmp[0, num]
            y = max_coor_tmp[1, num]
            fitx1 = np.asarray(range(x-7, x+8))
            fity1 = np.asarray(corr[range(x-7, x+8), y])
            popt, pcov = curve_fit(gauss1, fitx1, fity1, p0=[1, x, 3])
            max_coor[0, num] = popt[1]
            fitx2 = np.asarray(range(y-7, y+8))
            fity2 = np.asarray(corr[x, range(y-7, y+8)])
            popt,pcov = curve_fit(gauss1, fitx2, fity2, p0=[1, y, 3])
            max_coor[1, num] = popt[1]
    return max_coor, pk_value

if __name__ == '__main__':
    t1 = time.monotonic()
    img = io.imread('video.tif')
    mask = io.imread('maski.tif')
    num_images = img.shape[0]
    num_particles = 3
    nTotal = 100
    fid = open('xyt.dat', 'w')
    fid.write('X\tY\tframe\n')
    for num, frame in enumerate(img):
        print('Processing image ' + str(num) + ' ...')
        max_coor, pk_value = track_spheres(frame, mask, num_particles)
        for coor in max_coor.transpose():
            fid.write('%f\t%f\t%d\n' % (coor[0], coor[1], num))
        # plt.imshow(frame)
        # plt.plot(max_coor[1], max_coor[0], 'ro')
        # plt.show('block')
        if num >= nTotal - 1:
            break
    fid.close()
    t2 = time.monotonic()
    print('Python processed ' + str(nTotal) + ' images in ' + str(t2-t1) + ' secs')
