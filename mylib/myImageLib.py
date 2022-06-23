import os
import numpy as np
from scipy import fftpack
from scipy.signal import medfilt2d, convolve2d, fftconvolve
from scipy.optimize import curve_fit
from scipy import exp
import pandas as pd
import sys
import psutil
from skimage import io
import time
import shutil
from nd2reader import ND2Reader
# %% codecell

def dirrec(path, filename):
    """
    Recursively look for all the directories of files with name <filename>.
    ---
    *args
        path ... the directory where you want to look for files.
        filename ... name of the files you want to look for.
    Return
        dirList ... a list of full directories of files with name <filename>
    Note
        <filename> can be partially specified, e.g. '*.py' to search for all the
        .py files or 'track*' to search for all files starting with 'track'.
    """
    dirList = []
    for r, d, f in os.walk(path):
        for dir in d:
            tmp = dirrec(dir, filename)
            if tmp:
                dirList.append(tmp)
        for file in f:
            if filename.startswith('*'):
                if file.endswith(filename[1:]):
                    dirList.append(os.path.join(r, file))
            elif filename.endswith('*'):
                if file.startswith(filename[:-1]):
                    dirList.append(os.path.join(r, file))
            elif file == filename:
                dirList.append(os.path.join(r, file))
    return dirList
def to8bit(img16):
    """
    Enhance contrast and convert to 8-bit
    """
    # if img16.dtype != 'uint16':
        # raise ValueError('16-bit grayscale image is expected')
    maxx = img16.max()
    minn = img16.min()
    img8 = (img16 - minn) / (maxx - minn) * 255
    return img8.astype('uint8')
def bpass(*args):
    img8 = args[0]
    low = args[1]
    high = args[2]
    def gen_filter(img, low, high):
        filt = np.zeros(img.shape)
        h, w = img.shape
        center = [int(w/2), int(h/2)]
        Y, X = np.ogrid[:h, :w]
        dist = ((X - center[0])**2 + (Y-center[1])**2)**.5

        filt[(dist>low)&(dist<=high)] = 1
        return filt
    filt = gen_filter(img8, low, high)
    filt = fftpack.ifftshift(filt)
    im_fft = fftpack.fft2(img8)
    im_fft_filt = im_fft * filt
    im_new = fftpack.ifft2(im_fft_filt).real
    im_new = im_new - im_new.min()
    im_new = np.floor_divide(im_new, (im_new.max()+1)/256)
    return im_new.astype('uint8')
def bestcolor(n):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    return colors[n]
def wowcolor(n):
    colors = ['#C41F3B', '#A330C9', '#FF7D0A', '#A9D271', '#40C7EB',
              '#00FF96', '#F58CBA', '#FFF569', '#0070DE', '#8787ED',
              '#C79C6E', '#BBBBBB', '#1f77b4', '#ff7f0e', '#2ca02c',
              '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    return colors[n]
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
def FastPeakFind(data):
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
def minimal_peakfind(img):
    edg = 3
    shape = img.shape
    idx = np.nonzero(img[edg-1: shape[0]-edg-1, edg-1: shape[1]-edg-1])
    idx = np.transpose(idx)
    cent = []
    for xy in idx:
        x = xy[0]
        y = xy[1]
        if img[x, y] >= img[x-1, y-1] and \
            img[x, y] > img[x-1, y] and \
            img[x, y] >= img[x-1, y+1] and \
            img[x, y] > img[x, y-1] and \
            img[x, y] > img[x, y+1] and \
            img[x, y] >= img[x+1, y-1] and \
            img[x, y] > img[x+1, y] and \
            img[x, y] >= img[x+1, y+1]:
            cent.append(xy)
    cent = np.asarray(cent).transpose()
    return cent
def maxk(array, num_max):
    array = np.asarray(array)
    length = array.size
    array = array.reshape((1, length))
    idx = np.argsort(array)
    idx2 = np.flip(idx)
    return idx2[0, 0: num_max]
def track_spheres_dt(img, num_particles):
    def gauss1(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
    cent = FastPeakFind(img)
    num_particles = min(num_particles, cent.shape[1])
    peaks = img[cent[0], cent[1]]
    ind = maxk(peaks, num_particles)
    max_coor_tmp = cent[:, ind]
    max_coor = max_coor_tmp.astype('float32')
    pk_value = peaks[ind]
    for num in range(0, num_particles):
        try:
            x = max_coor_tmp[0, num]
            y = max_coor_tmp[1, num]
            fitx1 = np.asarray(range(x-7, x+8))
            fity1 = np.asarray(img[range(x-7, x+8), y])
            popt,pcov = curve_fit(gauss1, fitx1, fity1, p0=[1, x, 3])
            max_coor[0, num] = popt[1]
            fitx2 = np.asarray(range(y-7, y+8))
            fity2 = np.asarray(img[x, range(y-7, y+8)])
            popt,pcov = curve_fit(gauss1, fitx2, fity2, p0=[1, y, 3])
            max_coor[1, num] = popt[1]
        except:
            print('Fitting failed')
            max_coor[:, num] = max_coor_tmp[:, num]
            continue
    return max_coor, pk_value
def gauss1(x,a,x0,sigma,b):
    return a*exp(-(x-x0)**2/(2*sigma**2)) + b
def show_progress(progress, label='', bar_length=60):
    """ Display a progress bar
    Args:
    progress -- float between 0 and 1
    label -- a string displayed before the progress bar
    bar_length -- the length of the progress bar

    Returns:
    None

    Test:
    >>> show_progress(0.7, label='0812-data')
    """
    N_finish = int(progress*bar_length)
    N_unfinish = bar_length - N_finish
    print('{0} [{1}{2}] {3:.1f}%'.format(label, '#'*N_finish, '-'*N_unfinish, progress*100), end="\r")
def readdata(folder, ext='csv'):
    """
    Read data files with given extensions in a folder.

    Args:
    folder -- the folder to search in
    ext -- (optional) file extension of data files looked for, default to 'csv'

    Returns:
    fileList -- a DataFrame with columns 'Name' and 'Dir'

    Edit:
    11302020 -- reset the index of fileList, so that further trimming of data by index gets easier
    """
    dataDirs = dirrec(folder, '*.' + ext)
    nameList = []
    dirList = []
    for dataDir in dataDirs:
        path, file = os.path.split(dataDir)
        name, ext = os.path.splitext(file)
        nameList.append(name)
        dirList.append(dataDir)
    fileList = pd.DataFrame()
    fileList = fileList.assign(Name=nameList, Dir=dirList)
    fileList = fileList.sort_values(by=['Name']).reset_index(drop=True)
    return fileList

# %% codecell

class rawImage:
    """Implement raw image converting schemes."""
    def __init__(self, file_dir):
        self.file = file_dir
    def extract_tif(self):
        """Wrapper of all format-specific extractors."""
        file, ext = os.path.splitext(self.file)
        if ext == ".raw":
            self.extract_raw()
        elif ext == ".nd2":
            self.extract_nd2()
        else:
            raise ValueError("Unrecognized image format {}".format(ext))
    def extract_raw(self, cutoff=None):
        """Extract .tif sequence from .raw file.
        cutoff -- number of images extracted.
        Edit:
        Mar 14, 2022 -- Initial commit."""
        # read info from info_file
        folder, file = os.path.split(self.file)
        info_file = os.path.join(folder, "RawImageInfo.txt")
        if os.path.exists(info_file):
            fps, h, w = self.read_raw_image_info(info_file)
        else:
            print("Info file missing!")

        # create output folders, skip if both folders exist
        save_folder = folder
        out_raw_folder = os.path.join(save_folder, 'raw')
        out_8_folder = os.path.join(save_folder, '8-bit')
        if os.path.exists(out_raw_folder) and os.path.exists(out_8_folder):
            print(time.asctime() + " // {} tif folders exists, skipping".format(self.file))
            return None
        if os.path.exists(out_raw_folder) == False:
            os.makedirs(out_raw_folder)
        if os.path.exists(out_8_folder) == False:
            os.makedirs(out_8_folder)

        # check disk
        if self._disk_capacity_check() == False:
            raise SystemError("No enough disk capacity!")

        # calculate buffer size based on available memory, use half of it
        file_size = os.path.getsize(self.file)
        avail_mem = psutil.virtual_memory().available
        unit_size = (h * w + 2) * 2 # the size of a single unit in bytes (frame# + image)
        buffer_size = ((avail_mem // 2) // unit_size) * unit_size # integer number of units
        remaining_size = file_size

        # print("Memory information:")
        # print("Available memory: {:.1f} G".format(avail_mem / 2 ** 30))
        # print("File size: {:.1f} G".format(file_size / 2 ** 30))
        # print("Buffer size: {:.1f} G".format(buffer_size / 2 ** 30))

        # read the binary files, in partitions if needed
        num = 0
        n_images = int(file_size // unit_size)
        t0 = time.monotonic()
        while remaining_size > 0:
            # load np.array from buffer
            if remaining_size > buffer_size:
                read_size = buffer_size
            else:
                read_size = remaining_size
            with open(self.file, "rb") as f:
                f.seek(-remaining_size, 2) # offset bin file reading, counting remaining size from EOF
                bytestring = f.read(read_size)
                a = np.frombuffer(bytestring, dtype=np.uint16)
            remaining_size -= read_size

            # manipulate the np.array
            assert(a.shape[0] % (h*w+2) == 0)
            num_images = a.shape[0] // (h*w+2)
            img_in_row = a.reshape(num_images, h*w+2)
            labels = img_in_row[:, :2] # not in use currently
            images = img_in_row[:, 2:]
            images_reshape = images.reshape(num_images, h, w)

            # save the image sequence
            for label, img in zip(labels, images_reshape):
                # num = label[0] + label[1] * 2 ** 16 + 1 # convert image label to uint32 to match the info in StagePosition.txt
                io.imsave(os.path.join(save_folder, 'raw', '{:05d}.tif'.format(num)), img, check_contrast=False)
                io.imsave(os.path.join(save_folder, '8-bit', '{:05d}.tif'.format(num)), to8bit(img), check_contrast=False)
                t1 = time.monotonic() - t0
                show_progress(num / n_images, label="{:.1f} frame/s".format(num / t1))
                num += 1
                if cutoff is not None:
                    if num > cutoff:
                        return None
    def read_raw_image_info(self, info_file):
        """Read image info, such as fps and image dimensions, from RawImageInfo.txt.
        Helper function of extract_raw()."""
        with open(info_file, 'r') as f:
            a = f.read()
        fps, h, w = a.split('\n')[0:3]
        return int(fps), int(h), int(w)
    def extract_nd2(self):
        # check disk
        if self._disk_capacity_check() == False:
            raise SystemError("No enough disk capacity!")

        folder, file = os.path.split(self.file)
        name, ext = os.path.splitext(file)
        saveDir = os.path.join(folder, name, 'raw')
        saveDir8 = os.path.join(folder, name, '8-bit')
        if os.path.exists(saveDir) == False:
            os.makedirs(saveDir)
        if os.path.exists(saveDir8) == False:
            os.makedirs(saveDir8)
        t0 = time.monotonic()
        with ND2Reader(self.file) as images:
            n_images = len(images)
            for num, image in enumerate(images):
                io.imsave(os.path.join(saveDir8, '{:05d}.tif'.format(num)), to8bit(image))
                io.imsave(os.path.join(saveDir, '%05d.tif' % num), image, check_contrast=False)
                t1 = time.monotonic() - t0
                show_progress(num / n_images, label="{:.1f} frame/s".format(num/t1))
    def _disk_capacity_check(self):
        """Check if the capacity of disk is larger than twice of the file size.
        Args:
        file -- directory of the (.nd2) file being converted
        Returns:
        flag -- bool, True if capacity is enough."""
        d = os.path.split(self.file)[0]
        fs = os.path.getsize(self.file) / 2**30
        ds = shutil.disk_usage(d)[2] / 2**30
        print("File size {0:.1f} GB, Disk size {1:.1f} GB".format(fs, ds))
        return ds > 2 * fs
# %% codecell

if __name__=="__main__":
    # %% codecell
    # test rawImage class
    file = r"D:\swimming_droplets\2022-03-11_09h55m01s\RawImage.raw"
    raw = rawImage(file)
    raw.extract_raw()
