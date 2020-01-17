import os
import numpy as np
from scipy import fftpack
from scipy.signal import medfilt2d, convolve2d, fftconvolve
from scipy.optimize import curve_fit
from scipy import exp

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
    max = img16.max()
    min = img16.min()
    img8 = np.floor_divide(img16 - min , (max - min + 1) / 256)
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

