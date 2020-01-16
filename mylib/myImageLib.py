import os
import numpy as np
from scipy import fftpack
from scipy.signal import medfilt2d, convolve2d, fftconvolve
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