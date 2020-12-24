import numpy as np
import math
from skimage import io
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.signal import medfilt2d, convolve2d, fftconvolve
from scipy.ndimage import filters

import pdb

def xy_to_rc(img):
    pass
    
def sectionWindow(imgShape, windowShape):
    """
    imgShape: shape of image to be sectioned, tuple
    windowShape: shape of windowShape, tuple
    """
    Xnum = math.floor(imgShape[0]/windowShape[0])
    Ynum = math.floor(imgShape[1]/windowShape[1])
    Xrem = imgShape[0]%windowShape[0]
    Yrem = imgShape[1]%windowShape[1]
    Xstart = math.floor(Xrem/2)
    Ystart = math.floor(Yrem/2)
    Xend = imgShape[0] - Xstart + 1 
    Yend = imgShape[1] - Ystart + 1
    X = np.asarray(range(Xstart, Xend, windowShape[0]))
    Y = np.asarray(range(Ystart, Yend, windowShape[1]))
    return X, Y

def fft2_imagej(img):
    """
    Mimic the result of FFT method in ImageJ.
    Low frequency signals are moved to the center of the image.
    Logarithm is take for the original fft2 transform to enhance the visibility.
    The result is then normalized to make it ready for display. 
    """
    fft = fftpack.fft2(img)
    fft = np.log(fft)
    fft = np.real(fft)
    
    # pdb.set_trace()
    
    fft = fft.astype('float32')
    fft = fft - np.amin(fft)
    fft = fft/np.max(fft)
    fft_ij = fft.copy()
    h, w = img.shape
    h2 = math.floor(h/2)
    w2 = math.floor(w/2)
    for i in range(0, h):
        for j in range(0, w):
            i2 = i + h2
            j2 = j + w2
            if i2 > h - 1:
                i2 = i2 - h
            if j2 > w - 1:
                j2 = j2 - w
            fft_ij[i, j] = fft[i2, j2]
    return fft_ij

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
    
def img_smooth(img):
    """
    Denoising img
    """
    if str(img.dtype) != 'float32':
        img = img.astype('float32')
    img = medfilt2d(img, kernel_size=3)
    img = img.astype('float32')
    filt = matlab_style_gauss2D()
    conv = convolve2d(img, filt, mode='same')
    conv = filters.gaussian_filter(conv, 3)
    return conv

def imgAutoThreshold(img):
    if str(img.dtype) != 'float32':
        img = img.astype('float32')
    thres = max(min(np.amax(img,axis=0)), min(np.amax(img,axis=1)))
    w_idx = img > 1.2*thres
    img[w_idx] = 1
    img[~w_idx] = 0
    return img

def getOrientation(bw_img):
    try:
        label_img = label(thres).transpose()
        region = regionprops(label_img, coordinates='rc')
        area = 0
        for num in range(0, len(region)):
            area_tmp = region[num]['area']
            maj_tmp = region[num]['major_axis_length']
            min_tmp = region[num]['minor_axis_length']
            ori_tmp = region[num]['orientation']
            if area_tmp > area:
                area = area_tmp
                num_max = num
        maj = region[num_max]['major_axis_length']
        min = region[num_max]['minor_axis_length']
        ori = region[num_max]['orientation']
        img_ori = ori + math.pi/2
        # if img_ori > math.pi:
            # img_ori = img_ori - math.pi
        img_ori_magnitude = maj/min
    except:
        img_ori, img_ori_magnitude = 0, 0
    return img_ori, img_ori_magnitude


def imgOrientation(img, windowShape):
    X, Y = sectionWindow(img.shape, windowShape)
    y = X[0:X.size-1] + windowShape[0]/2
    x = Y[0:Y.size-1] + windowShape[1]/2
    x, y = np.meshgrid(x, y)
    u_ori = x.copy()
    v_ori = y.copy()
    for i in range(0, x.shape[0]): # col
        for j in range(0, y.shape[1]): # row
            window = img[X[i]:X[i+1], Y[j]:Y[j+1]]
            fft = fft2_imagej(window)
            smooth = img_smooth(fft)
            thres = imgAutoThreshold(smooth)
            ori, orim = getOrientation(thres)
            u_ori[i, j] = orim*math.cos(ori)
            v_ori[i, j] = orim*math.sin(ori)
            
    pdb.set_trace()
    return x, y, u_ori, v_ori      
    
    
    
def test():
    windowShape = (50, 50)
    img = io.imread('testFT_3.tif', as_gray=True)
    X, Y = sectionWindow(img.shape, windowShape)
    y = X[0:X.size-1] + windowShape[0]/2
    x = Y[0:Y.size-1] + windowShape[1]/2
    x, y = np.meshgrid(x, y)
    u_ori = x.copy()
    v_ori = y.copy()
    for i in range(0, x.shape[0]): # col
        for j in range(0, y.shape[1]): # row
            window = img[X[i]:X[i+1], Y[j]:Y[j+1]]
            fft = fft2_imagej(window)
            smooth = img_smooth(fft)
            thres = imgAutoThreshold(smooth)
            ori, orim = getOrientation(thres)
            u_ori[i, j] = orim*math.cos(ori)
            v_ori[i, j] = orim*math.sin(ori)
    plt.imshow(img, cmap='gray')
    plt.quiver(x, y, u_ori, v_ori, color='red') 
    plt.show()

def pre_processing(img):
    fft = fft2_imagej(img)
    smooth = img_smooth(fft)
    thres = imgAutoThreshold(smooth)
    return thres
    
if __name__ == '__main__':
    ### Input args ###
    windowShape = (50, 50)
    img = io.imread('testFT_4.jpg', as_gray=True)
    ##################
    X, Y = sectionWindow(img.shape, windowShape)
    y = X[0:X.size-1] + windowShape[0]/2
    x = Y[0:Y.size-1] + windowShape[1]/2
    x, y = np.meshgrid(x, y)
    u_ori = x.copy()
    v_ori = y.copy()
    for i in range(0, x.shape[0]): # col
        for j in range(0, y.shape[1]): # row
            window = img[X[i]:X[i+1], Y[j]:Y[j+1]]
            # fft = fft2_imagej(window)
            # smooth = img_smooth(fft)
            # thres = imgAutoThreshold(smooth)
            thres = pre_processing(window)
            ori, orim = getOrientation(thres)
            u_ori[i, j] = orim*math.cos(ori)
            v_ori[i, j] = orim*math.sin(ori)
    plt.imshow(img, cmap='gray')
    plt.quiver(x, y, u_ori, v_ori, color='red') 
    plt.axis('off')
    fig = plt.gcf()
    fig.savefig('FT_4.pdf', format='pdf', pad_inches=0)

    
    
