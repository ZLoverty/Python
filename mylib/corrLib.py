import numpy as np
import matplotlib.pyplot as plt
from myImageLib import dirrec, bpass
from skimage import io
import os
import pandas as pd
import sys
import time
import pdb
from numpy.polynomial.polynomial import polyvander

def corrS(X, Y, U, V):
    # X, Y, U, V represent a vector field
    # Return value C is a matrix representing spatial correlation distribution of given vector field
    row, col = X.shape
    vsq = 0
    CA = np.zeros((row, col))
    CV = np.zeros((row, col))
    for i in range(0, row):
        for j in  range(0, col):
            vsq += U[i, j]**2 + V[i, j]**2
    for xin in range(0, row):
        for yin in range(0, col):
            count = 0
            CAt = 0
            CVt = 0
            for i in range(0, col-xin):
                for j in range(0, row-yin):
                    ua = U[j, i]
                    va = V[j, i]
                    ub = U[j+yin, i+xin]
                    vb = V[j+yin, i+xin]
                    CAt += (ua*ub+va*vb)/((ua**2+va**2)*(ub**2+vb**2))**.5
                    CVt += ua*ub + va*vb
                    count += 1
            CA[yin, xin] = CAt / count
            CV[yin, xin] = CVt / vsq     
    return CA, CV

def corrI(X, Y, I):
    I = I - I.mean()
    row, col = I.shape
    Isq = 0
    for i in range(0, row):
        for j in range(0, col):
            Isq += I[i, j]**2
    Isq = Isq / row / col
    CI = np.zeros((row, col))
    for xin in range(0, col):
        for yin in range(0, row):
            count = 0
            CIt = 0
            for i in range(0, col-xin):
                for j in range(0, row-yin):
                    Ia = I[j, i]
                    Ib = I[j+yin, i+xin]
                    CIt += Ia * Ib
                    count += 1
            CI[yin, xin] = CIt / count / Isq
    return CI

def divide_windows(img, windowsize=[20, 20], step=10):
    row, col = img.shape
    windowsize[0] = int(windowsize[0])
    windowsize[1] = int(windowsize[1])
    step = int(step)
    X = np.array(range(0, col-windowsize[0], step))# + int(windowsize[0]/2)
    Y = np.array(range(0, row-windowsize[1], step))# + int(windowsize[1]/2)
#     X, Y = np.meshgrid(X, Y)
    I = np.zeros((len(Y), len(X)))
    for indx, x in enumerate(X):
        for indy, y in enumerate(Y):
            window = img[y:y+windowsize[1], x:x+windowsize[0]]
            I[indy, indx] = window.mean()
    X, Y = np.meshgrid(X, Y)
    return X, Y, I

def distance_corr(X, Y, C):
    rList = []
    cList = []
    table = pd.DataFrame()
    for xr, yr, cr in zip(X, Y, C):
        for x, y, c in zip(xr, yr, cr):
            rList.append((x**2 + y**2)**.5)
            cList.append(c)
    table = table.assign(R=rList, C=cList)
    table.sort_values(by=['R'], inplace=True)
    return table

def corrIseq(folder, **kwargs):
    # Default window settings
    wsize = [100, 100]
    step = 100
    # Process kwargs
    for kw in kwargs:
        if kw == 'windowsize':
            wsize = kwargs[kw]
        if kw == 'step':
            step = kwargs[kw]
    data_seq = pd.DataFrame()
    fileList = readseq(folder)
    for num, i in fileList.iterrows():
        # print('Processing frame {:04}'.format(num))
        imgDir = i.Dir
        img = io.imread(imgDir)
        X, Y, I = divide_windows(img, windowsize=wsize, step=step)
        C = corrI(X, Y, I)
#         A = distance_corr(X, Y, C)
        row, col = C.shape
        X = X.reshape(1, row*col).squeeze()
        Y = Y.reshape(1, row*col).squeeze()
        I = I.reshape(1, row*col).squeeze()
        C = C.reshape(1, row*col).squeeze()
        data_1 = pd.DataFrame(data=np.array([X, Y, I, C]).T, columns=['X', 'Y', 'I', 'C'])
        data_1 = data_1.assign(R=(data_1.X**2+data_1.Y**2)**.5, frame=num)
        data_seq = data_seq.append(data_1)
    return data_seq

def readseq(folder):
    imgDirs = dirrec(folder, '*.tif')
    nameList = []
    dirList = []
    for imgDir in imgDirs:
        path, file = os.path.split(imgDir)
        name, ext = os.path.splitext(file)
        nameList.append(name)
        dirList.append(imgDir)
    fileList = pd.DataFrame()
    fileList = fileList.assign(Name=nameList, Dir=dirList)
    fileList = fileList.sort_values(by=['Name'])
    return fileList

def boxsize_effect_spatial(img, boxsize, mpp):
    # img: the image to be tested, array-like
    # boxsize: a list of boxsize to be tested, list-like
    # mpp: microns per pixel, float
    data = {}
    for bs in boxsize:
        X, Y, I = divide_windows(img, windowsize=[bs, bs], step=bs)
        CI = corrI(X, Y, I)
        dc = distance_corr(X, Y, CI)
        bsm = bs * mpp # boxsize in microns
        dc.R = dc.R * mpp
        data['{0:.1f}'.format(bsm)] = dc
    for kw in data:
        dc = data[kw]
        length = len(dc)
        smooth_length = int(np.ceil(length/20)*2+1)
        plt.plot(dc.R, savgol_filter(dc.C, smooth_length, 3), label=kw)
    plt.legend()
    return data
    
def match_hist(im1, im2):
    # match the histogram of im1 to that of im2
    return (abs(((im1 - im1.mean()) / im1.std() * im2.std() + im2.mean()))+1).astype('uint8')

def density_fluctuation(img8):
    row, col = img8.shape
    l = min(row, col)
    size_min = 20    
    boxsize = np.unique(np.floor(np.logspace(np.log10(size_min), np.log10((l-size_min)/2), 100)))
    # Gradually increase box size and calculate dN=std(I) and N=mean(I)    
    # choose maximal box size to be (l-size_min)/2
    # to guarantee we have multiple boxes for each calculation, so that
    # the statistical quantities are meaningful.
    # Step is chosen as 5*size_min to guarantee speed as well as good statistics
    # instead of box size. When box size is large, number of boxes is too small
    # to get good statistics.
    # Papers by Narayan et al. used different method to calculate density fluctuation
    # Igor Aranson commented on the averaging methods saying that in a spatially
    # homogeneous system (small spatial temporal correlation) two methods should match.
    # This suggests that I need to test both methods.
    bp = bpass(img8, 3, 500)
    img8_mh = match_hist(bp, img8)
    NList = []
    dNList = []
    for bs in boxsize:
        X, Y, I = divide_windows(img8_mh, windowsize=[bs, bs], step=5*size_min)
        N = bs*bs
        dN = np.log10(I).std()*bs*bs
        NList.append(N)
        dNList.append(dN)
    df_data = pd.DataFrame().assign(n=NList, d=dNList)
    return df_data

def corrlength(corrData, fitting_range=1000):
    # Extract correlation length from spatial correlation DataFrame.
    xx = np.array(corrData.R)
    yy = np.array(corrData.C)
    x = xx[xx<fitting_range]
    y = yy[xx<fitting_range]
    p = np.polyfit(x, y, 8)
    xsolve = np.array(range(0, fitting_range))
    yfit = np.dot(polyvander(xsolve, 8), np.flip(p).transpose())
    try:
        corrlen = xsolve[yfit>1/np.e].max()
    except:
        corrlen = 0
    return corrlen

if __name__ == '__main__':
    img = io.imread(r'I:\Github\Python\Correlation\test_images\GNF\stat\40-1.tif')
    df_data = density_fluctuation(img)
    