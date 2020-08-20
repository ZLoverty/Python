import numpy as np
import matplotlib.pyplot as plt
from myImageLib import dirrec, bpass
from miscLib import label_slope
from skimage import io, util
import os
import pandas as pd
import sys
import time
import pdb
from numpy.polynomial.polynomial import polyvander

# def corrS(X, Y, U, V):
    # row, col = X.shape
    # vsq = 0
    # CA = np.zeros((row, col))
    # CV = np.zeros((row, col))
    # for i in range(0, row):
        # for j in  range(0, col):
            # vsq += U[i, j]**2 + V[i, j]**2
    # for xin in range(0, col):
        # for yin in range(0, row):
            # count = 0
            # CAt = 0
            # CVt = 0
            # for i in range(0, col-xin):
                # for j in range(0, row-yin):
                    # ua = U[j, i]
                    # va = V[j, i]
                    # ub = U[j+yin, i+xin]
                    # vb = V[j+yin, i+xin]
                    # CAt += (ua*ub+va*vb)/((ua**2+va**2)*(ub**2+vb**2))**.5
                    # CVt += ua*ub + va*vb
                    # count += 1
            # CA[yin, xin] = CAt / count
            # CV[yin, xin] = CVt / vsq     
    # return CA, CV

def corrS(X, Y, U, V):
    row, col = X.shape
    r = int(row/2)
    c = int(col/2)
    vsqrt = (U ** 2 + V ** 2) ** 0.5
    U = U - U.mean()
    V = V - V.mean()
    Ax = U / vsqrt
    Ay = V / vsqrt
    CA = np.ones((r, c))
    CV = np.ones((r, c))
    for xin in range(0, c):
        for yin in range(0, r):
            if xin != 0 or yin != 0:
                CA[yin, xin] = (Ax[0:row-yin, 0:col-xin] * Ax[yin:row, xin:col] + Ay[0:row-yin, 0:col-xin] * Ay[yin:row, xin:col]).mean()
                CV[yin, xin] = (U[0:row-yin, 0:col-xin] * U[yin:row, xin:col] + V[0:row-yin, 0:col-xin] * V[yin:row, xin:col]).mean() / (U.std()**2+V.std()**2)
    return X[0:r, 0:c], Y[0:r, 0:c], CA, CV

# def corrI(X, Y, I):
    # I = I - I.mean()
    # row, col = I.shape
    # Isq = 0
    # for i in range(0, row):
        # for j in range(0, col):
            # Isq += I[i, j]**2
    # Isq = Isq / row / col
    # CI = np.zeros((row, col))
    # for xin in range(0, col):
        # for yin in range(0, row):
            # count = 0
            # CIt = 0
            # for i in range(0, col-xin):
                # for j in range(0, row-yin):
                    # Ia = I[j, i]
                    # Ib = I[j+yin, i+xin]
                    # CIt += Ia * Ib
                    # count += 1
            # CI[yin, xin] = CIt / count / Isq
    # return CI

def corrI(X, Y, I):
    row, col = I.shape
    I = I - I.mean()
    # CI = np.ones(I.shape)
    r = int(row/2)
    c = int(col/2)
    CI = np.ones((r, c))
    XI = X[0: r, 0: c]
    YI = Y[0: r, 0: c]
    normalizer = I.std() ** 2
    for xin in range(0, int(col/2)):
        for yin in range(0, int(row/2)):
            if xin != 0 or yin != 0:
                I_shift_x = np.roll(I, xin, axis=1)
                I_shift = np.roll(I_shift_x, yin, axis=0)
                CI[yin, xin] = (I[yin:, xin:] * I_shift[yin:, xin:]).mean() / normalizer
    return XI, YI, CI
    
# def divide_windows(img, windowsize=[20, 20], step=10):
    # row, col = img.shape
    # windowsize[0] = int(windowsize[0])
    # windowsize[1] = int(windowsize[1])
    # step = int(step)
    # X = np.array(range(0, col-windowsize[0], step))# + int(windowsize[0]/2)
    # Y = np.array(range(0, row-windowsize[1], step))# + int(windowsize[1]/2)
    # I = np.zeros((len(Y), len(X)))
    # for indx, x in enumerate(X):
        # for indy, y in enumerate(Y):
            # window = img[y:y+windowsize[1], x:x+windowsize[0]]
            # I[indy, indx] = window.mean()
    # X, Y = np.meshgrid(X, Y)
    # return X, Y, I

def divide_windows(img, windowsize=[20, 20], step=10):
    row, col = img.shape
    windowsize[0] = int(windowsize[0])
    windowsize[1] = int(windowsize[1])
    step = int(step)
    if isinstance(windowsize, list):
        windowsize = tuple(windowsize)
    X = np.array(range(0, col-windowsize[0], step))
    Y = np.array(range(0, row-windowsize[1], step))
    X, Y = np.meshgrid(X, Y)
    I = util.view_as_windows(img, windowsize, step=step).mean(axis=(2, 3))
    return X, Y, I
    
# def distance_corr(X, Y, C):
    # rList = []
    # cList = []
    # table = pd.DataFrame()
    # for xr, yr, cr in zip(X, Y, C):
        # for x, y, c in zip(xr, yr, cr):
            # rList.append((x**2 + y**2)**.5)
            # cList.append(c)
    # table = table.assign(R=rList, C=cList)
    # table.sort_values(by=['R'], inplace=True)
    # return table
    
def distance_corr(X, Y, C):
    r_corr = pd.DataFrame({'R': (X.flatten()**2 + Y.flatten()**2) ** 0.5, 'C': C.flatten()}).sort_values(by='R')
    return r_corr

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
    size_min = 5    
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
    # bp = bpass(img8, 3, 100)
    # img8_mh = match_hist(bp, img8)
    NList = []
    dNList = []
    for bs in boxsize:
        X, Y, I = divide_windows(img8, windowsize=[bs, bs], step=bs)
        N = bs*bs
        # dN = np.log10(I).std()*bs*bs
        dN = I.std()*bs*bs
        NList.append(N)
        dNList.append(dN)
    df_data = pd.DataFrame().assign(n=NList, d=dNList)
    return df_data


def div_field(img, pivData, winsize, step):
    # A function that calculates the divergence field
    # img is the image from microscopy, pivData is a DataFrame with columns (x, y, u, v)
    # winsize and step should be consistent with the parameters of pivData, be extra coutious!
    
    # preprocessing, bpass and match hist for raw image, convert intensity field to density field
    
    # return value: intensity field (subtracted from I0)
    # bp = bpass(img, 3, 100)
    # bp_mh = match_hist(bp, img)
    # winsize = 10
    # step = 10
    X, Y, I = divide_windows(img, windowsize=[winsize, winsize], step=step)
    # concentration field
    I0 = 255
    c = I0 - I
    
    # calculation for divcn and divcv
    row, col = I.shape
    vx = np.array(pivData.u).reshape(I.shape)
    vy = np.array(pivData.v).reshape(I.shape)
    v = (vx**2 + vy**2)**.5
    nx = np.array(pivData.u / (pivData.u**2 + pivData.v**2)**.5).reshape(I.shape)
    ny = np.array(pivData.v / (pivData.u**2 + pivData.v**2)**.5).reshape(I.shape)
    cnx = c * nx
    cny = c * ny
    cvx = c * vx
    cvy = c * vy
    divcn = np.zeros(I.shape)
    divcv = np.zeros(I.shape)
    divv = np.zeros(I.shape)
    for x in range(0, col-1):
        for y in range(0, row-1):
            divcn[y, x] = cnx[y,x+1] - cnx[y,x] + cny[y+1,x] - cny[y,x]
            divcv[y, x] = cvx[y,x+1] - cvx[y,x] + cvy[y+1,x] - cvy[y,x]
            divv[y, x] = vx[y,x+1] - vx[y,x] + vy[y+1,x] - vy[y,x]
    return c, v, divcn, divcv, divv
    
def readdata(folder):
    dataDirs = dirrec(folder, '*.csv')
    nameList = []
    dirList = []
    for dataDir in dataDirs:
        path, file = os.path.split(dataDir)
        name, ext = os.path.splitext(file)
        nameList.append(name)
        dirList.append(dataDir)
    fileList = pd.DataFrame()
    fileList = fileList.assign(Name=nameList, Dir=dirList)
    fileList = fileList.sort_values(by=['Name'])
    return fileList

def df2(imgstack, size_min=5, step=250, method='linear'):
    """
    This is used for small scale test of temporal variation based density fluctuation analysis. 
    Here the input is a 3 dimensional array that contains all the images in a video [stack_index, height, width].
    For larger scale application, use the script df2.py or a later implementation of df2 which can take directory as argument 1.
    
    Args: 
    imgstack -- 3-D array with dimensions [stack_index, height, width]
    size_min -- minimal box size to sample
    step -- distance between adjacent boxes in a frame
    method -- use pixel intensity directly as concentration ('linear'), or use the log of it ('log')
    
    Returns:
    average -- the spatial average of temporal variations    
    """
    
    L = min(imgstack.shape[1:3])
    boxsize = np.unique(np.floor(np.logspace(np.log10(size_min),
                        np.log10((L-size_min)/2),100)))

    df = pd.DataFrame()
    for i, img in enumerate(imgstack):
        framedf = pd.DataFrame()
        for bs in boxsize: 
            X, Y, I = divide_windows(img, windowsize=[bs, bs], step=step)
            tempdf = pd.DataFrame().assign(I=I.flatten(), t=int(i), size=bs, 
                           number=range(0, len(I.flatten())))
            framedf = framedf.append(tempdf)
        df = df.append(framedf)
    
    if method == 'log':
        df['I'] = np.log(df['I'])
    
    df_out = pd.DataFrame()
    for number in df.number.drop_duplicates():
        subdata1 = df.loc[df.number==number]
        for s in subdata1['size'].drop_duplicates():
            subdata = subdata1.loc[subdata1['size']==s]
            
            d = s**2 * np.array(subdata.I).std()
            n = s**2 
            tempdf = pd.DataFrame().assign(n=[n], d=d, size=s, number=number)
            df_out = df_out.append(tempdf)

    average = pd.DataFrame()
    for s in df_out['size'].drop_duplicates():
        subdata = df_out.loc[df_out['size']==s]
        avg = subdata.drop(columns=['size', 'number']).mean().to_frame().T
        average = average.append(avg)
        
    return average

def plot_gnf(gnf_data):
    """
    Used for small scale test of gnf analysis. 
    It incorporates the guide of the eye slope already in the function, which usually needs further adjustment when preparing paper figures.
    
    Args:
    gnf_data -- gnf data generated by df2()
    
    Returns:
    ax -- axis on which the data are plotted
    slope -- the slope of the gnf_data curve
    """
    x = gnf_data.n / 100
    y = gnf_data.d / x ** 0.5
    y = y / y.iat[0]
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.plot(x, y)
    xf, yf, xt, yt, slope = label_slope(x, y, location='n')
    ax.plot(xf, yf, ls='--', color='black')
    ax.text(xt, yt, '{:.2f}'.format(slope))
    ax.loglog()
    ax.set_xlabel('$l^2/l_b^2$')
    ax.set_ylabel('$\Delta N/\sqrt{N}$')
    return ax, slope 
    
if __name__ == '__main__':
    img = io.imread(r'I:\Github\Python\Correlation\test_images\GNF\stat\40-1.tif')
    df_data = density_fluctuation(img)
    