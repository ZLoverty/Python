from openpiv import pyprocess
from openpiv.smoothn import smoothn
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pandas as pd
from corrLib import readseq
from scipy.signal import medfilt2d
import os

def PIV1(I0, I1, winsize, overlap, dt, smooth=True):
    u0, v0 = pyprocess.extended_search_area_piv(I0.astype(np.int32), I1.astype(np.int32), window_size=winsize, overlap=overlap, dt=dt, search_area_size=winsize)
    x, y = pyprocess.get_coordinates(image_size=I0.shape, search_area_size=winsize, window_size=winsize, overlap=overlap)
    if smooth == True:
        u1 = smoothn(u0)[0]
        v1 = smoothn(v0)[0]
        frame_data = pd.DataFrame(data=np.array([x.flatten(), y.flatten(), u1.flatten(), v1.flatten()]).T, columns=['x', 'y', 'u', 'v'])
    else:
        frame_data = pd.DataFrame(data=np.array([x.flatten(), y.flatten(), u0.flatten(), v0.flatten()]).T, columns=['x', 'y', 'u', 'v'])
    return frame_data
    
def imseqPIV(folder, winsize, overlap, dt):       
    data = pd.DataFrame()
    l = readseq(folder)
    for num, i in l.iterrows():
        # read 2 adjacent images
        if num % 2 == 0:
            I0 = io.imread(i.Dir)
            continue 
        I1 = io.imread(i.Dir)
        # run PIV function "extended_search_area_piv()"
        frame_data = PIV1(I0, I1, winsize, overlap, dt)
        if num < 2:
            data = frame_data.assign(frame=num)
        else:
            data = data.append(frame_data.assign(frame=num))
    return data

def read_piv(pivDir):
    """
    Read piv data from pivDir as X, Y, U, V
    
    X, Y, U, V = read_piv(pivDir)
    """
    pivData = pd.read_csv(pivDir)
    row = len(pivData.y.drop_duplicates())
    col = len(pivData.x.drop_duplicates())
    X = np.array(pivData.x).reshape((row, col))
    Y = np.array(pivData.y).reshape((row, col))
    U = np.array(pivData.u).reshape((row, col))
    V = np.array(pivData.v).reshape((row, col))
    return X, Y, U, V
    
if __name__ == '__main__':
    # set PIV parameters
    winsize = 50 # pixels
    searchsize = 100  # pixels, search in image B
    overlap = 25 # pixels
    dt = 0.033 # frame interval (sec)
    # imgDir = r'R:\Dip\DF\PIV_analysis\1.tif'
    # data = tiffstackPIV(imgDir, winsize, searchsize, overlap, dt)
    folder = r'D:\Wei\Dynamics_raw\60'
    data = imseqPIV(folder, winsize, overlap, dt)
    data.to_csv(os.path.join(folder, 'pivData.csv'), index=False)
    # plt.quiver(data.x, data.y, data.u, data.v)