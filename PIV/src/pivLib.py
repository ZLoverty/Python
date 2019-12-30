from openpiv import tools, process, validation, filters, scaling 
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pandas as pd
from corrLib import readseq
from scipy.signal import medfilt2d
import os

def tiffstackPIV(imgDir, winsize, searchsize, overlap, dt):    
    imgs = io.imread(imgDir)    
    data = pd.DataFrame()
    for num, img in enumerate(imgs):
        # read 2 adjacent images
        if num % 2 == 0:
            I0 = img
            continue 
        I1 = img
        # run PIV function "extended_search_area_piv()"
        u0, v0 = process.extended_search_area_piv(I0.astype(np.int32), I1.astype(np.int32), 
                                                             window_size=winsize, overlap=overlap, dt=dt, 
                                                             search_area_size=searchsize)
        x, y = process.get_coordinates(image_size=I0.shape, window_size=winsize, overlap=overlap) 
        u1 = medfilt2d(u0, kernel_size=3)
        v1 = medfilt2d(v0, kernel_size=3)
        u1[np.isnan(u1)]=0
        v1[np.isnan(v1)]=0
        u2 = medfilt2d(u1, kernel_size=3)
        v2 = medfilt2d(v1, kernel_size=3)
        frame_data = pd.DataFrame(data=np.array([x.flatten(), y.flatten(), u2.flatten(), v2.flatten()]).T,
                       columns=['x', 'y', 'u', 'v']).assign(frame=num)
        if num < 2:
            data = frame_data
        else:
            data = data.append(frame_data)
    return data
            
def imseqPIV(folder, winsize, searchsize, overlap, dt):  
    fileList = readseq(folder)
    frame = 0
    data = pd.DataFrame()
    for num, i in fileList.iterrows():    
        img = io.imread(i.Dir)
        if frame % 2 == 0:
            I0 = img
            frame += 1
            continue
        I1 = img    
        u0, v0 = process.extended_search_area_piv(I0.astype(np.int32), I1.astype(np.int32), 
                                                             window_size=winsize, overlap=overlap, dt=dt, 
                                                             search_area_size=searchsize)
        x, y = process.get_coordinates(image_size=I0.shape, window_size=winsize, overlap=overlap)    
        
        u1 = medfilt2d(u0, kernel_size=3)
        v1 = medfilt2d(v0, kernel_size=3)
        u1[np.isnan(u1)]=0
        v1[np.isnan(v1)]=0
        u2 = medfilt2d(u1, kernel_size=3)
        v2 = medfilt2d(v1, kernel_size=3)
        frame_data = pd.DataFrame(data=np.array([x.flatten(), y.flatten(), u2.flatten(), v2.flatten()]).T,
                       columns=['x', 'y', 'u', 'v']).assign(frame=frame)
        if num < 2:
            data = frame_data
        else:
            data = data.append(frame_data)
        frame += 1
    return data
    
if __name__ == '__main__':
    # set PIV parameters
    winsize = 50 # pixels
    searchsize = 100  # pixels, search in image B
    overlap = 25 # pixels
    dt = 0.033 # frame interval (sec)
    # imgDir = r'R:\Dip\DF\PIV_analysis\1.tif'
    # data = tiffstackPIV(imgDir, winsize, searchsize, overlap, dt)
    folder = r'I:\Data\Wei\transient\03'
    data = imseqPIV(folder, winsize, searchsize, overlap, dt)
    data.to_csv(os.path.join(folder, 'pivData.csv'), index=False)
    # plt.quiver(data.x, data.y, data.u, data.v)