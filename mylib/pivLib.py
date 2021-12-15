from openpiv import tools, pyprocess, validation, filters, scaling
from openpiv.smoothn import smoothn
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pandas as pd
from scipy.signal import medfilt2d
from corrLib import divide_windows
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

def PIV(I0, I1, winsize, overlap, dt):
    """ Normal PIV """
    u0, v0, sig2noise = pyprocess.extended_search_area_piv(
        I0.astype(np.int32),
        I1.astype(np.int32),
        window_size=winsize,
        overlap=overlap,
        dt=dt,
        search_area_size=winsize,
        sig2noise_method='peak2peak',
    )
    # get x, y
    x, y = pyprocess.get_coordinates(
        image_size=I0.shape,
        search_area_size=winsize,
        overlap=overlap,
        window_size=winsize
    )
    u1, v1, mask_s2n = validation.sig2noise_val(
        u0, v0,
        sig2noise,
        threshold = 1.05,
    )
    # replace_outliers
    u2, v2 = filters.replace_outliers(
        u1, v1,
        method='localmean',
        max_iter=3,
        kernel_size=3,
    )
    # median filter smoothing
    u3 = medfilt2d(u2, 3)
    v3 = medfilt2d(v2, 3)
    return x, y, u3, v3

def PIV_masked(I0, I1, winsize, overlap, dt, mask):
    """Apply PIV analysis on masked images
    Args:
    I0, I1 -- adjacent images in a sequence
    winsize, overlap, dt -- PIV parameters
    mask -- a boolean array, False marks masked region and True marks the region of interest
    mask_procedure -- the option chosen to apply the mask, used for testing, remove in the future.
    Returns:
    frame_data -- x, y, u, v DataFrame, here x, y is wrt original image, (u, v) are in px/s

    This function is rewritten based on the PIV_droplet() function in piv_droplet.py script.
    The intended usage is just to pass one additional `mask` parameter, on top of conventional parameter set.

    EDIT
    ====
    Dec 14, 2021 -- Initial commit.
    Dec 15, 2021 -- After testing 2 masking procedure, option 1 is better.
                    Two procedures produce similar results, but option 1 is faster.
                    So this function temporarily uses option 1, until a better procedure comes.

    MASKING PROCEDURE
    =================
    Option 1:
    i) Mask on raw image: I * mask, perform PIV
    ii) Divide mask into windows: mask_w
    iii) use mask_w to mask resulting velocity field: u[~mask_w] = np.nan
    ---
    Option 2:
    i) Perform PIV on raw images
    ii) Divide mask into windows:mask_w
    iii) use mask_w to mask resulting velocity field: u[~mask_w] = np.nan
    ---
    """
    assert(mask.shape==I0.shape)
    mask = mask >= mask.mean() # convert mask to boolean array
    I0_mask = I0 * mask
    I1_mask = I1 * mask
    x, y, u, v = PIV(I0_mask, I1_mask, winsize, overlap, dt)
    mask_w = divide_windows(mask, windowsize=[winsize, winsize], step=winsize-overlap)[2] >= 1
    assert(mask_w.shape==x.shape)
    u[~mask_w] = np.nan
    v[~mask_w] = np.nan
    return x, y, u, v

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
