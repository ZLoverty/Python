from openpiv import pyprocess, validation, filters, scaling
from openpiv.smoothn import smoothn
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pandas as pd
from scipy.signal import medfilt2d
from corrLib import divide_windows, readdata, autocorr1d, corrS, distance_corr, xy_bin
import os
import scipy
# %% codecell
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
    Jan 07, 2022 -- Change mask threshold from 1 to 0.5, this will include more velocities.

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
    mask_w = divide_windows(mask, windowsize=[winsize, winsize], step=winsize-overlap)[2] >= 0.5
    assert(mask_w.shape==x.shape)
    u[~mask_w] = np.nan
    v[~mask_w] = np.nan
    return x, y, u, v

def read_piv_stack(folder, cutoff=None):
    """Read PIV data in given folder and stack the velocity data
    Args:
    folder -- PIV data folder
    Returns:
    ustack, vstack -- 3D arrays of (t, x, y)"""
    l = readdata(folder, "csv")
    u_list = []
    v_list = []
    for num, i in l.iterrows():
        x, y, u, v = read_piv(i.Dir)
        u_list.append(u)
        v_list.append(v)
        if cutoff is not None:
            if num > cutoff:
                break
    return np.stack(u_list, axis=0), np.stack(v_list, axis=0)

def tangent_unit(point, center):
    """Compute tangent unit vector based on point coords and center coords.
    Args:
    point -- 2-tuple
    center -- 2-tuple
    Returns:
    tu -- tangent unit vector
    """
    point = np.array(point)
    # center = np.array(center)
    r = np.array((point[0] - center[0], point[1] - center[1]))
    # the following two lines set the initial value for the x of the tangent vector
    ind = np.logical_or(r[1] > 0, np.logical_and(r[1] == 0, r[0] > 0))
    x1 = np.ones(point.shape[1:])
    x1[ind] = -1
    y1 = np.zeros(point.shape[1:])
    x1[(r[1]==0)] = 0
    y1[(r[1]==0)&(r[0]>0)] = -1
    y1[(r[1]==0)&(r[0]<0)] = 1

    y1[r[1]!=0] = np.divide(x1 * r[0], r[1], where=r[1]!=0)[r[1]!=0]
    length = (x1**2 + y1**2) ** 0.5
    return np.divide(np.array([x1, y1]), length, out=np.zeros_like(np.array([x1, y1])), where=length!=0)

# %% codecell
class piv_data:
    """Tools for PIV data downstream analysis, such as correlation, mean velocity,
    derivative fields, energy, enstrophy, energy spectrum, etc."""
    def __init__(self, file_list, fps=50, cutoff=250):
        """file_list: return value of readdata"""
        self.piv_sequence = file_list
        self.dt = 2 / fps # time between two data files
        self.stack = self.load_stack(cutoff=cutoff)
    def load_stack(self, cutoff=None):
        u_list = []
        v_list = []
        for num, i in self.piv_sequence.iterrows():
            x, y, u, v = read_piv(i.Dir)
            if num == 0:
                shape = x.shape
            else:
                if x.shape != shape:
                    break
            u_list.append(u)
            v_list.append(v)
            if cutoff is not None:
                if num > cutoff:
                    break
        return np.stack(u_list, axis=0), np.stack(v_list, axis=0)
    def vacf(self, mode="direct", smooth_method="gaussian", smooth_window=3, xlim=None, plot=False):
        """Compute averaged vacf from PIV data.
        This is a wrapper of function autocorr1d(), adding the averaging over all the velocity spots.
        Args:
        mode -- the averaging method, can be "direct" or "weighted".
                "weighted" will use mean velocity as the averaging weight, whereas "direct" uses 1.
        smooth_window -- window size for gaussian smoothing in time
        xlim -- xlim for plotting the VACF, does not affect the return value
        Returns:
        corrData -- DataFrame of (t, corrx, corry)
        Edit:
        Mar 23, 2022 -- add smoothn smoothing option
        """
        # rearrange vstack from (f, h, w) to (f, h*w), then transpose
        corr_components = []
        for name, stack in zip(["corrx", "corry"], self.stack):
            stack_r = stack.reshape((stack.shape[0], -1)).T
            stack_r = stack_r[~np.isnan(stack_r).any(axis=1)]
            if smooth_method == "gaussian":
                stack_r = scipy.ndimage.gaussian_filter(stack_r, (0, smooth_window/4))
            elif smooth_method == "smoothn":
                stack_r = smoothn(stack_r, axis=1)[0]
            # compute autocorrelation
            corr_list = []
            weight = 1
            normalizer = 0
            for x in stack_r:
                if np.isnan(x[0]) == False: # masked out part has velocity as nan, which cannot be used for correlation computation
                    if mode == "weighted":
                        weight = abs(x).mean()
                    corr = autocorr1d(x) * weight
                    if np.isnan(corr.sum()) == False:
                        normalizer += weight
                        corr_list.append(corr)
            corr_mean = np.nansum(np.stack(corr_list, axis=0), axis=0) / normalizer
            corr_components.append(pd.DataFrame({"c": corr_mean, "t": np.arange(len(corr_mean)) * self.dt}).set_index("t").rename(columns={"c": name}))
        ac = pd.concat(corr_components, axis=1)
        # plot autocorrelation functions
        if plot == True:
            fig, ax = plt.subplots(figsize=(3.5, 3), dpi=100)
            ax.plot(ac.index, ac.corrx, label="$C_x$")
            ax.plot(ac.index, ac.corry, label="$C_y$")
            ax.plot(ac.index, ac.mean(axis=1), label="mean", color="black")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("VACF")
            ax.legend(frameon=False)
            ax.set_xlim([0, ac.index.max()])
            if xlim is not None:
                ax.set_xlim(xlim)
        return ac
    def corrS2d(self, mode="sample", n=10, plot=False):
        """Spatial correlation of velocity field.
        mode -- "sample" or "full"
                "sample" will sample n frames to compute the correlation
                "full" will sample all available frames to compute the correlation (could be computationally expensive)
        n -- number of frames to sample"""
        interval = max(len(self.piv_sequence) // n, 1)
        CV_list = []
        for num, i in self.piv_sequence[::interval].iterrows():
            x, y, u, v = read_piv(i.Dir)
            if num == 0:
                shape = x.shape
            else:
                if x.shape != shape:
                    break
            X, Y, CA, CV = corrS(x, y, u, v)
            CV_list.append(CV)
        CV_mean = np.stack(CV_list, axis=0).mean(axis=0)
        if plot == True:
            plt.imshow(CV_mean)
            plt.colorbar()
        return X, Y, CV_mean
    def corrS1d(self, mode="sample", n=10, xlim=None, plot=False):
        """Compute 2d correlation and convert to 1d. 1d correlation will be represented
        as pd.DataFrame of (R, C)."""
        X, Y, CV = self.corrS2d(mode=mode, n=n)
        dc = distance_corr(X, Y, CV)
        if plot == True:
            fig, ax = plt.subplots(figsize=(3.5, 3), dpi=100)
            ax.plot(dc.R, dc.C)
            ax.set_xlim([0, dc.R.max()])
            ax.set_xlabel("$r$ (pixel)")
            ax.set_ylabel("spatial correlation")
            if xlim is not None:
                ax.set_xlim(xlim)
        return dc
    def mean_velocity(self, mode="abs", plot=False):
        """Mean velocity time series.
        mode -- "abs" or "square"."""
        vm_list = []
        for num, i in self.piv_sequence.iterrows():
            x, y, u, v = read_piv(i.Dir)
            if mode == "abs":
                vm = np.nanmean((u ** 2 + v ** 2) ** 0.5)
            elif mode == "square":
                vm = np.nanmean((u ** 2 + v ** 2))  ** 0.5
            vm_list.append(vm)
        if plot == True:
            fig, ax = plt.subplots(figsize=(3.5, 3), dpi=100)
            ax.plot(np.arange(len(vm_list))*self.dt, vm_list)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("mean velocity (px/s)")
        return pd.DataFrame({"t": np.arange(len(vm_list))*self.dt, "v_mean": vm_list})
    def order_parameter(self, center, mode="wioland"):
        def wioland2013(pivData, center):
            """Compute order parameter with PIV data and droplet center coords using the method from wioland2013.
            Args:
            pivData -- DataFrame of x, y, u, v
            center -- 2-tuple droplet center coords
            Return:
            OP -- float, max to 1
            """
            pivData = pivData.dropna()
            point = (pivData.x, pivData.y)
            tu = tangent_unit(point, center)
            # \Sigma vt
            sum_vt = abs((pivData.u * tu[0] + pivData.v * tu[1])).sum()
            sum_v = ((pivData.u**2 + pivData.v**2) ** 0.5).sum()
            OP = (sum_vt/sum_v - 2/np.pi) / (1 - 2/np.pi)
            return OP
        def hamby2018(pivData, center):
            """Computes order parameter using the definition in Hamby 2018.
            Args:
            pivData - DataFrame of x, y, u, v
            center - 2-tuple center of droplet
            Returns:
            OP - order parameter
            """
            pivData = pivData.dropna()
            tu = tangent_unit((pivData.x, pivData.y), center)
            pivData = pivData.assign(tu=tu[0], tv=tu[1])
            OP = (pivData.u * pivData.tu + pivData.v * pivData.tv).sum() / ((pivData.u ** 2 + pivData.v ** 2) ** 0.5).sum()
            return OP
        OP_list = []
        if mode == "wioland":
            for num, i in self.piv_sequence.iterrows():
                pivData = pd.read_csv(i.Dir)
                OP = wioland2013(pivData, center)
                OP_list.append(OP)
        elif mode == "hamby":
            for num, i in self.piv_sequence.iterrows():
                pivData = pd.read_csv(i.Dir)
                OP = hamby2018(pivData, center)
                OP_list.append(OP)
        return pd.DataFrame({"t": np.arange(len(OP_list)) * self.dt, "OP": OP_list})
# %% codecell
if __name__ == '__main__':
    # %% codecell
    folder = r"test_images\moving_mask_piv\piv_result"
    l = readdata(folder, "csv")
    piv = piv_data(l, fps=50)
    # %% codecell
    vacf = piv.vacf(smooth_window=2, xlim=[0, 0.1])
    # autocorr1d(np.array([1,1,1]))
    # %% codecell
    corr1d = piv.corrS1d(n=600, xlim=[0, 170], plot=True)
    # %% codecell
    piv.mean_velocity(plot=True)
    # %% codecell
    op = piv.order_parameter((87, 87), mode="hamby")
    op
