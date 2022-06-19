# a set of functions for the circle fitting GUI
import numpy as np
import matplotlib.pyplot as plt
from corrLib import readdata
import os
from nd2reader import ND2Reader
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import ctypes

"""
Drying picture analyzing app utility functions
"""


def determine_figsize(img, ratio=0.9, dpi=100):
    """
    Determine the display size of image

    Args:
    img -- the image to display
    ratio -- the maximum fraction the image can take up, beyond which the image is rescaled
    dpi -- dot per inch

    Returns:
    shape -- determined figsize
    compressRatio -- how much image gets compressed

    Test:
    >>> img = io.imread(r'E:\Google Drive\Pictures and videos\research\DD\crack_pattern_of_bacterial_suspension_droplet.png', as_gray=True)
    >>> shape, cr = determine_figsize(img, ratio=0.9, dpi=100)
    """
    h, w = img.shape
    hcanvas, wcanvas = h, w
    compressRatio = 1
    user32 = ctypes.windll.user32
    wmax = np.floor(ratio*user32.GetSystemMetrics(0))
    hmax = np.floor(ratio*user32.GetSystemMetrics(1))
    if wcanvas > wmax:
        wcanvas = wmax
        hcanvas = h/w*wcanvas
        compressRatio = wmax/w
    if hcanvas > hmax:
        hcanvas = hmax
        wcanvas = w/h*hcanvas
        compressRatio = hmax/h

    return (wcanvas, hcanvas), compressRatio

def center_normalize_xy(x, y):
    """
    center and normalize x and y, to make position 0 mean and make length order 1.

    Args:
    x, y -- array-like, same length

    Returns:
    xr, yr -- centered and normalized x and y
    cache -- save the constant that will be used to scale back x and y

    Test:
    >>> xv = np.array([1, 1, -1, -0.8, 0.5])
    >>> yv = np.array([1, -1, -1, 0.8, -0.9])
    >>> center_normalize_xy(xv, yv)
    """

    assert(len(x)==len(y))

    xr = x - x.mean()
    yr = y - y.mean()
    S = ((xr**2).mean() + (yr**2).mean())**0.5
    xr = xr / S
    yr = yr / S
    cache = (x.mean(), y.mean(), S)

    return xr, yr, cache

def compute_gradient(params, x, y):
    """
    Compute the gradient of objective function at given point params

    Args:
    params -- the point where the gradient is evaluated

    Returns:
    grad -- gradients at the given point
    """

    assert(len(x)==len(y))
    a = params['a']
    b = params['b']

    r = ((x - a)**2 + (y-b)**2)**0.5
    u = (x - a) / r
    v = (y - b) / r
    da = 2 * (a + u.mean() * r.mean())
    db = 2 * (b + v.mean() * r.mean())

    grads = {'a': da, 'b': db}

    return grads

def update_params(params, grads, updating_rate):
    """
    Use grads to update params.

    Args:
    params -- fitting parameters
    grads -- gradient evaluated at params, by compute_gradient()
    updating_rate -- the speed of updating the parameters

    Returns:
    updated_params -- updated parameters
    """

    updated_params = {}

    for kw in params:
        updated_params[kw] = params[kw] - updating_rate * grads[kw]

    return updated_params

def scale_back(params, cache):
    """
    Scale back the parameters which were rescaled by center_normalize_xy()

    Args:
    params -- a and b, position of original points
    cache -- cached values of center_normalize_xy()

    Returns:
    rescaled_params -- rescaled parameters

    Test:
    >>> params = {'a': 0, 'b': 0.0}
    >>> cache = (1, 2, 3)
    >>> scale_back(params, cache)
    """

    xmean, ymean, S = cache
    a = params['a'] * S + xmean
    b = params['b'] * S + ymean

    rescaled_params = {'a': a, 'b': b}

    return rescaled_params

def compute_radius(params, x, y):
    """
    Calculate radius when center of circle is known

    Args:
    params -- center of circle
    x, y -- a set of points on edge of circle

    Returns:
    radius -- radius of circle
    """

    radius = (((x - params['a'])**2 + (y - params['b'])**2)**0.5).mean()

    return radius

def fit_circle(x, y, updating_rate=0.1):
    """
    Fit a set of points x and y with a circle (a, b, r), using gradient descent method

    Args:
    x, y -- coordinates of a set of points
    updating_rate -- rate of gradient descent

    Returns:
    output_params -- parameters of fitted circle
    """

    xr, yr, cache = center_normalize_xy(x, y)
    params = {'a': xr.mean(), 'b': yr.mean()}
    grad_norm = 1
    while grad_norm > 1e-12:
        grads = compute_gradient(params, xr, yr)
        params = update_params(params, grads, updating_rate)
        grad_norm = (grads['a']**2 + grads['b']**2)**0.5
    output_params = scale_back(params, cache)
    output_params['r'] = compute_radius(output_params, x, y)

    return output_params

if __name__ == '__main__':
    img = io.imread(r'E:\Google Drive\Pictures and videos\research\DD\crack_pattern_of_bacterial_suspension_droplet.png', as_gray=True)
    shape, cr = determine_figsize(img, ratio=0.9, dpi=100)
    print(shape)
