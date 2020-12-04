# Define some utility functions that are used in 'Correlation' notebooks
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from myImageLib import dirrec, bestcolor, wowcolor
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import savgol_filter, medfilt
from scipy.optimize import curve_fit
import corrLib
import os
from skimage import io
from matplotlib.patches import Rectangle

# general
def data_log_mapping(kw='aug'):
    """
    Returns the data log mapping. 
    My experiments are recorded in date/number fashion, without detailed parameters.
    All the parameters are logged in separated log files. 
    This function maps the parameters to date/number. 
    
    Args:
    kw -- keyword of data. Since I have done a new set of experiment in August, I have set 'aug' as one valid value. The old data may still be useful in the future. When needed, I will implement the mappings for the old data.
    
    Returns:
    dirs -- the data-log mapping.
    
    IMPORTANT: Whenever new experiments are added, this function needs to be updated.
    """
    if kw == 'aug':
        dirs = {}
        dirs['120'] = ['08062020-3', '08062020-4', '08062020-5']
        dirs['100'] = ['08062020-0', '08062020-1', '08062020-2']
        dirs['85'] = ['08052020-3', '08052020-4', '08052020-5']
        dirs['80'] = ['08032020-0', '08032020-1', '08032020-2']
        dirs['70'] = ['08042020-0', '08042020-1', '08042020-2']
        dirs['60'] = ['08032020-3', '08032020-4', '08032020-5']
        dirs['50'] = ['08042020-3', '08042020-4', '08042020-5']
        dirs['40'] = ['08032020-6', '08032020-7', '08032020-8']
        dirs['30'] = ['08042020-6', '08042020-7', '08042020-8']
        dirs['20'] = ['08032020-9', '08032020-10', '08032020-11']
        dirs['10'] = ['08042020-9', '08042020-10', '08042020-11']
        dirs['00'] = ['08032020-12', '08032020-13', '08032020-14']
    
    return dirs

def tentative_log():
    """
    Another log function of density fluctuations data.
    """
    conc = [120, 100, 85, 80, 70, 60, 50, 40, 30, 20, 10]
    folders = ['08062020', '08062020', '08052020', '08032020', '08042020', '08032020', '08042020', '08032020', '08042020', '08032020', '08042020']
    sample_num = [range(3, 6), range(0, 3), range(3, 6), range(0, 3), range(0, 3), range(3, 6), range(3, 6), range(6, 9), range(6, 9), range(9, 12), range(9, 12)]
    return conc, folders, sample_num


def illumination_correction(img, avg):
    """
    Correct the illumination inhomogeneity in microscope images.
    
    Args:
    img -- input image with illumination inhomogeneity
    avg -- average of (a large number of) raw images
    
    Returns:
    corrected -- corrected image
    """
    corrected = (img / avg * img.mean() / (img / avg).mean()).astype('uint8')
    return corrected

def data_log():
    """
    Return the data log: log[date][num, fps]    
    """
    log = {}
    log['08032020'] = {}
    log['08032020']['num'] = list(range(0, 15))
    log['08032020']['fps'] = [30, 30, 30, 30, 30, 30, 30, 10, 10, 10, 10, 10, 10, 10, 10]
    log['08042020'] = {}
    log['08042020']['num'] = list(range(0, 12))
    log['08042020']['fps'] = [30, 30, 30, 30, 30, 30, 30, 30, 30, 10, 10, 10]
    log['08052020'] = {}
    log['08052020']['num'] = list(range(0, 12))
    log['08052020']['fps'] = [30, 30, 30, 30, 30, 30, 30, 30, 30, 10, 10, 10]
    log['08062020'] = {}
    log['08062020']['num'] = list(range(0, 13))
    log['08062020']['fps'] = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 10]
    return log

def unified_symbols():    
    color_dict = {
        '0': 12,
        '10': 9,
        '20': 8,
        '30': 7,
        '40': 6,
        '50': 5,
        '60': 4,
        '70': 3,
        '80': 2,
        '85': 10,
        '100': 1,
        '120': 0
    }
    marker_list = ['o', 'p', 'P', '*', 'h', '+', 'x', 'D', 'd', 'v', '^', '<', '>', '1', '2', '3', '4']
    return color_dict, marker_list



# fig-1_experiment
def scalebar_shape_position(img_shape):
    """
    Args:
    img_shape -- tuple of 2 integers, (h_i, w_i)
    
    Returns:
    scalebar_shape -- tuple of 2 intergers, (h_s, w_s)
    position -- tuple of 2 integers (top left of scalebar), (x, y)
    
    Test:
    img_shape = (800, 1000)
    shape, xy = scalebar_shape_position(img_shape)
    print("position: " + str(xy) + '\nshape: ' + str(shape))
    """
    h, w = img_shape
    shape = (int(w/50), int(w/5))
    margin = shape[0]
    xy = (w - shape[1] - margin, h - shape[0] - margin)
    return shape, xy

def draw_scalebar(ax, shape, xy):
    """
    Args:
    ax -- the axis on which image is shown
    shape -- shape of scalebar
    xy -- position of scalebar
    
    Returns:
    None
    """
    
    h, w = shape
    rect = Rectangle(xy, w, h, color='white')
    ax.add_patch(rect)
    
    return None

def sparcify_piv(pivData, sparcity=2):
    """
    Args:
    pivData -- DataFrame (x, y, u, v)
    sparcity -- int, degree to which pivData is sparcified, higher is sparcer
    
    Returns:
    sparcified_pivData -- DataFrame (x, y, u, v)
    """
    temp = {}
    col = len(pivData.x.drop_duplicates())
    row = len(pivData.y.drop_duplicates())
    for c in pivData:
        temp[c] = np.array(pivData[c]).reshape(row, col)[0:row:sparcity, 0:col:sparcity].flatten()
    sparcified_pivData = pd.DataFrame(temp)
    
    return sparcified_pivData




# fig-2_GNF

def postprocess_gnf(gnf_data, lb, xlim=None, sparse=3, normalize='1', volume_fraction=None ,mpp=0.33):
    """
    Postprocess raw GNF data for plotting.
    
    Since we change the way of preparing GNF data, the corresponding function which is responsible for preparing ready-to-plot data needs to be modified. As far as I am concerned, the only function that needs to be changed is the `postprocess_gnf()`. To avoid issues, I want to keep the default behavior of the function, which rescale the starting point of all curves to 1. An additional keyword argument `normalize` will be added, and default to `'1'`, which standards for rescaling by the first point. Optionally, `normalize` can be set to `small-scale`, which applies the normalization described in Section 3.2. If `small-scale` is chosen, an additional keyword argument, `volume_fraction` will be required in order to calculate the rescaling factor. (implement after dinner)
    
    Args:
    gnf_data -- DataFrame containing columns ('n', 'd'), generated by df2_nobp.py or df2_kinetics.py
    lb -- size of bacteria (pixel, normalizing factor of x axis)
    xlim -- box size beyond which the data get cut off (pixel), can be either integer or a list of 2 integers
            if xlim is int, it is the upper limit, data above xlim will be cut off,
            if xlim is a list, data outside [xlim[0], xlim[1]] will be cut off
    sparse -- the degree to sparsify the data, 1 is doing nothing, 3 means only keep 1/3 of the orginal data
    normalize -- the method to normalize the data. Choose from '1', None or 'small-scale'.
                 '1': rescale y with y[0]
                 'small-scale': rescale y with y[0] / \sqrt{1 - volume_fraction}. Additional volume_fraction arg is required.
                 None: no normalization will be applied.
    
    Returns:
    x, y -- a tuple that can be plotted directly using plt.plot(x, y)
    
    Edit:
    12022020 -- Initial commit.
    
    Test:
    # test new postprocess_gnf(gnf_data, lb, xlim=None, sparse=3, normalize='1', volume_fraction=None ,mpp=0.33)
    data = pd.read_csv(r'E:\moreData\08032020\df2_kinetics\01\kinetics_data.csv')
    gnf_data = data.loc[data.segment==50]
    lb = 10
    # test normalize = '1'
    x, y = postprocess_gnf(gnf_data, lb, xlim=[10, 10000], sparse=3, normalize='1')
    plt.plot(x, y, label='1')
    # test normalize = 'small-scale'
    x, y = postprocess_gnf(gnf_data, lb, xlim=[1, 10000], sparse=3, normalize='small-scale', volume_fraction= 0.064)
    plt.plot(x, y, label='small-scale')
    # test normalize = '1'
    x, y = postprocess_gnf(gnf_data, lb, xlim=[1, 10000], sparse=3, normalize=None)
    plt.plot(x, y, label='None')
    plt.loglog()
    plt.legend(fontsize=5)
    plt.xlabel('$l^2/l_b^2$')
    plt.ylabel('$\Delta N/\sqrt N$')
    """    
    
    if xlim == None:
        data = gnf_data
    elif isinstance(xlim, int):
        data = gnf_data.loc[gnf_data.n < xlim*lb**2]
    elif isinstance(xlim, list) and len(xlim) == 2:
        data = gnf_data.loc[(gnf_data.n>=xlim[0]*lb**2)&(gnf_data.n < xlim[1]*lb**2)]  
    
    if normalize == '1':
        xx = data.n / lb**2
        yy = data.d / data.n**0.5
        yy = yy / yy.iat[0]
    elif normalize == None:
        xx = data.n / lb**2
        yy = data.d / data.n**0.5
    elif normalize == 'small-scale':
        assert(volume_fraction is not None)
        assert(volume_fraction < 1)
        assert(xlim[0] <= 1) # make sure the first data point is at a smaller scale than lb
        xx = data.n / lb**2
        yy = data.d / data.n**0.5
        yy = yy / yy.iat[0] * (1 - volume_fraction) ** 0.5        
    else:
        raise ValueError('Invalid normalize argument')
    
    # sparcify
    x = xx[0:len(xx):sparse]
    y = yy[0:len(xx):sparse]
    
    return x, y

def collapse_data(gnf_data_tuple, lb, xlim=None, sparse=3):
    """
    Args:
    gnf_data_tuple -- a tuple of gnf_data (dataframe) generated by df2_nobp.py, it has to be a tuple
    lb -- size of bacteria (pixel, normalizing factor of x axis)
    xlim -- box size beyond which the data get cut off (pixel), can be either integer or a list of 2 integers
            if xlim is int, it is the upper limit, data above xlim will be cut off,
            if xlim is a list, data outside [xlim[0], xlim[1]] will be cut off
    sparse -- the degree to sparsify the data, 1 is doing nothing, 3 means only keep 1/3 of the orginal data
    
    Returns:
    collapsed -- DataFrame containing ('x', 'avg', 'std')    
        'x' -- l**2/lb**2 used for plotting GNF, index
        'avg' -- average values of given dataset (gnf_data_tuple)
        'err' -- standard deviation of given dataset
    """
    
    L = len(gnf_data_tuple)
    for i in range(0, L):
        x, y = postprocess_gnf(gnf_data_tuple[i], lb, xlim=xlim, sparse=sparse)
        data = pd.DataFrame(data={'x': x, 'y': y}).set_index('x')
        if i == 0:
            data_merge = data
        else:
            data_merge = data_merge.join(data, rsuffix=str(i))
            
    x = data_merge.index                
    avg = data_merge.mean(axis=1)
    std = data_merge.std(axis=1)
    
    collapsed = pd.DataFrame(data={'x': x, 'avg': avg, 'std': std}).set_index('x')
    
    return collapsed

def prepare_multiple_data(dirs):
    """
    Args:
    dirs -- a list of directories of GNF data
    
    Returns:
    gnf_data_tuple -- a tuple of GNF DataFrame ('n', 'd')
    """
    
    data_list = []
    
    for d in dirs:
        data_list.append(pd.read_csv(d))
        
    gnf_data_tuple = tuple(data_list)
    
    return gnf_data_tuple

def plot_predictions(ax, key='M19'):
    """
    Plot predictions from theory and simulations along with my data in ax. 
    2D predictions will be '--', 3D predictions will be '.-'. 
    
    Args:
    ax -- axis where I plot my data
    key -- the prediction to plot, can be 'TT95', 'R03' or 'M19', default to 'M19'
    Returns:
    None    
    """
    
    pred = {'TT95': (0.3, 0.27),
           'R03': (0.5, 0.33),
           'M19': (0.33, 0.3)}
    
    x = np.array(ax.get_xlim())
    y2d = pred[key][0] * np.ones(2)
    y3d = pred[key][1] * np.ones(2)
    
    ax.plot(x, y2d, color='black', ls='--', lw=0.5)
    ax.plot(x, y3d, color='black', ls='-.', lw=0.5)
    
    return None

def plot_std(k_data, seg_length, tlim=None, xlim=None, lb=10, mpp=0.33, fps=10, num_curves=5):
    """
    Args:
    k_data -- kinetics data computed by df2_kinetics.py, has 3 columns (n, d, segment)
    seg_length -- segment length [frame] used in computing kinetics
    tlim -- bounds of time, only plot the data in the bounds (second)
            tlim can be None, int or list of 2 int
                None - plot all t
                int - plot all below tlim
                list - plot between tlim[0] and tlim[1]
    xlim -- box size beyond which the data get cut off (pixel), can be either integer or a list of 2 integers
    lb -- size of single bacterium [px]
    mpp -- microns per pixel
    fps -- frames per second
    num_curve -- number of curves in the final plot
    
    Returns:
    plot_data -- a dict containing (x, y)'s of all the curved plotted
                example {'l1': (x1, y1), 'l2': (x2, y2)} 
                where x1, y1, x2, y2 are all array-like object
    fig -- the figure handle of the plot, use for saving the figure
    """
    
    symbol_list = [ 'x', 's', 'P', '*', 'd', 'o', '^']
    
    plot_data = {}
    
    # filter out the data we don't need using tlim
    if tlim == None:
        data = k_data
    elif isinstance(tlim, int):
        data = k_data.loc[(k_data.segment-1) * seg_length < tlim * fps]
    elif isinstance(tlim, list) and len(tlim) == 2:
        data = k_data.loc[((k_data.segment-1) * seg_length < tlim[1] * fps) & ((k_data.segment-1) * seg_length >= tlim[0] * fps)]
    else:
        raise ValueError('tlim must be None, int or list of 2 int')
    
    
    # determine the number of curves we want
    num_total = len(data.segment.drop_duplicates())
    if num_total < num_curves:
        seg_list = data.segment.drop_duplicates()
    else:
        seg_list = np.floor(num_total / num_curves * (np.arange(num_curves))) +  data.segment.min()
    
    fig, ax = plt.subplots(dpi=300)
    for num, i in enumerate(seg_list):
        subdata = data.loc[data.segment==i]
        x, y = postprocess_gnf(subdata, lb, xlim=xlim, sparse=3)
        ax.plot(x, y, mec=bestcolor(num), label='{:d} s'.format(int(seg_length*(i-1)/fps)),
               ls='', marker=symbol_list[num], markersize=4, mfc=(0,0,0,0), mew=1)
        plot_data['l'+str(num)] = (x, y)
        
    ax.set_ylim([0.9, 11])
    ax.legend(ncol=2, loc='upper left')
    ax.loglog()
    ax.set_xlabel('$l^2/l_b^2$')
    ax.set_ylabel('$\Delta N/\sqrt{N}$')
    
    return plot_data, fig, ax

def plot_kinetics(k_data, i_data, tlim=None, xlim=None, lb=10, mpp=0.33, seg_length=100, fps=10, plot=True):
    """
    Plot evolution of number fluctuation exponents and light intensity on a same yyplot
    refer to https://matplotlib.org/gallery/api/two_scales.html
    
    Args:
    k_data -- kinetics data computed by df2_kinetics.py
    i_data -- light intensity evolution extracted by overall_intensity.py
    lb -- size of bacteria (pixel, normalizing factor of x axis)
    mpp -- microns per pixel
    seg_length -- segment length when computing kinetics [frame]
    fps -- frames per second
    
    Returns:
    fig -- figure object
    ax1 -- the axis of kinetics
    
    Edit:
    08222020 -- add optional argument plot, to turn on and off the automatic plotting 
                (sometimes only the calculation part is needed, for example batch processing,
                the autoplotting should be turned off)
    """
    
    t = [] 
    power = []
    
    # apply tlim
    if tlim == None:
        pass
    elif isinstance(tlim, int):
        tc = (k_data.segment-1)*seg_length/fps
        k_data = k_data.loc[ tc < tlim]
        i_data = i_data.loc[i_data.t / fps < tlim]
    elif isinstance(tlim, list) and len(tlim) == 2:
        assert(tlim[1]>tlim[0])
        tc = (k_data.segment-1)*seg_length/fps
        k_data = k_data.loc[ (tc < tlim[1]) & (tc >= tlim[0])]
        i_data = i_data.loc[(i_data.t / fps < tlim[1]) & (i_data.t / fps >= tlim[0])]
    else:
        raise ValueError('tlim should be None, int or list of 2 int')     
    
    # compute exponents at different time
    # t, power will be plotted on ax1
    for idx in k_data.segment.drop_duplicates():
        subdata = k_data.loc[k_data.segment==idx]
        xx, yy = postprocess_gnf(subdata, lb, xlim=xlim, sparse=3)
        x = np.log(xx)
        y = np.log(yy)
        p = np.polyfit(x, y, deg=1)
        t.append((idx-1)*seg_length/fps)
        power.append(p[0])

    # rescale light intensity to (0, 1)
    # t1, i will be plotted on ax2
    t1 = i_data.t / fps
    i = i_data.intensity - i_data.intensity.min()
    i = i / i.max()
    
    data = {'t0': t, 'alpha': power, 't1': t1, 'i': i}
    
    if plot == True:
        # set up fig and ax
        fig = plt.figure()
        ax1 = fig.add_axes([0,0,1,1])
        ax2 = ax1.twinx()

        # plot t, power
        color = wowcolor(0)
        ax1.set_xlabel('$t$ [s]')
        ax1.set_ylabel('$\\alpha$', color=color)
        ax1.plot(t, power, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # plot t1, intensity
        color = wowcolor(4)
        ax2.set_ylabel('$I$', color=color)
        ax2.plot(t1, i, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        return data, fig, ax1
    else:
        return data
    
    
    
    

def kinetics_from_light_on(data, plot=True):
    """
    Args:
    data -- dict of ('t0', 'alpha', 't1', 'i'), return value of plot_kinetics()
    
    Returns:
    new_data -- dict of ('t0', 'alpha', 't1', 'i'), where 't0' and 't1' are translated according to the light on time, so that light is on at time 0.
    """
    
    # find light on time
    i = data['i']
    i_thres = (i.max() + i.min()) / 2
    light_on_ind = (i>i_thres).replace(False, np.nan).idxmax()
    light_on_time = data['t1'][light_on_ind]
    
    # construct new_data
    new_data = {}
    for kw in data:
        if kw == 't0':
            new_data[kw] = np.array(data[kw])[data['t0']>=light_on_time] - light_on_time
        elif kw == 'alpha':
            new_data[kw] = np.array(data[kw])[data['t0']>=light_on_time]
        elif kw == 't1':
            new_data[kw] = np.array(data[kw])[data['t1']>=light_on_time] - light_on_time
        else:
            new_data[kw] = np.array(data[kw])[data['t1']>=light_on_time]
    
    if plot == True:
        # plot new_data
        fig = plt.figure()
        ax1 = fig.add_axes([0, 0, 1, 1])

        ax1.set_xlabel('$t$ [s]')
        ax1.set_ylabel('$\\alpha$')
        ax1.plot(new_data['t0'], new_data['alpha'])

        return new_data, fig, ax1
    else:
        return new_data

def plot_kinetics_eo(k_data, i_data, eo_data, tlim=None, xlim=None, lb=10, mpp=0.33, seg_length=100, fps=10, plot=True):
    """
    Plot evolution of number fluctuation exponents and light intensity on a same yyplot
    In addition, plot flow energy and flow order in the same figure as well
    
    Args:
    k_data -- kinetics data computed by df2_kinetics.py
    i_data -- light intensity evolution extracted by overall_intensity.py
    eo_data -- energy and order data (t, E, OP), t has unit second, computed by energy_order.py
    tlim -- time range in which data is plotted
    xlim -- range for fitting the gnf curve
    lb -- size of bacteria (pixel, normalizing factor of x axis)
    mpp -- microns per pixel
    seg_length -- segment length when computing kinetics [frame]
    fps -- frames per second
    plot -- plot the data or not, bool
    
    Returns:
    fig -- figure object
    ax1 -- the axis of kinetics
    
    Edit:
    11122020 -- add * mpp * mpp to E = eo_data.E, to make the unit of energy um^2/s^2
    """
    
    t = [] 
    power = []
    
    # apply tlim
    if tlim == None:
        pass
    elif isinstance(tlim, int):
        tc = (k_data.segment-1)*seg_length/fps
        k_data = k_data.loc[ tc < tlim]
        i_data = i_data.loc[i_data.t / fps < tlim]
        eo_data = eo_data.loc[eo_data.t < tlim]
    elif isinstance(tlim, list) and len(tlim) == 2:
        assert(tlim[1]>tlim[0])
        tc = (k_data.segment-1)*seg_length/fps
        k_data = k_data.loc[ (tc < tlim[1]) & (tc >= tlim[0])]
        i_data = i_data.loc[(i_data.t / fps < tlim[1]) & (i_data.t / fps >= tlim[0])]
        eo_data = eo_data.loc[(eo_data.t < tlim[1]) & (eo_data.t >= tlim[0])]
    else:
        raise ValueError('tlim should be None, int or list of 2 int')   
    
    # compute exponents at different time
    # t, power will be plotted on ax1
    for idx in k_data.segment.drop_duplicates():
        subdata = k_data.loc[k_data.segment==idx]
        xx, yy = postprocess_gnf(subdata, lb, xlim=xlim, sparse=3)
        x = np.log(xx)
        y = np.log(yy)
        p = np.polyfit(x, y, deg=1)
        t.append((idx-1)*seg_length/fps)
        power.append(p[0])

    # rescale light intensity to (0, 1)
    # t1, i will be plotted on ax2
    t1 = i_data.t / fps
    i = i_data.intensity - i_data.intensity.min()
    i = i / i.max()
    # t2, E will be plotted on ax3
    t2 = eo_data.t
    E = eo_data.E * mpp * mpp
    # t2, O will be plotted on ax4
    O = eo_data.OP
    
    data = {'t0': t, 'alpha': power, 't1': t1, 'i': i, 't2': t2, 'E': E, 'OP': O}
    
    if plot == True:
        # set up fig and ax
        fig = plt.figure()
        ax1 = fig.add_axes([0,0,1,1])
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax4 = ax1.twinx()

        # plot t, power
        color = 'black'
        ax1.set_xlabel('$t$ [s]')
        ax1.set_ylabel('$\\alpha$', color=color)
        ax1.plot(t, power, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # plot t1, intensity
        color = wowcolor(0)
        ax2.set_ylabel('$I$', color=color)
        ax2.plot(t1, i, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # plot t2, E
        color = wowcolor(2)
        ax3.set_ylabel('$E$ [$\mu$m$^2$/s$^2$]', color=color)
        ax3.plot(t2, E, color=color)
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.spines["right"].set_position(("axes", 1.1))

        # plot t2, O
        color = wowcolor(8)
        ax4.set_ylabel('$OP$', color=color)
        ax4.plot(t2, O, color=color)
        ax4.tick_params(axis='y', labelcolor=color)
        ax4.spines["right"].set_position(("axes", 1.2))

        ax = [ax1, ax2, ax3, ax4]   
        return data, fig, ax
    else:
        return data

def kinetics_eo_from_light_on(data, plot=True):
    """
    Args:
    data -- dict of (t0, alpha, t1, i, t2, E, OP), return value of plot_kinetics_eo()
    plot -- plot the data or not
    
    Returns:
    new_data -- dict of (t0, alpha, t1, i, t2, E, OP), modified so that light-on time is 0
    """
    
    # find light on time
    i = data['i']
    i_thres = (i.max() + i.min()) / 2
    light_on_ind = (i>i_thres).replace(False, np.nan).idxmax()
    light_on_time = data['t1'][light_on_ind]
    
    # construct new_data
    new_data = {}
    for kw in data:
        if kw == 't0' or kw == 't1' or kw == 't2':
            new_data[kw] = np.array(data[kw])[data[kw]>=light_on_time] - light_on_time
        elif kw == 'alpha':
            new_data[kw] = np.array(data[kw])[data['t0']>=light_on_time]
        elif kw == 'i':
            new_data[kw] = np.array(data[kw])[data['t1']>=light_on_time]
        else:
            new_data[kw] = np.array(data[kw])[data['t2']>=light_on_time]
    
    if plot == True:
        # plot new_data
        fig = plt.figure()
        ax1 = fig.add_axes([0, 0, 1, 1])

        color = 'black'
        ax1.set_xlabel('$t$ [s]')
        ax1.set_ylabel('$\\alpha$', color=color)
        ax1.plot(new_data['t0'], new_data['alpha'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        color = wowcolor(2)
        ax2 = ax1.twinx()
        ax2.set_ylabel('$E$ [$\mu$m$^2$/s$^2$]', color=color)
        ax2.plot(new_data['t2'], new_data['E'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)


        color = wowcolor(8)
        ax3 = ax1.twinx()
        ax3.set_ylabel('$OP$', color=color)
        ax3.plot(new_data['t2'], new_data['OP'], color=color)
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.spines["right"].set_position(("axes", 1.1))

        ax = [ax1, ax2, ax3]
    
        return new_data, fig, ax    
    else:
        return new_data

def kinetics_eo_smooth(data):
    """
    Generate smoothed data and plot them.
    
    Args:
    data -- dict of (t0, alpha, t1, i, t2, E, OP), return value of kinetics_eo_from_light_on(data) or plot_kinetics()
    
    Returns:
    new_data -- smoothed data, dict of (t0, alpha, t1, i, t2, E, OP)
    
    Note:
    Although there are many ways to smooth the curve, I apply here a gaussian filter with sigma=1/15*total_data_length to do the work.
    Also try uniform filter with same 'size'   
    """
    new_data = {}
    # Generate new_data
    for kw in data:
        if kw.startswith('t') == False:
            sigma = int(len(data[kw]) / 15) + 1
            new_data[kw] = gaussian_filter1d(data[kw], sigma)
#             new_data[kw] = uniform_filter1d(data[kw], sigma) 
        else:
            new_data[kw] = data[kw]
            
    # plot new_data
    fig, ax1 = plt.subplots(dpi=300)
    
    color = 'black'
    ax1.set_xlabel('$t$ [s]')
    ax1.set_ylabel('$\\alpha$', color=color)
    ax1.plot(new_data['t0'], new_data['alpha'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    color = wowcolor(2)
    ax2 = ax1.twinx()
    ax2.set_ylabel('$E$ [$\mu$m$^2$/s$^2$]', color=color)
    ax2.plot(new_data['t2'], new_data['E'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    
    color = wowcolor(8)
    ax3 = ax1.twinx()
    ax3.set_ylabel('$OP$', color=color)
    ax3.plot(new_data['t2'], new_data['OP'], color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.spines["right"].set_position(("axes", 1.2))
    
    ax = [ax1, ax2, ax3]
    
    return new_data, fig, ax

def df2(folder):
    l = readseq(folder)
    img = io.imread(l.Dir.loc[0])
    size_min = 5
    step = 50*size_min
    L = min(img.shape)
    boxsize = np.unique(np.floor(np.logspace(np.log10(size_min),
                        np.log10((L-size_min)/2),100)))
    
    df = pd.DataFrame()
    for num, i in l.iterrows():
        img = io.imread(i.Dir)
        framedf = pd.DataFrame()
        for bs in boxsize: 
            X, Y, I = divide_windows(img, windowsize=[bs, bs], step=step)
            tempdf = pd.DataFrame().assign(I=I.flatten(), t=int(i.Name), size=bs, 
                           number=range(0, len(I.flatten())))
            framedf = framedf.append(tempdf)
        df = df.append(framedf)

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


# fig3_spatial-correlations
def exp(x, a):
    return np.exp(-a*x)

def corr_length(data, fitting_range=None):
    """
    Args:
    data -- dataframe with columns (R, C), where R has pixel as unit
    fitting_range -- (optional) can be None, int or list of two int
    
    Returns:
    cl -- correlation length of given data (pixel)
    """
    if fitting_range == None:
        pass
    elif isinstance(fitting_range, int):
        data = data.loc[data['R'] < fitting_range]
    elif isinstance(fitting_range, list) and len(fitting_range) == 2:
        data = data.loc[(data['R'] < fitting_range[1])&(data['R'] >= fitting_range[0])]
    else:
        raise ValueError('fitting_range should be None, int or list of 2 int')
        
    fit = curve_fit(exp, data['R'], data['C'], p0=[0.01])
    cl = 1 / fit[0][0]
    return cl, fit

def xy_to_r(corr_xy):
    """
    Note, this version of function converts the xy data where x, y start from (step, step) instead of (0, 0).
    When the corr functions are changed, this function should not be used anymore. 
    Check carefully before using.
    
    Args:
    corr_xy -- DataFrame of (X, Y, ...)
    
    Returns:
    corr_r -- DataFrame (R, ...)
    """
    step_x = corr_xy.X.iloc[0]
    step_y = corr_xy.Y.iloc[0]
    corr_r = corr_xy.assign(R = ((corr_xy.X-step_x)**2 + (corr_xy.Y-step_y)**2)**0.5)    
    return corr_r

def average_data(directory, columns=['CA', 'CV']):
    """
    Take the average of all data in given directory
    
    Args:
    directory -- folder which contains *.csv data, with columns
    columns -- (optional) list of column labels of columns to be averaged
    
    Returns:
    averaged -- DataFrame with averaged data
    """
    k = 0
    
    l = corrLib.readdata(directory)
    for num, i in l.iterrows():
        data = pd.read_csv(i.Dir)
        # check if given label exists in data
        for label in columns:
            if label not in data:
                raise IndexError('Column \'{0}\' does not exist in given data'.format(label))
        if k == 0:
            temp = data[columns]
        else:
            temp += data[columns]
        k += 1                   
       
    # finally, append all other columns (in data but not columns) to averaged
    other_cols = []
    for label in data.columns:
        if label not in columns:
            other_cols.append(label) 
    
    averaged = pd.concat([temp / k, data[other_cols]], axis=1)       
    
    return averaged

def plot_correlation(data, plot_cols=['R', 'C'], xlim=None, mpp=0.33, lb=3, plot_raw=False):
    """
    Plot correlation data. Here we plot the exponential function fitting instead of raw data so that the curve look better.
    
    Args:
    data -- DataFrame (R, C, conc)
    plot_cols -- specify columns to plot. The first column should be distance and the second is correlation
    xlim -- trim the xdata, only use those in the range of xlim
    mpp -- microns per pixel 
    lb -- bacteria size in um
    
    Returns:
    ax -- the axis of plot, one can use this handle to add labels, title and other stuff   
    """
    
    # Initialization
    fig, ax = plt.subplots(dpi=300)
    cl_data = {'conc': [], 'cl': []}
    symbol_list = ['o', '^', 'x', 's', '+', 'p']
    data = data.sort_values(by=[plot_cols[0], 'conc'])
    
    # process data, apply xlim
    if xlim == None:
        pass
    elif isinstance(xlim, int):
        data = data.loc[data[plot_cols[0]] < xlim]
    elif isinstance(xlim, list) and len(xlim) == 2:
        data = data.loc[(data[plot_cols[0]] < xlim[1])&(data[plot_cols[0]] >= xlim[0])]
    else:
        raise ValueError('xlim must be None, int or list of 2 ints')
    
    for num, nt in enumerate(data.conc.drop_duplicates()):
        subdata = data.loc[data.conc==nt]
        x = subdata[plot_cols[0]]
        y = subdata[plot_cols[1]]
        p, po = curve_fit(exp, x, y, p0=[0.01])
        xfit = np.linspace(0, x.max(), num=50)
        yfit = exp(xfit, *p)
        if plot_raw:
            ax.plot(x*mpp/lb, y, color=wowcolor(num), lw=1, ls='--')
        ax.plot(xfit*mpp/lb, yfit, mec=wowcolor(num), label=str(nt), ls='',
                marker=symbol_list[num], mfc=(0,0,0,0), markersize=4, markeredgewidth=0.5)
        cl_data['conc'].append(int(nt))
        cl_data['cl'].append(1/p[0])   
        
    return fig, ax, pd.DataFrame(cl_data).sort_values(by='conc')


# fig-5 velocity and concentration
def retrieve_dxd_data(folder, log_list):
    """
    Args:
    folder -- folder containing dxd data
    log_list -- experiment log as a list object, format is ['date-num', ...]
    
    Returns:
    avg -- DataFrame with columns avg of given entry, adv_divv ... will be indices instead
    std -- DataFrame with columns std of given entry, adv_divv ... will be indices instead
    """
    for n, entry in enumerate(log_list):
        date, num = entry.split('-')
        temp = pd.read_csv(os.path.join(folder, date, 'div_x_dcadv', 'summary.csv'), index_col='sample').loc[[int(num)]]
        if n == 0:
            data = temp
        else:
            data = data.append(temp)
    data = data.transpose()
    avg = pd.DataFrame({'avg': data.mean(axis=1)})
    std = pd.DataFrame({'std': data.std(axis=1)})
    return avg, std

def corr2d(A, B):
    """
    Calculate the correlation between two matrices. 1 for perfect correlation and -1 for perfect anti-correlation. 
    
    Args:
    A, B -- Matrices of same shape
    
    Returns:
    correlation -- real number in [-1, 1]
    """
    assert(A.shape==B.shape)
    return ((A - A.mean())/A.std() * (B - B.mean())/B.std()).mean()



def local_df(img_folder, seg_length=50, winsize=50, step=25):
    """
    Compute local density fluctuations of given image sequence in img_folder
    
    Args:
    img_folder -- folder containing .tif image sequence
    seg_length -- number of frames of each segment of video, for evaluating standard deviations
    winsize --
    step --
    
    Returns:
    df -- dict containing 't' and 'local_df', 't' is a list of time (frame), 'std' is a list of 2d array 
          with local standard deviations corresponding to 't'
    """
    
    l = corrLib.readseq(img_folder)
    num_frames = len(l)
    assert(num_frames>seg_length)
    
    stdL = []
    tL = range(0, num_frames, seg_length)
    for n in tL:
        img_nums = range(n, min(n+seg_length, num_frames))
        l_sub = l.loc[img_nums]
        img_seq = []
        for num, i in l_sub.iterrows():
            img = io.imread(i.Dir)
            X, Y, I = corrLib.divide_windows(img, windowsize=[50, 50], step=25)
            img_seq.append(I)
        img_stack = np.stack([img_seq], axis=0)
        img_stack = np.squeeze(img_stack)
        std = np.std(img_stack, axis=0)
        stdL.append(std)
        
    return {'t': tL, 'std': stdL}

def vorticity(pivData, step=None, shape=None):
    """
    Compute vorticity field based on piv data (x, y, u, v)
    
    Args:
    pivData -- DataFrame of (x, y, u, v)
    step -- distance (pixel) between adjacent PIV vectors
    
    Returns:
    vort -- vorticity field of the velocity field. unit: [u]/pixel, [u] is the unit of u, usually px/s
    """
    x = pivData.sort_values(by=['x']).x.drop_duplicates()
    if step == None:
        # Need to infer the step size from pivData
        step = x.iat[1] - x.iat[0]
    
    if shape == None:
        # Need to infer shape from pivData
        y = pivData.y.drop_duplicates()
        shape = (len(y), len(x))
        
    X = np.array(pivData.x).reshape(shape)
    Y = np.array(pivData.y).reshape(shape)
    U = np.array(pivData.u).reshape(shape)
    V = np.array(pivData.v).reshape(shape)
    
    dudy = np.gradient(U, step, axis=0)
    dvdx = np.gradient(V, step, axis=1)
    vort = dvdx - dudy
    
    return vort

def convection(pivData, image, winsize, step=None, shape=None):
    """
    Compute convection term u.grad(c) based on piv data (x, y, u, v) and image.
    
    Args:
    pivData -- DataFrame of (x, y, u, v)
    image -- the image corresponding to pivData
    winsize -- coarse-graining scheme of image
    step -- (optional) distance (pixel) between adjacent PIV vectors
    shape -- (optional) shape of piv matrices
    
    Returns:
    udc -- convection term u.grad(c). unit: [u][c]/pixel, [u] is the unit of u, usually px/s, [c] is the unit of concentration 
           measured from image intensity, arbitrary.
    """
    x = pivData.sort_values(by=['x']).x.drop_duplicates()
    if step == None:
        # Need to infer the step size from pivData
        step = x.iat[1] - x.iat[0]
    
    if shape == None:
        # Need to infer shape from pivData
        y = pivData.y.drop_duplicates()
        shape = (len(y), len(x))
    
    # check coarse-grained image shape
    X, Y, I = corrLib.divide_windows(image, windowsize=[winsize, winsize], step=step)
    assert(I.shape==shape)
    
    X = np.array(pivData.x).reshape(shape)
    Y = np.array(pivData.y).reshape(shape)
    U = np.array(pivData.u).reshape(shape)
    V = np.array(pivData.v).reshape(shape)
    
    # compute gradient of concentration
    # NOTE: concentration is negatively correlated with intensity. 
    # When computing gradient of concentration, the shifting direction should reverse.
    
    dcx = np.gradient(I, -step, axis=1)
    dcy = np.gradient(I, -step, axis=0)
    
    udc = U * dcx + V * dcy
    
    return udc

def divergence(pivData, step=None, shape=None):
    """
    Compute divergence field based on piv data (x, y, u, v)
    
    Args:
    pivData -- DataFrame of (x, y, u, v)
    step -- distance (pixel) between adjacent PIV vectors
    
    Returns:
    vort -- vorticity field of the velocity field. unit: [u]/pixel, [u] is the unit of u, usually px/s
    """
    x = pivData.sort_values(by=['x']).x.drop_duplicates()
    if step == None:
        # Need to infer the step size from pivData
        step = x.iat[1] - x.iat[0]
    
    if shape == None:
        # Need to infer shape from pivData
        y = pivData.y.drop_duplicates()
        shape = (len(y), len(x))
        
    X = np.array(pivData.x).reshape(shape)
    Y = np.array(pivData.y).reshape(shape)
    U = np.array(pivData.u).reshape(shape)
    V = np.array(pivData.v).reshape(shape)
    
    dudx = np.gradient(U, step, axis=1)
    dvdy = np.gradient(V, step, axis=0)
    div = dudx + dvdy
    
    return div

def order_field(pivData):
    """
    Compute local order from pivData.
    
    Args:
    pivData -- DataFrame containing (x, y, u, v)
    
    Returns:
    order -- order field, an array of the shape of given velocity field
    """
    def inner(Ax, Ay, Bx, By):
        """
        define inner product between two matrices
        """
        return (Ax*Bx + Ay*By) / (Ax**2+Ay**2)**0.5 / (Bx**2+By**2)**0.5
    col = len(pivData.x.drop_duplicates())
    row = len(pivData.y.drop_duplicates())
    u = np.array(pivData.u).reshape((row, col))
    v = np.array(pivData.v).reshape((row, col))
    u1 = np.roll(u, -1, axis=0) # up
    v1 = np.roll(v, -1, axis=0) # up
    u2 = np.roll(u, 1, axis=0) # down
    v2 = np.roll(v, 1, axis=0) # down
    u3 = np.roll(u, 1, axis=1) # right
    v3 = np.roll(v, 1, axis=1) # right
    u4 = np.roll(u, -1, axis=1) # left
    v4 = np.roll(v, -1, axis=1) # left
    
    # do inner products with original matrix
    I1 = inner(u, v, u1, v1)
    I2 = inner(u, v, u2, v2)
    I3 = inner(u, v, u3, v3)
    I4 = inner(u, v, u4, v4)
    
    # average the products
    order = (I1 + I2 + I3 + I4) / 4
    
    return order

def order_df_correlation(df_folder, piv_folder, after=0.9):
    """
    Compute the local correlation between local density fluctuations and local flow order.
    
    Args:
    df_folder -- local density fluctuations data folder
    piv_folder -- piv data folder
    after -- only process data after certain percentage, when 0, process all data and when 1, process only the last data.
                Default is 0.9.
    
    Returns:
    corr_list -- an array of correlations at different frames
    """
    
    l = readdata(df_folder, 'npy')
    l_crop = l.loc[l.Name.astype('int')>l.Name.astype('int')*0.9]
    corr_list = []
    for num, i in l_crop.iterrows():
        f = int(i.Name)
        df = np.load(i.Dir)
        pivData = pd.read_csv(os.path.join(piv_folder, '{0:04d}-{1:04d}.csv'.format(f, f+1)))
        order = order_field(pivData)
        corr = corr2d(order, df)
        corr_list.append(corr)
    
    return np.array(corr_list)

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

def vspatial(X, Y, U, V):
    """
    Direct spatial velocity correlation (2D, not normalized)
    The reason I write this on top of corrS is because for energy spectrum calculation, normalization to (-1, 1) is not needed.
    
    X, Y, CA, CV = vspatial(X, Y, U, V)
    """
    row, col = X.shape
    r = row
    c = col
    vsqrt = (U ** 2 + V ** 2) ** 0.5
    Ax = U / vsqrt
    Ay = V / vsqrt
    CA = np.ones((r, c))
    CV = np.ones((r, c))
    for xin in range(0, c):
        for yin in range(0, r):
            CA[yin, xin] = (Ax[0:row-yin, 0:col-xin] * Ax[yin:row, xin:col] + Ay[0:row-yin, 0:col-xin] * Ay[yin:row, xin:col]).mean()
            CV[yin, xin] = (U[0:row-yin, 0:col-xin] * U[yin:row, xin:col] + V[0:row-yin, 0:col-xin] * V[yin:row, xin:col]).mean()
    return X, Y, CA, CV

def visualize_two_point_correlation(X, Y, U, V, CV):
    """
    Display PIV and velocity correlation field side by side
    
    visualize_two_point_correlation(X, Y, U, V, CV)
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, dpi=300, figsize=(7, 2.7))
    ax[0].quiver(X, Y, U, V, color='black', width=0.001)
    ax[0].axis('equal')
    # ax[0].axis('tight')
    ticks = np.array(range(0, 50, 10))
    ax[0].set_ylim(1080, 0)
    ax[0].set_xlim(0, 1280)
    ax[0].set_xticks(ticks*25)
    ax[0].set_xticklabels(ticks*25)
    ax[0].set_yticks(ticks*25)
    ax[0].set_yticklabels(ticks*25)
    ax[0].set_title('1 px = 0.33 um')
    ax[1].imshow(CV, cmap='seismic')
    
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(ticks*25)
    ax[1].set_yticks(ticks)
    ax[1].set_yticklabels(ticks*25)
    
def calculate_and_visualize_energy_spectrum(CV):
    """
    The goal of this function is to calculate energy spectrum from FFT of velocity two-point correlation function, and visualize the resultant spectrum in terms of real, imaginary and absolute value. The absolute value shows a decay with wavenumber k as k^(-4/3).
    
    calculate_and_visualize_energy_spectrum(CV)
    """
    E = 1 / (2 / np.pi)**2 * np.fft.fft2(CV) * 0.33 * 0.33
    # here the unit of CV is still the same as U and V (typically px/s), thus the unit of the correlation is px2/s2.
    # To convert the unit to um2/s2, multiply the correlation by mpp^2 (0.33^2 for 20x lens)
    k, K = corrLib.compute_wavenumber_field(E.shape, 25*0.33)

    ind = np.argsort(k.flatten())
    k_plot = k.flatten()[ind]
    E_plot = E.flatten()[ind]

    fig, ax = plt.subplots(nrows=1, ncols=2, dpi=300, figsize=(7, 3))
    ax[0].plot(k_plot, E_plot.real, lw=0.5, ls='--', alpha=0.5, label='real')
    ax[0].plot(k_plot, E_plot.imag, lw=0.5, ls='--', alpha=0.5, label='imag')
    ax[0].plot(k_plot, abs(E_plot), lw=0.5, label='abs') 
    ax[0].legend()
    # ax[1].plot(k_plot, E_plot.real, lw=0.5, ls='--', alpha=0.5, label='real')
    # ax[1].plot(k_plot, E_plot.imag, lw=0.5, ls='--', alpha=0.5, label='imag')
    ax[1].plot(k_plot, abs(E_plot), lw=0.5, label='abs', color=bestcolor(2))
    ax[1].loglog()
    ax[1].legend()

    # guide of the eye slope
    x = np.array([0.01,0.03])
    y = x ** -1.3 * 2e1
    ax[1].plot(x, y, lw=0.5, ls='--', color='black')
    ax[1].text(x.mean(), 1.1*y.mean(), '-1.3')
    


def spatial_correlation(A, B):
    """
    Compute the spatial correlation between two 2-D matrices A and B.
    A and B should have the same shape (m, n).
    The output correlation matrix coordinates will denote the displacement between A and B, 
    while the value denoting the correlation.
    """
    
    assert(A.shape==B.shape)
    
    r, c = A.shape
    corr = np.ones((r, c))
    for xin in range(0, c):
        for yin in range(0, r):
            corr[yin, xin] = (A[0:r-yin, 0:c-xin] * B[yin:r, xin:c]).mean()
    
    return corr

def xy_bin(xo, yo, n=100, mode='log', bins=None):
    """
    Bin x, y data on log scale
    
    Args:
    xo -- input x
    yo -- input y
    n -- points after binning
    mode -- 
    
    Returns:
    x -- binned x
    y -- means in bins
    
    Edit:
    11042020 -- Change function name to xy_bin, to incorporate the mode parameter, so that the function can do both log space binning and linear space binning.
    11172020 -- add bins kwarg, allow user to enter custom bins.
    
    Test:
    pivDir = r'D:\density_fluctuations\08032020\piv_imseq\01\3000-3001.csv'
    X, Y, U, V = read_piv(pivDir)
    XS, YS, CA, CV = vspatial(X, Y, U, V)
    k, E = energy_spectrum_2(CV)

    xo = k
    yo = abs(E)
    x, y = log_bin(xo, yo, n=100)
    plt.figure(dpi=300)

    plt.plot(k, abs(E), lw=0.5, color=bestcolor(0), label='corrFT', ls=':')
    plt.plot(x, y, lw=1, color=bestcolor(1), label='corrFT', marker='o', markersize=2, ls='')
    plt.loglog()
    """
    
    assert(len(xo)==len(yo))
    
    if bins is None:
        if mode == 'log':
            x = np.logspace(np.log10(xo[xo>0].min()), np.log10(xo.max()), n+1)
        elif mode == 'lin':
            x = np.linspace(xo.min(), xo.max(), n+1)
    else:
        x = np.sort(bins)
        
    y = (np.histogram(xo, x, weights=yo)[0] /
             np.histogram(xo, x)[0])
    
    return x[:-1], y

def efft(a, n=None, axis=-1, norm=None):
    """
    even function fourier transform
    """
    
    axes = np.arange(0, len(a.shape))
    
    if n == None:
        n = a.shape[axis]        
    
    k = np.arange(0, n)
    
    m = np.arange(0, n) / n
    
    if axis == -1:
        A = np.matmul(a, np.cos(-2 * np.pi * np.outer(m, k)))
    else:
        A = np.matmul(a.transpose(), np.cos(-2 * np.pi * np.outer(m, k))).transpose()
    
    return A

def autocorr_imseq(stack):
    """
    Compute intensity autocorrelation of an image sequence.
    
    Args:
    seq -- image sequence, a DataFrame table containing a set of image names and directories. Return value of corrLib.readseq()
    
    Returns:
    ac_mean -- the autocorrelation
    
    Test:
    stack = np.load(r'E:\moreData\08032020\small_imseq\06\stack.npy')[3000:3600]
    ac = autocorr_imseq(stack)
    plt.plot(np.arange(0, 600)/30, ac)
    """
    def autocorr(x):
        x = (x-x.mean()) / x.std()
        result = np.correlate(x, x, mode='full')/len(x)
        return result[len(result)//2:]
    
#     samples = []
#     for num, i in seq.iterrows():
#             X, Y, I = corrLib.divide_windows(io.imread(i.Dir), windowsize=[50, 50], step=300)
#             samples.append(I)
#     stack = np.stack(samples)
    r = stack.reshape((stack.shape[0], stack.shape[1]*stack.shape[2])).transpose()
    ac_list = []
    for x in r:
        ac = autocorr(x)
        ac_list.append(ac)
    ac_stack = np.stack(ac_list)
    ac_mean = ac_stack.mean(axis=0)
    return ac_mean

def structured_spectra(pivData, **kwargs):
    """
    Generate energy spectra that matches the length scales of GNF data, given by bins.
    
    Args:
    pivData -- PIV data
    bins -- the length scales in GNF data, l^2/l_b^2. 
            The wavenumber in energy spectra data should be convert to length first.
            Should load the GNF data column first, and convert the bins from l^2/l_b^2 to k, typically
            bins = 1 / (np.concatenate(([0.1], (np.array(k_data.index)))) ** 0.5 * 3)
    
    Returns:
    structured_spectra -- structured spectra
    
    Test:
    piv_folder = r'E:\moreData\08032020\piv_imseq\01'
    n = 2000
    pivData = pd.read_csv(os.path.join(piv_folder, '{0:04d}-{1:04d}.csv'.format(n, n+1)))
    k_data = pd.read_csv(os.path.join(data_master_dir, r'Research projects\DF\data\transient-nGNF-energy\08032020\df2_kinetics\00\nGNF_data.csv')).set_index('l_r')
    bins = 1 / (np.concatenate(([0.1], (np.array(k_data.index)))) ** 0.5 * 3)
    structured_spectra = compute_structured_spectra(pivData, bins=np.flip(bins))
    """
    
    es = corrLib.energy_spectrum(pivData)
    es = es.loc[es.k>0] # make sure the 1/es.k step won't encounter error
    
    x, y = xy_bin(es.k, es.E, **kwargs)
    y *= 2 * np.pi * x
    x = (2 * np.pi / x) ** 2 / 9
    spectra = pd.DataFrame({'l_r': x, 'E': y}).set_index('l_r').sort_index()
    
    return spectra

def construct_spectra_series(piv_folder, t_list, bins):
    """
    Construct energy spectra series data, in the same structure as that of rearranged GNF data.
    
    Args:
    piv_folder -- PIV folder
    t_list -- columns of rearranged GNF data
    bins -- indices (plus 1 more) of rearranged GNF data
    
    Returns:
    spectra_series -- energy spectra series data, in the same structure as that of rearranged GNF data
    
    Test:
    piv_folder = r'E:\moreData\08032020\piv_imseq\00'
    k_data = pd.read_csv(r'E:\Google Drive\Research projects\DF\data\transient-GNF-energy\08032020\df2_kinetics\00\kinetics_data.csv').set_index('l_r')
    t_list = k_data.keys().astype('int')
    bins = np.concatenate((np.array(k_data.index), [10000]))
    spectra_series = construct_spectra_series(piv_folder, t_list, bins)
    """
    
    
    for num, t in enumerate(t_list):
        pivData = pd.read_csv(os.path.join(piv_folder, '{0:04d}-{1:04d}.csv'.format(t, t+1)))
        spectra = structured_spectra(pivData, bins=bins).rename(columns={'E': t})
        if num == 0:
            master = spectra
        else:
            master = pd.concat([master, spectra], axis=1)
        
    return master