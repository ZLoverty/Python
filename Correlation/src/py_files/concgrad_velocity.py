import numpy as np
import os
from skimage import io
import corrLib
import sys
import time
import pandas as pd

def conc_grad(img):
    """
    Compute concentration gradient based on a given bright field image *img*.
    
    Args:
    img -- bright field image (or coarse-grained one), m*n
    
    Returns:
    grad -- concentration gradient, 2*m*n.
            grad[0] is the gradient in axis 0, i.e. the Y axis.
            grad[1] is the gradient in axis 1, i.e. the X axis.
            This needs to be double checked.
    """
    
    dcx = np.gradient(img, -1, axis=1)
    dcy = np.gradient(img, -1, axis=0)
    grad = np.stack([dcy, dcx], axis=0)
    
    return grad
    
def rearrange_pivdata(pivData, step=None, shape=None):
    """
    Rearrange pivData into np.array of shape (2, m, n).
    
    Args:
    pivData -- DataFrame, (x, y, u, v)
    
    Returns:
    rearranged_pivData -- 2*m*n
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
    
    rearranged_pivData = np.stack([V, U], axis=0)
    
    return rearranged_pivData

small_img_folder = sys.argv[1]
piv_folder = sys.argv[2]
out_folder = sys.argv[3]

if os.path.exists(out_folder) == False:
    os.makedirs(out_folder)
with open(os.path.join(out_folder, 'log.txt'), 'w') as f:
    f.write('small_img_folder ' + small_img_folder + '\n')
    f.write('piv_folder: ' + piv_folder + '\n')
    f.write('out_folder: ' + out_folder + '\n')
    f.write(time.asctime() + ' // Computation starts!\n')

stack = np.load(os.path.join(small_img_folder, 'stack.npy'))
l = corrLib.readdata(piv_folder, 'csv')
corr_list = [] # whole field
corr_sn_list = [] # single number
for num, i in l.iterrows():    
    pivData = pd.read_csv(i.Dir)    
    rearranged_pivData = rearrange_pivdata(pivData)
    n = int(i.Name.split('-')[0])
    I = stack[n]
    grad = conc_grad(I)
    corr = np.sum(grad * rearranged_pivData, axis=0)
    corr_list.append(corr)
    corr_sn_list.append(corr.mean()/corr.std())
    with open(os.path.join(out_folder, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // computing frame {:04d}\n'.format(n))
corr_stack = np.stack(corr_list, axis=0)
np.save(os.path.join(out_folder, 'corr_whole.npy'), corr_stack)
np.save(os.path.join(out_folder, 'corr_evolution.npy'), np.array(corr_sn_list))

with open(os.path.join(out_folder, 'log.txt'), 'a') as f:
    f.write(time.asctime() + ' // computing finishes!')
        
""" EDIT
09202020 -- First edit
"""

""" DESCRIPTION
Here, we compute the concentration gradient from bright field images. Since concentration is negetively proportional to image intensity, we use negative step size in function np.gradient() to make the value comply to convention. Once again, a bias towards positive correlation is desired.

In 2d space, concentration gradient and velocity both have two components (x and y). In our data, the dcx and dcy data will be stored in a 2*m*n array, where the first dimension denotes the component x or y. The PIV data are originally stored in DataFrame's, and I will rearrange the data to conform with the structure of concentration gradients.

In the final data, correlation on each frame will be the inner product between grad and piv, summing x and y direction up. In addition, the mean normalized by the standard deviation of the correlation matrix will be recorded as a single number measure of correlation for that moment. The normalized means will constitute a time series of correlation, which describes the evolution.
"""

""" SYNTAX
python concgrad_velocity.py small_img_folder piv_folder out_folder

small_img_folder -- small_imseq folder, containing coarse-grained image stack as .npy file
piv_folder -- piv folder
out_folder -- output folder, saving correlation data
"""

""" TEST PARAMS
small_img_folder -- E:\Github\Python\Correlation\test_images\test_corr\small_folder
piv_folder -- E:\Github\Python\Correlation\test_images\test_corr\piv_folder
out_folder -- E:\Github\Python\Correlation\test_images\test_corr\out_folder
"""

""" LOG
small_img_folder E:\Github\Python\Correlation\test_images\test_corr\small_folder
piv_folder: E:\Github\Python\Correlation\test_images\test_corr\piv_folder
out_folder: E:\Github\Python\Correlation\test_images\test_corr\out_folder
Sun Sep 20 15:42:52 2020 // Computation starts!
Sun Sep 20 15:42:52 2020 // computing frame 3000
Sun Sep 20 15:42:52 2020 // computing frame 3002
Sun Sep 20 15:42:52 2020 // computing frame 3004
Sun Sep 20 15:42:52 2020 // computing finishes!
"""        