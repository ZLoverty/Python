import numpy as np
import pandas as pd
import os
from corrLib import df2
import sys
import time
from miscLib import label_slope


amp_n = int(sys.argv[1])
folder_out = sys.argv[2]

if os.path.exists(folder_out) == False:
    os.makedirs(folder_out)
with open(os.path.join(folder_out, 'log.txt'), 'w') as f:
    f.write('amp_n = {}\n'.format(amp_n))
    f.write('folder_out = {}\n'.format(folder_out))
    
amp_iL = range(1, 3)

num_frames, h, w = 100, 100, 100
alpha = []
for amp_i in amp_iL:
    imgstack = np.random.randint(-amp_i, amp_i, (num_frames, h, w)) # generate new imgstack according to amp_i
    noise = np.random.randint(-amp_n, amp_n, (num_frames, 1, 1)) # generate new noise according to amp_n
    imgstack_add_n = imgstack + noise # compose noised imgstack
    gnf = df2(imgstack_add_n)
    x = gnf.n / 100
    y = gnf.d / x ** 0.5
    y = y / y.iat[0]
    xf, yf, xt, yt, slope = label_slope(x, y)
    alpha.append(slope)
    with open(os.path.join(folder_out, 'log.txt'), 'a') as f:
        f.write(time.asctime() + '\\ amp_i = {0}, amp_n = {1}: alpha = {2:.2f}\n'.format(amp_i, amp_n, slope))

data = pd.DataFrame().assign(amp_i=amp_iL, amp_n=amp_n, alpha=alpha)
data.to_csv(os.path.join(folder_out, 'amp_n={}.csv'.format(amp_n)))

""" SYNTAX
python illumination_noise.py amp_n folder_out

amp_n -- integer, noise amplitude
folder_out -- string folder of output file, the file will be named 'amp_n={}.csv'.format(amp_n)
"""

""" TEST PARAMS
amp_n = 2
folder_out = E:\Github\Python\Correlation\test_images\illumination_noise
"""

""" LOG
amp_n = 2
folder_out = E:\Github\Python\Correlation\test_images\illumination_noise
Fri Jul 31 17:23:50 2020\ amp_i = 1, amp_n = 2: alpha = 0.5
Fri Jul 31 17:24:09 2020\ amp_i = 2, amp_n = 2: alpha = 0.5
Fri Jul 31 17:24:29 2020\ amp_i = 3, amp_n = 2: alpha = 0.5
"""




