import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
import os
import corrLib as cl
import matplotlib
import myImageLib as mil
import sys
import time
import pdb
"""
intensity evolution
"""

folder = sys.argv[1]
folder_out = sys.argv[2]

if os.path.exists(folder_out) == False:
    os.makedirs(folder_out)
with open(os.path.join(folder_out, 'log.txt'), 'w') as f:
    pass

l = cl.readseq(folder)
t = []
I = []
for num, i in l.iterrows():
    img = io.imread(i.Dir)
    t.append(num)
    I.append(img.mean())
    with open(os.path.join(folder_out, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // image {} computed\n'.format(i.Name))
data = pd.DataFrame().assign(t=t, intensity=I)
data.to_csv(os.path.join(folder_out, 'intensity.csv'))

""" SYNTAX
python overall_intensity.py folder folder_out
"""

""" TEST PARAMS
folder = E:\Github\Python\Correlation\test_images\cl
folder_out = E:\Github\Python\Correlation\test_images\cl\overall_intensity
"""

""" LOG
Tue Jun 23 20:45:52 2020 // image 100-2 computed
Tue Jun 23 20:45:52 2020 // image 40-2 computed
Tue Jun 23 20:45:52 2020 // image 60-2 computed
Tue Jun 23 20:45:52 2020 // image 80-2 computed
"""