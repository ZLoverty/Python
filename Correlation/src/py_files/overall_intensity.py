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
data = pd.DataFrame().assign(t=t, intensity=I)
data.to_csv(os.path.join(folder_out, 'intensity.csv'))


