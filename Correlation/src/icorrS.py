import sys
import os
from corrLib import corrS, corrI, divide_windows, distance_corr, corrIseq
# import numpy
# folder = sys.argv[1]
# wsize = sys.argv[2]
# step = sys.argv[3]
folder = r'I:\Github\Python\Correlation\test_images\boxsize_effect'
wsize = 50
step = 50
t1 = time.monotonic()    
data_seq = corrIseq(folder, windowsize=[wsize, wsize], step=step)
data_seq.to_csv(os.path.join(folder, 'Icorrdata.dat'), index=False)
t2 = time.monotonic()
t = (t2 - t1) / 3600
print('Wall time: %.2f h' % t)
# print(1)