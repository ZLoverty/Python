import numpy as np
from corrLib import density_fluctuation, readseq
from skimage import io
import sys
import os
import time


input_folder = sys.argv[1]
output_folder = sys.argv[2]
interval = 1
if len(sys.argv) == 4:
    interval = int(sys.argv[3])

if os.path.exists(output_folder) == 0:
    os.makedirs(output_folder)

with open(os.path.join(output_folder, 'log.txt'), 'w') as f:
    pass

l = readseq(input_folder)
count = 0
for num, i in l.iterrows():
    if count % interval != 0:
        count += 1
        continue
    # print('Frame ' + i.Name)
    img = io.imread(i.Dir)
    df_data = density_fluctuation(img)
    df_data.to_csv(os.path.join(output_folder, i.Name+'.csv'), index=False)
    with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // ' + i.Name + ' calculated\n')
    count += 1
""" TEST COMMAND
python df_imseq.py input_folder output_folder [interval]
"""
        
"""  TEST PARAMS
input_folder = I:\Github\Python\Correlation\test_images\cl
output_folder = I:\Github\Python\Correlation\test_images\df_result
"""

""" LOG
Mon Jan 13 11:21:44 2020 // 100-2 calculated
Mon Jan 13 11:22:05 2020 // 40-2 calculated
Mon Jan 13 11:22:27 2020 // 60-2 calculated
Mon Jan 13 11:22:50 2020 // 80-2 calculated
"""