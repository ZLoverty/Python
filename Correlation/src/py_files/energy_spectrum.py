import pandas as pd
import numpy as np
import os
from corrLib import readdata, energy_spectrum
import sys
import time

piv_folder = sys.argv[1]
out_folder = sys.argv[2]
percentile = 0.8
if len(sys.argv) > 3:
    percentile = float(sys.argv[3])
if len(sys.argv) > 4:
    sample_spacing = float(sys.argv[4])
    
if os.path.exists(out_folder) == False:
    os.makedirs(out_folder)
with open(os.path.join(out_folder, 'log.txt'), 'w') as f:    
    f.write('piv_folder: ' + piv_folder + '\n')
    f.write('out_folder: ' + out_folder + '\n')
    f.write('percentile: ' + str(percentile) + '\n')
    f.write(time.asctime() + ' // Computation starts!\n')
l = readdata(piv_folder, 'csv')
l_crop = l[l.index>l.index.max()*percentile]
for num, i in l_crop.iterrows():
    pivData = pd.read_csv(i.Dir)
    es = energy_spectrum(pivData, sample_spacing)
    es.to_csv(os.path.join(out_folder, i.Name+'.csv'), index=False)
    with open(os.path.join(out_folder, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // {} finished!\n'.format(i.Name))
        
with open(os.path.join(out_folder, 'log.txt'), 'a') as f:    
    f.write(time.asctime() + ' // Computation ends!\n')
    
""" EDIT
10022020 -- First edit
10192020 -- Add sample_spacing argument
"""

""" DESCRIPTION
Compute energy spectrum from PIV data.
"""

""" SYNTAX
python energy_spectrum.py piv_folder out_folder percentile sample_spacing

piv_folder -- piv folder
out_folder -- folder to save energy spectrum data
percentile -- the videos are taken from low energy to high energy. For steady-state  analysis, only the frames towards the end of the videos should be    computed. percentile is between (0, 1). Default value 0.8 means the spectra of only the last 20% of frames are computed.
sample_spacing -- distance between adjacent velocity vector
"""

""" TEST PARAMS
piv_folder -- E:\Github\Python\Correlation\test_images\test_corr\piv_folder
out_folder -- E:\Github\Python\Correlation\test_images\energy_spectrum
percentile -- 0.9
sample_spacing -- 
"""

""" LOG
piv_folder: E:\Github\Python\Correlation\test_images\test_corr\piv_folder
out_folder: E:\Github\Python\Correlation\test_images\energy_spectrum
percentile: 0.8
Mon Oct 12 17:50:50 2020 // Computation starts!
Mon Oct 12 17:50:50 2020 // 3004-3005 finished!
Mon Oct 12 17:50:50 2020 // Computation ends!
"""        