from corrLib import corrS, corrI, divide_windows, distance_corr, corrIseq, readseq, density_fluctuation
from myImageLib import bpass, dirrec
import numpy as np
from scipy.signal import savgol_filter
from skimage import io
import pandas as pd
import os
import time
import sys
import pdb

def readdata(folder):
    dataDirs = dirrec(folder, '*.csv')
    nameList = []
    dirList = []
    for dataDir in dataDirs:
        path, file = os.path.split(dataDir)
        name, ext = os.path.splitext(file)
        nameList.append(name)
        dirList.append(dataDir)
    fileList = pd.DataFrame()
    fileList = fileList.assign(Name=nameList, Dir=dirList)
    fileList = fileList.sort_values(by=['Name'])
    return fileList

input_folder = sys.argv[1]
output_folder = sys.argv[2]

# check output dir existence
if os.path.exists(output_folder) == 0:
    os.makedirs(output_folder)
with open(os.path.join(output_folder, 'log.txt'), 'w') as f:
    pass

l = readdata(input_folder)
for num, i in l.iterrows():
    print('Frame ' + i.Name)
    pivData = pd.read_csv(i.Dir)
    col = len(pivData.x.drop_duplicates())
    row = len(pivData.y.drop_duplicates())
    X = np.array(pivData.x).reshape((row, col))
    Y = np.array(pivData.y).reshape((row, col))
    U = np.array(pivData.u).reshape((row, col))
    V = np.array(pivData.v).reshape((row, col))
    CA, CV = corrS(X, Y, U, V)        
    data = pd.DataFrame().assign(X=X.flatten(), Y=Y.flatten(), CA=CA.flatten(), CV=CV.flatten())
    # Save data
    data.to_csv(os.path.join(output_folder, i.Name+'.csv'), index=False)
    # Write log
    with open(os.path.join(output_folder, 'log.txt'), 'a') as f:
        f.write(time.asctime() + ' // ' + i.Name + ' calculated\n')


""" TEST COMMAND
python cav_imseq.py input_folder output_folder
"""
        
"""  TEST PARAMS
input_folder = I:\Github\Python\Correlation\test_images\CAV
output_folder = I:\Github\Python\Correlation\test_images\CAV\cav_result
"""

""" LOG
Thu Feb 13 11:39:47 2020 // 900-901 calculated
Thu Feb 13 11:40:35 2020 // 902-903 calculated
"""

""" SPEED 51 s/frame
Tue Feb 11 14:27:52 2020 // 900-901 calculated
Tue Feb 11 14:28:41 2020 // 902-903 calculated
"""
    
    
    
    