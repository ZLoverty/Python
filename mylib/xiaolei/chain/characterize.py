import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util, filters, measure
import os
from scipy import ndimage
from myImageLib import dirrec, to8bit, bpass, FastPeakFind
from corrLib import readseq


def avg_cos_1(traj, order_pNo):
    # average cos's of adjacent particles, for single frame
    cos = 0
    count = 0
    k = 0
    for i in order_pNo:
        if k == 0:
            try:
                x1 = traj.x.loc[traj.particle==i].values[0]
                y1 = traj.y.loc[traj.particle==i].values[0]
                k += 1
                continue
            except:
                continue
        if k == 1:
            try:
                x2 = traj.x.loc[traj.particle==i].values[0]
                y2 = traj.y.loc[traj.particle==i].values[0]
                k += 1
                continue
            except:
                continue

        v1x = x2 - x1
        v1y = y2 - y1

        xt = x2
        yt = y2
        try:
            x2t = traj.x.loc[traj.particle==i].values[0]
            y2t = traj.y.loc[traj.particle==i].values[0]
        except:
            continue
        v2x = x2t - xt
        v2y = y2t - yt

        x1 = xt
        y1 = yt
        x2 = x2t
        y2 = y2t
        cos = cos + (v1x*v2x+v1y*v2y)/((v1x**2+v1y**2)**0.5*(v2x**2+v2y**2)**0.5)
        count += 1
    cos = cos / count
    return cos

def avg_cos(traj, order_pNo):
    # average cos's of adjacent particles, for whole video
    t = []
    cosL = []
    for frame in traj.frame.drop_duplicates():
        subtraj = traj.loc[traj.frame==frame]
        cos = avg_cos_1(subtraj, order_pNo)
        t.append(frame)
        cosL.append(cos)
    df = pd.DataFrame().assign(frame=t, cos=cosL)
    return df

def get_angle_and_arc(traj, order_pNo):
    data = pd.DataFrame()
    for frame in traj.frame.drop_duplicates():
        subtraj = traj.loc[traj.frame==frame]        
        count = 0
        bond_length = 0
        theta = []
        s = []
        k = 0
        # handle imperfect tracking frames, by skipping the frames
        try:
            for i in order_pNo:
                if k == 0:
                    try:
                        x1 = subtraj.x.loc[subtraj.particle==i].values[0]
                        y1 = subtraj.y.loc[subtraj.particle==i].values[0]
                        xt = x1
                        yt = y1
                        k += 1 # make sure we have particle 1
                        continue
                    except:
                        continue
                x2 = subtraj.x.loc[subtraj.particle==i].values[0]
                y2 = subtraj.y.loc[subtraj.particle==i].values[0]
                theta.append(np.arctan((y2-y1)/(x2-x1)))
                bond_length += ((y2-yt)**2 + (x2-xt)**2)**.5
                s.append(bond_length)
                xt = x2
                yt = y2
            subdata = pd.DataFrame().assign(s=s, theta=theta, frame=frame)
            data = data.append(subdata)
        except:
            continue
    return data
    
def fourier_coef(data, n=10): # in unit same as input data (usually pixel)
    # data is pd.DataFrame containing theta and arc_length
    # n is number of expanded terms
    L = data.s.max() + data.s.min()
    data = data.assign(ds=data.s.diff())
    a = []
    for i in range(0, n):
        coef = 0
        for num, r in data.iterrows():
            if num == 0:
                coef += r.theta * np.cos(i*np.pi*r.s/2/L) * r.s 
                continue            
            coef += r.theta * np.cos(i*np.pi*(r.s-r.ds/2)/L) * r.ds 
        coef = coef * (2 / L)**0.5
        a.append(coef)
    return np.array(a)
    
if __name__ == '__main__':
    pass
    traj = pd.read_csv(r'E:\Github\Python\mylib\xiaolei\chain\test_files\lp\tracking1.csv')
    order_pNo = np.array([0,1,2,3,4,5,6])
    aaa = get_angle_and_arc(traj, order_pNo)
    # dt_track test code
    # traj = dt_track(r'I:\Github\Python\mylib\xiaolei\chain\test_files')
    
    # avg_cos test code 
    # traj = pd.read_csv(r'E:\Github\Python\ForFun\Misc\cos\data.csv')
    # order_pNo = [0, 13, 12, 11, 9, 8, 6, 7, 1, 2, 4, 3, 5, 14, 10]
    # df = avg_cos(traj, order_pNo)
    # plt.plot(df.frame, df.cos)
    # plt.show()
    