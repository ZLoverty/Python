import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util, filters, measure
import os
from scipy import ndimage, optimize
from myImageLib import dirrec, to8bit, bpass, FastPeakFind
from corrLib import readseq
import pdb

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
    # data is pd.DataFrame containing theta and arc_length of a single frame
    # n is number of expanded terms
    L = data.s.max() + data.s.min()
    data = data.assign(ds=data.s.diff())
    a = []
    number = []
    for i in range(1, n+1): # the expansion starts from n=1
        coef = 0
        for num, r in data.iterrows():
            if num == 0:
                coef += r.theta * np.cos(i*np.pi*r.s/2/L) * r.s 
                continue            
            coef += r.theta * np.cos(i*np.pi*(r.s-r.ds/2)/L) * r.ds 
        coef = coef * (2 / L)**0.5
        number.append(i)
        a.append(coef)
    return pd.DataFrame().assign(n=number, a=a)

def fourier_coef_video(data, n=10):
    # data is pd.DataFrame containing theta and arc_length of multiple frames
    # n is number of expanded terms
    data_all = pd.DataFrame()
    for frame in data.frame.drop_duplicates():
        subdata = data.loc[data.frame==frame]
        a = fourier_coef(subdata, n=12)
        subdata = a.assign(frame=frame)
        data_all = data_all.append(subdata)
    return data_all

def temp_var(data, dt):
    varan = []
    n = []
    nframe = len(data.frame.drop_duplicates())
    for i in data.n.drop_duplicates():
        varan_tmp = 0
        count = 0
        for t in range(1, nframe-dt):
            try:
                an1 = data.loc[(data.frame==t)&(data.n==i)].a.values[0]
                an2 = data.loc[(data.frame==t+dt)&(data.n==i)].a.values[0]
            except:
                continue
            varan_tmp += (an2 - an1)**2
            count += 1
        varan_tmp /= count
        varan.append(varan_tmp)
        n.append(i)        
    return pd.DataFrame().assign(n=n, var=varan)
    
def compute_lp(varan, L, nf=8):
    # varan is the temporal variance data, returned by function temp_var
    # L is the contour length of the chain, L should be in a unit that is consistent with all previous calculations
    # Need to verify!!!
    def lin(x, a):
        return -2 * x + a
    fitx = np.log(varan.loc[varan.n <= nf]['n'])
    fity =  np.log(varan.loc[varan.n <= nf]['var'])
    po, pc = optimize.curve_fit(lin, fitx, fity, p0=1)
    lp = L**2 / np.pi**2 / np.exp(po)
    return lp[0]
    
if __name__ == '__main__':
    pass
    dt = 100
    data = pd.read_csv(r'I:\Github\Python\mylib\xiaolei\chain\test_files\lp\coef.csv', index_col=0)
    varan = temp_var(data, dt)
    
    # data = pd.read_csv(r'E:\Github\Python\mylib\xiaolei\chain\test_files\lp\arc_and_angle.csv', index_col=0)
    # data_all = fourier_coef_video(data, n=12)
    
    # traj = pd.read_csv(r'E:\Github\Python\mylib\xiaolei\chain\test_files\lp\tracking1.csv')
    # order_pNo = np.array([0,1,2,3,4,5,6])
    # aaa = get_angle_and_arc(traj, order_pNo)
    # dt_track test code
    # traj = dt_track(r'I:\Github\Python\mylib\xiaolei\chain\test_files')
    
    # avg_cos test code 
    # traj = pd.read_csv(r'E:\Github\Python\ForFun\Misc\cos\data.csv')
    # order_pNo = [0, 13, 12, 11, 9, 8, 6, 7, 1, 2, 4, 3, 5, 14, 10]
    # df = avg_cos(traj, order_pNo)
    # plt.plot(df.frame, df.cos)
    # plt.show()
    