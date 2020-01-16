import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    t = []
    cosL = []
    for frame in traj.frame.drop_duplicates():
        subtraj = traj.loc[traj.frame==frame]
        cos = avg_cos_1(subtraj, order_pNo)
        t.append(frame)
        cosL.append(cos)
    df = pd.DataFrame().assign(frame=t, cos=cosL)
    return df

if __name__ == '__main__':
    traj = pd.read_csv(r'E:\Github\Python\ForFun\Misc\cos\data.csv')
    order_pNo = [0, 13, 12, 11, 9, 8, 6, 7, 1, 2, 4, 3, 5, 14, 10]
    df = avg_cos(traj, order_pNo)
    plt.plot(df.frame, df.cos)
    plt.show()