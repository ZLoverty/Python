import time
import matplotlib.pyplot as plt

def read_date(period_dir):
    L = []
    with open(period_dir, 'r') as f:
        while True:
            a = f.readline()
            if a == '':
                break        
            L.append(a.replace('\n', ''))
    return L
def compute_lap(date_list):
    count = 0
    lapL = []
    for l in date_list:
        date = l.split('/')
        t = (int(date[2]), int(date[0]), int(date[1]), 0, 0, 0, 0, 0, 0)
        if count == 0:        
            t0 = time.mktime(t)
            count += 1
        else:
            count += 1
            t1 = time.mktime(t)
            lap = (t1 - t0) / 24 / 3600
            lapL.append(lap)
            t0 = t1
    return lapL
def plot_period(date_list, lap):
    l = len(date_list)
    x = range(0, l-1)
    fig, ax = plt.subplots(dpi=300)
    ax.bar(x, lap)
    ax.set_xticklabels(date_list[0:l])
    ax.plot([0, l-2], [28, 28], ls='--', color='black')