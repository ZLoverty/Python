import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd

def label_slope(x, y, location='n'):
    def power(x, a, b):
        return a*x**b    
    popt, pcov = optimize.curve_fit(power, x, y)
    # label the slope of a power law curve in a log-log plot with a dashed line
    # location indicates the relative location of the dashed line as to the original data line (only 'n' is available, other options need to be implemented)
    xmin = np.log10(x.min())
    xmax = np.log10(x.max())
    ymin = np.log10(y.min())
    ymax = np.log10(y.max())
    if location == 'n':
        xf = 10**np.array([xmin + (xmax - xmin)*3/8, xmin + (xmax - xmin)*5/8])
    elif location == 'nw':
        xf = 10**np.array([xmin + (xmax - xmin)*2/8, xmin + (xmax - xmin)*4/8])
    elif location == 'ne':
        xf = 10**np.array([xmin + (xmax - xmin)*4/8, xmin + (xmax - xmin)*6/8])
    yf = 10**((ymax-ymin)/7)*popt[0]*xf**popt[1]
    xfmin = np.log10(xf)[0]
    xfmax = np.log10(xf)[1]
    xt = 10**(xfmin + (xfmax - xfmin)*3/8)
    yt = 10**((ymax-ymin)/20)*10**np.log10(yf).mean()
    return xf, yf, xt, yt, popt[1]

def reformat_sequence(original):
    """
    This is not done yet. The basic working principles are implemented. Input and output are not well defined.
    """
    upg = 5 # unit per group
    gpr = 4 # group per row
    with open(r'I:\Github\Python\ForFun\DNA\test_files\flic_formatted.txt', 'w') as fw:
        pass
    with open(r'I:\Github\Python\ForFun\DNA\test_files\flic.txt', 'r') as f:
        seq = f.read().replace('\n', '')
        with open(r'I:\Github\Python\ForFun\DNA\test_files\flic_formatted.txt', 'a') as fw:
            count = 1
            line_count = 1
            for i in range(0, int(len(seq)/upg)+1):
                if count % gpr == 1:
                    fw.write('{:05d}  '.format((line_count-1)*upg*gpr))
                    line_count += 1
                fw.write(seq[upg*i:upg*i+upg])
                if count % gpr != 0:
                    fw.write(' ')
                else:
                    fw.write('\n')
                count += 1
    return reformatted

if __name__ == '__main__':
    data = pd.read_csv(r'I:\Github\Python\Correlation\test_images\GNF\stat\data.csv')
    data = data.loc[data.Name=='100-1']
    x = data.n
    y = data.d
    xf, yf, xt, yt, slope = label_slope(x, y, location='ne')
    plt.plot(x, y)
    plt.plot(xf, yf, ls='--', color='black')
    plt.text(xt, yt, '{:.2f}'.format(slope))
    plt.xscale('log')
    plt.yscale('log')
    plt.show()