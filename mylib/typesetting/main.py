import matplotlib

def prl(preset='1-column-2-panel'):
    """
    Args:
    preset -- the type of preset to use (figure size and dpi)
    """
    presets = {}
    presets['1-column-2-panel'] = (1.75, 1.5, 400)
    presets['1-column-1-panel'] = (3.5, 1.5, 400)
    
    w, h, dpi = presets[preset]
    matplotlib.rcParams['figure.figsize'] = w, h
    matplotlib.rcParams['figure.dpi'] = dpi
    
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['text.usetex'] = True 

    matplotlib.rcParams['axes.labelpad'] = 1.0
    matplotlib.rcParams['axes.linewidth'] = 0.5

    matplotlib.rcParams['xtick.labelsize'] = 'small'
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['xtick.minor.visible'] = True
    matplotlib.rcParams['xtick.top'] = True
    matplotlib.rcParams['xtick.major.size'] = 2.5
    matplotlib.rcParams['xtick.minor.size'] = 1.6

    matplotlib.rcParams['ytick.labelsize'] = 'small'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['ytick.minor.visible'] = True
    matplotlib.rcParams['ytick.right'] = True
    matplotlib.rcParams['ytick.major.size'] = 2.5
    matplotlib.rcParams['ytick.minor.size'] = 1.6
    
    # LEGEND
    matplotlib.rcParams['legend.labelspacing'] = 0.2
    matplotlib.rcParams['legend.handlelength'] = 1
    matplotlib.rcParams['legend.fontsize'] = 'small'
    matplotlib.rcParams['legend.handletextpad'] = 0.2
    matplotlib.rcParams['legend.columnspacing'] = 0.5
    matplotlib.rcParams['legend.frameon'] = False
    
    # FONT
    matplotlib.rcParams['font.size'] = 8
    return None
    