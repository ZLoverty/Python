import pandas as pd
import numpy as np

def experiment_log(verbose=False):
    """
    Save the experiment logs of a research project in a DataFrame. See "About-experiment-log" notebook for more details.
    
    Args:
    verbose -- print the integrity check result to find mistakes in the log. 
                Default to False, which only prints "The data looks OK!" if integrity check is passed.
    
    Returns:
    log_df -- DataFrame of experiment logs
    """
    
    log_dict = {} # dict log, with date (or folder name, str) as keys
    log_dict['08032020'] = { # transform daily log to a dict, each parameter forms a list
        'run_number': range(0, 15),
        'conc': [80, 80, 80, 60, 60, 60, 40, 40, 40, 20, 20, 20, 0, 0, 0],        
        'FPS': [30, 30, 30, 30, 30, 30, 30, 10, 10, 10, 10, 10, 10, 10, 10],
        'MPP': np.ones(15) * 0.33,
        'length': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 1800, 1800, 1800, 1800, 1800, 100, 100, 100],
        'exposure_time': [1, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        'thickness': np.ones(15) * 140
    }
    log_dict['08042020'] = { # transform daily log to a dict, each parameter forms a list
        'run_number': range(0, 12),
        'conc': [70, 70, 70, 50, 50, 50, 30, 30, 30, 10, 10, 10],        
        'FPS': [30, 30, 30, 30, 30, 30, 30, 30, 30, 10, 10, 10],
        'MPP': np.ones(12) * 0.33,
        'length': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 1800, 1800, 1800],
        'exposure_time': [2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3],
        'thickness': np.ones(12) * 140
    }
    log_dict['08052020'] = { # transform daily log to a dict, each parameter forms a list
        'run_number': range(0, 12),
        'conc': np.ones(12) * 85,        
        'FPS': [30, 30, 30, 30, 30, 30, 30, 30, 30, 10, 10, 10],
        'MPP': np.ones(12) * 0.33,
        'length': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 1800, 1800, 1800],
        'exposure_time': [4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        'thickness': [200, 200, 200, 140, 140, 140, 100, 100, 100, 20, 20, 20]
    }
    log_dict['08062020'] = { # transform daily log to a dict, each parameter forms a list
        'run_number': range(0, 13),
        'conc': [100, 100, 100, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120],        
        'FPS': [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 10],
        'MPP': np.ones(13) * 0.33,
        'length': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600],
        'exposure_time': [4, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
        'thickness': [140, 140, 140, 140, 140, 140, 100, 100, 100, 200, 200, 200, 20]
    }
    
    # Check integrity: each column from the same day should be of the same length
    for kw in log_dict:
        if verbose == True:
            print('---------{}----------'.format(kw))
        for count, param in enumerate(log_dict[kw]):
            l = len(log_dict[kw][param])
            if verbose == True:
                print('length of {0:15s}: {1:d}'.format(param, l))
            if count > 0:
                assert(l==l_temp)
            l_temp = l
        
    print("-------The log looks OK!--------")
    
    
    log_df = pd.DataFrame()
    for kw in log_dict:
        log_df_temp = pd.DataFrame(log_dict[kw]).assign(date=kw)
        log_df = log_df.append(log_df_temp)
        
    return log_df