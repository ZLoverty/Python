def header(walltime, mem, code_path, to_file=None, nodes=1):
    # hrs, gb
    if to_file == None:
        print('#!/bin/bash -l')
        print('#PBS -l walltime={0}:00:00,nodes={2:d}:ppn=8,mem={1}gb'.format(walltime, mem, nodes))
        print('#PBS -m abe')
        print('#PBS -M liux3141@umn.edu\n')
        print('cd {}'.format(code_path))
        print('module load python3')
        print('source activate pythonEnv')
    else:
        to_file.write('#!/bin/bash -l\n')
        to_file.write('#PBS -l walltime={0}:00:00,nodes={2:d}:ppn=8,mem={1}gb\n'.format(walltime, mem, nodes))
        to_file.write('#PBS -m abe\n')
        to_file.write('#PBS -M liux3141@umn.edu\n')
        to_file.write('cd {}\n\n'.format(code_path))
        to_file.write('module load python3\n')
        to_file.write('source activate pythonEnv\n')

def data_log():
    log = {}
    log['08032020'] = {}
    log['08032020']['num'] = list(range(0, 15))
    log['08032020']['fps'] = [30, 30, 30, 30, 30, 30, 30, 10, 10, 10, 10, 10, 10, 10, 10]
    log['08042020'] = {}
    log['08042020']['num'] = list(range(0, 12))
    log['08042020']['fps'] = [30, 30, 30, 30, 30, 30, 30, 30, 30, 10, 10, 10]
    log['08052020'] = {}
    log['08052020']['num'] = list(range(0, 12))
    log['08052020']['fps'] = [30, 30, 30, 30, 30, 30, 30, 30, 30, 10, 10, 10]
    log['08062020'] = {}
    log['08062020']['num'] = list(range(0, 13))
    log['08062020']['fps'] = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 10]
    return log