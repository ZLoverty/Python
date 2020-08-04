def header(walltime, mem, code_path, to_file=None):
    # hrs, gb
    if to_file == None:
        print('#!/bin/bash -l')
        print('#PBS -l walltime={0}:00:00,nodes=1:ppn=8,mem={1}gb'.format(walltime, mem))
        print('#PBS -m abe')
        print('#PBS -M liux3141@umn.edu\n')
        print('cd {}'.format(code_path))
        print('module load python3')
        print('source activate pythonEnv')
    else:
        f.write('#!/bin/bash -l\n')
        f.write('#PBS -l walltime={0}:00:00,nodes=1:ppn=8,mem={1}gb\n'.format(walltime, mem))
        f.write('#PBS -m abe\n')
        f.write('#PBS -M liux3141@umn.edu\n')
        f.write('cd {}\n\n'.format(code_path))
        f.write('module load python3\n')
        f.write('source activate pythonEnv\n')