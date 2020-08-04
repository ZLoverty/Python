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
        to_file.write('#!/bin/bash -l\n')
        to_file.write('#PBS -l walltime={0}:00:00,nodes=1:ppn=8,mem={1}gb\n'.format(walltime, mem))
        to_file.write('#PBS -m abe\n')
        to_file.write('#PBS -M liux3141@umn.edu\n')
        to_file.write('cd {}\n\n'.format(code_path))
        to_file.write('module load python3\n')
        to_file.write('source activate pythonEnv\n')