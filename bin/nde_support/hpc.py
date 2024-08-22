'''

python script to deploy jobs on della-gpu


'''
import os, sys 


def train_NDE_nonpart(iseed, hr=12, gpu=True): 
    ''' train NDE training
    '''
    jname = "NDE_nonpart.%i" % iseed
    ofile = "o/_NDE_nonpart.%i" % iseed

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --export=ALL", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        "#SBATCH --output=%s" % ofile, 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python nde_nonpart.py", 
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


def train_NDE_binary(iseed, binary, hr=12, gpu=True): 
    ''' train NDE training
    '''
    jname = "NDE_binary%i.%i" % (binary, iseed)
    ofile = "o/_NDE_binary%i.%i" % (binary, iseed)

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --export=ALL", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        "#SBATCH --output=%s" % ofile, 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python nde_binary.py %i" % binary, 
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None


train_NDE_nonpart(0, hr=1, gpu=True)
for code in [ 0,  2,  8,  9, 10, 27, 64, 66, 72, 75]: 
    train_NDE_binary(0, code, hr=1, gpu=True)
