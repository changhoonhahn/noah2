'''

python script to deploy jobs on della-gpu


'''
import os, sys 


def hull(fcovar, fhull, hr=12): 
    ''' train NDE training
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % os.path.basename(fhull),
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --export=ALL", 
        "#SBATCH --output=_%s.o" % os.path.basename(fhull), 
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=chhahn@princeton.edu",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python hull.py %s %s" % (fcovar, fhull), 
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

hull('/scratch/gpfs/chhahn/noah/noah2/covars_w.npy', '/scratch/gpfs/chhahn/noah/noah2/hull_covars_w.pkl')
hull('/scratch/gpfs/chhahn/noah/noah2/covars_pt.npy', '/scratch/gpfs/chhahn/noah/noah2/hull_covars_pt.pkl')
