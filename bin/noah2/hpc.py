'''

python script to deploy flow training on a HPC. The script in this case is for
the Della cluster on the Princeton Research Computing 


'''
import os, sys 


def binary_flows(arch, act_code, output_dir='.', hr=12, gpu=True): 
    ''' write, deploy, and delete script to train flows using optuna. 
    '''
    jname = "flow.%s.%i" % (arch, act_code) 
    ofile = "o/_flow.%s.%i" % (arch, act_code) 
    while os.path.isfile(ofile): 
        jname += '_'
        ofile += '_'

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --mem=8G", 
        "#SBATCH --export=ALL", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        "#SBATCH --output=%s" % ofile, 
        "#SBATCH --mail-type=all",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python binary_combo.py %s %i %s" % (arch, act_code, output_dir), 
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


def binary_support(arch, act_code, output_dir='.', hr=12, gpu=True): 
    ''' write, deploy, and delete script to train flows using optuna. 
    '''
    jname = "supp.%s.%i" % (arch, act_code) 
    ofile = "o/_supp.%s.%i" % (arch, act_code) 
    while os.path.isfile(ofile): 
        jname += '_'
        ofile += '_'

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --mem=8G", 
        "#SBATCH --export=ALL", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        "#SBATCH --output=%s" % ofile, 
        "#SBATCH --mail-type=all",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python binary_combo_support.py %s %i %s" % (arch, act_code, output_dir), 
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


def nonpart_flows(output_dir='.', hr=12, gpu=True): 
    ''' write, deploy, and delete script to train flows using optuna. 
    '''
    jname = "flow.nonpart" 
    ofile = "o/_flow.nonpart"  
    while os.path.isfile(ofile): 
        jname += '_'
        ofile += '_'

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --mem=8G", 
        "#SBATCH --export=ALL", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        "#SBATCH --output=%s" % ofile, 
        "#SBATCH --mail-type=all",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python nonpart.py %s" % output_dir, 
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


def nonpart_support(output_dir='.', hr=12, gpu=True): 
    ''' write, deploy, and delete script to train flows using optuna. 
    '''
    jname = "supp.nonpart" 
    ofile = "o/_supp.nonpart" 
    while os.path.isfile(ofile): 
        jname += '_'
        ofile += '_'

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --mem=8G", 
        "#SBATCH --export=ALL", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        "#SBATCH --output=%s" % ofile, 
        "#SBATCH --mail-type=all",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python nonpart_support.py %s" % output_dir, 
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


def binary_activites(arch, exp0, exp1, hr=1): 
    ''' write, deploy, and delete script to train flows using optuna. 
    '''
    jname = "%s.%i_%i" % (arch, exp0, exp1) 
    ofile = "o/_%s.%i_%i" % (arch, exp0, exp1) 

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % jname,
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --mem=16G", 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=%s" % ofile, 
        "#SBATCH --mail-type=all",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python binary_act.py %s %i %i" % (arch, exp0, exp1), 
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


def _deploy_base_slurm(job_name, output_name, cmd, hr=1): 
    '''
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % job_name,
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --mem=16G", 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=%s" % output_name, 
        "#SBATCH --mail-type=all",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        cmd,
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


def binary_metro_base0(hr=1): 
    cmd = "python binary_act.py metro 0 '1, 2, 4, 8, 16, 64'"
    _deploy_base_slurm("metro.base0", "o/_metro.base0", cmd, hr=hr)
    return None


def binary_metro_base8(hr=1): 
    cmd = "python binary_act.py metro 8 '9'"
    _deploy_base_slurm("metro.base8", "o/_metro.base8", cmd, hr=hr)
    return None


def binary_metro_base9(hr=1): 
    cmd = "python binary_act.py metro 9 '8, 11, 13, 1, 25, 73'"
    _deploy_base_slurm("metro.base9", "o/_metro.base9", cmd, hr=hr)
    return None

def binary_metro_base64(hr=1): 
    cmd = "python binary_act.py metro 64 '65, 72, 80'"
    _deploy_base_slurm("metro.base64", "o/_metro.base64", cmd, hr=hr)
    return None


def binary_metro_base66(hr=1): 
    cmd = "python binary_act.py metro 66 '2'" #'67, 64, 70, 74'"
    _deploy_base_slurm("metro.base66", "o/_metro.base66", cmd, hr=hr)
    return None


def binary_metro_base75(hr=1): 
    cmd = "python binary_act.py metro 75 '107'"
    _deploy_base_slurm("metro.base66", "o/_metro.base66", cmd, hr=hr)
    return None


def binary_micro_base0(hr=1): 
    cmd = "python binary_act.py micro 0 '1, 2, 4, 8, 16, 64'"
    _deploy_base_slurm("micro.base0", "o/_micro.base0", cmd, hr=hr)
    return None


if __name__=='__main__': 
    #-------- 2024.09.25 --------
    #for code in [0, 1, 2, 3, 8, 9, 10, 11, 17, 18, 24, 25, 27, 29, 31, 64, 66, 67, 72, 73, 75, 91, 95]:
    #    binary_flows('metro', code, output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/',
    #            hr=6, gpu=False)
    #for code in [0, 64]:  
    #    binary_flows('micro', code, output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/',


    #-------- 2024.09.25 --------
    #for code in [0, 1, 2, 3, 8, 9, 10, 11, 17, 18, 24, 25, 27, 29, 31, 64, 66, 67, 72, 73, 75, 91, 95]:
    #    binary_flows('metro', code, output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/',
    #            hr=6, gpu=False)
    #for code in [0, 64]:  
    #    binary_flows('micro', code, output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/',
    #            hr=6, gpu=False)

    #-------- 2024.09.26 --------
    #for code in [0, 2, 8, 9, 10, 11, 24, 25, 27, 64, 66, 67, 72, 73, 75]:  
    #    binary_support('metro', code, output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/', hr=6, gpu=False)
    #
    #for code in [0, 64]:  
    #    binary_support('micro', code, output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/', hr=6, gpu=False)
    #
    #binary_support('rural', 0, output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/', hr=6, gpu=False)
    #
    #for code in [6, 16, 17, 18, 29, 31, 89, 91, 95]:  
    #    binary_support('metro', code, output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/', 
    #            hr=6, gpu=False)
    #for code in [1, 2, 8, 16]:
    #    binary_support('micro', code, output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/', 
    #            hr=6, gpu=False)
    #for code in [0, 64]:
    #    binary_support('small', code, output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/', 
    #            hr=6, gpu=False)
    #for code in [10]:  
    #    binary_support('metro', code, output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/', hr=6, gpu=False)

    #-------- for low impact --------
    #nonpart_flows(output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/', hr=12, gpu=False)
    #nonpart_flows(output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/', hr=12, gpu=False)
    #nonpart_flows(output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/', hr=12, gpu=False)
    #nonpart_support(output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/', hr=12, gpu=False)
    #nonpart_support(output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/', hr=12, gpu=False)
    #nonpart_support(output_dir='/scratch/gpfs/chhahn/noah/noah2/qphi/', hr=12, gpu=False)

    # --- evaluate CTEs for binary activities ---- 
    #binary_metro_base0(hr=1)
    #binary_metro_base8(hr=1)
    #binary_metro_base9(hr=2)
    #binary_metro_base64(hr=2)
    #binary_metro_base66(hr=2)
    #binary_metro_base75(hr=2)
    binary_micro_base0(hr=2)
