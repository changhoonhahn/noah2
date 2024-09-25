'''

script to train series of flows for different combination of the ``binary'' CRS
activities ('c360', 'c520', 'c530', 'c540', 'c610', 'c620', 'c630')


'''
import os, sys
import numpy as np 

from noah2 import data as D
from causalflow import causalflow

import torch
##################################################################
# input 
##################################################################
arch = sys.argv[1] # archetype 
code = int(sys.argv[2])
output_dir = sys.argv[3]

##################################################################
cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")
##################################################################
# read CRS participant data 
DNoah = D.Noah2()
fema = DNoah._read_data_full()
is_participant = DNoah._participants(fema)
fema = fema[is_participant]

metro = ((fema['RUCA1'] == 1))# | (fema['RUCA1'] == 2) | (fema['RUCA1'] == 3))
micro = ((fema['RUCA1'] == 4) | (fema['RUCA1'] == 5) | (fema['RUCA1'] == 6))
small = ((fema['RUCA1'] == 7) | (fema['RUCA1'] == 8) | (fema['RUCA1'] == 9))
rural = ((fema['RUCA1'] == 10))
arch_dict = {'metro': metro, 'micro': micro, 'small': small, 'rural': rural}
if arch is not in arch_dict.keys(): 
    raise ValueError

columns = DNoah._columns()[:8] + ['s_%s' % c for c in ['c350', 'c420', 'c450']]
column_labels = np.array(DNoah._column_labels()[:8] + ['c350', 'c420', 'c450'])

# binary activity codes 
binary_activities = ['c360', 'c520', 'c530', 'c540', 'c610', 'c620', 'c630']

binary_data = np.array([np.array(fema[col]) for col in binary_activities]).T
binary_data = (binary_data > 0).astype(int)

binary_act_codes = np.zeros(len(fema)).astype(int) 
for i in range(binary_data.shape[1]):
    binary_act_codes += 2**i * binary_data[:,i]

# compile training data
train_data = np.array([np.array(fema[col]) for col in columns]).T
# reduce dynamical range  
train_data[:,0] = np.log10(train_data[:,0])
train_data[:,3] = np.log10(train_data[:,3])
train_data[:,4] = np.log10(train_data[:,4])

# only keep data of specified archetype that performs set of binary activities  
train_data = train_data[arch_dict[arch] & (binary_act_code == code)]

##################################################################################
# OPTUNA
##################################################################################
# declare Scenario A CausalFlow
Cflow = causalflow.CausalFlowA(device=device)

# Optuna Parameters
n_trials    = 1000
n_jobs     = 1
study_name = '%s.%i' % (arch, code) 
if not os.path.isdir(os.path.join(output_dir, study_name)):
    os.system('mkdir %s' % os.path.join(output_dir, study_name))
storage    = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
n_startup_trials = 20

n_blocks_min, n_blocks_max = 2, 5
n_transf_min, n_transf_max = 2, 5
n_hidden_min, n_hidden_max = 32, 128
n_comp_min, n_comp_max = 1, 5
n_lr_min, n_lr_max = 5e-6, 1e-3


def Objective(trial):
    ''' bojective function for optuna
    '''
    # Generate the model
    n_blocks = trial.suggest_int("n_blocks", n_blocks_min, n_blocks_max)
    n_transf = trial.suggest_int("n_transf", n_transf_min,  n_transf_max)
    n_hidden = trial.suggest_int("n_hidden", n_hidden_min, n_hidden_max, log=True)
    lr = trial.suggest_float("lr", n_lr_min, n_lr_max, log=True)
    n_comp = trial.suggest_int("n_comp", n_comp_min, n_comp_max)

    Cflow.set_architecture(
            arch='made',
            nhidden=n_hidden,
            ntransform=n_transf,
            nblocks=n_blocks,
            num_mixture_components=n_comp,
            batch_norm=True)


   flow, best_valid_log_prob = Cflow._train_flow(train_data[:,0], train_data[:,1:],
            outcome_range=[[-1.], [6.]],
            training_batch_size=50,
            learning_rate=lr,
            verbose=False)

    # save trained NPE
    fflow = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(flow, fflow)

    return -1*best_valid_log_prob

sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, directions=["minimize"], load_if_exists=True)

study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)
print("  Number of finished trials: %i" % len(study.trials))
