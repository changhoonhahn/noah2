'''

train flow for non-participants to estimate 
p( Y | community properties )  


'''
import os, sys
import numpy as np 

from noah2 import data as D
from causalflow import causalflow

import torch
import optuna 
##################################################################
# input 
##################################################################
output_dir = sys.argv[1]

##################################################################
cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")
##################################################################
# read CRS non-participant data 
DNoah = D.Noah2()
fema = DNoah._read_data_full()
is_participant = DNoah._participants(fema)
fema = fema[~is_participant]

# compile training data
columns = DNoah._columns()[:8]
train_data = np.array([np.array(fema[col]) for col in columns]).T
# reduce dynamical range  
#train_data[:,0] = np.log10(train_data[:,0])
train_data[:,3] = np.log10(train_data[:,3])
train_data[:,4] = np.log10(train_data[:,4])

Ntrain = int(train_data.shape[0] * 0.9)

# shufftle training data 
ishfl = np.arange(train_data.shape[0])
np.random.seed(42) 
np.random.shuffle(ishfl) 
train_data = train_data[ishfl][:Ntrain] # reserve 10% for testing

##################################################################################
# OPTUNA
##################################################################################
# declare Scenario A CausalFlow
Cflow = causalflow.CausalFlowA(device=device)

# Optuna Parameters
n_trials    = 1000
n_jobs      = 1
study_name  = 'flow.nonpart.nolog' 
if not os.path.isdir(os.path.join(output_dir, study_name)):
    os.system('mkdir %s' % os.path.join(output_dir, study_name))
storage     = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
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
           outcome_range=[[0.], [1.e6]],
           #outcome_range=[[-1.], [6.]],
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
