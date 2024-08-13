'''
'''
import os, sys 
import numpy as np
from noah2 import data as D
from noah2 import util as U

import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter

import otpuna 
from sbi import utils as Ut
from sbi import inference as Inference

cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")

#####################################################################
# input 
#####################################################################
output_dir = sys.argv[1]

#####################################################################
# load training data
#####################################################################
DNoah = D.Noah2()
data_train = DNoah._read_data(version='0.1', split='train') 

# log10 scale some of the columns to reduce dynamical scale
data_train[:,0] = np.log10(data_train[:,0])
data_train[:,3] = np.log10(data_train[:,3])
data_train[:,4] = np.log10(data_train[:,4])


#####################################################################
# initializing  qphi training
#####################################################################
# set prior 
lower_bounds = torch.tensor([1])
upper_bounds = torch.tensor([9])
prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=device)

# set up optuna 
n_trials    = 1000
study_name  = 'noah2.qphi'

n_jobs = 1
if not os.path.isdir(os.path.join(output_dir, study_name)):
    os.system('mkdir %s' % os.path.join(output_dir, study_name))
storage    = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
n_startup_trials = 20

n_blocks_min, n_blocks_max = 2, 5
n_transf_min, n_transf_max = 2, 5
n_hidden_min, n_hidden_max = 32, 128
l_rate = 1e-3


def Objective(trial):
    ''' bojective function for optuna
    '''
    # Generate the model
    n_blocks = trial.suggest_int("n_blocks", n_blocks_min, n_blocks_max)
    n_transf = trial.suggest_int("n_transf", n_transf_min,  n_transf_max)
    n_hidden = trial.suggest_int("n_hidden", n_hidden_min, n_hidden_max, log=True)
    n_comp = trial.suggest_int("n_comp", 2, 20)
    
    # define NPE 
    neural_posterior = Ut.posterior_nn('made', 
            hidden_features=n_hidden, 
            num_transforms=n_transf, 
            num_blocks=n_blocks, 
            num_mixture_components=n_comp, 
            use_batch_norm=True)
    
    # setup NPE training 
    anpe = Inference.SNPE(prior=prior,
            density_estimator=neural_posterior,
            device=device)

    # load training data 
    anpe.append_simulations( 
            torch.tensor(data_train[:,:1], dtype=torch.float32).to(device), 
            torch.tensor(data_train[:,1:], dtype=torch.float32).to(device))
    
    # train 
    qphi_y_x = anpe.train(
            training_batch_size=50,
            learning_rate=l_rate, 
            show_train_summary=True)

    # save trained NPE  
    qphi    = anpe.build_posterior(qphi_y_x)
    fqphi   = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(qphi, fqphi)

    best_valid_log_prob = anpe._summary['best_validation_log_prob'][0]

    return -1*best_valid_log_prob

sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, directions=["minimize"], load_if_exists=True)

study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)
print("  Number of finished trials: %i" % len(study.trials))
