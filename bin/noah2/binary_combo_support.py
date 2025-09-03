'''

script to train series of flows for different combination of the ``binary'' CRS
activities ('c360', 'c520', 'c530', 'c540', 'c610', 'c620', 'c630')


'''
import os, sys
import numpy as np 

from noah2 import data as D
from causalflow import support as Support

import torch
import optuna 
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
if arch not in arch_dict.keys(): 
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
train_data = np.array([np.array(fema[col]) for col in columns[1:]]).T
# reduce dynamical range  
train_data[:,2] = np.log10(train_data[:,2])
train_data[:,3] = np.log10(train_data[:,3])

# only keep data of specified archetype that performs set of binary activities  
select = arch_dict[arch] & (binary_act_codes == code)
train_data = train_data[select]

# downsample based on zipcode to prevent overfittin 
zuniq, iuniq, nuniq = np.unique(fema['zipcode'][select], return_index=True, return_counts=True)

n_repeat = 1
# loop through zipcodes and only keep n_repeat entry per zipcode  
remove = np.zeros(np.sum(select)).astype(bool)
for z in zuniq: 
    is_zip = (np.array(fema['zipcode'])[select] == z)

    if np.sum(is_zip) > n_repeat:
        _remove = np.random.choice(np.arange(np.sum(is_zip)), np.sum(is_zip) - n_repeat, replace=False)
        remove[np.arange(np.sum(select))[is_zip][_remove]] = True

train_data = train_data[~remove]

# shufftle training data 
ishfl = np.arange(train_data.shape[0])
np.random.seed(42) 
np.random.shuffle(ishfl) 
train_data = train_data[ishfl]
ndim = train_data.shape[1]

##################################################################################
# OPTUNA
##################################################################################
Sup = Support.Support(device=device)

# Optuna Parameters
n_trials   = 1000
n_jobs     = 1
study_name = 'support.%s.%i' % (arch, code) 
if not os.path.isdir(os.path.join(output_dir, study_name)):
    os.system('mkdir %s' % os.path.join(output_dir, study_name))
storage    = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
n_startup_trials = 20

n_blocks_min, n_blocks_max = 2, 5
n_hidden_min, n_hidden_max = 32, 128
n_lr_min, n_lr_max = 5e-6, 1e-3


def Objective(trial):
    ''' bojective function for optuna
    '''
    # Generate the model
    n_blocks = trial.suggest_int("n_blocks", n_blocks_min, n_blocks_max)
    n_hidden = trial.suggest_int("n_hidden", n_hidden_min, n_hidden_max, log=True)
    lr = trial.suggest_float("lr", n_lr_min, n_lr_max, log=True)

    # set architecture
    Sup.set_architecture(ndim,
            nhidden=n_hidden,
            nblock=n_blocks)

    # run trianing
    flow, best_valid_loss = Sup._train(train_data,
            batch_size=50,
            learning_rate=lr,
            num_iter=300,
            clip_max_norm=1,
            verbose=False)

    # save trained flow
    fflow = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(flow, fflow)

    return best_valid_loss

sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, directions=["minimize"], load_if_exists=True)

study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)
print("  Number of finished trials: %i" % len(study.trials))
