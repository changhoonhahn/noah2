import os,sys
import numpy as np
from tqdm.notebook import tqdm, trange

from noah2 import data as D
from noah2 import util as U

import copy
import torch
from nflows import transforms, distributions, flows
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_

import optuna 

cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")

##################################################################################
# read in training and validation  data 
##################################################################################
cx_train = np.load('../../dat/nde_support/data.train.inv_cdf.npy')
cx_valid = np.load('../../dat/nde_support/data.valid.inv_cdf.npy')

ndim = cx_train.shape[1]

train_loader = torch.utils.data.DataLoader(torch.tensor(cx_train.astype(np.float32)).to(device), batch_size=512, shuffle=True)
valid_loader = torch.utils.data.DataLoader(torch.tensor(cx_valid.astype(np.float32)).to(device), batch_size=512, shuffle=False)
##################################################################################
# OPTUNA
##################################################################################
# Optuna Parameters
n_trials    = 1000
study_name  = 'nde_support.v0'

output_dir = '/scratch/gpfs/chhahn/noah/noah2/nde_support/'

n_jobs     = 1
if not os.path.isdir(os.path.join(output_dir, study_name)): 
    os.system('mkdir %s' % os.path.join(output_dir, study_name))
storage    = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
n_startup_trials = 20 

num_iter = 1000
patience = 20
lrate    = 1e-3

n_blocks_min, n_blocks_max = 2, 15 
n_hidden_min, n_hidden_max = 32, 512

def Objective(trial):
    ''' bojective function for optuna 
    '''
    n_blocks = trial.suggest_int("n_blocks", n_blocks_min, n_blocks_max)
    n_hidden = trial.suggest_int("n_hidden", n_hidden_min, n_hidden_max, log=True)
    
    # initialize flow 
    blocks = []
    for iblock in range(n_blocks): 
        blocks += [transforms.MaskedAffineAutoregressiveTransform(features=ndim, hidden_features=n_hidden),
                transforms.RandomPermutation(features=ndim)]
    transform = transforms.CompositeTransform(blocks)

    base_distribution = distributions.StandardNormal(shape=[ndim])
    flow = flows.Flow(transform=transform, distribution=base_distribution)
    flow.to(device)
    
    # train flow
    optimizer = optim.Adam(flow.parameters(), lr=lrate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, min_lr=1e-5)
    #optim.lr_scheduler.OneCycleLR(optimizer, lrate, total_steps=num_iter)

    best_epoch, best_valid_loss, valid_losses = 0, np.inf, []
    for epoch in range(num_iter):
        # train 
        train_loss = 0.
        for batch in train_loader: 
            optimizer.zero_grad()
            loss = -flow.log_prob(batch).mean()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss = train_loss/float(len(train_loader))
    
        # validate
        with torch.no_grad():
            valid_loss = 0.
            for batch in valid_loader: 
                loss = -flow.log_prob(batch).mean()
                valid_loss += loss.item()
            valid_loss = valid_loss/len(valid_loader)

            if np.isnan(valid_loss): break

            valid_losses.append(valid_loss)
        scheduler.step(valid_loss)

        if epoch % 100 == 0: 
            print('Epoch: %i LR %.2e TRAINING Loss: %.2e VALIDATION Loss: %.2e' % 
                          (epoch, scheduler._last_lr[0], train_loss, valid_loss))
            
        if valid_loss < best_valid_loss: 
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_flow = copy.deepcopy(flow)
        else: 
            if epoch > best_epoch + patience: 
                break 

    # save trained flow 
    fflow = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(best_flow, fflow)

    floss = os.path.join(output_dir, study_name, '%s.%i.loss' % (study_name, trial.number))
    with open(floss,'w') as f:
        f.write(best_valid_loss)
    f.close()
    return best_valid_loss

sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials) 
study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, directions=["minimize"], load_if_exists=True) 

study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)
print("  Number of finished trials: %i" % len(study.trials))
