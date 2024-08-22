'''

estimate $Q_\phi(X)$ for CRS non-participant communities

X consists of community properities: Mean Rainfall, Flood Risk, Median Income,
Population, Renter Fraction, Educational Attainment, White Fraction


'''
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
# input
##################################################################################
binary = int(sys.argv[1])

##################################################################################
# read in training and validation  data 
##################################################################################
# read full data set
DNoah = D.Noah2()

data = DNoah.data(sample='binary%i' % binary, 
                  columns='props+', 
                  unique_zipcode=True) 

# reduce dynamic scale
data[:,2] = np.log10(data[:,2])
data[:,3] = np.log10(data[:,3])

# preprocess the data 
data[:,0] = U.inv_cdf_transform(data[:,0], [-1e-3, 1500])
data[:,1] = U.inv_cdf_transform(data[:,1], [-1e-3, 10])
data[:,2] = U.inv_cdf_transform(data[:,2], [3, 6])        
data[:,3] = U.inv_cdf_transform(data[:,3], [1, 6])        
data[:,4] = U.inv_cdf_transform(data[:,4], [-1e-3, 1+1e-3])            
data[:,5] = U.inv_cdf_transform(data[:,5], [-1e-3, 1+1e-3])            
data[:,6] = U.inv_cdf_transform(data[:,6], [-1e-3, 1+1e-3])      
# cdf transfer the CRS activities column
data[:,-3] = U.inv_cdf_transform(data[:,-3], [-1e-3, 100.+1e-3])
data[:,-2] = U.inv_cdf_transform(data[:,-2], [-1e-3, 100.+1e-3])
data[:,-1] = U.inv_cdf_transform(data[:,-1], [-1e-3, 100.+1e-3])

# shuffle up the data 
np.random.seed(0)
ind = np.arange(data.shape[0])
np.random.shuffle(ind)

# set up training/testing data
Ntrain = int(0.9 * data.shape[0])
print('Ntrain= %i, Nvalid= %i' % (Ntrain, data.shape[0] - Ntrain))
data_train = data[ind][:Ntrain]
data_valid = data[ind][Ntrain:]

ndim = data_train.shape[1]

train_loader = torch.utils.data.DataLoader(
        torch.tensor(data_train.astype(np.float32)).to(device),
        batch_size=50, shuffle=True)
valid_loader = torch.utils.data.DataLoader(
        torch.tensor(data_valid.astype(np.float32)).to(device),
        batch_size=50, shuffle=False)
##################################################################################
# OPTUNA
##################################################################################
# Optuna Parameters
n_trials    = 1000
study_name  = 'nde_binary%i.v0' % binary

output_dir = '/scratch/gpfs/chhahn/noah/noah2/nde_support/'
if not os.path.isdir(output_dir): 
    output_dir = '/Users/chahah/data/noah/noah2/nde_support/'

n_jobs     = 1
if not os.path.isdir(os.path.join(output_dir, study_name)): 
    os.system('mkdir %s' % os.path.join(output_dir, study_name))
storage    = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
n_startup_trials = 20 

num_iter = 1000
patience = 20
lrate    = 1e-3
clip_max_norm = 5.

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
        blocks += [
                transforms.MaskedAffineAutoregressiveTransform(
                    features=ndim, hidden_features=n_hidden),
                transforms.RandomPermutation(features=ndim)]
    transform = transforms.CompositeTransform(blocks)

    base_distribution = distributions.StandardNormal(shape=[ndim])
    flow = flows.Flow(transform=transform, distribution=base_distribution)
    flow.to(device)
    
    # train flow
    optimizer = optim.Adam(flow.parameters(), lr=lrate)

    best_epoch, best_valid_loss, valid_losses = 0, np.inf, []
    for epoch in range(num_iter):
        # train 
        train_loss = 0.
        for batch in train_loader: 
            optimizer.zero_grad()
            loss = -flow.log_prob(batch).mean()
            loss.backward()
            train_loss += loss.item()
            clip_grad_norm_(flow.parameters(), max_norm=clip_max_norm)
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

        if epoch % 10 == 0: 
            print('Epoch: %i TRAINING Loss: %.2e VALIDATION Loss: %.2e' %
                  (epoch, train_loss, valid_loss))
            
        if valid_loss < best_valid_loss: 
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_flow = copy.deepcopy(flow)
        else: 
            if epoch > best_epoch + patience: 
                print('DONE: EPOCH %i, BEST EPOCH %i BEST VALIDATION Loss: %.2e' %
                      (epoch, best_epoch, best_valid_loss))
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
