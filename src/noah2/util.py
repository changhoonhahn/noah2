'''

utility functions 


'''
import os
import glob
import numpy as np
from scipy.special import erf, erfinv

import torch
import optuna 


def cdf_transform(x, bounds):
    """ Transform from a Gaussian (which is x) to a Uniform with bounds as input.
    """
    return _gaussian_cdf(x, 0, 1) * (bounds[1] - bounds[0]) + bounds[0]


def inv_cdf_transform(x, bounds):
    """ Transform from a Uniform with bounds (which is x) to a Gaussian
    """
    return _inv_gaussian_cdf((x - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)


def _gaussian_cdf(x, mu, sigma):
    """ CDF of a Gaussian distribution.

    :math:`F(x) = \\frac{1}{2}(1 + erf(\\frac{x - \\mu}{\\sigma}))`
    """
    return 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * sigma)))


def _inv_gaussian_cdf(x, mu, sigma):
    """ Inverse CDF of a Gaussian distribution.

    :math:`F^{-1}(x) = \\mu + \\sigma \\sqrt{2} \\text{erfinv}(2 \\times x - 1)`

    """
    return mu + sigma * np.sqrt(2) * erfinv(2 * x - 1)


def read_best_ndes(study_dir, n_ensemble=5, device='cpu', verbose=False):
    '''
    '''
    output_dir = os.path.dirname(study_dir)
    study_name = os.path.basename(study_dir) 
    storage    = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)

    study = optuna.load_study(study_name=study_name, storage=storage)

    nums = np.array([trial.number for trial in study.trials])
    vals = np.array([trial.value for trial in study.trials])

    nums = nums[vals != None]
    vals = vals[vals != None]
    if verbose: print('%i models trained' % len(nums))
   
    ndes = [] 
    for itrial in nums[np.argsort(vals)][:n_ensemble]: 
        fnde = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, itrial))
        if verbose: print(fnde)
        ndes.append(torch.load(fnde, map_location=device))

    return ndes 
