'''

module for interfacing with q_phi(Y | X) to calculate causal effects.


'''
import os
import glob
import numpy as np 

import torch


class qphi_Y_X(object):
    ''' class object for loading and using q_phi(Y|X) estimates to evaluate the 
    causal effect of CRS activites for the NOAH2 project. 


    '''
    def __init__(self, dir_qphi=None, device=None): 
        if dir_qphi is None:  
            raise ValueError('specify the location of the saved q_phi(Y|X)')
        self._dir = dir_qphi 

        if device is None: 
            if torch.cuda.is_available(): device = 'cuda'
            else: device = 'cpu'
        self.device = device

        self.qphis = self._load_qphis()


    def _load_qphis(self): 
        ''' load all qphis in the specified directory 
        '''
        qphis = []
        for fqphi in glob.glob(os.path.join(self._dir, '*.pt')):
            qphi = torch.load(fqphi, map_location=self.device)
            qphis.append(qphi)

        self._n_ensemble = len(qphis)
        if self._n_ensemble == 0: raise ValueError("no qphis in directory!") 

        return qphis

    def Y_sample(self, x, n_sample=10000): 
        ''' for given X, use q_phi to sample `n_sample` Y' values. This can be
        used to estimate: e.g. 

        $$<Y> = \int Y p(Y\,|\,X) {\rm d}Y \approx \frac{1}{N} \sum_{Y'\sim q_\phi} Y'$$

        '''
        # preprocess x to be consistent with qphi training 
        x = np.atleast_2d(x) 
        x[:,0] = np.log10(x[:,0])
        x[:,3] = np.log10(x[:,3])
        x[:,4] = np.log10(x[:,4])
        
        y_samp = [] 
        for qphi in self.qphis:
            _samp = qphi.sample((int(n_sample/self._n_ensemble),),
                                x=torch.tensor(x, dtype=torch.float32).to(device),
                                show_progress_bars=False)
            y_samp.append(_samp.detach().cpu().numpy())

        return 10**np.array(y_samp).flatten()  
