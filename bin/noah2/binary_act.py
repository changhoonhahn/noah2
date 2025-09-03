'''
'''
import os, sys 
import numpy as np

from astropy.table import Table

from noah2 import data as D
from noah2 import util as U

from causalflow import causalflow
from causalflow import support as Support

import torch
################################################
# input 
################################################
arch = sys.argv[1]
exp0 = int(sys.argv[2])
exp1s = [int(exp) for exp in (sys.argv[3]).split(',')]

################################################
# read data 
################################################
# read CRS participant data
DNoah = D.Noah2()
fema = DNoah._read_data_full()
is_participant = DNoah._participants(fema)
fema = fema[is_participant]

metro = ((fema['RUCA1'] == 1))# | (fema['RUCA1'] == 2) | (fema['RUCA1'] == 3)) 
micro = ((fema['RUCA1'] == 4) | (fema['RUCA1'] == 5) | (fema['RUCA1'] == 6))
small = ((fema['RUCA1'] == 7) | (fema['RUCA1'] == 8) | (fema['RUCA1'] == 9))
rural = ((fema['RUCA1'] == 10))

arch_dict =  {'rural': rural, 'small': small, 'micro': micro, 'metro': metro,
              'all': np.ones(len(fema)).astype(bool)}

columns = DNoah._columns()[:8] + ['s_%s' % c for c in ['c350', 'c420', 'c450']]
column_labels = np.array(DNoah._column_labels()[:8] + ['c350', 'c420', 'c450'])

data = np.array([np.array(fema[col]) for col in columns]).T
_data = data.copy()
_data[:,0] = np.log10(_data[:,0])
_data[:,3] = np.log10(_data[:,3])
_data[:,4] = np.log10(_data[:,4])


binary_activities = ['c360', 'c520', 'c530', 'c540', 'c610', 'c620', 'c630']

binary_data = np.array([np.array(fema[col]) for col in binary_activities]).T
binary_data = (binary_data > 0).astype(int)

binary_act_codes = np.zeros(len(fema))
for i in range(binary_data.shape[1]):
    binary_act_codes += 2**i * binary_data[:,i]

################################################
# 
################################################
for exp1 in exp1s: 
    exp = [exp0, exp1]

    print('%i - % i' % (exp[0], exp[1]))

    # load support for base sample
    Supp = Support.Support()
    Supp.load_optuna('support.%s.%i' % (arch, exp[0]), '/scratch/gpfs/chhahn/noah/noah2/qphi/', verbose=True)

    if exp[0] < exp[1]: base = 'control'
    else: base = 'treated'
    Cflow = causalflow.CausalFlowB(base)

    # load support
    Cflow.load_support(Supp)

    # load flows
    Cflow.load_flows_optuna('%s.%i' % (arch, exp[0]), '/scratch/gpfs/chhahn/noah/noah2/qphi/', verbose=True)

    # get reference data
    ref_data = _data[(binary_act_codes == exp[1]) & arch_dict[arch]]
    Y_ref = ref_data[:,0][:,None]
    X_ref = ref_data[:,1:]
    zipcode_ref = fema['zipcode'][(binary_act_codes == exp[1]) & arch_dict[arch]]

    print('%i CTE evaluations' % X_ref.shape[0])

    # check support
    support = Cflow.support_base.check_support(X_ref, Nsample=10000, return_support=True)

    # calculate Conditional Treatment Effect
    ctes = []
    for _X, _Y in zip(X_ref, Y_ref):
        cte = Cflow.CTE(_X, _Y, Nsample=10000, Nsupport=10000,
                       support_threshold=0.95, transf=lambda x: 10**x)
        ctes.append(cte)
    ctes = np.array(ctes)
    
    output = Table() 
    output['Y_ref']     = Y_ref
    output['X_ref']     = X_ref
    output['zipcode']   = zipcode_ref
    output['CTE']       = ctes
    output['p_support'] = support
    
    output.write("/scratch/gpfs/chhahn/noah/noah2/exps/%s.%i_%i.hdf5" % (arch, exp0, exp1)) 
