'''



'''
import os, sys
import pickle 
import numpy as np 
from scipy.spatial import ConvexHull

fcovars = sys.argv[1]
hull_name = sys.argv[2]


# read covariates 
covars = np.load(fcovars)

# get convexhull
hull = ConvexHull(covars)

# save hull (overwrites by default) 
pickle.dump(hull, open(hull_name, 'wb'))
