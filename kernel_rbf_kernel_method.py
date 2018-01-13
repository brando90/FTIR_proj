import time
start_time = time.time()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
import os
import sys

import torch
from torch.autograd import Variable
from torch import autograd

from mdl_trainer import *
from maps import NamedDict

import pdb

font = {'weight' : 'normal',
        'size'   : 14}
labelsize = 18
matplotlib.rc('font', **font)

norm_factor = 0.15

''' First import A matrices & y-values
'''
path = '../BroadbandMZIdata_to_Brando'

df=pd.read_csv(path+'/A1.csv', sep=',')
A1 = df.values
df=pd.read_csv(path+'/A2.csv', sep=',')
A2 = df.values

## get y1's
yfile = path+'/12-14-17_broadband_src_MZI/interferogram_MZI1_v2.txt'
yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
yval1, OPL = yf.values[:,1]/norm_factor, yf.values[:,0]
## get y2's
yfile = path+'/12-14-17_broadband_src_MZI/interferogram_MZI1_v5.txt'
yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
yval2, OPL = yf.values[:,1]/norm_factor, yf.values[:,0]

wavelengths = np.linspace(1550,1570,A1.shape[1])

# wavelengths = np.linspace(1550,1570,10)
# A1 = np.arange(64*10).reshape(64,10)
'''  train '''
A, sigma = 0.85/70., 5.0
center = 1560.0
x_real = [A*np.exp(-(wl-center)**2/sigma**2) for wl in wavelengths]
y_real= np.dot(A1, x_real)
#plt.show()

''' Pseudo-inverse method (for reference) '''
Ainv = np.linalg.pinv(A1)
#x_pinv = np.dot(Ainv, yval1)
#x_pinv = np.dot(Ainv, y_real)
##
train_error_pinv = np.linalg.norm( np.dot(A1,x_pinv) - y_real,2)
print(f'train_error_pinv = ||Xw_pinv - y||^2 = {train_error_pinv}')
''' '''
# A, sigma = 0.85/70., 5.0
# x_real = [A*np.exp(-(wl-1560.0)**2/sigma**2) for wl in wavelengths]
# y_real= np.dot(A1, x_real)
# plt.plot(wavelengths, x_real)
# plt.show()
xfile = os.getcwd()+'../BroadbandMZIdata_to_Brando/12-14-17_broadband_src_MZI/MZI1.CSV'
xf = pd.read_csv(xfile, header=30)
xval, xwl = xf.values[:,1], xf.values[:,0]
xwl = np.array([x - 0.7 for x in xwl])
x_real = np.interp(wavelengths, xwl, xval)
y_real = np.dot(A1, x_real)

plt.plot(xwl, xval, 'ro')
plt.plot(wavelengths, x_real, 'bo')
plt.show()

#########
''' plot training results '''
''' reconstructions '''
plt.figure()
plt.title('reconstructions')
plt_real_recon,= plt.plot(wavelengths, x_real)
y_pred = mdl_x_recon(a.t()).t().data.numpy()
plt_mdl_recon, = plt.plot(wavelengths, y_pred)
plt.legend([plt_real_recon,plt_mdl_recon],['plt_real_recon','plt_mdl_recon'])
''' plot and end script show '''
seconds = (time.time() - start_time)
minutes = seconds/ 60
hours = minutes/ 60
print("--- %s seconds ---" % seconds )
print("--- %s minutes ---" % minutes )
print("--- %s hours ---" % hours )
print('\a \a \a')
plt.show()
