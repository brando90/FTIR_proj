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

from mdl_trainer import train_SGD
from mdl_trainer import train_SGD2
from mdl_trainer import train_SGD3
from mdl_trainer import train
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
x_pinv = np.dot(Ainv, y_real)

##
train_error_pinv = np.linalg.norm( np.dot(A1,x_pinv) - y_real,2)
print(f'train_error_pinv = ||Xw_pinv - y||^2 = {train_error_pinv}')

''' SGD mdl '''
##dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
dtype = torch.FloatTensor
##
N,D = A1.shape
D_out = 1
''' Data set '''
a=(wavelengths - np.mean(wavelengths))/np.std(wavelengths)
a=Variable(torch.FloatTensor(a.reshape(D,1)), requires_grad=False)
A1 = Variable(torch.FloatTensor(A1),requires_grad=False)
X_train = A1
Y_train = Variable(torch.FloatTensor(y_real.reshape(N,1)),requires_grad=False)
x_real = np.array(x_real)
''' NN mdl W_L[W_L-1[...[W1a = y'''
H=1500
mdl_x_recon = torch.nn.Sequential(
        torch.nn.Linear(D, H), # L1 = W1
        torch.nn.ReLU(), # L2 = ReLU
        torch.nn.Linear(H, D,bias=False) # L3 = W2
    )
#mdl_sgd[0].weight.data.fill_(0)
#mdl_sgd[1].weight.data.fill_(0)
full_mdl = torch.nn.Sequential(
    mdl_x_recon,
    ##
    torch.nn.Linear(D, N,bias=False) ## L2 = W3
)
full_mdl[1].weight.data = A1.data
full_mdl[1].weight.requires_grad = False
#full_mdl[1].bias.requires_grad = False
''' train SGM '''
M = int(N)
eta = 0.001
nb_iter = 200
train_errors,erm_errors = train_SGD3(full_mdl,a, M,eta,nb_iter, dtype, X_train,Y_train)
#########
''' plot training results '''
plt.figure()
plt.title('train error vs iterations')
plt_erm, = plt.plot(np.arange(0,nb_iter+1),erm_errors)
plt_train, = plt.plot(np.arange(0,nb_iter+1),train_errors)
plt.legend([plt_erm,plt_train],['ERM','Train'])
''' reconstructions '''
plt.figure()
plt.title('reconstructions')
plt_real_recon,= plt.plot(wavelengths, x_real)
y_pred = full_mdl(a.t()).t().data.numpy()
plt_mdl_recon, = plt.plot(wavelengths,  )
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
