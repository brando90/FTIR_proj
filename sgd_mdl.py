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

from mdl_trainer import train_SGD

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

wavelengths = np.linspace(1550,1570,len(A1[0]))

'''  train '''
A, sigma = 0.85/70., 5.0
center = 1560.0
x_real = [A*np.exp(-(wl-center)**2/sigma**2) for wl in wavelengths]
y_real= np.dot(A1, x_real)
plt.plot(wavelengths, x_real)
#plt.show()

''' Pseudo-inverse method (for reference) '''
Ainv = np.linalg.pinv(A1)
x_pinv = np.dot(Ainv, yval1)
plt.plot(wavelengths, x_pinv)

''' SGD '''
##dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
dtype = torch.FloatTensor
##
N,D = A1.shape
D_out = 1
bias=True
''' Data set '''
a = Variable(torch.FloatTensor(wavelengths), requires_grad=False)
X_train, Y_train = Variable(torch.FloatTensor(A1),requires_grad=False) , Variable(torch.FloatTensor(y_real.reshape(N,1)),requires_grad=False)
## reg params
reg_l = 1
A_param = Variable(torch.FloatTensor([A]), requires_grad=False)
sigma_param = Variable(torch.FloatTensor([sigma]), requires_grad=True)
t_param = Variable(torch.FloatTensor([center]), requires_grad=True)
def get_reg(a,A_param,t_param,sigma_param):
    R_x = A_param*torch.exp(-(a - t_param)**2/sigma_param**2)
    R_x = 1/R_x
    return R_x
R_x = None
R_x = get_reg(a,A_param,t_param,sigma_param)
## SGD mdl
M = N
eta = 0.01
nb_iter = 1000
mdl_sgd = torch.nn.Sequential(torch.nn.Linear(D,D_out,bias=bias))
#mdl_sgd.linear_layers[1].weight.data.fill_(0)
##
train_SGD(mdl_sgd, M,eta,nb_iter, dtype, X_train,Y_train, reg_l,R_x)


#
# ratio = np.max(yval1)/np.max(y_real)
# print("power ratio = "+str(ratio))
# ''' See how close real interferogram is to measured interferogram '''
# plt.plot(y_real*ratio, 'ro--', label="expected")
# plt.plot(yval1, 'ko-', label="measured")
# plt.legend(loc='best')
# plt.show()
#
# ''' Begin RIDGE parameter search
# '''
# alpha_list = np.logspace(-3,5,400)
# r2_list = []
# for alpha in alpha_list:
#     ''' Search for a suitable alpha '''
#     ridge = Ridge(alpha=alpha)
#     y_pred_ridge = ridge.fit(A1, yval1).predict(A1)
# #    r2_list.append(r2_score(yval2, np.dot(A2, ridge.coef_)))
# #    Uncomment below to optimize wrt real spectrum
#     r2_list.append(r2_score(x_real/np.max(x_real), ridge.coef_/np.max(ridge.coef_)))
#
# plt.semilogx(alpha_list, r2_list)
# plt.title("Ridge Regression Model Fitting")
# plt.xlabel("Alpha")
# plt.ylabel("R2 value")
# plt.show()
#
# max_index = r2_list.index(max(r2_list))
# alpha_max = alpha_list[max_index]
#
# ridge = Ridge(alpha=alpha_max)
# print r2_list
# print "alphamax = "+str(alpha_max)
# y_pred_ridge = ridge.fit(A2, yval2).predict(A2)
# r2_score_ridge = r2_score(x_real/np.max(x_real), ridge.coef_/np.max(ridge.coef_))
#
# print(ridge)
# print("r^2 on test data : %f" % r2_score_ridge)
#
# plt.figure(figsize=(8, 6))
# plt.subplot(2,1,1)
# plt.plot(wavelengths, ridge.coef_/np.max(ridge.coef_), 'r-', label="Ridge regression", linewidth=2.0)
# plt.legend(loc='best')
# plt.subplot(2,1,2)
# plt.plot(wavelengths, x_real/np.max(x_real), 'k-', label="Measured", linewidth=2.0)
# plt.xlabel("Wavelength [nm]", fontsize=labelsize)
# plt.ylabel("Amplitude [a.u.]", fontsize=labelsize)
# maxy, miny = 1.0, 0.0
# plt.ylim([miny - 0.05*abs(maxy-miny), maxy + 0.05*abs(maxy-miny)])
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()
