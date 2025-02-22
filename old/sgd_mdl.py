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

wavelengths = np.linspace(1550,1570,len(A1[0]))

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
print(f'||Xw - y||^2 = {train_error_pinv}')

''' SGD mdl '''
##dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
dtype = torch.FloatTensor
##
N,D = A1.shape
D_out = 1
''' Data set '''
a=Variable(torch.FloatTensor(wavelengths), requires_grad=False)
a_norm=a/a.sum()
X_train, Y_train = Variable(torch.FloatTensor(A1),requires_grad=False) , Variable(torch.FloatTensor(y_real.reshape(N,1)),requires_grad=False)
## reg params
''' RBF regularization '''
# center = 1558.0
# reg_l = 0.1
# A_param = Variable(torch.FloatTensor([A]), requires_grad=False)
# sigma_param = Variable(torch.FloatTensor([sigma]), requires_grad=False)
# t_param = Variable(torch.FloatTensor([center]), requires_grad=True)
# def get_reg(x, a,A_param,t_param,sigma_param,print_rx=False):
#     #pdb.set_trace()
#     D = len(a)
#     ''' compute weights according to traget function '''
#     R_x = A_param*torch.exp(-(a - t_param)**2/sigma_param**2)
#     R_x = 1/R_x
#     R_x = R_x.view(1,D)
#     if print_rx:
#         print(f'R_x={R_x.norm(2).data.numpy()}')
#     ''' compute x.^2 = [...,|x_i|^2,...]'''
#     x_2 = (x**2).t()
#     ''' Regularization R(f) = <R_f,x.^2>'''
#     R_f = R_x.mm(x_2)
#     return R_f
#     #return Variable( R_f, requires_grad=False)
''' NN weighting'''
H=1500
R_a_mdl = torch.nn.Sequential(
        torch.nn.Linear(D, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D)
    )
# H=D
# L1 = torch.nn.Linear(D, H)
# softmax = torch.nn.Softmax()
# R_a_mdl = torch.nn.Sequential(
#         L1,
#         softmax
#     )
M=1
eta=0.1
nb_iter = 500
#criterion = torch.nn.MSELoss()
#optimizer = torch.optim.SGD(R_a_mdl.parameters(), lr=0.1)
x_real = np.array(x_real)
# reciprocal_x_real = 1/x_real
# x_real_norm = (reciprocal_x_real/sum(reciprocal_x_real)).reshape(D,1)
#x_real = x_real/sum(x_real)

#R_x_works = 1/Variable(torch.Tensor(x_real/sum(x_real)))
R_x_works = 1/Variable(torch.Tensor(x_real))
#print(f'sum(x_real_norm)={sum(x_real_norm)}')
print(f'sum(R_x_works)={sum(R_x_works)}')

print(f'np.mean(y_real)={np.mean(y_real)}')
print(f'np.std(y_real)={np.std(y_real)}')

X_train_R_x = a_norm.view(1,D)
Y_train_R_x = Variable(torch.FloatTensor(1/x_real.reshape(D,1)),requires_grad=False)
#A, sigma = 0.85/70., 5.0 # 0.85/70., 5.0
#center = 1555.0 # 1560.0
#x_real_shifted = np.array( [A*np.exp(-(wl-center)**2/sigma**2) for wl in wavelengths] )
#Y_train_R_x = Variable(torch.FloatTensor( (1/x_real_shifted).reshape(D,1)),requires_grad=False)
#print(f'R_a_mdl={R_a_mdl(X_train_R_x).t()}')
#f_init_error = train(R_a_mdl, M,eta,nb_iter, dtype, X_train=X_train_R_x,Y_train=Y_train_R_x)
#print(f'f_init_error={f_init_error}')
# print(f'Y_train_R_x={Y_train_R_x}')
# print(f'R_a_mdl={R_a_mdl(X_train_R_x).t()}')
##
# reg_l = float(sum(x_real))
# reg_l = float(sum(y_real**2)/sum(x_real))
reg_l = 1
reciprocal_x_real = 1/1
print(f'reg_l={reg_l}')
def get_reg_softmax(x, a,R_a_mdl):
    #pdb.set_trace()
    D = len(a)
    ''' compute weighting R_x(a) = NN(x)'''
    R_x = R_a_mdl(a)
    R_x = R_x.view(1,D)
    ''' compute x.^2 = [...,|x_i|^2,...]'''
    x_2 = x**2
    x_2 = x_2.view(D,1)
    ''' Regularization R(f) = <R_f,x.^2>'''
    R_f = R_x.mm(x_2)
    #pdb.set_trace()
    return R_f
def fix_softmax(x, a,R_a_mdl):
    #R_x = Y_train_R_x
    R_x = R_a_mdl(a)
    R_x = R_x.view(1,D)
    ''' compute x.^2 = [...,|x_i|^2,...]'''
    x_2 = x**2
    x_2 = x_2.view(D,1)
    ''' Regularization R(f) = <R_f,x.^2>'''
    R_f = R_x.mm(x_2)
    return R_f
#R_a_mdl = lambda a_norm: 1/Y_train_R_x
#R_a_mdl = lambda a_norm: 1/Variable(torch.Tensor(x_real))
#R_a_mdl = lambda a_norm: 1/Variable(torch.Tensor(x_real**2))
R_a_mdl = lambda a_norm: 1/Variable(torch.Tensor(x_real/sum(x_real)))
R_x_params = NamedDict(a=a,R_a_mdl=R_a_mdl)
R_x = fix_softmax
# R_x_params = NamedDict(a=a_norm,R_a_mdl=R_a_mdl)
# R_x = get_reg_softmax
# R_x_params = NamedDict({'a':a,'A_param':A_param,'t_param':t_param,'sigma_param':sigma_param})
# R_x = get_reg
''' SGD mdl '''
bias=False
mdl_sgd = torch.nn.Sequential(torch.nn.Linear(D,D_out,bias=bias))
#mdl_sgd[0].weight.data.fill_(0)
#print(f'mdl_sgd[0].weight.data = {mdl_sgd[0].weight.data.numpy().shape}')
#mdl_sgd[0].weight.data = torch.FloatTensor(x_pinv.reshape(1,D))
''' train SGM '''
M = int(N/4)
eta = 0.000001
eta_R_x = 0.000000
nb_iter = 20000
R_x_np_before = R_a_mdl(a_norm).data.numpy().reshape(D,1)
train_errors,erm_errors = train_SGD2(mdl_sgd, M,eta,nb_iter, dtype, X_train,Y_train, reg_l,eta_R_x, R_x,R_x_params)
R_x_np_after = R_a_mdl(a_norm).data.numpy().reshape(D,1)
##
print(f'reg_l={reg_l}')

''' learned params '''
#print(f't_param={t_param.data.numpy()}')
##
''' plot weight 1/R_x'''
plt.figure()
plt.title('1/R_x') #Gaussian
# print(f'Y_train={x_real_norm}')
# print(f'R_x_np_before={R_x_np_before}')
plt_R_x_works, = plt.plot(wavelengths,1/R_x_works.data.numpy())
plt_before, = plt.plot(wavelengths,reciprocal_x_real*1/R_x_np_before)
plt_after, = plt.plot(wavelengths,reciprocal_x_real*1/R_x_np_after)
plt_x_real, = plt.plot(wavelengths,x_real) # note x_real is the RBF
plt.legend([plt_R_x_works, plt_before,plt_after,plt_x_real],['plt_R_x_works','plt_before','plt_after','plt_x_real'])
''' plot weight R_x'''
plt.figure()
plt.title('R_x = 1/x')
# print(f'Y_train={x_real_norm}')
# print(f'R_x_np_before={R_x_np_before}')
plt_R_x_works, = plt.plot(wavelengths,R_x_works.data.numpy())
plt_before, = plt.plot(wavelengths,R_x_np_before)
plt_after, = plt.plot(wavelengths,R_x_np_after)
plt_x_real, = plt.plot(wavelengths,1/x_real) # note 1/x_real is 1/RBF
plt.legend([plt_R_x_works, plt_before,plt_after,plt_x_real],['plt_R_x_works','plt_before','plt_after','plt_x_real'])
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
plt_mdl_recon, = plt.plot(wavelengths, mdl_sgd[0].weight.data.numpy().reshape((D,)) )
plt.legend([plt_real_recon,plt_mdl_recon],['plt_real_recon','plt_mdl_recon'])
# ''' plot params stats '''
# plt.figure()
# plt.plot(np.arange(0,nb_iter+1),t_params)
''' plot show '''
seconds = (time.time() - start_time)
minutes = seconds/ 60
hours = minutes/ 60
print("--- %s seconds ---" % seconds )
print("--- %s minutes ---" % minutes )
print("--- %s hours ---" % hours )
print('\a \a \a')
plt.show()

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
