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
from scipy.optimize import minimize
import os
import sys

from models import *

import pdb

font = {'weight' : 'normal',
        'size'   : 14}
labelsize = 18
matplotlib.rc('font', **font)

norm_factor = 0.15

''' First import A matrices & y-values
'''
path = './BroadbandMZIdata_to_Brando'

df=pd.read_csv(path+'/A1.csv', sep=',')
A1 = df.values
wavelengths = np.linspace(1550,1570,A1.shape[1])

df=pd.read_csv(path+'/A2.csv', sep=',')
A2 = df.values

""" Choose type of signal we want to use:
"""
options = ["MZI1", "MZI2", "MZI1MZI2", "W0000"]
signal_train = options[3]
signal_validate = options[0]

## get y1's
yfile = path+'/12-14-17_broadband_src_MZI/interferogram_'+str(signal_train)+'_v1.txt'
yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
yval_train, OPL = yf.values[:,1]/norm_factor, yf.values[:,0]
## get y2's
yfile = path+'/12-14-17_broadband_src_MZI/interferogram_'+str(signal_validate)+'_v1.txt'
yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
yval_validate, OPL = yf.values[:,1]/norm_factor, yf.values[:,0]

xfile = os.getcwd()+'/BroadbandMZIdata_to_Brando/12-14-17_broadband_src_MZI/'+str(signal_train)+'.CSV'
xf = pd.read_csv(xfile, header=30)
xval_train, xwl = xf.values[:,1], xf.values[:,0]
xwl = np.array([x - 0.7 for x in xwl])
x_real_train = np.interp(wavelengths, xwl, xval_train)
y_real_train = np.dot(A1, x_real_train)

xfile = os.getcwd()+'/BroadbandMZIdata_to_Brando/12-14-17_broadband_src_MZI/'+str(signal_validate)+'.CSV'
xf = pd.read_csv(xfile, header=30)
xval_validate, xwl = xf.values[:,1], xf.values[:,0]
xwl = np.array([x - 0.7 for x in xwl])
x_real_validate = np.interp(wavelengths, xwl, xval_validate)
y_real_validate = np.dot(A1, x_real_validate)


''' Pseudo-inverse method (for reference) '''
Ainv = np.linalg.pinv(A1)
x_pinv_train = np.dot(Ainv, yval_train)
x_pinv_validate = np.dot(Ainv, yval_validate)
x_pinv_train = normalize_vector(x_pinv_train, x_real_train)
x_pinv_validate = normalize_vector(x_pinv_validate, x_real_validate)

''' RBF training'''

N,D = A1.shape
std = (wavelengths[1] - wavelengths[0])
print('std = '+str(std))

def getRBFspectrum(y_input, step, wavelengths, x_real):
    wavelengths = wavelengths.reshape(D,1)
    C = get_rbf_coefficients(A=A2,X=wavelengths,centers=wavelengths,Y=y_input,std=step)
    def rbf(X,centers,std):
        beta = np.power(1.0/step,2)
        Kern = np.exp(-beta*euclidean_distances_manual(x=X,W=centers.transpose()))
        return np.dot(Kern,C)
    f_rbf = lambda a: rbf(a,wavelengths,step)
    x_rbf = f_rbf(wavelengths)
    x_rbf = normalize_vector(x_rbf, x_real)
    return (x_rbf, np.linalg.norm(x_rbf - x_real))

# Create the objective function to be optimized
objective = lambda s: getRBFspectrum(yval_train, s, wavelengths, x_real_train)[1]

res = minimize(objective, 1*std)
std_optimal = res.x[0]
print("std_optimal = "+str(std_optimal))
x_rbf_train = getRBFspectrum(yval_train, std_optimal, wavelengths, x_real_train)[0]

""" With std_optimal calculated, get RBF spectrum for validation set
"""
x_rbf_validate = getRBFspectrum(yval_validate, std_optimal, wavelengths, x_real_validate)[0]


''' plot errors from y ||x_pred - x_real||^2'''
error_real = np.linalg.norm( x_real_validate - x_real_validate,2)
error_pinv = np.linalg.norm( x_pinv_validate - x_real_validate,2)
error_rbf = np.linalg.norm( x_rbf_validate - x_real_validate,2)
print('Errors of reconstructions')
print('train_error_real = ||w_real - y||^2 = '+str(error_real))
print('train_error_pinv = ||w_pinv - y||^2 = '+str(error_pinv))
print('train_error_rbf = ||x_rbf - y||^2 = '+str(error_rbf))

''' plot errors from y ||Y_pred - Y_real||^2'''
train_error_real = np.linalg.norm( np.dot(A1,x_real_validate) - y_real_validate,2)
train_error_pinv = np.linalg.norm( np.dot(A1,x_pinv_validate) - y_real_validate,2)
train_error_rbf = np.linalg.norm( np.dot(A1,x_rbf_validate) - y_real_validate,2)
print('Train Errors')
print('train_error_real = ||A*x_real - y||^2 = '+str(train_error_real))
print('train_error_pinv = ||A*x_pinv - y||^2 = '+str(train_error_pinv))
print('train_error_rbf = ||A*x_rbf - y||^2 = '+str(train_error_rbf))
''' '''
plt_x_real, = plt.plot(wavelengths, x_real_validate, 'ro')
plt_x_rbf, = plt.plot(wavelengths, x_rbf_validate, 'bo')
plt_x_pinv, = plt.plot(wavelengths, x_pinv_validate, 'co')
plt.legend([plt_x_real,plt_x_rbf, plt_x_pinv],['x_real','x_rbf', 'x_pinv'])
plt.show()
