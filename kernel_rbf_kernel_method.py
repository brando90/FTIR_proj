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
df=pd.read_csv(path+'/A2.csv', sep=',')
A2 = df.values

""" Choose type of signal we want to use:
"""
options = ["MZI1", "MZI2", "MZI1MZI2"]
signal = options[2]

## get y1's
yfile = path+'/12-14-17_broadband_src_MZI/interferogram_'+str(signal)+'_v2.txt'
yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
yval1, OPL = yf.values[:,1]/norm_factor, yf.values[:,0]
## get y2's
yfile = path+'/12-14-17_broadband_src_MZI/interferogram_'+str(signal)+'_v5.txt'
yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
yval2, OPL = yf.values[:,1]/norm_factor, yf.values[:,0]

wavelengths = np.linspace(1550,1570,A1.shape[1])
xfile = os.getcwd()+'/BroadbandMZIdata_to_Brando/12-14-17_broadband_src_MZI/'+str(signal)+'.CSV'
xf = pd.read_csv(xfile, header=30)
xval, xwl = xf.values[:,1], xf.values[:,0]
xwl = np.array([x - 0.7 for x in xwl])

x_real = np.interp(wavelengths, xwl, xval)
y_real = np.dot(A1, x_real)
''' Pseudo-inverse method (for reference) '''
Ainv = np.linalg.pinv(A1)
x_pinv = np.dot(Ainv, yval1)

''' RBF training'''
N,D = A1.shape
std = (wavelengths[1] - wavelengths[0])
print('std='+str(std))

wavelengths = wavelengths.reshape(D,1)
C = get_rbf_coefficients(A=A1,X=wavelengths,centers=wavelengths,Y=yval1,std=std)
def rbf(X,centers,std):
    beta = np.power(1.0/std,2)
    Kern = np.exp(-beta*euclidean_distances_manual(x=X,W=centers.transpose()))
    return np.dot(Kern,C)
f_rbf = lambda a: rbf(a,wavelengths,std)
x_rbf = f_rbf(wavelengths)
x_rbf = normalize_vector(x_rbf, x_real)

''' plot errors from y ||Y_pred - Y_real||^2'''
error_real = np.linalg.norm( x_real - x_real,2)
error_pinv = np.linalg.norm( x_pinv - x_real,2)
error_rbf = np.linalg.norm( x_rbf - x_real,2)
print('Errors of reconstructions')
print('train_error_pinv = ||w_real - y||^2 = '+str(error_real))
print('train_error_pinv = ||w_pinv - y||^2 = '+str(error_pinv))
print('train_error_rbf = ||x_rbf - y||^2 = '+str(error_rbf))
''' plot errors from y ||Y_pred - Y_real||^2'''
train_error_real = np.linalg.norm( np.dot(A1,x_real) - y_real,2)
train_error_pinv = np.linalg.norm( np.dot(A1,x_pinv) - y_real,2)
train_error_rbf = np.linalg.norm( np.dot(A1,x_rbf) - y_real,2)
print('Train Errors')
print('train_error_pinv = ||A*w_real - y||^2 = '+str(train_error_real))
print('train_error_pinv = ||A*w_pinv - y||^2 = '+str(train_error_pinv))
print('train_error_rbf = ||A*x_rbf - y||^2 = '+str(train_error_rbf))
''' '''
plt_x_real, = plt.plot(wavelengths, x_real, 'ro')
plt_x_rbf, = plt.plot(wavelengths, x_rbf, 'bo')
plt.legend([plt_x_real,plt_x_rbf],['x_real','x_rbf'])
#plt_x_pinv, = plt.plot(wavelengths, x_pinv, 'co')
#plt.legend([plt_xval,plt_x_real,plt_x_rbf,plt_x_pinv],['xval','x_real','x_rbf','plt_x_pinv'])
plt.show()
