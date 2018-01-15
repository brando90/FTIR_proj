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

## get y1's
yfile = path+'/12-14-17_broadband_src_MZI/interferogram_MZI1_v2.txt'
yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
yval1, OPL = yf.values[:,1]/norm_factor, yf.values[:,0]
## get y2's
yfile = path+'/12-14-17_broadband_src_MZI/interferogram_MZI1_v5.txt'
yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
yval2, OPL = yf.values[:,1]/norm_factor, yf.values[:,0]

wavelengths = np.linspace(1550,1570,A1.shape[1])
''' '''
xfile = os.getcwd()+'/BroadbandMZIdata_to_Brando/12-14-17_broadband_src_MZI/MZI1.CSV'
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
std = wavelengths[1] - wavelengths[0]
print(f'std={std}')
wavelengths = wavelengths.reshape(D,1)
C = get_rbf_coefficients(A=A1,X=wavelengths,centers=wavelengths,Y=y_real,std=std)
def rbf(X,centers,std):
    beta = np.power(1.0/std,2)
    Kern = np.exp(-beta*euclidean_distances_manual(x=X,W=centers.transpose()))
    return np.dot(Kern,C)
f_rbf = lambda a: rbf(a,wavelengths,std)
x_rbf = f_rbf(wavelengths)
''' plot errors from y ||Y_pred - Y_real||^2'''
train_error_pinv = np.linalg.norm( np.dot(A1,x_pinv) - y_real,2)
train_error_rbf = np.linalg.norm( np.dot(A1,x_rbf) - y_real,2)
print(f'train_error_pinv = ||Xw_pinv - y||^2 = {train_error_pinv}')
print(f'train_error_rbf = ||Xw_rbf - y||^2 = {train_error_rbf}')
''' '''
plt_xval, = plt.plot(xwl, xval, 'ro')
plt_x_real, = plt.plot(wavelengths, x_real, 'bo')
plt_x_rbf, = plt.plot(wavelengths, x_real, 'go')
plt.legend([plt_xval,plt_x_real,plt_x_rbf],['xval','x_real','x_rbf'])
plt.show()
