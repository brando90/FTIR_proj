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
signal_train = options[2]
signal_validate = options[0]

## get y1's
yfile = path+'/12-14-17_broadband_src_MZI/interferogram_'+str(signal_train)+'_v1.txt'
yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
yval_train, OPL = yf.values[:,1], yf.values[:,0]
## get y2's
yfile = path+'/12-14-17_broadband_src_MZI/interferogram_'+str(signal_validate)+'_v1.txt'
yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
yval_validate, OPL = yf.values[:,1], yf.values[:,0]

xfile = path+'/12-14-17_broadband_src_MZI/'+str(signal_train)+'.CSV'
xf = pd.read_csv(xfile, header=30)
xval_train, xwl = xf.values[:,1], xf.values[:,0]
xwl = np.array([x - 0.7 for x in xwl])
x_real_train = np.interp(wavelengths, xwl, xval_train)

xfile = path+'/12-14-17_broadband_src_MZI/'+str(signal_validate)+'.CSV'
xf = pd.read_csv(xfile, header=30)
xval_validate, xwl = xf.values[:,1], xf.values[:,0]
xwl = np.array([x - 0.7 for x in xwl])
x_real_validate = np.interp(wavelengths, xwl, xval_validate)


''' Pseudo-inverse method (for reference) '''
Ainv = np.linalg.pinv(A1)
x_pinv_train = np.dot(Ainv, yval_train)
x_pinv_validate = np.dot(Ainv, yval_validate)

''' RBF training via cross-validation'''
N,D = A1.shape
std = (wavelengths[1] - wavelengths[0])
print('std = '+str(std))

def getRBFspectrum(y_input1, step, wavelengths):
    wavelengths = wavelengths.reshape(D,1)
    
    C = get_rbf_coefficients(A=A1,X=wavelengths,centers=wavelengths,Y=y_input1,std=step)
    def rbf(X, c, centers,std):
        beta = np.power(1.0/step,2)
        Kern = np.exp(-beta*euclidean_distances_manual(x=X,W=centers.transpose()))
        return np.dot(Kern,c)
    f_rbf = lambda a: rbf(a, C, wavelengths,step)
    x_rbf = f_rbf(wavelengths)
    x_trained = normalize_vector(x_rbf, x_real_train)
    
    return (x_rbf, r2_score(x_trained, x_real_train))

score_list = []
srange = np.logspace(-2,1.5,200)*std
mx, smax = 0, 0
for s in srange:
    score = getRBFspectrum(yval_train, s, wavelengths)[1]
    score_list.append(score)
    if score > mx:
        mx = score
        smax = s

std_optimal = smax
print("std_optimal = "+str(std_optimal))

""" With std_optimal calculated, get RBF spectrum for validation set
"""
x_rbf_validate = getRBFspectrum(yval_validate, std_optimal, wavelengths)[0]

""" Normalize the real spectrum to reconstructed spectra
"""
x_real_validate = normalize_vector(x_real_validate, x_rbf_validate)

''' plot errors from y ||x_pred - x_real||^2'''
error_real = np.linalg.norm( x_real_validate - x_real_validate,2)
error_pinv = np.linalg.norm( x_pinv_train - x_real_validate,2)
error_rbf = np.linalg.norm( x_rbf_validate - x_real_validate,2)
print('Errors of reconstructions')
print('train_error_real = ||x_real - x||^2 = '+str(error_real))
print('train_error_pinv = ||x_pinv - x||^2 = '+str(error_pinv))
print('train_error_rbf = ||x_rbf - x||^2 = '+str(error_rbf))
print('R2 train_error_real = R^2 = '+str(r2_score(x_real_validate, x_real_validate)))
print('R2 train_error_pinv = R^2 = '+str(r2_score(x_pinv_train, x_real_validate)))
print('R2 train_error_rbf = R^2 = '+str(r2_score(x_rbf_validate, x_real_validate)))

''' '''
#plt_x_real, = plt.plot(wavelengths, x_real, 'ro')
#plt_x_rbf, = plt.plot(wavelengths, x_rbf_validate, 'bo')
#plt_x_pinv, = plt.plot(wavelengths, x_pinv_validate, 'co')
#plt.legend([plt_x_real,plt_x_rbf, plt_x_pinv],['x_real','x_rbf', 'x_pinv'])
#plt.show()

##plt_y_real, = plt.plot( y_real, 'ro')
#plt_y_rbf, = plt.plot( np.dot(A1, x_rbf_validate), 'bo')
#plt_y_pinv, = plt.plot( np.dot(A1, x_pinv_validate), 'co')
#plt.legend([plt_y_rbf, plt_y_pinv],['y_rbf', 'y_pinv'])
#plt.show()

""" Uncomment below for separate plot (comparison) """
plt.subplot(2,1,1)
plt_x_real, = plt.plot(wavelengths, x_real_validate/max(x_real_validate), 'k-', linewidth=2)
plt.legend([plt_x_real],['Reference'])
plt.ylabel("Intensity [a.u.]")
plt.subplot(2,1,2)
plt_x_rbf, = plt.plot(wavelengths, x_rbf_validate/max(x_rbf_validate), 'r-', linewidth=2)
#plt_x_pinv, = plt.plot(wavelengths, x_pinv_validate, 'co')
plt.legend([plt_x_rbf],['RBF Network'])
plt.xlabel("Wavelength [nm]")
plt.ylabel("Intensity [a.u.]")
plt.show()
