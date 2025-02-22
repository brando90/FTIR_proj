#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:07:33 2018

@author: dkita
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
import os
import sys
from models import *
    
font = {'weight' : 'normal',
#        'name' : 'Arial',
        'size'   : 14}
labelsize = 16
ticksize=14
matplotlib.rc('font', **font)

""" First import A matrices & y-values 
"""
path = './BroadbandMZIdata_to_Brando'

df=pd.read_csv(path+'/A1.csv', sep=',')
A1 = df.values
df=pd.read_csv(path+'/A2.csv', sep=',')
A2 = df.values

wavelengths = np.linspace(1550,1570,A1.shape[1])

""" Choose type of signal we want to use:
"""
options = ["MZI1", "MZI2", "MZI1MZI2", "W0000"]
signal_train = options[0]
signal_validate = options[2]

yfile = path+'/12-14-17_broadband_src_MZI/interferogram_'+signal_train+'_v1.txt'
yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
yval_train, OPL = yf.values[:,1], yf.values[:,0]
yfile = path+'/12-14-17_broadband_src_MZI/interferogram_'+signal_validate+'_v1.txt'
yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
yval_validate, OPL = yf.values[:,1], yf.values[:,0]


""" UNCOMMENT area below only if desired spectrum is KNOWN beforehand """
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

#""" Begin ELASTIC NET parameter search
#"""
#l1_list = np.logspace(-4, 2, 50)
#alpha_list = np.logspace(-4, 2, 50)
#mv = 0.0 #max value of R2
#amax, l1max = 0, 0
#r2_list = []
#for l1 in l1_list:
#    r2_list.append([])
#    for alpha in alpha_list:
#        enet = ElasticNet(alpha=alpha, l1_ratio=l1, positive=True)
#        y_pred_enet = enet.fit(A1, yval_train).predict(A1)
##        score = r2_score(yval_validate, np.dot(A2, enet.coef_))
#        
##        Uncomment below to optimize wrt real spectrum
#        if np.max(enet.coef_) >= 1E-8:
#            x_trained = normalize_vector(enet.coef_, x_real_train)
#            score = r2_score(x_real_train, x_trained)
#        else:
#            score = 0
#            
#        r2_list[-1].append(score)
#        if score>mv:
#            mv = score
#            amax, l1max = alpha, l1
#            
#if amax==0 and l1max==0:
#    sys.exit("No maximum found.  All hyperparameter values gave r2 values < 0")
#            
#print "alphamax = "+str(amax)+",  l1max = "+str(l1max)
#    
#fig, ax = plt.subplots()
#ax.matshow(np.array(r2_list), aspect=1, cmap=matplotlib.cm.afmhot)
#plt.xlabel("Alpha")
#ax.xaxis.tick_top()
#ax.xaxis.set_label_position('top')
#plt.ylabel("L1")
#plt.show()
#
#""" ELASTIC NET LEARNED PARAMETERS
#    Learned from MZI1MZI2.csv """
amax = 0.152641796718
l1max = 0.000542867543932

#   Learned from MZI1
#amax = 0.065512855686
#l1max = 0.000175751062485

enet = ElasticNet(alpha=amax, l1_ratio=l1max, positive=True)
y_pred_enet = enet.fit(A1, yval_validate).predict(A1)

""" Normalize the real spectrum to reconstructed spectra
"""
x_real = normalize_vector(x_real_validate, enet.coef_)

r2_score_enet = r2_score(x_real/max(x_real), enet.coef_/max(enet.coef_))
l2_score_enet = np.linalg.norm(x_real/max(x_real) - enet.coef_/max(enet.coef_), 2)

print(enet)
print("r^2 result for enet : %f" % r2_score_enet)
print("L2-norm result for enet: %f" % l2_score_enet)

r2_score_pinv = r2_score(x_real, x_pinv_train)
l2_score_pinv = np.linalg.norm(x_real - x_pinv_train, 2)
print("r^2 result for pseudoinverse : %f" % r2_score_pinv)
print("L2-norm result for pseudoinverse: %f" % l2_score_pinv)

"""============================================================================
Plot on separate axes
============================================================================"""
#plt.figure(figsize=(8,6))
#plt.subplot(2,1,1)
#plt.plot(wavelengths, enet.coef_/np.max(enet.coef_), 'r-', label="64-ch dFT", linewidth=2.0)
#plt.ylabel("Intensity [a.u.]", fontsize=labelsize)
#plt.xticks(fontsize=ticksize)
#plt.yticks(fontsize=ticksize)
#plt.legend(loc='best')
#maxy, miny = 1.0, 0.0
#plt.ylim([miny - 0.05*abs(maxy-miny), maxy + 0.05*abs(maxy-miny)])
#plt.subplot(2,1,2)
#plt.plot(wavelengths, x_real/np.max(x_real), 'k-', label="reference", linewidth=2.0)
#plt.xticks(fontsize=ticksize)
#plt.yticks(fontsize=ticksize)
#plt.xlabel("Wavelength [nm]", fontsize=labelsize)
#plt.ylabel("Intensity [a.u.]", fontsize=labelsize)
#maxy, miny = 1.0, 0.0
#plt.ylim([miny - 0.05*abs(maxy-miny), maxy + 0.05*abs(maxy-miny)])
#plt.legend(loc='best')
#plt.tight_layout()

#plt.subplot(3,1,3)
#plt.plot(wavelengths, x_pinv_validate/np.max(x_pinv_validate), 'b-', label="PINV", linewidth=2.0)
#plt.xticks(fontsize=ticksize)
#plt.yticks(fontsize=ticksize)
#plt.xlabel("Wavelength [nm]", fontsize=labelsize)
#plt.ylabel("Intensity [a.u.]", fontsize=labelsize)
#maxy, miny = 1.0, 0.0
#plt.ylim([miny - 0.05*abs(maxy-miny), maxy + 0.05*abs(maxy-miny)])
#plt.legend(loc='best')

"""============================================================================
Plot overlapping one another
============================================================================"""
plt.figure(figsize=(6,4))

plt.plot(wavelengths, x_real/np.max(x_real), 'k-', label="OSA", linewidth=2.0)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.xlabel("Wavelength [nm]", fontsize=labelsize)
plt.ylabel("Intensity [a.u.]", fontsize=labelsize)
maxy, miny = 1.0, 0.0
#plt.ylim([miny - 0.05*abs(maxy-miny), maxy + 0.05*abs(maxy-miny)])
plt.xlim([1550,1570])
plt.ylim([0,1])

plt.plot(wavelengths, enet.coef_/np.max(enet.coef_), 'r-', label="64-ch dFT", linewidth=2.0)
plt.ylabel("Intensity [a.u.]", fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.legend(loc='best')
maxy, miny = 1.0, 0.0
#plt.ylim([miny - 0.05*abs(maxy-miny), maxy + 0.05*abs(maxy-miny)])
plt.xlim([1550,1570])
plt.ylim([0,1])


plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(3,2))
plt.plot(wavelengths, x_real_train/np.max(x_real_train), 'k-', linewidth=2.0)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
#plt.xlabel("Wavelength [nm]", fontsize=labelsize)
#plt.ylabel("Intensity [a.u.]", fontsize=labelsize)
plt.xlim([1550,1570])
plt.ylim([0,1])
plt.tight_layout()
plt.show()