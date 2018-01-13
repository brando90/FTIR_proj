#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:15:46 2017

@author: dkita
"""

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

font = {'weight' : 'normal',
        'size'   : 14}
labelsize = 18
matplotlib.rc('font', **font)

norm_factor = 0.15

def get_index(wl, wavelengths):
    # for an float "wl" on range[1550, 1570] and list "wavelengths" from [1550,1570],
    # returns the index of "wavelengths" that is closest to the input "wl"
    i, = np.where(wavelengths==(min(wavelengths, key=lambda x:abs(x-wl))))
    return i[0]

def set_negative_to_zeros(vector):
    for i in xrange(len(vector)):
        if vector[i]<0:
            vector[i]=0
    return vector

""" First import A matrices & y-values
"""
df=pd.read_csv('A1.csv', sep=',')
A1 = df.values
df=pd.read_csv('A3.csv', sep=',')
A2 = df.values

#yfile = os.getcwd()+'/12-12-17_broadband_src_1560cent/1560src_filtered_2nd_v1.txt'
yfile = os.getcwd()+'/12-14-17_broadband_src_MZI/interferogram_MZI1_v2.txt'
yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
yval1, OPL = yf.values[:,1]/norm_factor, yf.values[:,0]
#yfile = os.getcwd()+'/12-12-17_broadband_src_1560cent/1560src_filtered_2nd_v1.txt'
yfile = os.getcwd()+'/12-14-17_broadband_src_MZI/interferogram_MZI1_v5.txt'
yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
yval2, OPL = yf.values[:,1]/norm_factor, yf.values[:,0]

wavelengths = np.linspace(1550,1570,len(A1[0]))


""" UNCOMMENT area below only if desired spectrum is KNOWN beforehand """
A, sigma = 0.85/70., 5.0
x_real = [A*np.exp(-(wl-1560.0)**2/sigma**2) for wl in wavelengths]
y_real= np.dot(A1, x_real)
plt.plot(wavelengths, x_real)
plt.show()

xfile = os.getcwd()+'/12-14-17_broadband_src_MZI/MZI1.CSV'
xf = pd.read_csv(xfile, header=30)
xval, xwl = xf.values[:,1], xf.values[:,0]
xwl = np.array([x - 0.7 for x in xwl])
x_real = np.interp(wavelengths, xwl, xval)
y_real = np.dot(A1, x_real)

plt.plot(xwl, xval, 'ro')
plt.plot(wavelengths, x_real, 'bo')
plt.show()

#xfile = os.getcwd()+'/12-14-17_broadband_src_MZI/W0003.CSV'
#xf = pd.read_csv(xfile, header=30)
#xval2, xwl2 = xf.values[:,1], xf.values[:,0]
#xwl = np.array([x - 0.7 for x in xwl])
#x_real2 = np.interp(wavelengths, xwl, xval)
#y_real2 = np.dot(A1, x_real)
#
#plt.plot(xwl, xval/np.max(xval), 'ro')
#plt.plot(xwl2, xval2/np.max(xval2), 'bo')
#plt.plot(wavelengths, x_real, 'bo')
#plt.show()
#plt.plot(xwl, [xval[i]*xval2[i] for i in xrange(len(xval))], 'ro')
#plt.show()

""" Pseudo-inverse method (for reference) """
Ainv = np.linalg.pinv(A1)
x_pinv = np.dot(Ainv, yval1)
plt.plot(wavelengths, x_pinv)
plt.show()

ratio = np.max(yval1)/np.max(y_real)
print("power ratio = "+str(ratio))
""" See how close real interferogram is to measured interferogram """
plt.plot(y_real*ratio, 'ro--', label="expected")
plt.plot(yval1, 'ko-', label="measured")
plt.legend(loc='best')
plt.show()


""" Begin LASSO parameter search
"""
alpha_list = np.logspace(-2,2,100)
r2_list = []
for alpha in alpha_list:
    """ Search for a suitable alpha """
    lasso = Lasso(alpha=alpha)
    y_pred_lasso = lasso.fit(A1, yval1).predict(A1)
    r2_list.append(r2_score(yval2, np.dot(A2, lasso.coef_)))
#    Uncomment below to optimize wrt real spectrum
#    if np.max(lasso.coef_) >= 1E-8:
#        r2_list.append(r2_score(x_real/np.max(x_real), lasso.coef_/np.max(lasso.coef_)))
#    else:
#        r2_list.append(-1)

plt.semilogx(alpha_list, r2_list)
plt.title("Lasso Regression Model Fitting")
plt.xlabel("Alpha")
plt.ylabel("R2 value")
plt.show()

max_index = r2_list.index(max(r2_list))
alpha_max = alpha_list[max_index]

lasso = Lasso(alpha=alpha_max)
print r2_list
print "alphamax = "+str(alpha_max)
y_pred_lasso = lasso.fit(A2, yval2).predict(A2)
r2_score_lasso = r2_score(x_real/np.max(x_real), lasso.coef_/np.max(lasso.coef_))

print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)

plt.figure(figsize=(6, 4))
plt.subplot(2,1,1)
plt.plot(wavelengths, lasso.coef_, 'r-', label="LASSO", linewidth=2.0)
plt.legend(loc='best')
plt.subplot(2,1,2)
plt.plot(wavelengths, x_real*np.max(lasso.coef_)/np.max(x_real), 'k-', label="expected", linewidth=2.0)
plt.xlabel("Wavelength [nm]", fontsize=labelsize)
plt.ylabel("Amplitude [a.u.]", fontsize=labelsize)
plt.legend(loc='best')
plt.tight_layout()
plt.show()


""" Begin RIDGE parameter search
"""
alpha_list = np.logspace(-3,5,400)
r2_list = []
for alpha in alpha_list:
    """ Search for a suitable alpha """
    ridge = Ridge(alpha=alpha)
    y_pred_ridge = ridge.fit(A1, yval1).predict(A1)
#    r2_list.append(r2_score(yval2, np.dot(A2, ridge.coef_)))
#    Uncomment below to optimize wrt real spectrum
    r2_list.append(r2_score(x_real/np.max(x_real), ridge.coef_/np.max(ridge.coef_)))

plt.semilogx(alpha_list, r2_list)
plt.title("Ridge Regression Model Fitting")
plt.xlabel("Alpha")
plt.ylabel("R2 value")
plt.show()

max_index = r2_list.index(max(r2_list))
alpha_max = alpha_list[max_index]

ridge = Ridge(alpha=alpha_max)
print r2_list
print "alphamax = "+str(alpha_max)
y_pred_ridge = ridge.fit(A2, yval2).predict(A2)
r2_score_ridge = r2_score(x_real/np.max(x_real), ridge.coef_/np.max(ridge.coef_))

print(ridge)
print("r^2 on test data : %f" % r2_score_ridge)

plt.figure(figsize=(8, 6))
plt.subplot(2,1,1)
plt.plot(wavelengths, ridge.coef_/np.max(ridge.coef_), 'r-', label="Ridge regression", linewidth=2.0)
plt.legend(loc='best')
plt.subplot(2,1,2)
plt.plot(wavelengths, x_real/np.max(x_real), 'k-', label="Measured", linewidth=2.0)
plt.xlabel("Wavelength [nm]", fontsize=labelsize)
plt.ylabel("Amplitude [a.u.]", fontsize=labelsize)
maxy, miny = 1.0, 0.0
plt.ylim([miny - 0.05*abs(maxy-miny), maxy + 0.05*abs(maxy-miny)])
plt.legend(loc='best')
plt.tight_layout()
plt.show()


""" Begin ELASTIC NET parameter search
"""
l1_list = np.logspace(-3, 2, 50)
alpha_list = np.logspace(-3, 2, 50)
mv = 0.0
amax, l1max = 0, 0
r2_list = []
for l1 in l1_list:
    r2_list.append([])
    for alpha in alpha_list:
        enet = ElasticNet(alpha=alpha, l1_ratio=l1, positive=True)
        y_pred_enet = enet.fit(A1, yval1).predict(A1)
#        score = r2_score(yval2, np.dot(A2, enet.coef_))
#        Uncomment below to optimize wrt real spectrum
        if np.max(enet.coef_) >= 1E-8:
            score = r2_score(x_real/np.max(x_real), enet.coef_/np.max(enet.coef_))
        else:
            score = 0
        r2_list[-1].append(score)
        if score>mv:
            mv = score
            amax, l1max = alpha, l1

print "alphamax = "+str(amax)+",  l1max = "+str(l1max)

fig, ax = plt.subplots()
ax.matshow(np.array(r2_list), aspect=1, cmap=matplotlib.cm.afmhot)
plt.xlabel("L1")
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.ylabel("Alpha")
plt.show()

#amax = 0.0686648845004
#amax=0.138949549437
#l1max=0.00409491506238
#l1max = 0.00202358964773

enet = ElasticNet(alpha=amax, l1_ratio=l1max, positive=True)
y_pred_enet = enet.fit(A2, yval2).predict(A2)
r2_score_enet = r2_score(x_real/np.max(x_real), enet.coef_/np.max(enet.coef_))

print(enet)
print("r^2 on test data : %f" % r2_score_enet)

plt.figure(figsize=(8,6))
plt.subplot(2,1,1)
plt.plot(wavelengths, enet.coef_/np.max(enet.coef_), 'r-', label="64-ch FTIR", linewidth=2.0)
plt.ylabel("Amplitude [a.u.]", fontsize=labelsize)
plt.legend(loc='best')
plt.subplot(2,1,2)
plt.plot(wavelengths, x_real/np.max(x_real), 'k-', label="Measured", linewidth=2.0)
plt.xlabel("Wavelength [nm]", fontsize=labelsize)
plt.ylabel("Amplitude [a.u.]", fontsize=labelsize)
maxy, miny = 1.0, 0.0
plt.ylim([miny - 0.05*abs(maxy-miny), maxy + 0.05*abs(maxy-miny)])
plt.legend(loc='best')
plt.tight_layout()
plt.show()
