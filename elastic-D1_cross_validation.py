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
from sklearn.linear_model import ElasticNet
import os
import sys
from models import *
import scipy.optimize as so
from cvxopt import matrix, solvers
solvers.options['show_progress']=False
    
font = {'weight' : 'normal',
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

savedfilename = "broadband_signals_elastic-D1.csv"
with open(savedfilename, 'a') as wf:
    wf.write("signal-type, train_version/validate_version, R2-val, L2-norm, l1, l2, l3\n")
    wf.close()

""" Choose type of signal we want to use:
"""
options = ["MZI1", "MZI2", "MZI1MZI2"]#, "W0000"]
combinations = [[1,2], [2,3], [3,4], [4,5]]
for signal in options:
    for combo in combinations:
#        signal = options[3]
        
        yfile = path+'/12-14-17_broadband_src_MZI/interferogram_'+signal+'_v'+str(combo[0])+'.txt'
        yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
        yval_train, OPL = yf.values[:,1], yf.values[:,0]
        yfile = path+'/12-14-17_broadband_src_MZI/interferogram_'+signal+'_v'+str(combo[1])+'.txt'
        yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
        yval_validate, OPL = yf.values[:,1], yf.values[:,0]

        #yfile = path+'/Narrowband_2laser_data/2laser_dlambda=0.2nm_v1.txt'
        #yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
        #yval_train, OPL = yf.values[:,1], yf.values[:,0]
        #yfile = path+'/Narrowband_2laser_data/2laser_dlambda=0.2nm_v2.txt'
        #yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
        #yval_validate, OPL = yf.values[:,1], yf.values[:,0]

        """ UNCOMMENT area below only if desired spectrum is KNOWN beforehand """
        xfile = path+'/12-14-17_broadband_src_MZI/'+str(signal)+'.CSV'
        xf = pd.read_csv(xfile, header=30)
        xval_train, xwl = xf.values[:,1], xf.values[:,0]
        xwl = np.array([x - 0.7 for x in xwl])
        x_real = np.interp(wavelengths, xwl, xval_train)
        y_real = np.dot(A1, x_real)
        
        
        ''' Pseudo-inverse method (for reference) '''
        Ainv = np.linalg.pinv(A1)
        x_pinv_train = np.dot(Ainv, yval_train)
        x_pinv_validate = np.dot(Ainv, yval_validate)
        
        """ Begin ELASTIC NET parameter search
        """
        Dsize = len(A1[0])
        D1 = np.zeros((Dsize+1, Dsize))
        for i in xrange(Dsize):
            D1[i][i] = 1
            D1[i+1][i] = -1
            
        def elastic_D1_smoothing(l1, l2, l3, A, D1, Dsize, yval):
            """ p = 2(AT.A + l2.I + l3.DT.D) """
            p = 2*(np.dot(np.transpose(A), A) + l2*np.identity(Dsize) + l3*np.dot(np.transpose(D1), D1))
            
            """ Q = (l1.vec1 - 2 AT.y) """
            Q = l1*np.ones(Dsize) - 2*np.dot(np.transpose(A), yval)
            
            """ G = -I """
            G = -1*np.identity(Dsize)
            
            """ h = zero-vector """
            h = np.zeros(Dsize)
            
            sol = solvers.qp(matrix(p.tolist()), matrix(Q.tolist()),  matrix(G.tolist()), matrix(h.tolist()))
            x = np.transpose(np.array(sol['x']))[0]
               
            return x
            
        d1, d2, d3 = 12,12,12
        l1_list = np.logspace(-2, 4, d1)
        l2_list = np.logspace(-3, 5, d2)
        l3_list = np.logspace(2, 7, d3)
        r2_list = []
        mv = -1E10
        l1max, l2max, l3max = 0.0, 0.0, 0.0
        print "Running sweep...",
        for l1 in l1_list:
            r2_list.append([])
            print "l1="+str(l1),
            for l2 in l2_list:
                r2_list[-1].append([])
                print ".",
                for l3 in l3_list:
                    x = elastic_D1_smoothing(l1, l2, l3, A1, D1, Dsize, yval_train)
                    if np.isfinite(x.all()) and max(x) !=0.0:
                        score = r2_score(yval_validate, np.dot(A2, x))
                        r2_list[-1][-1].append(score)
                        if score > mv:
                            mv = score
                            l1max = l1
                            l2max = l2
                            l3max = l3
        
        print "l1 max = "+str(l1max)
        print "l2 max = "+str(l2max)
        print "l3 max = "+str(l3max)
        
        xmax = elastic_D1_smoothing(l1max, l2max, l3max, A1, D1, Dsize, yval_train)
        
        x_real = normalize_vector(x_real, xmax)
        
        r2_score_ed1 = r2_score(x_real/max(x_real), xmax/max(xmax))
        l2_score_ed1 = np.linalg.norm((x_real/max(x_real)) - (xmax/max(xmax)), 2)
        print("r^2 result for D1 : %f" % r2_score_ed1)
        print("L2-norm result for D1: %f" % l2_score_ed1)
        
        r2_score_pinv = r2_score(x_real/max(x_real), x_pinv_validate/max(x_pinv_validate))
        l2_score_pinv = np.linalg.norm((x_real/max(x_real)) - (x_pinv_validate/max(x_pinv_validate)), 2)
        print("r^2 result for PINV : %f" % r2_score_pinv)
        print("L2-norm result for PINV: %f" % l2_score_pinv)
        
        with open(savedfilename, 'as') as wf:
            wf.write(str(signal)+', '+str(combo[0])+'/'+str(combo[1])+', '+str(r2_score_ed1)+', '+str(l2_score_ed1)+', '+str(l1max)+', '+str(l2max)+', '+str(l3max)+'\n')
            wf.close()
        
#        fig, ax = plt.subplots()
#        ax.matshow(np.array(r2_list).reshape((d1, d2*d3)), aspect=1, cmap=matplotlib.cm.afmhot)
#        plt.xlabel("L2")
#        ax.xaxis.tick_top()
#        ax.xaxis.set_label_position('top')
#        plt.ylabel("L1")
#        plt.show()

#        plt.figure(figsize=(6,4))
#        plt.plot(wavelengths, xmax/max(xmax), 'r', label="D1 method")
#        plt.plot(wavelengths, x_pinv_validate/max(x_pinv_validate), 'b', label="pseudo-inverse")
#        plt.plot(wavelengths, x_real/max(x_real), 'k', label="real")
#        plt.legend(loc='best')
#        plt.show()
#        sys.exit()

sys.exit()

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
#plt.show()

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
plt.ylim([miny - 0.05*abs(maxy-miny), maxy + 0.05*abs(maxy-miny)])

plt.plot(wavelengths, xmax/np.max(xmax), 'r-', label="64-ch dFT", linewidth=2.0)
plt.ylabel("Intensity [a.u.]", fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.legend(loc='best')
maxy, miny = 1.0, 0.0
plt.ylim([miny - 0.05*abs(maxy-miny), maxy + 0.05*abs(maxy-miny)])


plt.legend(loc='best', fontsize=12)
plt.tight_layout()


plt.show()
