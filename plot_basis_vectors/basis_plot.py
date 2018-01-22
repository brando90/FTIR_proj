# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 17:56:11 2017

@author: dkita
"""

import matplotlib.pyplot as plt
import scipy.io
from matplotlib.pyplot import cm 
import matplotlib
import numpy as np
import sys
import os
from Labview_basis_data import LBD
import h5py
import pandas as pd

gap = 25.554 #um
numstates=64
cur_dir = os.getcwd()+'/dataset4_11-30-2017/'
norm_factor = 1E6

font = {'family' : 'normal',
#        'weight' : 'none',
        'size'   : 14}
matplotlib.rc('font', **font)
    
def getBasis(filename):
    with h5py.File(filename,'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        """ Extract the data from the .h5 files """
        M = np.array(hf.get('M'))
        M2 = np.array(hf.get('M_ideal'))
        lambda_vector = np.array(hf.get('lambda_vector'))
        hf.close()
    return M, M2, lambda_vector
    
def calculateAndSaveBasis(filename):
    M = []
    M2 = []
    lambda_vector = []
    for i in xrange(numstates):
        print "getting dataset "+str(i)+"/63"
        if i<10:
            m1 = LBD(cur_dir+"data  "+str(i)+'.txt', norm=norm_factor)
        else:
            m1 = LBD(cur_dir+"data "+str(i)+'.txt', norm=norm_factor)
        lambda_vector, max_loss_trunc = m1.wavelength, m1.signal
#        print(max(max_loss_trunc))
#        max_loss_trunc = np.divide(max_loss_trunc,max(max_loss_trunc)) #normalization
        
        """ Ideal case """
        ideal_loss = [np.cos(2*np.pi*2.14*i*gap/(wl/1000.0))**2 for wl in lambda_vector]
        if i==0:
            ideal_loss = [1.0 for wl in lambda_vector]
        
        if i==27:
#            plt.plot(lambda_vector, ideal_loss, 'r')
#            fig = plt.figure(figsize=(6,3))
            fig, ax = plt.subplots(figsize=(6,2.8))
            ax.semilogy(lambda_vector, max_loss_trunc, 'r')
            plt.ylabel("Intensity [a.u.]")
            plt.xlabel("Wavelength")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.xlim([1550,1570])
            plt.tight_layout()
            fig.show()
        print(max(max_loss_trunc))
        M.append(max_loss_trunc)
        M2.append(ideal_loss)
#        sys.exit()

    with h5py.File(filename,'w') as hf:
        hf.create_dataset('M', data=np.array(M))
        hf.create_dataset('M_ideal', data=np.array(M2))
        hf.create_dataset('lambda_vector', data=np.array(lambda_vector))
        hf.close()

if __name__=="__main__":
    calculateAndSaveBasis('basis.h5')
    M, M2, lambda_vector = getBasis('basis.h5')
    import matplotlib
    font = {'family' : 'normal',
#        'weight' : 'none',
        'size'   : 16}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.matshow(M[::-1], aspect=20./1.609902, cmap=matplotlib.cm.afmhot, extent=[1550.0,1570.0,0,1.609902])
    plt.xlabel("Wavelength [nm]")
    ax.xaxis.tick_bottom()
    plt.yticks(np.linspace(0,1.6,9))
    ax.xaxis.set_label_position('bottom')
    plt.ylabel("OPL [mm]", va='bottom', rotation=270, labelpad=20)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    plt.tight_layout()
    plt.show()
    
    sys.exit()
    plt.matshow(np.transpose(M2), aspect=64/20.0, cmap=matplotlib.cm.afmhot, extent=[0,64,1550.0,1570.0])
    plt.show()
    dL_vector = np.array([i*gap for i in xrange(numstates)])

    print "Computing inverse..."
    Minv = np.linalg.pinv(M)
    M2inv = np.linalg.pinv(M2)
    print(np.allclose(M, np.dot(M, np.dot(Minv, M))))
    print(np.allclose(Minv, np.dot(Minv, np.dot(M, Minv))))
    print(np.shape(M))
    print(np.shape(Minv))
    print(np.allclose(M2, np.dot(M2, np.dot(M2inv, M2))))
    print(np.allclose(M2inv, np.dot(M2inv, np.dot(M2, M2inv))))
    print(np.shape(M2))
    print(np.shape(M2inv))
    
#    """ Test scenario -- single frequency source at 1555nm IDEAL """
#    
    """ Test scenario -- single frequency source at 1555nm """
    testvector = np.zeros(len(lambda_vector))
    desired_lambda = 1553.0
    idx = (np.abs(lambda_vector-desired_lambda)).argmin()
    testvector[idx] = 1.0
    desired_lambda = 1567.0
    idx = (np.abs(lambda_vector-desired_lambda)).argmin()
    testvector[idx] = 1.0

#    print "idx="+str(idx)
#    print "lambda_val = "+str(lambda_vector[idx])
#    sys.exit()
    plt.plot(lambda_vector, testvector)
    plt.show()
    
    testvector_dL = np.dot(M, testvector)
    testvector_dL_theoretical = testvector_dL
    plt.plot(dL_vector, testvector_dL)
    plt.show()
    
    reconstruction = np.dot(Minv, testvector_dL)
    plt.plot(lambda_vector, reconstruction)
    plt.ylabel('Intensity [a.u.]')
    plt.xlabel('Wavelength [nm]')
    plt.show()
    

    