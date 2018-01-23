#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:08:51 2018

@author: dkita
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import matplotlib
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
import os
import sys
from models import *
    
font = {'weight' : 'normal',
        'size'   : 18}
labelsize = 20
ticksize=18
matplotlib.rc('font', **font)

def get_index(wl, wavelengths):
    # for an float "wl" on range[1550, 1570] and list "wavelengths" from [1550,1570],
    # returns the index of "wavelengths" that is closest to the input "wl"
    i, = np.where(wavelengths==(min(wavelengths, key=lambda x:abs(x-wl))))
    return i[0]

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
options = ["0.1", "0.2", "0.3", "0.4", "0.5", "5.0", "10.0", "15.0"]
signals = [options[1], options[2], options[3], options[5], options[7]] #5 MAX
x_vals = []
x_real_vals = []

""" ELASTIC NET LEARNED PARAMETERS
    Learned from dl=10.0nm """
amax = 0.0464158883361
l1max = 2.53536449397

for i in xrange(len(signals)):
    yfile = path+'/Narrowband_2laser_data/2laser_dlambda='+signals[i]+'nm_v1.txt'
    yf = pd.read_csv(yfile, sep='\t', usecols=[0,1])
    yval, OPL = yf.values[:,1], yf.values[:,0]

    enet = ElasticNet(alpha=amax, l1_ratio=l1max, positive=True)
    y_pred_enet = enet.fit(A1, yval).predict(A1)
    x_vals.append(enet.coef_)
    
    x_real_validate = np.zeros(len(A1[0]))
    x_real_validate[get_index(1560+float(signals[i])/2.0, wavelengths)] = 0.8
    x_real_validate[get_index(1560-float(signals[i])/2.0, wavelengths)] = 1.0
    x_real_vals.append(x_real_validate)

""" ---------------------------------------------------------------------------
PLOT THE RESULTS FROM x_vals & x_real_vals BELOW
--------------------------------------------------------------------------- """
font = {'size' : 16}
matplotlib.rc('font', **font)

f, ax = plt.subplots(ncols=1,nrows=len(x_vals), figsize=(6,7))
for i in xrange(len(x_vals)):
    dl = float(signals[i])
    plt.subplot(len(x_vals), 1, i+1)
    color=cm.brg(np.linspace(0.0,0.5,len(x_vals)))
    plt.plot(wavelengths, x_vals[i]/max(x_vals[i]), color=color[i], label="spacing = "+str(dl))
#    plt.plot(wavelengths, x_real_vals[i]/max(x_real_vals[i]), 'ko')
    if i==len(x_vals)//2:
        plt.ylabel("Intensity [a.u.]")
    plt.xlim([1550,1570])
    plt.ylim([-0.05, 1.05])
    frame = plt.gca()
    if i!=len(x_vals)-1:
        frame.axes.get_xaxis().set_ticklabels([])
    frame.axes.get_yaxis().set_ticklabels([])
    plt.yticks(np.linspace(0,1.,6))
    hl, hw = 0.7, 0.07
    arrow_height = 0.5
    if i==2:
        plt.text(1555.0, arrow_height, '400 pm', horizontalalignment='center', verticalalignment='center')
        plt.arrow(1560.0-3.0, arrow_height, +1.6, 0.0, fc='k', head_width=hw, head_length=hl)
        plt.arrow(1560.0+3.0, arrow_height, -1.6, 0.0, fc='k', head_width=hw, head_length=hl)
    elif i==4:
        plt.arrow(1560.0-2.0, arrow_height, -4.5, 0.0, fc='k', head_width=hw, head_length=hl)
        plt.arrow(1560.0+2.0, arrow_height, 4.5, 0.0, fc='k', head_width=hw, head_length=hl)
        plt.text(1560.0, arrow_height, '15 nm', horizontalalignment='center', verticalalignment='center')
    elif i==0:
        plt.text(1555.0, arrow_height, '200 pm', horizontalalignment='center', verticalalignment='center')
        plt.arrow(1560.0-3.0, arrow_height, +1.8, 0.0, fc='k', head_width=hw, head_length=hl)
        plt.arrow(1560.0+3.0, arrow_height, -1.8, 0.0, fc='k', head_width=hw, head_length=hl)
    elif i==1:
        plt.text(1555.0, arrow_height, '300 pm', horizontalalignment='center', verticalalignment='center')
        plt.arrow(1560.0-3.0, arrow_height, +1.6, 0.0, fc='k', head_width=hw, head_length=hl)
        plt.arrow(1560.0+3.0, arrow_height, -1.6, 0.0, fc='k', head_width=hw, head_length=hl)
    elif i==3:
        plt.arrow(1560.0-5.0, arrow_height, +1.5, 0.0, fc='k', head_width=hw, head_length=hl)
        plt.arrow(1560.0+5.0, arrow_height, -1.5, 0.0, fc='k', head_width=hw, head_length=hl)
        plt.text(1560.0, arrow_height, '5 nm', horizontalalignment='center', verticalalignment='center')
plt.xlabel("Wavelength [nm]")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()