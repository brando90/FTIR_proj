# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 16:52:04 2017

@author: Derek Kita
"""
import numpy as np
import sys
import os
cur_dir = os.getcwd()

class LBD:
    def __init__(self, filename, norm=1):
        fileData = open(filename, 'r')
                
        self.wavelength = []        # [V]
        self.signal = []           # [um]
        
        past_wavelength = False
        for line in fileData:
            if (line != '\n'):
                if not past_wavelength:
                    subdata = line.strip('\n').split('\t')
                    for sd in subdata:
                        self.wavelength.append(float(sd))
                    past_wavelength = True
                else:
                    subdata = line.strip('\n').split('\t')
                    for sd in subdata:
                        self.signal.append(float(sd)/norm)
    
if __name__ == "__main__":
    data = LBD(cur_dir+"\\basis_data\\good_data_8-25-17\\data  6.txt")
    print len(data.signal)
    import matplotlib.pyplot as plt
    plt.plot(data.wavelength, data.signal, 'ro-')
    plt.show()