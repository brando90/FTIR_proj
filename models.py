import numpy as np

import pdb

def get_gauss_coeffs(A,X,centers,std):
    '''
    A = calibration matrix
    X = input
    center = centers of RBF
    std = standard dev of Gaussians
    '''
    beta = np.power(1.0/stddev,2)
    Kern = np.exp(-beta*euclidean_distances(X=X,Y=centers,squared=True))
    (C,_,_,_) = np.linalg.lstsq(A*Kern,Y)
    return C

def get_gauss_coeffs(X,centers,std):
    '''
    X = input
    center = centers of RBF
    std = standard dev of Gaussians
    '''
    #indices=np.random.choice(a=N,size=K,replace=replace) # choose numbers from 0 to D^(1)
    #subsampled_data_points=X[indices,:] # M_sub x D
    beta = np.power(1.0/stddev,2)
    Kern = np.exp(-beta*euclidean_distances(X=X,Y=subsampled_data_points,squared=True))
    (C,_,_,_) = np.linalg.lstsq(Kern,Y)
    return C

def get_kernel_matrix(x,W,S):
    beta = get_beta_np(S)
    #beta = 0.5*tf.pow(tf.div( tf.constant(1.0,dtype=tf.float64),S), 2)
    Z = -beta*euclidean_distances(X=x,Y=W,squared=True)
    K = np.exp(Z)
    return K

def get_z_np(x,W):
    WW = np.sum(np.multiply(W,W), axis=0, dtype=None, keepdims=True)
    XX = np.sum(np.multiply(x,x), axis=1, dtype=None, keepdims=True)
    Delta_tilde = 2.0*np.dot(x,W) - (WW + XX)
    return Delta_tilde
