import numpy as np

import pdb

def get_rbf_coefficients(A,X,centers,std):
    '''
    A = calibration matrix
    X = input
    center = centers of RBF
    std = standard dev of Gaussians

    We want to solve ||Af(a) - y||^2 s.t. f(a) is smooth. Thus use RBF kernel
    with appropriate standard deviation.
    With that we solve:

    ||AKc - y||^2 where K is the kernel matrix K=exp(-beta|a-t|^2) where t are
    centers of the RBFs.
    To solve it do:
    c=(AK)^+y
    '''
    beta = np.power(1.0/std,2)
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
    beta = np.power(1.0/std,2)
    Kern = np.exp(-beta*euclidean_distances(X=X,Y=centers,squared=True))
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
