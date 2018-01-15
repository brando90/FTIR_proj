import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import pdb

def get_rbf_coefficients(A,X,centers,Y,std):
    '''
    A = calibration matrix, NxD
    X = input, Dx1
    center = centers of RBF, Dx1
    std = standard dev of Gaussians, 1x1

    We want to solve ||Af(a) - y||^2 s.t. f(a) is smooth. Thus use RBF kernel
    with appropriate standard deviation.
    With that we solve:

    ||AKc - y||^2 where K is the kernel matrix K=exp(-beta|a-t|^2) where t are
    centers of the RBFs.
    To solve it do:
    c=(AK)^+y
    '''
    beta = np.power(1.0/std,2)
    Kern = get_kernel_matrix(X,centers.transpose(),beta)
    #(C,_,_,_) = np.linalg.lstsq( np.(A,Kern),Y)
    AKern_pinv = np.linalg.pinv( np.dot(A,Kern) )
    C = np.dot(AKern_pinv,Y)
    return C

def get_kernel_matrix(x,W,beta):
    '''
    x = input, M data points of size D^(l-1), MxD^(l-1)
    W = params/centers, D^(l-1)xD^(l)
    beta = 1x1
    '''
    #Kern = np.exp(-beta*euclidean_distances(X=x,Y=W,squared=True))
    Kern = np.exp(-beta*euclidean_distances_manual(x=x,W=W))
    return Kern

def euclidean_distances_manual(x,W):
    '''
    x = input, M data points of size D^(l-1), MxD^(l-1)
    W = params/centers, D^(l-1) x D^(l) means that each center is of dimension D^(l-1) since
        they are being computed with respect to an element from the previous layer.
        Thus, they are D^(l) of them.

    return:
    Delta_tilde = (M x D^(l))
    '''
    WW = np.sum(np.multiply(W,W), axis=0, dtype=None, keepdims=True) #(1 x D^(l))= sum( (D^(l-1) x D^(l)), 0 )
    XX = np.sum(np.multiply(x,x), axis=1, dtype=None, keepdims=True) #(M x 1) = sum( (M x D^(l-1)), 1 )
    # || x - w ||^2 = (||x||^2 + ||w||^2) - 2<x,w>
    #Delta_tilde = 2.0*np.dot(x,W) - (WW + XX)
    xW = np.dot(x,W) #(M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l))
    Delta_tilde = (WW + XX) - 2.0*xW #(M x D^(l)) = (M x D^(l)) + ( (M x 1) + (1 x D^(l)) )
    return Delta_tilde
