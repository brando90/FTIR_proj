import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import unittest

import pdb

##

def get_Z_tf(x,W,l='layer'):
    ## https://github.com/brando90/hbf_tensorflow_code/tree/master/my_tf_proj/my_tf_pkg
    W = tf.Variable(W, name='W'+l, trainable=True, dtype=tf.float64)
    WW =  tf.reduce_sum(W*W, reduction_indices=0, keep_dims=True) #( 1 x D^(l)= sum( (D^(l-1) x D^(l)), 0 )
    XX =  tf.reduce_sum(x*x, reduction_indices=1, keep_dims=True) # (M x 1) = sum( (M x D^(l-1)), 1 )
    # -|| x - w ||^2 = -(-2<x,w> + ||x||^2 + ||w||^2) = 2<x,w> - (||x||^2 + ||w||^2)
    Delta_tilde = 2.0*tf.matmul(x,W) - tf.add(WW, XX)
    return Delta_tilde

def Gaussian_kernel():
    WW = (W * W).sum(axis=0, keepdim=True) #( 1 x D^(l)= sum( (D^(l-1) x D^(l)), 0 )
    XX = (X * X).sum(axis=0, keepdim=True) # (M x 1) = sum( (M x D^(l-1)), 1 )
    ## -|| x - w ||^2 = -(-2<x,w> + ||x||^2 + ||w||^2) = 2<x,w> - (||x||^2 + ||w||^2)
    Delta_tilde = 2.0*x.mm(W) - (WW+XX)
    ##

    ##
    activations = torch.exp(Delta_tilde)
    return activations
