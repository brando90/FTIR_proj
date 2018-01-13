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

##

class NN(torch.nn.Module):
    # http://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html#sphx-glr-beginner-examples-nn-two-layer-net-module-py
    # http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-nn
    def __init__(self, D_layers,act,w_inits,b_inits,biases):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.

        D_layers = [D^(0),D^(1),...,D^(L)]
        act = activation func
        w_inits = [None,W_f1,...,W_fL]
        b_inits = [None,b_f1,...,b_fL]
        bias = True
        """
        super(type(self), self).__init__()
        # if bias is false then we don't need any init for it (if we do have an init for it and bias=False throw an error)
        #if not bias and (b_inits != [] or b_inits != None):
        #    raise ValueError('bias is {} but b_inits is not empty nor None but isntead is {}'.format(bias,b_inits))
        print('biases = {}'.format(biases))
        self.biases = biases
        # actiaction func
        self.act = act
        #create linear layers
        self.linear_layers = torch.nn.ModuleList([None])
        #self.linear_layers = torch.nn.ParameterList([None])
        for d in range(1,len(D_layers)):
            bias = biases[d]
            print( 'D_layers[{}], D_layers[{}] = {},{} '.format(d-1,d,D_layers[d-1], D_layers[d]) )
            print( 'biases[{}] = {}'.format(d,bias))
            linear_layer = torch.nn.Linear(D_layers[d-1], D_layers[d],bias=bias)
            self.linear_layers.append(linear_layer)
        #pdb.set_trace()
        # initialize model
        for d in range(1,len(D_layers)):
            ## weight params
            weight_init = w_inits[d]
            m = self.linear_layers[d]
            weight_init(m)
            ## bias params
            bias = biases[d]
            if bias:
                bias_init = b_inits[d]
                bias_init(m)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        a = x
        for d in range(1,len(self.linear_layers)-1):
            W_d = self.linear_layers[d]
            z = W_d(a)
            a = self.act(z)
        d = len(self.linear_layers)-1
        y_pred = self.linear_layers[d](a)
        return y_pred

    # def numpy_forward(self,x,dtype):
    #     if type(x) == np.ndarray:
    #         X = Variable(torch.FloatTensor(X).type(dtype), requires_grad=False)
    #     y_pred = self.forward(x)
    #     return y_pred.data.numpy()

    def to_gpu(self,device_id=None):
        torch.nn.Module.cuda(device_id=device_id)

    def get_parameters(self):
        return list(self.parameters())

    def get_nb_params(self):
        return sum(p.numel() for p in model.parameters())

    def __getitem__(self, key):
        return self.linear_layers[key]

##
