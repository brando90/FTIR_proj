import time
from datetime import date
import calendar

import os
import sys

sys.path.append(os.getcwd())

#from ''' import *
#from models_pytorch import *
#from inits import *
#from sympy_poly import *
#from poly_checks_on_deep_net_coeffs import *
#from data_file import *
#from plotting_utils import *

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy
from numpy.polynomial.hermite import hermvander

#from maps import NamedDict

import pdb

import unittest

def index_batch(X,batch_indices,dtype):
    '''
    returns the batch indexed/sliced batch
    '''
    if len(X.shape) == 1: # i.e. dimension (M,) just a vector
        batch_xs = torch.FloatTensor(X[batch_indices]).type(dtype)
    else:
        batch_xs = torch.FloatTensor(X[batch_indices,:]).type(dtype)
    return batch_xs

def get_batch2(X,Y,M,dtype):
    '''
    get batch for pytorch model
    '''
    # TODO fix and make it nicer, there is pytorch forum question
    X,Y = X.data.numpy(), Y.data.numpy()
    N = len(Y)
    valid_indices = np.array( range(N) )
    batch_indices = np.random.choice(valid_indices,size=M,replace=False)
    batch_xs = index_batch(X,batch_indices,dtype)
    batch_ys = index_batch(Y,batch_indices,dtype)
    return Variable(batch_xs, requires_grad=False), Variable(batch_ys, requires_grad=False)

def get_sequential_lifted_mdl(nb_monomials,D_out, bias=False):
    return torch.nn.Sequential(torch.nn.Linear(nb_monomials,D_out,bias=bias))

def vectors_dims_dont_match(Y,Y_):
    '''
    Checks that vector Y and Y_ have the same dimensions. If they don't
    then there might be an error that could be caused due to wrong broadcasting.
    '''
    DY = tuple( Y.size() )
    DY_ = tuple( Y_.size() )
    if len(DY) != len(DY_):
        return True
    for i in range(len(DY)):
        if DY[i] != DY_[i]:
            return True
    return False

def train_SGD(mdl, M,eta,nb_iter, dtype, X_train,Y_train, reg_l,R_x,R_x_params):
    N_train,_ = tuple( X_train.size() )
    ''' stats to collect from training'''
    erm_errors = np.zeros(nb_iter+1)
    train_errors = np.zeros(nb_iter+1)
    t_params = np.zeros(nb_iter+1)
    ''' error before training'''
    current_train_loss = (1/N_train)*(mdl.forward(X_train) - Y_train).pow(2).sum().data.numpy()
    print(f'i = 0')
    print(f'current_train_loss = 1/n||Xw - y||^2 = {current_train_loss}')
    ''' SGD train '''
    t_param = R_x_params['t_param']
    sigma_param = R_x_params['sigma_param']
    A_param = R_x_params['A_param']
    erm_errors[0] = current_train_loss
    t_params[0] = t_param
    for i in range(1,nb_iter+1):
        # Forward pass: compute predicted Y using operations on Variables
        batch_xs, batch_ys = get_batch2(X_train,Y_train,M,dtype) # [M, D], [M, 1]
        ## FORWARD PASS
        y_pred = mdl(batch_xs)
        ## Check vectors have same dimension
        if vectors_dims_dont_match(batch_ys,y_pred):
            pdb.set_trace()
            raise ValueError('You vectors don\'t have matching dimensions. It will lead to errors.')
        ## LOSS + Regularization
        if R_x is None:
            batch_loss = (1.0/M)*(y_pred - batch_ys).pow(2).sum()
        else:
            batch_loss = (1.0/M)*(y_pred - batch_ys).pow(2).sum()
            batch_loss = batch_loss + reg_l*R_x(x=mdl[0].weight, **R_x_params)
        ## BACKARD PASS
        batch_loss.backward() # Use autograd to compute the backward pass. Now w will have gradients
        ## SGD update
        for W in mdl.parameters():
            delta = eta*W.grad.data
            W.data.copy_(W.data - delta)
        t_param.data.copy_(t_param.data - t_param.grad.data)
        # train stats
        if i % (nb_iter/nb_iter) == 0 or i == 0:
            #X_train_, Y_train_ = Variable(X_train), Variable(Y_train)
            X_train_, Y_train_ = X_train, Y_train
            current_train_loss = (1/N_train)*(mdl.forward(X_train_) - Y_train_).pow(2).sum().data.numpy()
            R_f = R_x(x=mdl[0].weight, **R_x_params,print_rx=True).data.numpy()
            erm = current_train_loss+reg_l*R_f
            print('\n-------------')
            print(f'i = {i}')
            print(f'current_train_loss = 1/n||Xw - y||^2 = {current_train_loss}')
            print(f'erm = {erm}')
            print('----')
            print(f't_param={t_param.data.numpy()}')
            # print(f'eta*W.grad.data = {eta*W.grad.data}')
            # print(f'W.grad.data = {W.grad.data}')
            '''  '''
            erm_errors[i] = erm
            train_errors[i] = current_train_loss
            t_params[i] = t_param.data.numpy()
        ## Manually zero the gradients after updating weights
        mdl.zero_grad()
    return train_errors,erm_errors,t_params

def train(mdl, M,eta,nb_iter, dtype, X_train,Y_train):
    for i in range(1,nb_iter+1):
        # Forward pass: compute predicted Y using operations on Variables
        batch_xs, batch_ys = X_train,Y_train
        #batch_xs, batch_ys = get_batch2(X_train,Y_train,M,dtype) # [M, D], [M, 1]
        ## FORWARD PASS
        pdb.set_trace()
        y_pred = mdl(batch_xs).t()
        ## Check vectors have same dimension
        if vectors_dims_dont_match(batch_ys,y_pred):
            pdb.set_trace()
            raise ValueError('You vectors don\'t have matching dimensions. It will lead to errors.')
        ## loss
        batch_loss = (1.0/M)*(y_pred - batch_ys).pow(2).sum()
        batch_loss.backward()
        for W in mdl.parameters():
            delta = eta*W.grad.data
            #print(f'delta={delta.norm(2)}')
            W.data.copy_(W.data - delta)
        # train stats
        # if i % (nb_iter/nb_iter) == 0 or i == 0:
        #     print('\n-------------')
        #     print(f'i = {i}')
        #     print(f'batch_loss={batch_loss}')
        # Manually zero the gradients after updating weights
        mdl.zero_grad()
    return batch_loss.data.numpy()

def train_SGD2(mdl, M,eta,nb_iter, dtype, X_train,Y_train, reg_l,eta_R_x, R_x,R_x_params):
    N_train,_ = tuple( X_train.size() )
    ''' stats to collect from training'''
    erm_errors = np.zeros(nb_iter+1)
    train_errors = np.zeros(nb_iter+1)
    t_params = np.zeros(nb_iter+1)
    ''' error before training'''
    current_train_loss = (1/N_train)*(mdl.forward(X_train) - Y_train).pow(2).sum().data.numpy()
    print(f'i = 0')
    print(f'current_train_loss = 1/n||Xw - y||^2 = {current_train_loss}')
    ''' SGD train '''
    erm_errors[0] = current_train_loss
    for i in range(1,nb_iter+1):
        print('\n-------------')
        # Forward pass: compute predicted Y using operations on Variables
        batch_xs, batch_ys = get_batch2(X_train,Y_train,M,dtype) # [M, D], [M, 1]
        ## FORWARD PASS
        y_pred = mdl(batch_xs)
        ## Check vectors have same dimension
        if vectors_dims_dont_match(batch_ys,y_pred):
            pdb.set_trace()
            raise ValueError('You vectors don\'t have matching dimensions. It will lead to errors.')
        ## LOSS + Regularization
        if R_x is None:
            batch_loss = (1.0/M)*(y_pred - batch_ys).pow(2).sum()
        else:
            batch_loss = (1.0/M)*(y_pred - batch_ys).pow(2).sum()
            #R_f = R_x(x=mdl[0].weight,a=R_x_params.a,R_a_mdl=R_x_params.R_a_mdl)
            R_f = R_x(x=mdl[0].weight,**R_x_params)
            batch_loss = batch_loss + reg_l*R_f + 0*mdl[0].weight.norm(2)**2
        ## BACKARD PASS
        batch_loss.backward() # Use autograd to compute the backward pass. Now w will have gradients
        ## SGD update
        for W in mdl.parameters():
            delta = eta*W.grad.data
            print(f'delta_loss={delta.norm(2)}')
            W.data.copy_(W.data - delta)
        if eta_R_x != 0:
            for W in R_x_params.R_a_mdl.parameters():
                delta = eta_R_x*W.grad.data
                print(f'delta_R_x={delta.norm(2)}')
                W.data.copy_(W.data - delta)
        # train stats
        if i % (nb_iter/nb_iter) == 0 or i == 0:
            #X_train_, Y_train_ = Variable(X_train), Variable(Y_train)
            X_train_, Y_train_ = X_train, Y_train
            current_train_loss = (1/N_train)*(mdl.forward(X_train_) - Y_train_).pow(2).sum().data.numpy()
            R_f = R_x(x=mdl[0].weight, **R_x_params).data.numpy()
            erm = current_train_loss+reg_l*R_f
            print(f'i = {i}')
            print(f'R_f={R_f}')
            print(f'current_train_loss = 1/n||Xw - y||^2 = {current_train_loss}')
            print(f'erm = {erm}')
            print('----')
            # print(f'eta*W.grad.data = {eta*W.grad.data}')
            # print(f'W.grad.data = {W.grad.data}')
            '''  '''
            erm_errors[i] = erm
            train_errors[i] = current_train_loss
        ## Manually zero the gradients after updating weights
        mdl.zero_grad()
    return train_errors,erm_errors

def train_SGD3(mdl,a, M,eta,nb_iter, dtype, X_train,Y_train):
    N_train,_ = tuple( X_train.size() )
    ''' stats to collect from training'''
    erm_errors = np.zeros(nb_iter+1)
    train_errors = np.zeros(nb_iter+1)
    t_params = np.zeros(nb_iter+1)
    ''' error before training'''
    current_train_loss = (1/N_train)*(mdl.forward(X_train) - Y_train).pow(2).sum().data.numpy()
    print(f'i = 0')
    print(f'current_train_loss = 1/n||Xw - y||^2 = {current_train_loss}')
    ''' SGD train '''
    erm_errors[0] = current_train_loss
    for i in range(1,nb_iter+1):
        # Forward pass: compute predicted Y using operations on Variables
        #batch_xs, batch_ys = X_train,Y_train
        batch_xs, batch_ys = get_batch2(X_train,Y_train,M,dtype) # [M, D], [M, 1]
        ## FORWARD PASS
        mdl[1].weight.data = batch_xs.data
        #pdb.set_trace()
        y_pred = mdl(a.t())
        y_pred = y_pred.t()
        ## Check vectors have same dimension
        if vectors_dims_dont_match(batch_ys,y_pred):
            pdb.set_trace()
            raise ValueError('You vectors don\'t have matching dimensions. It will lead to errors.')
        ## loss
        batch_loss = (1.0/M)*(y_pred - batch_ys).pow(2).sum()
        batch_loss.backward()
        for W in mdl[0].parameters():
            delta = eta*W.grad.data
            #print(f'delta={delta.norm(2)}')
            W.data.copy_(W.data - delta)
        ## train stats
        if i % (nb_iter/nb_iter) == 0 or i == 0:
            print('\n-------------')
            print(f'i = {i}')
            print(f'batch_loss={batch_loss}')
        ## Manually zero the gradients after updating weights
        mdl.zero_grad()
    return train_errors,erm_errors
