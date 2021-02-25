'''

Numerical calculations of Lyapunov Exponents, using the QR-decomposition.

Authors : Giancarlo Kerg, Victor Geadah

'''
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg 
import math
import requests
import pickle
import gzip
import math
import torch
import pickle
import definitions as defs

'''
Define the gamma function for numpy and import it for pytorch
'''

def batch_sigmoid(x, n, threshold):
    lins = torch.mul(n, x)
    val = torch.sigmoid(torch.mul(x,n))
    # print(val)
    # print('sigmoid : out of bounds :',np.sum(((torch.abs(lins) > threshold).int()).numpy()))
    val = torch.where(torch.abs(lins) > threshold, (lins.sign() >0).float(), val)
    # print(val)
    return val.t()

def batch_softplus(x, beta, threshold):
    beta = torch.tensor(beta)
    x = torch.tensor(x)

    lins = torch.mul(x, beta)
    val = torch.div(torch.log(1 + torch.exp(torch.mul(beta,x))), beta)
    # print('out of bounds :',np.sum(((lins > threshold).int()).numpy()))
    val = torch.where(lins > threshold, torch.mul(beta.sign(), x), val)
    return val.t()

# In Numpy
def gam(x,n):
    return (1 / n)*(np.log(1 + np.exp(n*x)))
def gam_prime(x,n):
    return 1/(1 + np.exp(-n*x))

def Gam(x, n, s):
    return (1-s)*gam(x,n) + s*gam_prime(x,n)

def Gam_prime(x, n, s):
    return (1-s)*gam_prime(x,n) + s*n*gam_prime(x,n)*(1-gam_prime(x,n))

# Pytorch, defined for tensors
def gamma(x,n,s):
    gamma_one = batch_softplus(x, n, 20)
    # gamma_two = batch_sigmoid(x, n, 10)
    gamma_two = torch.sigmoid(torch.mul(n,x))
    output = torch.mul((1-s), gamma_one.t()) + torch.mul(s, gamma_two.t())
    return output

def gamma_prime(x,n,s):
    gamma_two = torch.sigmoid(torch.mul(n,x))
    output = (1-s)*gamma_two + s*n*gamma_two*(1-gamma_two)
    return output

class Gamma_prime(torch.autograd.Function):

    @staticmethod
    def forward(input, n, s):
        gamma_two = torch.sigmoid(torch.mul(n,input))
        output = (1-s)*gamma_two + s*n*gamma_two*(1-gamma_two)
        
        return output

class local_gamma2(torch.autograd.Function):
    @staticmethod
    def forward(input, n, s):
        gamma_one = batch_softplus(input, n, 5)
        # gamma_two = torch.sigmoid(torch.mul(n,input))
        gamma_two = batch_sigmoid(input, n, 10)
        output = torch.mul((1-s), gamma_one.t()) + torch.mul(s, gamma_two.t())

        return output

class local_gamma2_prime(torch.autograd.Function):

    @staticmethod
    def forward(input, n, s):
        gamma_two = torch.sigmoid(torch.mul(n,input))
        gamma_two = batch_sigmoid(input, n, 10)
        output = (1-s)*gamma_two + s*n*gamma_two*(1-gamma_two)
        
        return output

from Gamma import Gamma, Gamma2

# gamma = Gamma.apply
# gamma_prime = Gamma_prime.apply
# gamma = local_gamma2.apply
# gamma_prime = local_gamma2_prime.apply

'''
Define useful functions
'''
# def oneStep(x, W, n, s, Gam=Gam):
#     return Gam(W.dot(x), n, s)

def oneStep(x, W, n, s):
    value  = gamma(torch.tensor(W @ x), n, s).numpy()
    return value
    
def oneStepVar(x, Y, W, n, s, Gam=Gam, Gam_prime=Gam_prime):
    v = gamma_prime(torch.tensor(W.dot(x)), n, s).numpy()
    A = np.multiply(v,W.T).T
    return A.dot(Y)

def oneStepVarQR(x, Q, W, n, s, drive=0, Gam=Gam, Gam_prime=Gam_prime, output="complex"):
    v = gamma_prime(torch.tensor(W.dot(x)), n, s).numpy()
    # print(v)
    A = np.diag(v).dot(W)
    Z = A.dot(Q)
    q,r = np.linalg.qr(Z,mode='reduced')
    diag_r = np.diag(r)
    if output=="real": 
        angle_r = np.sign(diag_r)
    else: 
        norm_r = np.abs(diag_r)
        angle_r = diag_r / norm_r

    return q.dot(np.diag(angle_r)), np.diag(1/angle_r).dot(diag_r)


# function to generate a random orthogonal matrix
def rvs(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim-n+1,))
        D[n-1] = np.sign(x[0])
        x[0] -= D[n-1]*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1-(dim % 2))*D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D*H.T).T
    return H

# # check that PP^T is close to identity (off diag terms should be very small)
# F = rvs(dim=3)
# print(F@F.T)

#generating complex unitary matrix
def u_rvs(dim=3):
    N=dim
    G = np.random.normal(0,g/np.sqrt(N),[N,N])
    P= scipy.linalg.schur(G, output="complex")[1]
    return P


'''
Main method
'''
from Gamma import gamma2

def calculateLEs(k, n, s, h_0, V, num_steps=1000):
    Q = rvs(V.shape[0])
    Q = Q[:,:k]

    if (type(h_0) or type(V) is not np.ndarray):
        x = h_0.detach().numpy() # assuming its a torch tensor
        Z = V.numpy()
    else:
        x = h_0
        Z = V

    rvals = []
    for t in range(num_steps):
        Q,r = oneStepVarQR(x, Q, Z, n, s, output="real")
        x = oneStep(x, Z, n, s)
        rvals.append(r)

    rvals = np.vstack(rvals)
    # LEs = np.sum(np.log2(rvals),axis=0)/num_steps
    LEs = np.nanmean(np.log2(rvals), axis=0)
    return LEs
	# print('[{}, {}], LEs = {}'.format(n, s, [round(i,5) for i in LEs]))

# test_n, test_s, test_V = defs.load_net(10.0, 0.0, 'psMNIST', lp=True, nonlin='gamma2')

# print(test_n, test_s)
# print(test_V)
# print(defs.get_accuracy(10.0, 0.0, 'psMNIST', lp=True, nonlin='gamma2')[0][-1])
# test_h0 = 0.5*torch.ones([400])

# print(calculateLEs(2, test_n, test_s, test_h0, V=test_V, num_steps=1000))

