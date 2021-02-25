'''
============================================

Description : PyTorch autograd function gamma(x;n,s) parametrised by :
    n > 0      : neuronal gain
    s in [0,1] : degree of saturation

============================================
'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Module

# ---------------------------
#   Homogeneous adaptation
# ---------------------------

class Gamma(torch.autograd.Function):
    '''
    Gamma autograd function for homogeneous adaptation. 

    Forward params : 
    - input : torch tensor
    - n     : neuronal gain, scalar torch tensor of shape (1,)
    - s     : saturaiton, scalar torch tensor of shape (1,)
    '''
    @staticmethod
    def forward(ctx, input, n, s):

        if not (isinstance(n, float) or isinstance(n, int)): n = n.item()
        if not (isinstance(s, float) or isinstance(s, int)): s = s.item()

        ctx.n = n
        ctx.s = s

        gamma_one = F.softplus(input, beta = n)
        gamma_two = torch.sigmoid(torch.mul(n,input))
        output = torch.mul((1-s), gamma_one) + torch.mul(s, gamma_two)

        ctx.save_for_backward(input, gamma_one, gamma_two)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma_one, gamma_two = ctx.saved_tensors
        n = ctx.n
        s = ctx.s

        grad_input = grad_n = grad_s = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.mul(grad_output, ((1-s)*gamma_two + s*n*gamma_two*(1-gamma_two)))
        if ctx.needs_input_grad[1]:
            grad_n = grad_output * ((1-s)/n * (input * gamma_two - gamma_one) + s*input*gamma_two*(1-gamma_two))
        if ctx.needs_input_grad[2]:
            grad_s = grad_output * (gamma_two - gamma_one)

        return grad_input, grad_n, grad_s


class gamma(nn.Module):
    def __init__(self, n, s):
        super(gamma, self).__init__()
        self.n = n
        self.s = s

    def forward(self, input):
        return Gamma.apply(input, self.n, self.s)

# ---------------------------
#   Heterogeneous adaptation
# ---------------------------

def batch_softplus(x, beta, threshold):
    '''
    Softplus reformulation for overflow handling.
    '''
    lins = torch.mul(x, beta)
    val = torch.div(torch.log(1 + torch.exp(torch.mul(beta,x))), beta)
    val = torch.where(lins > threshold, torch.mul(beta.sign(), x), val)
    return val.t()

class Gamma2(torch.autograd.Function):
    '''
    Gamma autograd function for heteogeneous adaptation

    Forward params : 
    - input : torch tensor of shape (batch_size, input_dimension)
    - n     : neuronal gain, torch tensor of shape (input_dimension,)
    - s     : saturaiton, torch tensor of shape (input_dimension,) 
    '''
    @staticmethod
    def forward(ctx, input, n, s):
        ctx.n = n
        ctx.s = s

        gamma_one = batch_softplus(input, n, 20)
        gamma_two = torch.sigmoid(torch.mul(n, input).t())
        output = torch.mul((1-s), gamma_one.t()) + torch.mul(s, gamma_two.t())

        ctx.save_for_backward(input, gamma_one.t(), gamma_two.t())

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma_one, gamma_two = ctx.saved_tensors
        
        n = ctx.n
        s = ctx.s

        grad_input = grad_n = grad_s = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output * ((1-s)*gamma_two + s*n*gamma_two*(1-gamma_two))
        if ctx.needs_input_grad[1]:
            grad_n = grad_output * ((1-s)/n * (input * gamma_two - gamma_one) + s*input*gamma_two*(1-gamma_two))
        if ctx.needs_input_grad[2]:
            grad_s = grad_output * (gamma_two - gamma_one)

        return grad_input, grad_n, grad_s

class gamma2(nn.Module):
    def __init__(self, n, s, hidden_size, random_init=False):
        super(gamma2, self).__init__()
        self.n = n
        self.s = s
        if type(n)==int or type(n)==float:
            self.n = n*torch.ones(hidden_size)

        if type(s)==int or type(s)==float:
            self.s = s*torch.ones(hidden_size)

    def forward(self, input):
        #print(self.n.shape)
        return Gamma2.apply(input, self.n, self.s)

class batch_gamma(nn.Module):
    def __init__(self, n, s, hidden_size, batchsize=100, random_init=False):
        super(batch_gamma, self).__init__()
        self.n = n
        self.s = s

        # print(self.n.shape)
        self.n = n.repeat(hidden_size, 1).t()
        # print(self.n.shape)
        self.s = s.repeat(hidden_size, 1).t()

        if type(n)==int or type(n)==float:
            self.n = n*torch.ones(hidden_size)

        if type(s)==int or type(s)==float:
            self.s = s*torch.ones(hidden_size)

    def forward(self, input):
        return Gamma2.apply(input, self.n, self.s)

class batch_gamma2(nn.Module):
    def __init__(self, n, s, hidden_size, batchsize=100, random_init=False):
        super(batch_gamma, self).__init__()
        self.n = n
        self.s = s

        # print(self.n.shape)
        self.n = n.repeat(hidden_size, 1).t()
        # print(self.n.shape)
        self.s = s.repeat(hidden_size, 1).t()

        if type(n)==int or type(n)==float:
            self.n = n*torch.ones(hidden_size)

        if type(s)==int or type(s)==float:
            self.s = s*torch.ones(hidden_size)

    def forward(self, input):
        return Gamma2.apply(input, self.n, self.s)

# ---------------------------
#   Verification
# ---------------------------

def verify_autograd(function_class):
    '''
    Verify the Pytorch backward pass.
    '''
    from torch.autograd import gradcheck

    # Create artificial data 
    np.random.seed(1)
    (test_batch, test_hid) = np.random.randint(10, size=2) # minimize magic numbers
    if function_class==Gamma: test_hid = 1
    test_n = torch.randn([test_hid], dtype=torch.double, requires_grad=True)
    test_s = torch.randn([test_hid], dtype=torch.double, requires_grad=True)
    test_x = torch.randn([test_batch, test_hid], dtype=torch.double, requires_grad=True)

    # Test autograd
    test_function = function_class.apply
    test_input = (test_x, test_n, test_s)
    try:
        test = gradcheck(test_function, test_input, eps=1e-6, atol=1e-4)
        print(f"{function_class} passed gradcheck: {test}")
    except Exception as e:
        print('GradcheckError:',e)
    return

# verify_autograd(Gamma)


# ---------------------------
#   Stand-alone
# ---------------------------

def torch_dgamma(input, n, s):

    if not (isinstance(n, float) or isinstance(n, int)): n = n.item()
    if not (isinstance(s, float) or isinstance(s, int)): s = s.item()

    gamma_one = F.softplus(input, beta = n)
    gamma_two = torch.sigmoid(torch.mul(n,input))
    output = ((1-s)*gamma_two + s*n*gamma_two*(1-gamma_two))

    return output

