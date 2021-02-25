import numpy as np 
import torch
import torch.nn as nn

from .common import modrelu, henaff_init,cayley_init,random_orthogonal_init
from .exp_numpy import expm
import sys
import math
from .gamma_function import gamma, gamma2, batch_gamma
verbose = False

ReLU = torch.nn.ReLU()
tanh = torch.nn.Tanh()

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, nonlin='gamma', n_init=1.0, s_init=1.0, bias=False, cuda=False,
                r_initializer=None, i_initializer=nn.init.xavier_normal_, learn_params=False):
        super(RNN, self).__init__()
        self.cudafy = cuda
        self.hidden_dim = hidden_dim
        self.r_initializer = r_initializer
        self.i_initializer = i_initializer

        # Linear layers
        self.U = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        if learn_params:
            self.n = nn.Parameter(torch.tensor([n_init]))
            self.s = nn.Parameter(torch.tensor([s_init]))
        else:
            self.n = torch.tensor([n_init])
            self.s = torch.tensor([s_init])

        if nonlin == 'ReLU':
            self.nonlinearity = nn.ReLU()
        if nonlin == 'modrelu':
            self.nonlinearity = modrelu(hidden_dim)
        elif nonlin == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlin == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        elif nonlin == 'gamma':
            self.nonlinearity = gamma(self.n, self.s)
        elif nonlin == 'gamma2':
            self.nonlinearity = gamma2(self.n, self.s, hidden_dim)
        else:
            self.nonlinearity = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.r_initializer == "cayley":
            self.V.weight.data = torch.as_tensor(cayley_init(self.hidden_size))
            A = self.V.weight.data.triu(diagonal=1)
            A = A - A.t()
            self.V.weight.data = expm(A)
        elif self.r_initializer == "henaff":
            self.V.weight.data = torch.as_tensor(henaff_init(self.hidden_dim))
            A = self.V.weight.data.triu(diagonal=1)
            A = A - A.t()
            self.V.weight.data = expm(A)
        elif self.r_initializer == "random":
            self.V.weight.data = torch.as_tensor(random_orthogonal_init(self.hidden_dim))
            A = self.V.weight.data.triu(diagonal=1)
            A = A - A.t()
            self.V.weight.data = expm(A)
        elif self.r_initializer == 'xavier':
            nn.init.xavier_normal_(self.V.weight.data)
        elif self.r_initializer == 'kaiming':
            nn.init.kaiming_normal_(self.V.weight.data)
        elif self.r_initializer == 'normal':
            stdv = 1 / math.sqrt(self.hidden_dim)
            for weight in self.parameters():
                weight.data.normal_(0, stdv)

        if self.i_initializer == "xavier":
            nn.init.xavier_normal_(self.U.weight.data)
        elif self.i_initializer == 'kaiming':
            nn.init.kaiming_normal_(self.U.weight.data)
        elif self.i_initializer == 'normal':
            stdv = 1 / math.sqrt(self.hidden_dim)
            for weight in self.parameters():
                weight.data.normal_(0, stdv)


    def forward(self, x, hidden, external_drive=0.0):
        if hidden is None:
            hidden = x.new_zeros(x.shape[0], self.hidden_dim, requires_grad=True)

        h = a = self.U(x) + self.V(hidden) + external_drive #*torch.ones_like(x.clone())
        if self.nonlinearity:
            h = self.nonlinearity(h)
        return h

class GammaRNN(RNN):
    '''
    Variant of the Elmann RNN above with a fixed gamma nonlinearity, 
    with shape parameters [n,s] provided as inputs in the forward pass.
    '''
    def __init__(self, input_dim, hidden_dim, bias=True, cuda=False,
                r_initializer=None, i_initializer=nn.init.xavier_normal_):
        RNN.__init__(self, input_dim, hidden_dim, bias=bias, cuda=cuda, learn_params=False,
                    r_initializer=r_initializer, i_initializer=i_initializer)
        self.reset_parameters()

    def forward(self, x, hidden, n, s, cont=False):
        if hidden is None:
            hidden = x.new_zeros(x.shape[0], self.hidden_dim, requires_grad=True)

        h = self.U(x) + self.V(hidden)
        nonlinearity = gamma2(n, s, self.hidden_dim)
        h = nonlinearity(h)

        return hion

class ANRU(nn.Module):
    r"""
    ANRU = Adaptive Nonlinearity Recurrent Unit. 
    Composed of a main processing unit (net0, module of type GammaRNN) and 
        a second RNN (net1) tasked with modulating the nonlinearity of net0.
    Args:
        input_size: (int)
        main_hidden_size: (int)
        supervisor_hidden_size: (int)

        __optional__
        adaptation_type: (str)
        cuda: (bool)

    Equations: For neuron i

            a^i_t = U^{i,:}_{xh} x_t + W^{i,:}_{hh} h_{t-1} +  b_a
              g_t = \phi( U_{ag} a_t + W_{gg} g_{t-1} + b_g )
    [n^i_t s^i_t] = \phi( W_{gc} g_t +  b_c )
            h^i_t = \gamma ( a^i_t ; n^i_t , s^i_t)

    TODO: cleanup self.init_states, merge self.adaptation_mechanism with self.forward
    """
    def __init__(self, input_size, main_hidden_size, supervisor_hidden_size, 
                adaptation_type='heterogeneous', cuda=False, r_initializer=nn.init.xavier_normal_, i_initializer=nn.init.xavier_normal_,
                verbose=False):
        super(ANRU, self).__init__()
        self.net0_hidden_size = main_hidden_size 
        self.net1_hidden_size = supervisor_hidden_size
        self.input_size = input_size
        self.CUDA = cuda
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.net0_r_init = r_initializer
        self.net0_i_init = i_initializer
        
        self.verbose = verbose

        self.signal_dim = 1
        if adaptation_type=='heterogeneous':
            self.signal_dim = main_hidden_size

        # Modules
        ## main rnn
        self.Uxh = nn.Linear(input_size, main_hidden_size, bias=True)
        self.Whh = nn.Linear(main_hidden_size, main_hidden_size, bias=True)
        ## supervisor
        self.Uag = nn.Linear(1, supervisor_hidden_size, bias=True)
        self.Wgg = nn.Linear(supervisor_hidden_size, supervisor_hidden_size, bias=True)
        self.Wgn = nn.Linear(supervisor_hidden_size, 1, bias=True)
        self.Wgs = nn.Linear(supervisor_hidden_size, 1, bias=True)

        self.params = [self.Uxh, self.Whh, self.Uag, self.Wgg, self.Wgn, self.Wgs]
        #print(self.Uag.weight)
        self.init_weights()

    def init_from_scheme(self, module, init_scheme):
        if init_scheme == "xavier":
            nn.init.xavier_normal_(module.weight.data)
        elif init_scheme == 'kaiming':
            nn.init.kaiming_normal_(module.weight.data)
        elif init_scheme == 'normal':
            torch.nn.init.normal_(self.V.weight.data)
        elif init_scheme == "henaff":
            module.weight.data = torch.as_tensor(henaff_init(module.weight.shape[0]))
            A = module.weight.data.triu(diagonal=1)
            A = A - A.t()
            module.weight.data = expm(A)
        else:
            print("Unrecognised init_scheme, using xavier")
            nn.init.xavier_normal_(module.weight.data)

    def init_weights(self):
        '''
        Specific weight initalisation here.
        '''
        for input_module in [self.Uxh, self.Uag]:
            self.init_from_scheme(input_module, self.net0_i_init)
        for recurrent_module in [self.Whh, self.Wgg]:
            self.init_from_scheme(recurrent_module, self.net0_r_init)
        for module in [self.Uxh, self.Uag, self.Whh, self.Wgg]:
           torch.nn.init.constant_(module.bias, 0.0)

        # (n^i_t,s^i_t) decoders weights
        nn.init.xavier_normal_(self.Wgn.weight.data, gain=0.1)
        nn.init.xavier_normal_(self.Wgs.weight.data, gain=0.01)
        self.Wgn.bias.data = torch.ones_like(self.Wgn.bias)* (5+2*torch.rand(1))
        self.Wgs.bias.data = torch.zeros_like(self.Wgs.bias)


    def init_states(self, batch_size):
        self.gt = torch.zeros((batch_size, self.net0_hidden_size, self.net1_hidden_size), requires_grad=True)
        if self.CUDA:
            self.gt = self.gt.cuda()

    def adaptation_mechanism(self, net0_preactivation):
        '''
        Conceptually, maps the pre-activation to a nonlinear activation function
        Args:
            net0_preactivation: input of shape (batch_size, 1)
            neuron_index:  int
        '''
        # Verification
        if self.gt is None:
            self.gt = torch.zeros((x.shape[0], self.net0_hidden_size, self.net1_hidden_size), requires_grad=True)
        if self.CUDA:
            self.gt = self.gt.cuda()
        
        # Forward pass
        net1_preactivation = self.Uag(net0_preactivation.unsqueeze(-1)) + self.Wgg(self.gt)
        self.gt = self.tanh(net1_preactivation)
        nt, st = self.Wgn(self.gt).squeeze(-1), self.Wgs(self.gt).squeeze(-1)
        return nt, st

    def forward(self, x, ht, return_ns=False, external_drive=0.0):
        # Verification
        if ht is None:
            ht = torch.zeros((x.shape[0], self.net0_hidden_size), requires_grad=True)
            if self.CUDA: 
                ht = ht.cuda()
        
        shape_parameters=[]        
        
        # Forward pass
        at = self.Uxh(x) + self.Whh(ht) + external_drive
        nt, st = self.adaptation_mechanism(at)
        nonlin = gamma2(nt, st, self.net0_hidden_size)
        ht = nonlin(at)
        
        shape_parameters.append(torch.cat((nt.clone(),st.clone())).cpu().detach().numpy())
        if return_ns:
            return ht, (shape_parameters, at)
        else:
            return ht

class LSTM(nn.Module):
    '''
    Single step LSTM.
    '''
    def __init__(self, input_size, hidden_size,cuda=True):
        super(LSTM, self).__init__()
        self.CUDA = cuda
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.Wf = nn.Linear(input_size+hidden_size, hidden_size,bias=True)
        self.Wi = nn.Linear(input_size+hidden_size, hidden_size,bias=True)
        self.Wo = nn.Linear(input_size+hidden_size, hidden_size,bias=True)
        self.Wg = nn.Linear(input_size+hidden_size, hidden_size,bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        torch.nn.init.xavier_normal_(self.Wf.weight)
        torch.nn.init.xavier_normal_(self.Wi.weight)
        torch.nn.init.xavier_normal_(self.Wo.weight)
        torch.nn.init.xavier_normal_(self.Wg.weight)
        self.params = [self.Wf.weight,self.Wf.bias,self.Wi.weight,self.Wi.bias,self.Wo.weight,self.Wo.bias,self.Wg.weight,self.Wg.bias]
        self.orthogonal_params = []
        
    def init_states(self,batch_size):
        self.ct = torch.zeros((batch_size,self.hidden_size))
        if self.CUDA:
            self.ct = self.ct.cuda()
        
    def forward(self,x,hidden=None, external_drive=0.0):
        if hidden is None:
            hidden = x.new_zeros(x.shape[0],self.hidden_size)
        #if self.ct is None:
        #    self.init_states(x.shape[0])

        inp = torch.cat((hidden,x),1)
        #ext = external_drive*torch.cat((torch.ones_like(hidden.clone()), torch.zeros_like(x.clone())),1)
        ft = self.sigmoid(self.Wf(inp))
        it = self.sigmoid(self.Wi(inp))
        ot = self.sigmoid(self.Wo(inp)+external_drive)
        gt = self.tanh(self.Wg(inp))
        self.ft = ft
        self.it = it
        self.gt = gt
        self.ot = ot
        self.ct = torch.mul(ft,self.ct) + torch.mul(it, gt)
        hidden = torch.mul(ot, self.tanh(self.ct))
        return hidden

class GRU(nn.Module):
    '''
    Single step GRU.

    TODO: cat hidden and inputs in forward.
    '''
    def __init__(self, input_size, hidden_size,cuda=True):
        super(GRU, self).__init__()
        self.CUDA = cuda
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.Wir = nn.Linear(input_size, hidden_size, bias=True)
        self.Whr = nn.Linear(hidden_size, hidden_size, bias=True)
        self.Wiz = nn.Linear(input_size, hidden_size, bias=True)
        self.Whz = nn.Linear(hidden_size, hidden_size, bias=True)
        self.Win = nn.Linear(input_size, hidden_size, bias=True)
        self.Whn = nn.Linear(hidden_size, hidden_size, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        # torch.nn.init.xavier_normal_(self.Wf.weight)
        # torch.nn.init.xavier_normal_(self.Wi.weight)
        # torch.nn.init.xavier_normal_(self.Wo.weight)
        # torch.nn.init.xavier_normal_(self.Wg.weight)
        self.params = []
        for lin_module in [self.Wir, self.Whr, self.Wiz, self.Whz, self.Win, self.Whn]:
            self.params.append(lin_module.weight)
            self.params.append(lin_module.bias)

        # self.params = [self.Wir.weight,self.Wir.bias,self.Wi.weight,self.Wi.bias,self.Wo.weight,self.Wo.bias,self.Wg.weight,self.Wg.bias]
        self.orthogonal_params = []
        
    def init_states(self,batch_size):
        
        self.ct = torch.zeros((batch_size,self.hidden_size))
        if self.CUDA:
            self.ct = self.ct.cuda()
        
    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = x.new_zeros(x.shape[0],self.hidden_size)
        # inp = torch.cat((hidden,x),1)

        rt = self.sigmoid(self.Wir(x)+self.Whr(hidden))
        zt = self.sigmoid(self.Wiz(x)+self.Whz(hidden))
        nt = self.tanh(self.Win(x)+torch.mul(rt, self.Whn(hidden)))
        ht = torch.mul((1-zt), nt)+torch.mul(zt, hidden)
        return ht