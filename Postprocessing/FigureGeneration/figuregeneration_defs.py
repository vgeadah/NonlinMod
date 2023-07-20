import torch
import torch.nn as nn
import numpy as np
import sys
HOMEPATH = '/Users/victorgeadah-mac-ii/Documents/Documents - Victor’s MacBook Pro/3_Research/Adaptation'
sys.path.append(HOMEPATH)
from NetworkCreation.Networks import RNN
import torch.distributions as distributions

model_dirs={
    'MODELDIR':{
        'ANRU':'ANRU_lr0.0001_p0.0_hs400/',
        'RNN+gamma2':'RNN_lr0.0001_p0.0_hs400_gamma2_LP/',
        'RNN+gamma':'RNN_lr0.0001_p0.0_hs400_gamma_LP/',
        'RNN+ReLU':'RNN_lr0.0001_p0.0_hs400_ReLU/',
        'RNN+softplus':'RNN_lr0.0001_p0.0_hs400_softplus/'
    },
    'seed400_date':{
        'ANRU':'2021-03-12--1/',
        'RNN+gamma2': '2021-03-09/',
        'RNN+gamma':'2021-03-09--2/',
        'RNN+ReLU':'2021-03-08--0/',
        'RNN+softplus':'2021-03-21/'
    },
    'seed600_date':{
        'ANRU':'2021-03-27--0/',
        'RNN+gamma2': '2021-03-13/',
        'RNN+gamma':'2021-03-16/'
    }
}

def file_dir(seed, alpha=0.0, content='ns', net='ANRU', epoch=99) -> str:
    SAVEDIR = f"/Volumes/Geadah_2/raw_data/ssh_DMS/gamma/Training/SavedModels/psMNIST/{seed}/"
    if content=='ns':
        file_name = f'shapesignals_Step_a{alpha}_l200_p200.npy'
    elif content=='a':
        if alpha>0.0:
            file_name = f'net0_preactivations_Step_a{alpha}_l200_p200.npy'
        else:
            file_name = f'net0_preactivations__e{epoch}.npy'
    elif content=='h':
        if alpha>0.0:
            file_name = f'net0_hiddenstates_Step_a{alpha}_l200_p200.npy'
        else:
            file_name = f'net0_hiddenstates__e{epoch}.npy'
    else:
        file_name=content
    seed_date = f'seed{seed}_date'
    return SAVEDIR+model_dirs['MODELDIR'][net]+model_dirs[seed_date][net]+file_name



class Model(nn.Module):
    def __init__(self, hidden_size, rnn):#, adapt_module):
        super(Model, self).__init__()
        self.hidden_size = hidden_size

        # Modules
        self.lin = nn.Linear(hidden_size, 10)
        self.rnn = rnn

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs, y, order, 
                transform=None, return_ns=False, external_drive=None):
        h_net0, h_net1, a = None, None, None
        hiddens, shape_signals, pre_activations = [], [], []

        inputs = inputs[:, order] # permute inputs

        for i, input in enumerate(torch.unbind(inputs, dim=1)):
            x = input.unsqueeze(1)
            if transform is not None:
                x = transform(i, x)
            if external_drive is None:
                shift = 0.0
            elif isinstance(external_drive, distributions.multivariate_normal.MultivariateNormal):
                if (i>=200 and i<400):
                    shift = external_drive.sample().cuda()
                    shift.requires_grad=False
                    #print(shift[0])
                    #print('Drive sampled from :', external_drive)
                else: shift=0.0
            else:
                shift = external_drive.get_factor(i)-1
                #if i==300 and args.verbose: print('Shift:',shift)
            if return_ns and args.net_type=='ANRU': 
                h_net0, (shape_parameters, pre_activs) = self.rnn(x, h_net0, return_ns=return_ns, external_drive=shift)
            elif return_ns:
                h_net0, pre_activs = self.rnn(x, h_net0, external_drive=shift, return_ns=return_ns)
            else:
                h_net0 = self.rnn(x, h_net0, external_drive=shift)
            for temp in [h_net0, h_net1]:
                if temp is not None and temp.requires_grad: 
                    temp.retain_grad()

            #hiddens.append(h_net0.cpu().detach().numpy())
            if return_ns:
                if args.net_type=='ANRU': 
                    shape_signals.append(shape_parameters)
                pre_activations.append(pre_activs.cpu().detach().numpy())
                hiddens.append(h_net0.cpu().detach().numpy())

        if return_ns:
            if transform is not None: suffix=transform.name
            elif isinstance(external_drive, distributions.multivariate_normal.MultivariateNormal): suffix='mvnormal-m5.0-cov2.0-l200-p200'
            elif external_drive is not None: suffix=external_drive.name
            else: suffix=''
            if recover is not None:suffix+=f'_e{recover}'
            shape_signals_label = 'shapesignals_'+suffix+'.npy'
            hiddens_label = 'net0_hiddenstates_'+suffix+'.npy'
            preactivs_label = 'net0_preactivations_'+suffix+'.npy'

        #print(f'Norm/max/min of ht: {torch.norm(h_net0.clone().cpu().detach()).item()}/ {torch.max(h_net0.clone().cpu().detach()).item()}/ {torch.min(h_net0.clone().cpu().detach()).item()}')
        out = self.lin(h_net0)  # decode
        #if args.verbose: print('out',out)
        loss = self.loss_func(out, y)
        preds = torch.argmax(out, dim=1)
        correct = torch.eq(preds, y).sum().item()
        return loss, correct, hiddens

class AdaptModule(RNN):
    def __init__(self, input_dim, hidden_dim, bias=True, cuda=False,
                r_initializer=None, i_initializer=nn.init.xavier_normal_):
        RNN.__init__(self, input_dim, hidden_dim, bias=bias, cuda=cuda, learn_params=False,
                    r_initializer=r_initializer, i_initializer=i_initializer, nonlin='tanh')
        self.reset_parameters()
        self.counter= 0

    def load_from_ANRU(self, ANRU_net):
        self.U = ANRU_net.rnn.Uag
        self.V = ANRU_net.rnn.Wgg
        self.Wgn = ANRU_net.rnn.Wgn
        self.Wgs = ANRU_net.rnn.Wgs

    def decode(self, ht):
        nt, st = self.Wgn(ht), self.Wgs(ht)
        return nt, st

def npgam1(x,n):
    if np.linalg.norm(x*n)>20:
        return np.max(x,0)
    else:
        return (1 / n)*(np.log(1 + np.exp(n*x)))
def npgam2(x,n):
    if np.linalg.norm(x*n)>20:
        return 1.0
    elif np.linalg.norm(x*n)<-20:
        return 0.0
    else:
        return (np.exp(n*x))/(1 + np.exp(n*x))
def np_gamma(x, n, s):
    return (1-s)*npgam1(x,n) + s*npgam2(x,n)
def np_gamma_prime(x, n, s):
    return (1-s)*npgam2(x,n) + s*n*npgam2(x,n)*(1-npgam2(x,n))

def freqdomain_filter(f, alpha=1.0):
    # print(f)
    # out = (1j*2*np.pi*f)**(alpha)
    # print(out)
    out = []
    for fi in f:
        if fi==0.:
            out.append(0.0)
        else:
            out.append((1j*2*np.pi*fi)**(alpha))
    return np.array(out)

def frac_diff(input_signal, time_domain, alpha=1.0):
    Xf = np.fft.fft(input_signal)
    freq_domain = np.fft.fftfreq(time_domain.shape[-1], d=time_domain[-1]-time_domain[-2]) # assumes evenly spaced-grid
    Hf = freqdomain_filter(freq_domain, alpha=alpha)

    Rf = [Hf[i]*Xf[i] for i in range(len(freq_domain))]
    rt = np.fft.ifft(Rf)
    return rt

def frac_int(frac_diffd_signal, time_domain, alpha=1.0):
    '''
    Reverse frac diff
    '''
    Rf = np.fft.fft(frac_diffd_signal)
    freq_domain = np.fft.fftfreq(time_domain.shape[-1], d=time_domain[-1]-time_domain[-2]) # assumes evenly spaced-grid
    Hf = freqdomain_filter(freq_domain, alpha=alpha)
    Xf = []
    for i in range(len(freq_domain)):
        try:
            # a, b = Rf[i].real, Rf[i].imag
            # c, d = Hf[i].real, Hf[i].imag
            assert ~np.isnan(Hf[i])
            if (np.linalg.norm(Hf[i]))==0:
                if freq_domain[i]!=0:
                    print(f'Division by 0 at f={freq_domain[i]}')
                raise RuntimeError
            else:
                # Xf.append(((a*c+b*d)+1j*(b*c-a*d))/(c**2+d**2))
                Xf.append(Rf[i]/Hf[i])
        except RuntimeError:
            Xf.append(0.0)
        except AssertionError:
            #! check if better to handle as missing and remove from domain
            Xf.append(0.0) 
    # Xf = [Rf[i]/Hf[i] for i in range(len(freq_domain))]
    xt = np.fft.ifft(Xf)
    return xt


def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c
def power_func(x, a, b, c):
    return a * x**(-b) + c
def lin_func(x, a, b):
    return a * x + b

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap