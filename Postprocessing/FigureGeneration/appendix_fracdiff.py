
import numpy as np
import scipy
import torch
import seaborn as sns
import sys
from tqdm import tqdm
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as colors

HOMEPATH = '.' #! specify
sys.path.append(HOMEPATH)
from NetworkCreation.Networks import ANRU

import figuregeneration_defs as figdefs

# SAVEDIR = '/Volumes/Geadah_2/raw_data/ssh_DMS/gamma/Training'
SAVEDIR = '.' #! specify


# =================================================
# RECOVER 
# =================================================

task = 'psMNIST'
seed = 400
CONV_ENCODER = False

print(f'{task}, seed={seed}')

nhid = 400
rnn = ANRU(1, main_hidden_size=nhid, supervisor_hidden_size=50, cuda=False,
                    r_initializer='henaff', i_initializer='kaiming', adaptation_type='heterogeneous', supervision_type='local', verbose=False)
net = figdefs.Model(nhid, rnn)
if task=='psMNIST':
    if seed==200: last_model = torch.load(SAVEDIR+'/SavedModels/psMNIST/200/ANRU_lr0.0001_p0.0_hs400/e_99.pth.tar', map_location='cpu')
    elif seed==400: last_model = torch.load(SAVEDIR+'/SavedModels/psMNIST/400/ANRU_lr0.0001_p0.0_hs400/2021-03-12--1/e_99.pth.tar', map_location='cpu')
    elif seed==600: last_model = torch.load(SAVEDIR+'/SavedModels/psMNIST/600/ANRU_lr0.0001_p0.0_hs400/2021-03-27--0/e_99.pth.tar', map_location='cpu')
    elif seed==401: last_model = torch.load(SAVEDIR+'/SavedModels/psMNIST/401/ANRU/2022-02-23/e_99.pth.tar', map_location='cpu')
    elif seed==402: last_model = torch.load(SAVEDIR+'/SavedModels/psMNIST/402/ANRU/2022-02-27/e_99.pth.tar', map_location='cpu')
    elif seed==500: last_model = torch.load(SAVEDIR+'/SavedModels/psMNIST/500/ARUN/2021-07-28/e_99.pth.tar', map_location='cpu')
    else:
        raise KeyboardInterrupt('Unrecognized seed.')
    # last_model = torch.load('/Volumes/Geadah_2/raw_data/ssh_DMS/gamma/Training/SavedModels/psMNIST/600/ANRU_lr0.0001_p0.0_hs400/2021-03-27--0/e_99.pth.tar', map_location='cpu')
elif task=='gsCIFAR10': 
    if CONV_ENCODER:
        model_dir = SAVEDIR+f'/SavedModels/gsCIFAR10/ConvEncoder/{seed}/ANRU_lr0.0001_p0.0_hs400/'
        last_model = torch.load(model_dir+'e_99.pth.tar', map_location='cpu')
        pretrained_conv_encoder = torch.load(model_dir+'conv_block_e99.pth.tar', map_location='cpu')
        last_model['state_dict']['rnn.Uxh.weight'] = torch.ones(400,1)
        last_model['state_dict']['rnn.Uxh.bias'] = torch.zeros(400)
    else:
        last_model = torch.load(SAVEDIR+f'/SavedModels/gsCIFAR10/ANRU/ANRU_seed{seed}_lr1e-05_p0.0_hs400/e_99.pth.tar', map_location='cpu')

net_dict = net.state_dict()
last_model_dict = last_model['state_dict']
# 1. filter out unnecessary keys
last_model_dict = {k: v for k, v in last_model_dict.items() if k in net_dict}
# 2. overwrite entries in the existing state dict
net_dict.update(last_model_dict) 
# 3. load the new state dict
net.load_state_dict(last_model_dict)


# net.load_state_dict(last_model['state_dict'])

supervisor = figdefs.AdaptModule(1, 50)
supervisor.load_from_ANRU(net)

# neuron_number = 0 # 100 for cosyne
# Uxh = last_model['state_dict']['rnn.Uxh.weight'][neuron_number,0]
# bx = last_model['state_dict']['rnn.Uxh.bias'][neuron_number]
# Whh = last_model['state_dict']['rnn.Whh.weight'][neuron_number,neuron_number]
# bh = last_model['state_dict']['rnn.Whh.bias'][neuron_number]


parallel_net = net
# print(parallel_net.rnn.Uxh.weight.shape)
# print(parallel_net.rnn.Whh.weight)
# print(net.rnn.Whh.weight)
# print(torch.diag(torch.diag(parallel_net.rnn.Whh.weight)))

ISOLATED = True
print('Isolated',ISOLATED)

if CONV_ENCODER:
    from NetworkCreation.conv_architectures import ConvBlock_v1
    conv_encoder = ConvBlock_v1()
    conv_encoder.load_state_dict(pretrained_conv_encoder['state_dict'])
    parallel_net.rnn.Uxh.weight = torch.nn.Parameter(torch.diag(parallel_net.rnn.Uxh.weight[:,0]))
else:
    parallel_net.rnn.Uxh.weight = torch.nn.Parameter(torch.diag(parallel_net.rnn.Uxh.weight[:,0]))
if ISOLATED:
    parallel_net.rnn.Whh.weight = torch.nn.Parameter(torch.diag(torch.diag(parallel_net.rnn.Whh.weight)))
    # print(parallel_net.rnn.Uxh.weight.shape)
# print(parallel_net.rnn.Whh.weight)
# print(net.rnn.Whh.weight)
print('Recovered')

# ================================================
# COMPUTE 
# ================================================

NET0_RECURRENCE = False
PARALLEL = True
USE_LEARNED_NS = False
sin_input = False

hidden_size = [400 if PARALLEL else 1]
xis= [0.0, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0]
A,  H, G, N, S = [], [], [], [], []
n_iters = 20
resolution = 1
if task=='gsCIFAR10' and CONV_ENCODER:
    # x = torch.tensor(np.random.normal(loc=0.5, scale=0.1, size=[n_iters, 3, 32, 32])).type(torch.FloatTensor)
    # x_convd = conv_encoder(x)
    # from scipy.stats import norm
    # mean, std = norm.fit(x_convd.detach().cpu().numpy().flatten())
    # print(mean, std)
    # external_inputs = np.random.normal(loc=mean, scale=std, size=[n_iters, 400])
    '''
    For loading convenience, we set Uxh in this model to be the identity
    and feed the inputs x already encoded by the convolutional encoder.
    '''
    external_inputs = np.random.normal(loc=0.05, scale=0.4, size=[n_iters, 400]) # approximation of output of conv encoder
else:
    if NET0_RECURRENCE:
        external_inputs = np.random.normal(loc=0.5, scale=0.1, size=[n_iters,]) # = x, inputs
    elif PARALLEL:
        external_inputs = np.random.normal(loc=0.8, scale=0.1, size=[n_iters, 400])
    else:
        external_inputs = np.random.normal(loc=2.149376 , scale=3.2600198) # = a, preactivation
epsilons = np.random.normal(loc=0, scale=0.1, size=[n_iters, 400])

for j in tqdm(range(n_iters), desc='Get hiddens'):
    cnt=0
    x = external_inputs[j,:]
    if sin_input:
        # initial conditions
        # h, g = torch.tensor([x], dtype=torch.float32), torch.randn(50) #torch.zeros(400, dtype=torch.float32)
        h, g = torch.zeros(hidden_size, dtype=torch.float32), torch.randn(50)
        if PARALLEL: parallel_net.rnn.init_states(1)
        inps=[]
        for t in range(int(784/resolution)):
            eps = epsilons[j,:]
            offset = x#+eps #+0.1*np.sin(t*2*np.pi/(int(784/resolution)))
            if PARALLEL: inp = torch.tensor([offset], dtype=torch.float32).reshape(1,400)# offset*torch.ones(*(1,400), dtype=torch.float32)
            else: inp = offset
            # inps.append(inp[0,0].item())
            sin_drive =  0.1*np.sin(t*2*np.pi/(int(784/resolution)))+np.random.normal(loc=0, scale=0.01)
            # sin_drive = np.random.normal(loc=0, scale=0.2) if (t>int(400/resolution) and t<int(604/resolution)) else 0.0
            if NET0_RECURRENCE:
                a = Uxh*inp + bx + Whh*h + bh
                a += sin_drive
                g = supervisor(a, g)
                n, s = supervisor.decode(g)
                h = figdefs.np_gamma(a.detach().numpy(), n.detach().numpy(), s.detach().numpy())

                H.append(h)
                N.append(n.detach().numpy())
                S.append(s.detach().numpy())
            elif PARALLEL:
                print(inp.shape)
                h, (ct, _) = parallel_net.rnn(inp,h, 
                                return_ns=True,
                                external_drive=sin_drive)

                H.append(h.detach().numpy()[0,:])
                N.append(np.mean(torch.tensor(ct).squeeze().detach().numpy()[0,:]))
                S.append(np.mean(torch.tensor(ct).squeeze().detach().numpy()[1,:]))
    else:
        for xi in xis:
            if USE_LEARNED_NS:
                temp = np.load('/Volumes/Geadah_2/raw_data/ssh_DMS/gamma/Training/SavedModels/psMNIST/400/ANRU_lr0.0001_p0.0_hs400/2021-03-12--1/'\
                                +'shapesignals_Step_a{:2.1f}_l200_p200.npy'.format(xi))
                learned_n, learned_s = np.mean(temp[:,0,0,:], axis=1), np.mean(temp[:,0,100,:], axis=1)

            # initial conditions
            h, g = torch.tensor([0.0], dtype=torch.float32), torch.randn(50)
            if PARALLEL: 
                h = torch.zeros(400, dtype=torch.float32)
                # h = torch.tensor(np.array([x]), dtype=torch.float32)
                parallel_net.rnn.init_states(1)
            inps=[]
            for t in range(150):
                if sin_input:
                    eps = np.random.normal(loc=0, scale=0.02)
                    offset = x+eps+0.2*np.sin(t*2*np.pi/(int(784/resolution)))
                    inp = torch.tensor(offset, dtype=torch.float32).reshape(1,400) #*torch.ones(*(1,400), dtype=torch.float32)
                    inps.append(inp[0,0].item())
                else:
                    # eps = np.random.normal(loc=0, scale=0.1, size=[400])
                    eps = epsilons[j,:]
                    if PARALLEL:
                        inp = torch.tensor(np.array([x+eps])).reshape(1,400).type(torch.FloatTensor)
                    # if CONV_ENCODER: 
                    #     inp = encoder(torch.tensor([x])) + torch.tensor(eps)
                    elif NET0_RECURRENCE: 
                        inp = torch.tensor([x])
                    offset = xi if (t>50 and t<101) else 0.0
                if NET0_RECURRENCE:
                    # print(Whh.shape, torch.tensor(h).shape)
                    # a = torch.mul(Whh, torch.tensor(h))#+bh+torch.mul(Uxh, inp)+bx
                    a = Uxh*inp + bx + Whh*h + bh + offset
                    # a = a.reshape(400,-1).type(torch.FloatTensor)
                    # print(a.shape)
                    if USE_LEARNED_NS:
                        h = np_gamma(a.detach().numpy(), learned_n[t], learned_s[t]).item()
                    else: # calculate n, s
                        g = supervisor(a, g)
                        n, s = supervisor.decode(g)
                        h = np_gamma(a.detach().numpy(), n.detach().numpy(), s.detach().numpy())

                    H.append(h)
                    N.append(n.detach().numpy())
                    S.append(s.detach().numpy())
                elif PARALLEL:
                    h, (ct, _) = net.rnn(inp,h, 
                                return_ns=True,
                                external_drive=offset)
                    H.append(h.detach().numpy()[0,:])
                    N.append(np.mean(torch.tensor(np.array(ct)).squeeze().detach().numpy()[0,:]))
                    S.append(np.mean(torch.tensor(np.array(ct)).squeeze().detach().numpy()[1,:]))
                else:
                    # print(a)
                    g = supervisor(inp, g)
                    n, s = supervisor.decode(g)
                    h = np_gamma(inp.detach().numpy(), n.detach().numpy(), s.detach().numpy())


# print(net.rnn.Whh.weight - parallel_net.rnn.Whh.weight)

# A = np.array(A).reshape(n_iters, len(xis), int(784/resolution))
second_dim = 1 if sin_input else len(xis)
if PARALLEL:
    H = np.array(H).reshape(n_iters, second_dim, int(150/resolution), 400)
    # N = np.array(N).reshape(n_iters, 1, int(784/resolution), 400)
    # S = np.array(S).reshape(n_iters, 1, int(784/resolution), 400)
else:
    H = np.mean(np.array(H).reshape(n_iters, second_dim, int(784/resolution), 1), axis=0)
N = np.mean(np.array(N).reshape(n_iters, second_dim, int(150/resolution)), axis=0)
S = np.mean(np.array(S).reshape(n_iters, second_dim, int(150/resolution)), axis=0)
del A, G

# np.save(HOMEPATH+f'/Postprocessing/calculations/fracdiff/{task}_seed{seed}_ht_res{resolution}.npy', H)
# np.save(HOMEPATH+f'/Postprocessing/calculations/fracdiff/{task}_seed{seed}_nt_res{resolution}.npy', N)
# np.save(HOMEPATH+f'/Postprocessing/calculations/fracdiff/{task}_seed{seed}_st_res{resolution}.npy', S)

# ================================================
# PLOT 
# ================================================


fig = plt.figure(figsize=[6,6], constrained_layout=True)
plt.rcParams['xtick.labelsize']=7
plt.rcParams['ytick.labelsize']=7
plt.rcParams['axes.labelsize']=8
plt.rcParams['axes.titlesize']=9
plt.rcParams['legend.fontsize']=7
plt.rcParams['lines.markersize']=4
plt.rcParams['boxplot.meanprops.markersize'] = 4
plt.rcParams['legend.markerscale'] = 1.0
plt.rcParams['legend.title_fontsize'] = plt.rcParams['axes.titlesize']

gs = fig.add_gridspec(nrows=3, ncols=2)#, height_ratios=(1.5,1.,1.))
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])

ax_mse = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1,1])
# ax2.set_axis_off()

ax_mse2 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2,1])

x = np.arange(150)[::resolution] #H.shape[2]
if x.shape[0] < H.shape[2]:
    x = np.arange(150+resolution)[::resolution]
elif x.shape[0] > H.shape[2]:
    x = np.arange(150-resolution)[::resolution]
cmap = plt.get_cmap('Greys')
new_cmap = figdefs.truncate_colormap(cmap, 0.1,0.9)
colors = new_cmap(np.linspace(0.0,1.0,len(xis)))

alphas = np.linspace(0.0,1.0,1001)
offset = int(25/resolution)

# Average first over neurons
MSEs = []
for i, signal in  enumerate(tqdm(np.mean(np.mean(H, axis=0), axis=-1), desc='Frac diff per xi')):
    if i==0: continue
    step_signal = np.where(np.abs(x-75)<25, 0.8+xis[i], 0.8)
    MSEs_sublevel = []
    for alpha in alphas:
        # print(f'\ni={i}, alpha={alpha}')
        # signal_fracdiffd_func = figdefs.fractional_derivative_function(signal, x, alpha=alpha)
        # signal_fracdiffd = np.array([signal_fracdiffd_func(t) for t in x])

        # # correction
        # step_signal_fracdiffd = figdefs.frac_diff(step_signal, x, alpha=alpha).real
        # step_signal_fracdiffd_fracintd = figdefs.frac_diff(step_signal_fracdiffd, x, alpha=alpha).real
        # scalar_offset = step_signal-step_signal_fracdiffd_fracintd.real

        # signal_fracdiffd = figdefs.frac_diff(signal, x, alpha=alpha).real
        # y1 = signal_fracdiffd+scalar_offset

        # positional offset 
        signal_fracdiffd = figdefs.frac_diff(signal, x, alpha=alpha).real
        y1 = signal_fracdiffd - signal_fracdiffd[offset]

        MSEs_sublevel.append(MSE(y1[int(40/resolution):int(110/resolution)], step_signal[int(40/resolution):int(110/resolution)]))
        # MSEs_sublevel.append(MSE(y1, step_signal))
    MSEs.append(MSEs_sublevel)
    ax_mse.plot(alphas, MSEs_sublevel, c=colors[i])
    # msesum_mins.append(np.min(np.sum(MSEs, axis=0)))
ax_mse.plot(alphas, np.sum(MSEs, axis=0)/len(xis), c='r', label='average')

# Z = data_df.pivot_table(index='offset', columns='alpha', values='MSE').T.values
# print(np.argmin(data_df['MSE'].values))
# print(data_df.iloc[np.argmin(data_df['MSE'].values)])
# X_unique = np.sort(data_df.offset.unique())
# Y_unique = np.sort(data_df.alpha.unique())
# X, Y = np.meshgrid(X_unique, Y_unique)
# CS = ax2.contourf(X, Y, Z, cmap='viridis', levels=10);
# # ax2.clabel(CS)
# fig.colorbar(CS, ax=ax2, location='top') 

ax_mse.set_yscale('log');
ax_mse.set_xlabel(r'Frac diff order $\alpha$')
ax_mse.set_ylabel('MSE')
ax_mse.set_title('MSE between ARU hidden states\n and frac. diff. hidden states')
sns.despine(ax=ax_mse, offset=5, trim=True);

# for _xi, _alpha in zip(xis[1:], [alphas[i] for i in np.argmin(MSEs, axis=1)]):
#     print('xi = {:2.2f}, alpha = {:1.4f}'.format(_xi, _alpha))

alpha = alphas[np.argmin(np.sum(MSEs, axis=0))]
ax_mse.axvline(x=alpha, ls=':', c='r', label='argmin');
ax_mse.legend(loc='lower right');

# alpha = data_df.iloc[np.argmin(data_df['MSE'].values)]['alpha']
# scalar_offset = data_df.iloc[np.argmin(data_df['MSE'].values)]['offset']

def adaptive_index(signal, res=1, window=15):
    # signal : (n_steps, )
    L_early, L_late = int(100/res), int(149/res)
    r_early = np.mean(signal[L_early:L_early+window])
    r_late = np.mean(signal[L_late-window:L_late])
    if (r_early+r_late)<1e-10:
        return 0.0
    else:
        return (r_early-r_late)/(r_early+r_late)

As, A0s, A1s = [], [], []
adaptive_indices = pd.DataFrame(columns=['All','Even','Odd'])

for signal in np.mean(H[:,-1,:,:], axis=0).T:
    # adaptive_indices = adaptive_indices.append({
    #     'All':adaptive_index(signal), 
    #     'Even':adaptive_index(signal[::2], res=2*resolution), 
    #     'Odd':adaptive_index(signal[1::2], res=2*resolution)
    #     }, ignore_index=True)
    adaptive_indices = pd.concat([
            adaptive_indices, 
            pd.DataFrame(data={
                'All':[adaptive_index(signal)], 
                'Even':[adaptive_index(signal[::2], res=2*resolution)], 
                'Odd':[adaptive_index(signal[1::2], res=2*resolution)]
                })
        ], ignore_index=True)
            # adaptive_exp = adaptive_exp.append({
    # ax0.plot(x, signal/np.mean(signal[int(300/resolution):int(400/resolution)]), ',', c='k')
# ax0.hist(np.array(As), density=True, bins=np.linspace(-1,1,15), alpha=0.4)
# ax0.hist(np.array(A0s), density=True, bins=np.linspace(-1,1,15), alpha=0.4)
# ax0.hist(np.array(A1s), density=True, bins=np.linspace(-1,1,15), alpha=0.4)
# sns.histplot(data=np.array(As), ax=ax0, label='all', element="step",stat="density")
# sns.histplot(data=np.array(A0s), ax=ax0, label='even', element="step",stat="density")
# sns.histplot(data=np.array(A1s), ax=ax0, label='odd', element="step",stat="density")
# ax0.set_xlim([-1,1])

# hp = sns.histplot(data=adaptive_indices, element="step", ax=ax0, bins=20)
hp = sns.kdeplot(data=adaptive_indices, ax=ax0, fill=True)
sns.move_legend(hp, "upper left", title='Indices')
ax0.set_xlim([-1,1])

for i, signal in enumerate(np.mean(np.mean(H, axis=0), axis=-1)):
    step_signal = np.where(np.abs(x-75)<25, 0.8+xis[i], 0.8)

    # step_signal_fracdiffd = figdefs.frac_diff(step_signal, x, alpha=alpha).real
    # step_signal_fracdiffd_fracintd = figdefs.frac_diff(step_signal_fracdiffd, x, alpha=alpha).real
    # scalar_offset = step_signal-step_signal_fracdiffd_fracintd.real

    # signal_fracdiffd = figdefs.frac_diff(signal, x, alpha=alpha).real
    # # signal_fracdiffd_fracintd = figdefs.frac_int(signal_fracdiffd, x, alpha=alpha).real
    # # scalar_offset = step_signal-signal_fracdiffd_fracintd
    # y1 = signal_fracdiffd+scalar_offset

    # positional offset 
    signal_fracdiffd = figdefs.frac_diff(signal, x, alpha=alpha).real
    y1 = signal_fracdiffd - signal_fracdiffd[offset]
        
    # signal_fracdiffd_func = figdefs.fractional_derivative_function(signal, x, alpha=alpha)
    # signal_fracdiffd = np.array([signal_fracdiffd_func(t) for t in x])

    # ax0.plot(x, signal, c=colors[i]);
    ax1.plot(x, signal,  c=colors[i]);# 
    ax2.plot(x, y1,  c=colors[i]);# 
    label = None
    if i==len(xis)-1:
        label='drive'
    ax2.plot(x, step_signal,  c='tab:orange', ls='--', label=label);

# ax0.set_ylabel('Hidden states $h_t$')
# ax0.set_title('Average ARU activity')
# ax0.set_title('Adaptive index')
# ax0.legend(title='Indices')
ax0.set_xlabel('Adaptive Index')
ax1.set_ylabel('Hidden states $h_t$')

# ax1.set_xlabel('Time steps $t$')
ax2.set_xlabel('Time steps $t$')
ax2.set_ylabel('Signal')
ax2.set_title('Frac. diff. hidden states '+r'($\alpha = $'+'${:1.3f}$)'.format(alpha)+'\noverlayed with original step drive')
ax2.legend();


# ---------------------------------------------------------------------

# Look at dist of measured alpha over neurons
resolution=1
MSEs = []
alphas = np.linspace(-1.0,1.0,100)
for neuron, signal in  enumerate(tqdm((np.mean(H, axis=0)[-1]).T, desc='Frac diff per neuron')):
    step_signal = np.where(np.abs(x-75)<25, 0.8+xis[-1], 0.8)
    MSEs_sublevel = []
    for alpha in alphas:
        signal_fracdiffd = figdefs.frac_diff(signal, x, alpha=alpha).real
        y1 = signal_fracdiffd - signal_fracdiffd[offset]
        MSEs_sublevel.append(MSE(y1[int(40/resolution):int(110/resolution)], step_signal[int(40/resolution):int(110/resolution)]))

    MSEs.append(MSEs_sublevel)
    ax_mse2.plot(alphas, MSEs_sublevel, c=colors[-1], alpha=0.1)
    # msesum_mins.append(np.min(np.sum(MSEs, axis=0)))

ax_mse2.plot(alphas, np.sum(MSEs, axis=0)/400, c='r', label='average')
ax_mse2.set_yscale('log')
ax_mse2.set_ylabel("MSE")
ax_mse2.set_xlabel(r'Frac diff order $\alpha$')

alpha_dist_indices = np.argmin(MSEs, axis=1)
alpha_dist = [alphas[i] for i in alpha_dist_indices]
sns.histplot(data=alpha_dist, ax=ax5, element="step", stat="density", alpha=0.5, bins=np.linspace(-1,1,9)) #fill=True,
ax5.set_xlim([-1,1])

# ax2.clear()
# hidds=[]
# for i in range(400):
#     if -0.25<= alpha_dist[i] <-0.0:
#         hidds.append(np.mean(H, axis=0)[-1,:,i])
# ax2.plot(np.mean(hidds, axis=0)[::3])

ax2.clear()
hidds=[]
for i in range(400):
    if 0.0<= alpha_dist[i] <0.25:
        hidds.append(np.mean(H, axis=0)[-1,:,i])
ax2.plot(np.mean(hidds, axis=0)[2::3])


# ---------------------------------------------------------------------

for ax in [ax0, ax1, ax_mse]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=plt.Normalize(vmin=xis[0], vmax=xis[-1]))
cbar = fig.colorbar(sm, ax=[ax1, ax2], location='right', shrink=1.0, aspect=30) 
cbar.ax.set_title(r"$\xi$", pad=10)

if CONV_ENCODER: task += '_conv'
fig.suptitle(f'Fractional integration by ARUs.\n Isolated: {ISOLATED}, weights from: {task}, seed={seed}.')
# plt.savefig(HOMEPATH+f'/Postprocessing/figures/isolated{ISOLATED}_{task}_seed{seed}_hidsfracdiffd_res{resolution}_Ai{15}_kde.pdf', format='pdf');
plt.show();
