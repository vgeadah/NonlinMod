import numpy as np
import pickle
# import torch
# import torch.nn as nn
import os 
import traceback
# from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.optimize import curve_fit
import pandas as pd 
from sklearn.metrics import mean_squared_error as MSE

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# plt.style.use('default')
import seaborn as sns
import torch

from scipy import stats

import sys
HOMEPATH = '.' #! specify
sys.path.append(HOMEPATH)
from NetworkCreation.Networks import ARUN
import figuregeneration_defs as figdefs
SAVEDIR = '' #! specify

# ======================================================================================

ANRUv2_shapesignals = pickle.load(open(HOMEPATH+'/Postprocessing/notebooks/ANRUv2_shapesignals_seed400.json','rb'))

# xis = [ 0.0, 0.5,1.0,2.5,5.0,7.5,10.0,15.0,20.0,30.0]
# cmap = plt.get_cmap('plasma_r') #plasma_r, Greys
# new_cmap = figdefs.truncate_colormap(cmap, 0.2,1.0)
# col = new_cmap(np.linspace(0.0,1.0,len(xis)))
# col_dict = {str(xis[i]): col[i] for i in range(len(xis))}
xis = [0.5,1.0,2.5,5.0,7.5,10.0,15.0,20.0, 30.0]
cmap = plt.get_cmap('plasma_r') #plasma_r, Greys
new_cmap = figdefs.truncate_colormap(cmap, 0.1,1.0)

def lognormal_xi(xi):
    return (np.log(xi)-np.log(xis)[0])/np.log(xis)[-1]

xis_lognormalized = lognormal_xi(xis)
col = new_cmap(xis_lognormalized)
col_dict = {str(xis[i]): new_cmap(xis_lognormalized[i]) for i in range(len(xis))}
col_dict['0.0'] = 'tab:grey'
Adaptive_index = True

fig = plt.figure(figsize=[7., 4.5], constrained_layout=True);
# fig.set_constrained_layout_pads(w_pad=1./72., h_pad=2./72.,
#             hspace=-1./72., wspace=2./72.)

plt.rcParams['xtick.labelsize']=8
plt.rcParams['ytick.labelsize']=8
plt.rcParams['axes.labelsize']=8
plt.rcParams['axes.titlesize']=9
plt.rcParams['legend.fontsize']=7
plt.rcParams['lines.markersize']=4
plt.rcParams['boxplot.meanprops.markersize'] = 4
plt.rcParams['legend.markerscale'] = 1.0
plt.rcParams['legend.title_fontsize'] = plt.rcParams['axes.titlesize']

gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.0, wspace=1.0, width_ratios=[1,2])

gs0 = gs[0].subgridspec(3, 1, hspace=0.2, wspace=0.1, height_ratios=[1,1,2.2])
gs1 = gs[1].subgridspec(3, 2, hspace=0.2, wspace=0.01, height_ratios=[2.2,1,1])

ax_nt = fig.add_subplot(gs0[0])
ax_st = fig.add_subplot(gs0[1])

ax_alpha_connected = fig.add_subplot(gs1[0, 0])
ax_alpha_isolated = fig.add_subplot(gs1[0, 1])

ax_ht0 = fig.add_subplot(gs1[1, 0])
ax_ht1 = fig.add_subplot(gs1[2, 0])
ax_ht2 = fig.add_subplot(gs1[1, 1])
ax_ht3 = fig.add_subplot(gs1[2, 1])

ax_fracdiff = fig.add_subplot(gs0[2])

for label, ax in zip(
        ['b', 'c', 'd'],
        [ax_alpha_connected, ax_alpha_isolated, ax_fracdiff]
    ):
    # print(label, ax)
    ax.text(-0.15, 1.2, label, transform=ax.transAxes,
      fontsize=13, fontweight='bold', va='top', ha='right')
ax_nt.text(-0.15, 1.4, 'a', transform=ax_nt.transAxes,
      fontsize=13, fontweight='bold', va='top', ha='right')
# if not Adaptive_index:
#     ax_fracdiff.text(-0.2, 1.15, 'e', transform=ax_fracdiff.transAxes,
#         fontsize=13, fontweight='bold', va='top', ha='right') #'            c'


# ===============================================================================================================
#                   Fractional order
# ===============================================================================================================
def p_to_label(p, label_type='star'):
    index = 1 if label_type=='p' else 0
    if 1.00e-02 < p <= 5.00e-02: return ["*","p<0.05"][index]
    elif 1.00e-03 < p <= 1.00e-02: return ["**","p<0.01"][index]
    elif 1.00e-04 < p <= 1.00e-03: return ["***","p<0.001"][index]
    elif p <= 1.00e-04: return ["****","p<0.0001"][index]
    else: return "ns"

data = pd.read_csv(HOMEPATH+'/Postprocessing/calculations/fracdiff/fracdifforder.csv')
df = data.query("connectivity == 'isolated'")

# sns.lineplot(x='xi', y='alpha', data=data.query("connectivity == 'isolated'"), hue='task', style='task',
#             ci=68, err_style="bars", err_kws={'capsize':3}, markers=['o','o'], dashes=False, ax=ax_fracdiff);
sns.lineplot(x='xi', y='alpha', data=df.query("task != 'gsCIFAR10'"), hue='task', style='task',
            markers=['o','^'], dashes=False, ax=ax_fracdiff, palette=['k', 'tab:grey']);

# t test tasks
a = df.query("task == 'psMNIST' and xi == 30.0")['alpha']
b = df.query("task == 'sCIFAR10+conv' and xi == 30.0")['alpha']
ttest = stats.ttest_ind(a,b)

y1, y2 = a.mean(), b.mean()
x, w = df['xi'].max() + 2, 1
ax_fracdiff.plot([x, x+w, x+w, x], [y1, y1, y2, y2], lw=1.0, c='k')
ax_fracdiff.text(x+2*w, (y1+y2)*.5, p_to_label(ttest.pvalue, label_type='star'), ha='left', va='center', color='k', fontsize=10);

ax_fracdiff.set_xlabel(r'Amplitude $\xi$');
ax_fracdiff.set_ylabel(r'Frac. order $\alpha$');
ax_fracdiff.spines['top'].set_visible(False)
ax_fracdiff.spines['right'].set_visible(False)

h, l = ax_fracdiff.get_legend_handles_labels()
ax_fracdiff.legend(h, ['psMNIST', 'sCIFAR10'])


# ============================================================================
#                   FRAC DIFF
# ============================================================================

CONV_ENCODER = True
def get_net(task, seed):
    nhid = 400
    rnn = ANRU(1, main_hidden_size=nhid, supervisor_hidden_size=50, cuda=False,
                        r_initializer='henaff', i_initializer='kaiming', adaptation_type='heterogeneous', supervision_type='local', verbose=False)
    net = figdefs.Model(nhid, rnn)
    if task=='psMNIST':
        if seed==200: pretrained = torch.load(SAVEDIR+'/SavedModels/psMNIST/200/ANRU_lr0.0001_p0.0_hs400/e_99.pth.tar', map_location='cpu')
        elif seed==400: pretrained = torch.load(SAVEDIR+'/SavedModels/psMNIST/400/ANRU_lr0.0001_p0.0_hs400/2021-03-12--1/e_99.pth.tar', map_location='cpu')
        elif seed==600: pretrained = torch.load(SAVEDIR+'/SavedModels/psMNIST/600/ANRU_lr0.0001_p0.0_hs400/2021-03-27--0/e_99.pth.tar', map_location='cpu')
        else:
            raise KeyboardInterrupt('Unrecognized seed.')
        # pretrained = torch.load('/Volumes/Geadah_2/raw_data/ssh_DMS/gamma/Training/SavedModels/psMNIST/600/ANRU_lr0.0001_p0.0_hs400/2021-03-27--0/e_99.pth.tar', map_location='cpu')
    elif task=='gsCIFAR10': 
        if CONV_ENCODER:
            model_dir = SAVEDIR+f'/SavedModels/gsCIFAR10/ConvEncoder/{seed}/ANRU_lr0.0001_p0.0_hs400/'
            pretrained = torch.load(model_dir+'e_99.pth.tar', map_location='cpu')
            pretrained_conv_encoder = torch.load(model_dir+'conv_block_e99.pth.tar', map_location='cpu')
            pretrained['state_dict']['rnn.Uxh.weight'] = torch.ones(400,1)
            pretrained['state_dict']['rnn.Uxh.bias'] = torch.zeros(400)
        else:
            pretrained = torch.load(SAVEDIR+f'/SavedModels/gsCIFAR10/ANRU/ANRU_seed{seed}_lr1e-05_p0.0_hs400/e_99.pth.tar', map_location='cpu')
    
    pretrained_dict = pretrained['state_dict']
    net_dict = net.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict} # filter out unnecessary keys
    net_dict.update(pretrained_dict) # overwrite entries in the existing state dict
    net.load_state_dict(pretrained_dict) # load the new state dict

    if task=='gsCIFAR10' and CONV_ENCODER:
        from NetworkCreation.conv_architectures import ConvBlock_v1
        conv_encoder = ConvBlock_v1()
        conv_encoder.load_state_dict(pretrained_conv_encoder['state_dict'])

    torch.manual_seed(seed)
    np.random.seed(seed)
    return net

def artificial_forward(net, xis, n_iters=100):
    H, N, S = [], [], []
    for j in tqdm(range(n_iters), desc='Get hiddens'):
        if task=='gsCIFAR10' and CONV_ENCODER:
            '''
            For loading convenience, we set Uxh in this model to be the identity
            and feed the inputs x already encoded by the convolutional encoder.
            '''
            x = np.random.normal(loc=0.05, scale=0.4, size=[400]) # approximation of output of conv encoder
        else:
            x = np.random.normal(loc=0.8, scale=0.1, size=[400])
        for xi in xis:
            # initial conditions
            g =  torch.randn(50)
            h = torch.zeros(400, dtype=torch.float32)
            parallel_net.rnn.init_states(1)
            inps=[]
            for t in range(300):
                eps = np.random.normal(loc=0, scale=5.0, size=[400])
                inp = torch.tensor(np.array([x+eps])).reshape(1,400).type(torch.FloatTensor)
                offset = xi if (t>101 and t<202) else 0.0

                h, (ct, _) = parallel_net.rnn(inp,h, 
                            return_ns=True,
                            external_drive=offset)
                H.append(h.detach().numpy()[0,:])
                N.append(np.mean(torch.tensor(np.array(ct)).squeeze().detach().numpy()[0,:]))
                S.append(np.mean(torch.tensor(np.array(ct)).squeeze().detach().numpy()[1,:]))
    H = np.array(H).reshape(n_iters, len(xis), 300, 400)
    N = np.mean(np.array(N).reshape(n_iters, len(xis), 300), axis=0)
    S = np.mean(np.array(S).reshape(n_iters, len(xis), 300), axis=0)
    return H, N, S


task = 'psMNIST'
CONV_ENCODER=False
seed = 400

x = np.arange(300)
resolution = 1
alphas = np.linspace(0,1.0,100)
offset = int(25/resolution)

xis= [0.0, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0]

axs_alpha = [ax_alpha_connected, ax_alpha_isolated]
axs_negalpha = [ax_ht0, ax_ht2]
axs_posalpha = [ax_ht1, ax_ht3]
isolated_labels = ['Interacting', 'Isolated']

for col, ISOLATED in enumerate([False, True]):
    print(f'\nIsolated : {ISOLATED}')
    # Get net
    net = get_net(task, seed)
    supervisor = figdefs.AdaptModule(1, 50)
    supervisor.load_from_ANRU(net)

    parallel_net = net
    parallel_net.rnn.Uxh.weight = torch.nn.Parameter(torch.diag(parallel_net.rnn.Uxh.weight[:,0]))
    if ISOLATED:
        parallel_net.rnn.Whh.weight = torch.nn.Parameter(torch.diag(torch.diag(parallel_net.rnn.Whh.weight)))

    # Compute
    H, _, _ = artificial_forward(parallel_net, xis, n_iters=5)

    # Compute MSE accross xis
    MSEs = []
    for i, signal in  enumerate(tqdm(np.mean(np.mean(H, axis=0), axis=-1), desc='Frac diff per xi')):
        if i<(len(xis)-1): continue
        y2 = np.where(np.abs(x-150)<50, xis[i], 0.0)
        MSEs_sublevel = []
        for alpha in alphas:

            signal_fracdiffd = figdefs.frac_diff(signal, x, alpha=alpha)
            y1 = signal_fracdiffd.real- signal_fracdiffd.real[offset]
            MSEs_sublevel.append(MSE(y1[int(80/resolution):int(220/resolution)], y2[int(80/resolution):int(220/resolution)]))
        MSEs.append(MSEs_sublevel)
    alpha_mean = alphas[np.argmin(np.sum(MSEs, axis=0))]

    # Look at dist of measured alpha over neurons
    resolution=1
    MSEs = []
    alphas = np.linspace(-1.0,1.0,100)
    for neuron, signal in  enumerate(tqdm((np.mean(H, axis=0)[-1]).T, desc='Frac diff per neuron')):
        step_signal = np.where(np.abs(x-150)<50, 0.8+xis[-1], 0.8)
        MSEs_sublevel = []
        for alpha in alphas:
            signal_fracdiffd = figdefs.frac_diff(signal, x, alpha=alpha).real
            y1 = signal_fracdiffd - signal_fracdiffd[offset]
            MSEs_sublevel.append(MSE(y1[int(80/resolution):int(220/resolution)], step_signal[int(80/resolution):int(220/resolution)]))

        MSEs.append(MSEs_sublevel)
        # ax_mse2.plot(alphas, MSEs_sublevel, c=colors[-1], alpha=0.1)
        # msesum_mins.append(np.min(np.sum(MSEs, axis=0)))

    # ax_mse2.plot(alphas, np.sum(MSEs, axis=0)/400, c='r', label='average')

    alpha_dist_indices = np.argmin(MSEs, axis=1)
    alpha_dist = [alphas[i] for i in alpha_dist_indices]
    sns.kdeplot(data=alpha_dist, ax=axs_alpha[col], color='tab:blue', alpha=0.3, fill=True) #edgecolor='tab:orange', stat="count",  bins=np.linspace(-1,1,9)
    # sns.histplot(data=alpha_dist, ax=axs_alpha[col], color='tab:blue', alpha=0.3, edgecolor='tab:blue', stat="count",  bins=np.linspace(-1,1,9))
    axs_alpha[col].set_xlim([-1,1])
    axs_alpha[col].set_xticks([-1., -0.5, 0., 0.5, 1.])
    axs_alpha[col].axvline(x=alpha_mean, c='k', label=r'$\alpha$ of mean resp.')
    axs_alpha[col].axvline(x=0., c='tab:grey', zorder=-1, alpha=0.3, ls='--')
    axs_alpha[col].set_title(isolated_labels[col])
    axs_alpha[col].set_yticks([])
    axs_alpha[col].set_yticklabels([])
    axs_alpha[col].set_ylim([-0.02,2.5])
    if col==0: axs_alpha[col].legend();

    sns.despine(ax=axs_alpha[col], left=True, offset=5, trim=True)


    hidds, counter = [], 0
    for i in range(400):
        # if col==0:
        #     if -0.5<= alpha_dist[i] <-0.25:
        #         hidds.append(np.mean(H, axis=0)[-1,:,i])
        # elif col==1:
        if alpha_dist[i] <= -0.0:
            counter += 1
            hidds.append(np.mean(H, axis=0)[-1,:,i])
    axs_negalpha[col].plot(np.mean(hidds, axis=0))
    axs_negalpha[col].text(0.95, 0.9, f'N={counter}', transform=axs_negalpha[col].transAxes,
                            fontsize=plt.rcParams['legend.fontsize'], va='top', ha='right')

    axs_negalpha[col].set_title(r"$\alpha < 0$")
    sns.despine(ax=axs_negalpha[col], offset=0, trim=False)
    axs_negalpha[col].set_xticklabels([])


    hidds, counter = [], 0
    for i in range(400):
        if 0.0 < alpha_dist[i]:
            counter += 1
            hidds.append(np.mean(H, axis=0)[-1,:,i])
    axs_posalpha[col].plot(np.mean(hidds, axis=0))
    axs_posalpha[col].text(0.95, 0.9, f'N={counter}', transform=axs_posalpha[col].transAxes,
                            fontsize=plt.rcParams['legend.fontsize'], va='top', ha='right')

    axs_posalpha[col].set_title(r"$\alpha > 0$")
    sns.despine(ax=axs_posalpha[col], offset=0, trim=False)

    # # Add arrows
    # con = patches.ConnectionPatch(
    #     xyA=(0.25, 0.1), coordsA=axs_alpha[col].transAxes,
    #     xyB=(0.5, 1.0), coordsB=axs_negalpha[col].transAxes,
    #     arrowstyle="-|>", color='k')
    # fig.add_artist(con)
    # con = patches.ConnectionPatch(
    #     xyA=(0.75, 0.1), coordsA=axs_alpha[col].transAxes,
    #     xyB=(0.5, 1.0), coordsB=axs_posalpha[col].transAxes,
    #     arrowstyle="-|>", color='k')
    # fig.add_artist(con)

    axs_alpha[col].set_xlabel(r"Frac. order $\alpha$")

ax_ht1.set_ylabel('Hidden states $h_t$')
ax_ht1.set_xlabel('Time step $t$')

ax_alpha_isolated.set_ylabel('')

# =========

ANRUv2_shapesignals = pickle.load(open(HOMEPATH+'/Postprocessing/notebooks/ANRUv2_shapesignals_seed400.json','rb'))

f_0s, f_infs = [], []
for i, xi in enumerate(tqdm(xis, desc='Shape signals')):
    temp = np.load(SAVEDIR+'/SavedModels/psMNIST/400/ANRU_lr0.0001_p0.0_hs400/2021-03-12--1/'\
                +'shapesignals_Step_a{:2.1f}_l200_p200.npy'.format(xi)).squeeze()
    nt_mean = np.mean(temp[:,:100,:].reshape(784,-1), axis=-1)
    st_mean = np.mean(temp[:,100:,:].reshape(784,-1), axis=-1)
    # st_mean = np.mean(
    #     np.load('/Volumes/Geadah_2/raw_data/ssh_DMS/gamma/Training/SavedModels/psMNIST/400/ANRU_lr0.0001_p0.0_hs400/2021-03-12--1/'\
    #             +'net0_preactivations_Step_a{:2.1f}_l200_p200.npy'.format(5.0)).reshape(784,-1),
    #     axis=-1)

    ax_nt.plot(nt_mean, color=col_dict[str(xi)]);
    ax_st.plot(st_mean, color=col_dict[str(xi)]);


ax_nt.set_ylabel("Gain $n_t$")
ax_st.set_ylabel("Saturation $s_t$")
for ax in [ax_nt, ax_st]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([0,200,400,600])
ax_nt.set_xticklabels([])
ax_nt.spines['bottom'].set_visible(False)
ax_st.set_xlabel("Time step $t$")

ax_nt.fill_between(np.arange(200,400), 2.5*np.ones(200), 2.61*np.ones(200), color='lime')
ax_st.fill_between(np.arange(200,400), -0.022*np.ones(200), -0.02*np.ones(200), color='lime')

# ============================================================================
# finalise
# ============================================================================

# for ax in [ax_ht, ax_nt, ax_st, ax_ht_fracdiffd, ax_ht_isolated]:
#     ax.set_xticks(xticks)
# Big colorbar
sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=plt.Normalize(vmin=xis[0], vmax=xis[-1]))
# sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=colors.LogNorm(vmin=0.5, vmax=30))#norm=plt.Normalize(vmin=xis[0], vmax=xis[-1]))
# cax = plt.axes([0.95, 0.05, 0.01, 0.82])
cbar = fig.colorbar(sm, ax=[ax_nt, ax_st], shrink=1.0, aspect=30) #, cax=cax, pad=-0.3 
cbar.ax.set_title(r"Drive ($\xi$)", pad=10)

# Save & plot
# fig.subplots_adjust(hspace=2)
plt.savefig(HOMEPATH+'/Postprocessing/figures/fig4_fracdiff.png', format='png', dpi=300);
plt.show();
