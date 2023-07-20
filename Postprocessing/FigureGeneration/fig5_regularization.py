import numpy as np
# import scipy
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib import patches

import seaborn as sns
import pickle
import sys
# HOMEPATH = '.' #! specify your own
# sys.path.append(HOMEPATH)
from NetworkCreation.Networks import ARUN, RNN
# import torch.nn as nn
import pandas as pd
from tqdm import tqdm
# from sklearn.decomposition import PCA
# from scipy.optimize import curve_fit
# from scipy import stats

import figuregeneration_defs as figdefs

# ======================================================================================

fig = plt.figure(figsize=[7,4], constrained_layout=True)
plt.rcParams['xtick.labelsize']=7
plt.rcParams['ytick.labelsize']=7
plt.rcParams['axes.labelsize']=8
plt.rcParams['axes.titlesize']=9
plt.rcParams['legend.fontsize']=7
plt.rcParams['lines.markersize']=4
plt.rcParams['boxplot.meanprops.markersize'] = 4
plt.rcParams['legend.markerscale'] = 1.0
plt.rcParams['legend.title_fontsize'] = plt.rcParams['axes.titlesize']

# gs = fig.add_gridspec(3,3, height_ratios=[1.0,1.0,1.9])

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05, width_ratios=[2,1])

gs0 = gs[0].subgridspec(2, 2, hspace=0.05, wspace=0.05, height_ratios=[1.7,1])
gs1 = gs[1].subgridspec(2, 1, hspace=0.1, wspace=0.01, height_ratios=[1.0,1.])

# gs = gridspec.GridSpec(2,2, figure=fig, wspace=0., hspace=0.1, height_ratios=[1,1.5])
# gs0 = gs[0,0].subgridspec(2, 1, hspace=0.)

ax_LE0 = fig.add_subplot(gs0[1,0])
ax_LE = fig.add_subplot(gs0[1,1])
ax_JN = fig.add_subplot(gs0[0,1])
ax_pca = fig.add_subplot(gs0[0,0])
ax_spectrum = fig.add_subplot(gs1[1])
ax_spectrum_init = fig.add_subplot(gs1[0])

for label, ax in zip(
        ['a', 'b', 'e'],
        [ax_pca, ax_JN, ax_spectrum_init]
    ):
    ax.text(-0.15, 1.15, label, transform=ax.transAxes,
      fontsize=12, fontweight='bold',  va='top', ha='right')
for label, ax in zip(
        ['c', 'd'],
        [ax_LE0, ax_LE]
    ):
    ax.text(-0.15, 1.3, label, transform=ax.transAxes,
      fontsize=12, fontweight='bold',  va='top', ha='right')

xis = [0.0, 0.5,1.0,2.5,5.0,7.5,10.0,15.0,20.0,30.0]
cmap = plt.get_cmap('plasma_r') #plasma_r, Greys
new_cmap = figdefs.truncate_colormap(cmap, 0.1,1.0)

def lognormal_xi(xi):
    return (np.log(xi)-np.log(xis[1]))/np.log(xis[-1])

xis_lognormalized = lognormal_xi(xis[1:])
col = new_cmap(xis_lognormalized)
col_dict = {str(xi): new_cmap(lognormal_xi(xi)) for xi in xis[1:]}
col_dict['0.0'] = 'k' #cmap(0.09)

SAVEDIR = '/Volumes/GEADAH_3/3_Research/Adaptation'

# ======================================================================================

LEs_df_full = pd.read_csv(HOMEPATH+'/Postprocessing/calculations/LEs/LEs_psMNIST.csv')
# print(LEs_df_full.query("seed == 400 and shape_params == 'adaptive'"))

# MLE for varying time steps
sns.lineplot(x='time_step', y='MLE', data=LEs_df_full.query("seed == 400 and shape_params == 'adaptive' and xi in [0.0, 1.0, 5.0, 30.0]"), hue='xi',
            marker='o', ax=ax_LE0, palette=[col_dict[str(i)] for i in [0.0,1.0,5.0,30.0]], legend=False);
ax_LE0.set_xlabel('Time from onset');
ax_LE0.set_xticks([200,250,300,350]);
ax_LE0.set_xticklabels([0,50,100,150]);
# ax_LE.set_yscale('log')
# ax_LE0.spines['top'].set_visible(False)
# ax_LE0.spines['right'].set_visible(False)
ax_LE0.set_ylabel('$\lambda_1$')
# ax_LE0.set_title("                                                         Maximum Lyapunov Exponent\n")
ax_LE0.set_title('                                                         Nonlinear regularization\n\n')
print('ax_LE0 done')

# MLE for varying xi
sns.lineplot(x='xi', y='MLE', data=LEs_df_full.query("seed == 400 and time_step == 350.0"), hue='shape_params',
             marker='o', palette=['tab:red', col_dict['0.0']], ax=ax_LE);
ax_LE.set_xlabel(r'Amplitude $\xi$');
# ax_LE.set_ylabel("$\lambda_1$")
# ax_LE.set_yticklabels([])
ax_LE.set_ylabel('')
# ax_LE.spines['top'].set_visible(False)
# ax_LE.spines['right'].set_visible(False)
h, l = ax_LE.get_legend_handles_labels()
ax_LE.legend(h,  ['Adaptive','Fixed'])
print('ax_LE done')


for ax in [ax_LE0, ax_LE]:
    ax.set_ylim([-0.005,0.04])
    ax.set_yticks([0.0, .02, .04])
    # ax.axhline(y=0, c="tab:grey", zorder=-1)
    sns.despine(ax=ax, offset=5, trim=True)

ax_LE.text(1.09, 0.3, 'chaotic', rotation=90, transform=ax_LE.transAxes, 
            c='tab:grey', fontsize=plt.rcParams['legend.fontsize'])
ax_LE.text(1.05, 0.08, '-------', transform=ax_LE.transAxes, 
            c='tab:grey', fontsize=plt.rcParams['legend.fontsize'])
ax_LE.text(1.06, -0.2, 'fixed\npoint', rotation=90, transform=ax_LE.transAxes, 
            c='tab:grey', fontsize=plt.rcParams['legend.fontsize'])

# # ======================================================================================

# norms_df = pd.read_csv(HOMEPATH+'/Postprocessing/calculations/grad_norms_df.csv')

# sns.lineplot(x='t', y='norm', data=norms_df.query("model in ['RNN+ReLU','RNN+gamma2','ARUN']"), hue='model',
#             palette=['tab:blue','tab:green','tab:red'], ax=ax_grad);
# ax_grad.set_yscale("log")
# ax_grad.set_xlabel("Input sequence length")
# ax_grad.set_ylabel("Norm of $W_{hh}$ gradient")

# ax_grad.legend(labels=['RNN+ReLU','RNN+$\gamma$ (het.)', 'ARUN']);
# ax_grad.set_title("Gradient propagation")
# print('ax_grad done')

# ======================================================================================
from sklearn.decomposition import PCA

task = 'psMNIST'
seed = 400

# SAVEDIR = '/Volumes/Geadah_2/raw_data/ssh_DMS/gamma/Training'
SAVEDIR = '/Volumes/GEADAH_3/3_Research/Adaptation'

nhid = 400
rnn = ANRU(1, main_hidden_size=nhid, supervisor_hidden_size=50, cuda=False,
                    r_initializer='henaff', i_initializer='kaiming', adaptation_type='heterogeneous', supervision_type='local', verbose=False)
net = figdefs.Model(nhid, rnn)
if task=='psMNIST':
    if seed==200: last_model = torch.load(SAVEDIR+'/SavedModels/psMNIST/200/ANRU_lr0.0001_p0.0_hs400/e_99.pth.tar', map_location='cpu')
    elif seed==400: last_model = torch.load(SAVEDIR+'/SavedModels/psMNIST/400/ANRU_lr0.0001_p0.0_hs400/2021-03-12--1/e_99.pth.tar', map_location='cpu')
    elif seed==600: last_model = torch.load(SAVEDIR+'/SavedModels/psMNIST/600/ANRU_lr0.0001_p0.0_hs400/2021-03-27--0/e_99.pth.tar', map_location='cpu')
    else:
        raise KeyboardInterrupt('Unrecognized seed.')
    # last_model = torch.load('/Volumes/Geadah_2/raw_data/ssh_DMS/gamma/Training/SavedModels/psMNIST/600/ANRU_lr0.0001_p0.0_hs400/2021-03-27--0/e_99.pth.tar', map_location='cpu')
elif task=='gsCIFAR10': 
    last_model = torch.load(SAVEDIR+f'/SavedModels/gsCIFAR10/ANRU/ANRU_seed{seed}_lr1e-05_p0.0_hs400/e_99.pth.tar', map_location='cpu')
net.load_state_dict(last_model['state_dict'])

supervisor = figdefs.AdaptModule(1, 50)
supervisor.load_from_ANRU(net)
print('Recovered')


steps = np.exp(np.linspace(-2,2,10))
# Staircase input
controller_hiddens = []
for _ in range(10):
    temp = []
    g = torch.randn(50)
    # steps = []
    for step in steps:
        temp2 = []
        for t in range(400):
            if t<=200:
                inp = 0.0
            else:
                inp = float(step)
            # step = float(t-t%100)/100
            # steps.append(step)
            a = torch.tensor([inp], requires_grad=False)
            g = supervisor(a, g)
            temp2.append(g.detach().numpy())
        temp.append(temp2)
    controller_hiddens.append(temp)

controller_hiddens = np.mean(np.array(controller_hiddens), axis=0)
print(controller_hiddens.shape)

pca2 = PCA(n_components=2)
pca2.fit(controller_hiddens.reshape(-1,50))

# controller_hiddens_pca = pca2.transform(controller_hiddens)
for signal, step in zip(controller_hiddens, steps):
    # for t in 100*np.arange(7):
    controller_hiddens_pca = pca2.transform(signal)
    ax_pca.plot(*controller_hiddens_pca[200:,:].T, '-', c=new_cmap([lognormal_xi(step)]), zorder=0)
    # ax2.plot(controller_hiddens_pca[:,0], '-', c=new_cmap([step/30]), zorder=0)
    ax_pca.scatter(*controller_hiddens_pca[-1,:], s=30, ec='k', c=new_cmap([lognormal_xi(step)]), zorder=2)
    # ax2.plot(i*np.ones(784))

# ax_pca.set_title('Constant input');
# ax_pca.set_title('Limit points')
ax_pca.set_xlabel('PC1')
ax_pca.set_ylabel('PC2')
sns.despine(ax=ax_pca, offset=5, trim=True)

# ax_pca.set_xticks([])
# ax_pca.set_yticks([])
ax_pca.set_xticklabels([])
ax_pca.set_yticklabels([])
ax_pca.set_title("Controller activity")

# ======================================================================================

ANRUv2_shapesignals = pickle.load(open(HOMEPATH+'/Postprocessing/notebooks/ANRUv2_shapesignals_seed400.json','rb'))

f_0s, f_infs = [], []
for i, xi in enumerate(tqdm(xis[1::])):
    # nt_mean = ANRUv2_shapesignals['variable_amplitude'][f'a{xi}']['nt']['mean'][201:400]
    # st_mean = ANRUv2_shapesignals['variable_amplitude'][f'a{xi}']['st']['mean'][201:400]
    temp = np.load(SAVEDIR+'/SavedModels/psMNIST/400/ANRU_lr0.0001_p0.0_hs400/2021-03-12--1/'\
                +'shapesignals_Step_a{:2.1f}_l200_p200.npy'.format(xi)).squeeze()
    nt_mean = np.mean(temp[:,:100,:].reshape(784,-1), axis=-1)
    st_mean = np.mean(temp[:,100:,:].reshape(784,-1), axis=-1)

    f_0s.append([nt_mean[201], st_mean[201]])
    f_infs.append([np.mean(nt_mean[350:399]), np.mean(st_mean[350:399])])

    if i>0:
        ax_JN.plot(nt_mean[201:400], st_mean[201:400], color=col_dict[str(xi)], alpha=1.0, zorder=0) #s=8, 
        ax_JN.scatter([nt_mean[201]], [st_mean[201]],
                marker='^', ec='k', s=20, color=col_dict[str(xi)], alpha=1.0, zorder=1) #s=8, 
        ax_JN.scatter([np.mean(nt_mean[300:380])], [np.mean(st_mean[300:380])],
                marker='s', ec='k', s=20, color=col_dict[str(xi)], alpha=1.0, zorder=1) #s=8, 

f_0s = np.array(f_0s) ; f_infs = np.array(f_infs)

del temp, nt_mean, st_mean

N, S = np.linspace(2.0,5.0,100), np.linspace(-0.015,0.025,100)
for xi2 in  [0.9, 1.1,1.5,2.0, 3.0, 5.0, 30.0]:
    Z = np.zeros(shape=[len(N), len(S)])
    for i in range(len(N)):
        for j in range(len(S)):
            Z[i,j] = figdefs.np_gamma_prime(xi2, N[i],S[j])+0.013
    C3 = ax_JN.contour(N, S, Z.transpose(), levels=[1.0], colors=new_cmap([lognormal_xi(xi2)]), linestyles=['--'], zorder=-1)

ax_JN.set_title(r'Linearization')
# ax_JN.set_xticklabels([])
ax_JN.set_ylabel('Saturation $s$')
ax_JN.set_xlabel('Gain $n$')
# sns.despine(ax=ax_JN, offset=5, trim=True)

from matplotlib.lines import Line2D
legend_elements = [
                Line2D([0], [0], marker='^', label='Onset', ls='',
                          markerfacecolor='tab:grey', markeredgecolor='k'),
                Line2D([0], [0], marker='s', label='Steady-state', ls='',
                          markerfacecolor='tab:grey', markeredgecolor='k'),
                Line2D([0], [0], color='tab:grey', ls='--', label='Jacobian Norm'),]

# Create the figure
ax_JN.legend(handles=legend_elements, loc=4)


# # ======================================================================================
def henaff_matrix(N: int=400) -> torch.Tensor:
    from NetworkCreation.common import henaff_init
    from NetworkCreation.exp_numpy import expm
    W = torch.as_tensor(henaff_init(N))
    A = W.triu(diagonal=1)
    A = A - A.t()
    W = expm(A)
    return W

W_init_ARUN = henaff_matrix(400)
W_init_ReLU = henaff_matrix(400)
ax_spectrum_init.plot((np.linalg.eig(W_init_ARUN)[0]).real, (np.linalg.eig(W_init_ARUN)[0]).imag, '.', c='tab:red', label='ARUN')
ax_spectrum_init.plot((np.linalg.eig(W_init_ReLU)[0]).real, (np.linalg.eig(W_init_ReLU)[0]).imag, '.', c='tab:blue', label='RNN+ReLU')

last_model = torch.load(SAVEDIR+'/SavedModels/psMNIST/400/ANRU_lr0.0001_p0.0_hs400/2021-03-12--1/e_99.pth.tar', map_location='cpu')
Whh = last_model['state_dict']['rnn.Whh.weight']
eigs_ARUN = np.linalg.eig(Whh)[0]
ax_spectrum.plot(eigs_ARUN.real, eigs_ARUN.imag, '.', c='tab:red', label='ARUN')

last_model = torch.load(SAVEDIR+'/SavedModels/psMNIST/400/RNN_lr0.0001_p0.0_hs400_ReLU/2021-03-08--0/e_99.pth.tar', map_location='cpu')
W = last_model['state_dict']['rnn.V.weight']
eigs_ReLU = np.linalg.eig(W)[0]
ax_spectrum.plot(eigs_ReLU.real, eigs_ReLU.imag, '.', c='tab:blue', label='RNN+ReLU')

for ax in [ax_spectrum_init, ax_spectrum]:
    ax.text(0.1, 0.9, "$\mathbb{C}$", transform=ax.transAxes, fontsize=13)
    # ax.axis('off')
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])
    sns.despine(ax=ax, offset=5, trim=True)

ax_spectrum_init.get_xaxis().set_visible(False)
ax_spectrum_init.spines['bottom'].set_visible(False)
ax_spectrum_init.set_title("Eigenspectrum")
ax_spectrum.legend(loc='lower right');
# ax_spectrum_init.set_title("Init.")
print('ax_spectrum done')


# # Draw a line between the different points, defined in different coordinate
# # systems.
con = patches.ConnectionPatch(
    xyA=(0.5, 0.1), coordsA=ax_spectrum_init.transAxes,
    xyB=(0.5, 1.0), coordsB=ax_spectrum.transAxes,
    arrowstyle="-|>", color='k')
fig.add_artist(con)
plt.figtext(0.88, 0.52, 'Training', fontdict={'fontsize':plt.rcParams['legend.fontsize']})

# plt.arrow(0.8,0.7,0.,-0.1);

# ============================================================================
# finalise
# ============================================================================

# colorbar
sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=colors.LogNorm(vmin=xis[1], vmax=xis[-1]))#norm=plt.Normalize(vmin=xis[0], vmax=xis[-1]))
# cax = plt.axes([0.95, -0.05, 0.6, 0.04])
# cbar = fig.colorbar(sm, cax=cax, shrink=1.0, orientation='horizontal', aspect=30) #, cax=cax, pad=-0.3 
cbar = fig.colorbar(sm, ax=[ax_JN], location='right', shrink=1.0, aspect=30, pad=-.1) 
cbar.ax.set_title(r"$\xi$", pad=5)

# Save & plot
plt.savefig(HOMEPATH+'/Postprocessing/figures/fig5_v2_regularization.pdf', format='pdf');
plt.show();