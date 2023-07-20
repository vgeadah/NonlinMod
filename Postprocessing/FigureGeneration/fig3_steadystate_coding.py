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
from NetworkCreation.Networks import ANRU
import figuregeneration_defs as figdefs
SAVEDIR = '.' #! specify

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
Adaptive_index = False

fig = plt.figure(figsize=[7.0,6]);
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

gs0 = gridspec.GridSpec(2,1, figure=fig, hspace=0.6)

gs00 = gs0[0].subgridspec(4, 3, hspace=2, wspace=0.5)

ax_hetero = fig.add_subplot(gs00[0:2, 0])
ax_hetero2 = fig.add_subplot(gs00[2:, 0])
ax1 = fig.add_subplot(gs00[0, 1])
ax1.axis('off')
ax1.set_title("                                                 Gain scaling")
ax_gainscaling = fig.add_subplot(gs00[1:, 1])
ax_gainscaling2 = fig.add_subplot(gs00[1:, -1])


gs01 = gs0[1].subgridspec(1, 2, width_ratios=[2,1], wspace=0.3)
gs010 = gs01[0].subgridspec(2, 2, wspace=0.4)

# ax_periodic = fig.add_subplot(gs[1, 0])
ax_ht= fig.add_subplot(gs010[0, 0])
ax_ht_fracdiffd = fig.add_subplot(gs010[0, 1])
# ax_Ai = fig.add_subplot(gs[-2, -1])

ax_ht_isolated = fig.add_subplot(gs010[-1, 0])
ax_ht_isolated_fracdiffd = fig.add_subplot(gs010[-1, 1])
# ax_Ai_isolated = fig.add_subplot(gs[-1, -1])

# gs010 = gs01[1].subgridspec(2, 2)
ax_fracdiff = fig.add_subplot(gs01[1])

for label, ax in zip(
        ['a', 'b', 'c', 'd'],
        [ax_hetero, ax1, ax_ht, ax_ht_fracdiffd]
    ):
    # print(label, ax)
    ax.text(-0.3, 1.3, label, transform=ax.transAxes,
      fontsize=13, fontweight='bold', va='top', ha='right')

# ax_gainscaling.text(-0.2, 1.3, 'b', transform=ax_gainscaling.transAxes,
#       fontsize=13, fontweight='bold', va='top', ha='right')    
ax_fracdiff.text(-0.2, 1.15, 'e', transform=ax_fracdiff.transAxes,
      fontsize=13, fontweight='bold', va='top', ha='right') #'            c'

# plt.margins(0.3)
# xticks = [0,200,400,600]


# ax_infographic = ax_ht.inset_axes([0.8, 0.75, 0.4, 0.3]) #inset_axes(ax_ht, width="40%", height="30%", loc='upper right')
# ax_infographic.plot(np.arange(784), np.where(np.abs(np.arange(784)-300)<100, 1, 0),c='k')
# ax_infographic.set_ylim([-0.1,1.5])
# ax_infographic.text(280, 1.1, r'$\tau$', fontsize=6)
# ax_infographic.text(90, 0.5, r'$\xi$', fontsize=6)
# ax_infographic.set_xticks([])
# ax_infographic.set_xticklabels([])
# ax_infographic.set_yticks([])
# ax_infographic.set_yticklabels([])

# ===============================================================================================================
#                   HETEROGENEITY
# ===============================================================================================================

torch.manual_seed(400)
np.random.seed(400)

RNNgamma2 = torch.load(SAVEDIR+'/SavedModels/psMNIST/400/RNN_lr0.0001_p0.0_hs400_gamma2_LP/2021-03-09/e_99.pth.tar', map_location='cpu')
hetero_N = RNNgamma2['state_dict']['rnn.n'].detach().numpy()
hetero_S = RNNgamma2['state_dict']['rnn.s'].detach().numpy()
n0 = 5+2*torch.rand(1).item() ; s0 = 0.0

ax_hetero.scatter(hetero_N, hetero_S, color='k', s=5)
# ax_hetero.scatter([np.mean(hetero_N)], [np.mean(hetero_S)], c='tab:blue', label='mean')
# ax_hetero.scatter([n0], [s0], label='init', c='tab:grey')

# ax_hetero.plot(hetero_N, hetero_S, 'o', color='k', markeredgecolor='w', alpha=1.0, markeredgewidth=0.3)
ax_hetero.plot([np.mean(hetero_N)], [np.mean(hetero_S)], 'o', c='tab:blue', markeredgecolor='w', alpha=1.0, markeredgewidth=0.3, label='COM')
ax_hetero.plot([n0], [s0], 'o', markeredgecolor='w', c='tab:grey', alpha=1.0, markeredgewidth=0.3, label='init')

del hetero_N, hetero_S

ax_hetero.legend();

ax_hetero2.set_ylabel('                                    Saturation $s$')
ax_hetero.set_title("Heterogeneity")


# ----------------------------------------------------------------------
run = 1
epoch = 99
mod = 45
adapt_type='shapeonly'
n, s = 5.0, 0.0

modeldict_preadapt = torch.load(SAVEDIR+'/SavedModels/psMNIST/400/RNN_lr0.0001_p0.0_hs400_gamma2_LP/gamma2/run{}_n_5.0_s_0.0/RNN_{}.pth.tar'.format(run,epoch), map_location=torch.device('cpu'))['state_dict']
modeldict_postadapt = torch.load(SAVEDIR+'/SavedModels/psMNIST/400/RNN_lr0.0001_p0.0_hs400_gamma2_LP/gamma2/data/{}_RNN_run1_n{}_s{}_lr0.0001_mod{}_epoch{}.pth.tar'.format(adapt_type,n,s,mod,epoch), map_location=torch.device('cpu'))['state_dict']

pre_adapt_shape = np.array([modeldict_preadapt['rnn.n'].data.numpy(),modeldict_preadapt['rnn.s'].data.numpy()])
post_adapt_shape = np.array([modeldict_postadapt['rnn.n'].data.numpy(),modeldict_postadapt['rnn.s'].data.numpy()])

# df = pd.DataFrame(columns={"n","s","t"})
# for i in range(pre_adapt_shape.shape[1]):
#     df = df.append({'n':pre_adapt_shape[0,i], 's':pre_adapt_shape[1,i], 't':'before'}, ignore_index=True)
# for i in range(post_adapt_shape.shape[1]):
#     df = df.append({'n':post_adapt_shape[0,i], 's':post_adapt_shape[1,i], 't':'after'}, ignore_index=True)
# sns.scatterplot(x='n', y='s', data=df, hue='t', hue_order=['after','before'], ax=ax_hetero2)

# ax_hetero2.plot(*pre_adapt_shape, 'o', markeredgecolor='w', alpha=1.0, markeredgewidth=0.3, label='before', zorder=1)
# ax_hetero2.plot(*post_adapt_shape, 'o', markeredgecolor='w', alpha=1.0, markeredgewidth=0.3, label='after', zorder=-1)
ax_hetero2.scatter(*pre_adapt_shape, s=5, label='before', c='k', zorder=1)
ax_hetero2.scatter(*post_adapt_shape, s=5, label='after', c='tab:grey', zorder=-1)

# ax_hetero2.set_ylabel('Saturation $s$')
ax_hetero2.set_xlabel('Gain $n$')
ax_hetero2.legend();

# # ===============================================================================================================
# #                   Periodic step
# # ===============================================================================================================

# ax_periodic.plot([11,25,31],[4.456,8.181,9.707], '-o', c='k')
# ax_periodic.set_xlabel('T')
# ax_periodic.set_ylabel(r'$\tau$')


# ===============================================================================================================
#                   Fractional order
# ===============================================================================================================
def p_to_label(p, label_type='star'):
    index=0
    if label_type=='p':
        index=1
    
    if 1.00e-02 < p <= 5.00e-02:
        return ["*","p<0.05"][index]
    elif 1.00e-03 < p <= 1.00e-02:
        return ["**","p<0.01"][index]
    elif 1.00e-04 < p <= 1.00e-03:
        return ["***","p<0.001"][index]
    elif p <= 1.00e-04:
        return ["****","p<0.0001"][index]
    else:
        return "ns"
    
data = pd.read_csv(HOMEPATH+'/Postprocessing/calculations/fracdiff/fracdifforder.csv')
df = data.query("connectivity == 'isolated'")

# sns.lineplot(x='xi', y='alpha', data=data.query("connectivity == 'isolated'"), hue='task', style='task',
#             ci=68, err_style="bars", err_kws={'capsize':3}, markers=['o','o'], dashes=False, ax=ax_fracdiff);
sns.lineplot(x='xi', y='alpha', data=df, hue='task', style='task',
             marker='o', dashes=False, ax=ax_fracdiff, palette=['k','tab:grey']);

# t test tasks
a = df.query("task == 'psMNIST' and xi == 30.0")['alpha']
b = df.query("task == 'gsCIFAR10' and xi == 30.0")['alpha']
ttest = stats.ttest_ind(a,b)

y1, y2 = a.mean(), b.mean()
x, w = df['xi'].max() + 2, 1
ax_fracdiff.plot([x, x+w, x+w, x], [y1, y1, y2, y2], lw=1.0, c='k')
ax_fracdiff.text(x+2*w, (y1+y2)*.5, p_to_label(ttest.pvalue, label_type='star'), ha='left', va='center', color='k', fontsize=10);

ax_fracdiff.set_xlabel(r'Amplitude $\xi$');
ax_fracdiff.set_ylabel(r'Frac. order $\alpha$');
ax_fracdiff.spines['top'].set_visible(False)
ax_fracdiff.spines['right'].set_visible(False)


# ===============================================================================================================
#                   GAIN SCALING
# ===============================================================================================================

import pickle
xis_reduced = [0.0, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0]

varianceoffset_shapesignals = pickle.load(open(HOMEPATH+'/Postprocessing/varianceoffset_shapesignals.pkl','rb'))

nt, st  = [], []
for seed_id in ['200','400','600']:
    temp1 = np.mean(varianceoffset_shapesignals['nt'][seed_id][:,110:200], axis=-1)
    nt.append(temp1)
    temp2 = np.mean(varianceoffset_shapesignals['st'][seed_id][:,110:200], axis=-1)
    st.append(temp2)

nt = np.mean(nt, axis=0); st = np.mean(st, axis=0);     


ax_gainscaling.plot(xis_reduced[1:], nt[1:], '-o', c='k')
popt2, pcov2 = curve_fit(figdefs.power_func, xis_reduced[1:], nt[1:])
ax_gainscaling.plot(np.linspace(0.8,31,100), figdefs.power_func(np.linspace(0.8,31,100), *popt2), ls='--', c='k');
# ax_gainscaling.set_yscale('log');
# ax_gainscaling.set_xscale('log');
ax_gainscaling.set_ylabel('Gain $n$', color='k')
ax_gainscaling.tick_params(axis='y', labelcolor='k')

# ax_gainscaling2 = ax_gainscaling.twinx() 
ax_gainscaling2.plot(xis_reduced[1:], st[1:], '-o', c='k')
popt2, pcov2 = curve_fit(figdefs.exp_func, xis_reduced[1:], st[1:])
ax_gainscaling2.plot(np.linspace(0,31,100), figdefs.exp_func(np.linspace(0,31,100), *popt2), ls='--', c='k');
# ax_gainscaling2.set_xscale('log');
ax_gainscaling2.set_ylabel('Saturation $s$', color='k');
ax_gainscaling2.tick_params(axis='y', labelcolor='k');

ax_gainscaling.set_xlabel(r'Std. dev. $\xi$');
# ax_gainscaling.set_title('Gain scaling')

a=0.0
ax_infographic2 = inset_axes(ax_gainscaling2, width="40%", height="35%", loc='lower right')
ax_infographic2.plot(np.arange(784)[::16], np.where(np.abs(np.arange(784)[::16]-300)<100, np.random.normal(0,0.4, size=[49]), 0), c='k', lw=1.0)
ax_infographic2.set_ylim([-1,1])
ax_infographic2.annotate(r'$\mathcal{N}(0,\xi^2)$', (420, 0.4), fontsize=6)
# ax_infographic2.text(90, 0.5, r'$\xi$', fontsize=6)
# ax_infographic2.set_xticks([])
# ax_infographic2.set_xticklabels([])
# ax_infographic2.set_yticks([])
# ax_infographic2.set_yticklabels([])
# sns.despine(ax=ax_infographic2, offset=5, trim=True)
ax_infographic2.axis('off')

# ax_gammashapes = inset_axes(ax1, width="40%", height="35%", loc='upper right')
# ax_gammashapes.plot(np.linspace(-1.5,2,100), [figdefs.np_gamma(x, 8.0, 0.0) for x in np.linspace(-1.5,2,100)], c=col_dict[str(1.0)])
# ax_gammashapes.plot(np.linspace(-1.5,2,100), [figdefs.np_gamma(x, 2.0, 0.4) for x in np.linspace(-1.5,2,100)], c=col_dict[str(30.0)])
# ax_gammashapes.axis('off')


# ax_gainscaling.axvline(x=1.0, color=col_dict[str(1.0)], lw=2, zorder=-1)
# ax_gainscaling.axvline(x=30.0, color=col_dict[str(30.0)], lw=2, zorder=-1)

ax_gainscaling.set_ylim([5.4,5.73])
ax_gainscaling2.set_ylim([-0.004, 0.0035])

# for ax in [ax_gainscaling, ax_gainscaling2]:
#     ax.annotate("", xy=(1.0, 0.0024), xytext=(1.0, 5.73),
#             arrowprops=dict(arrowstyle="-|>", color=col_dict[str(1.0)]), xycoords='data')
#     ax.annotate("", xy=(30.0, 5.68), xytext=(30.0, 5.73),
#             arrowprops=dict(arrowstyle="-|>", color=col_dict[str(30.0)]), xycoords='data')

# sns.despine(ax=ax_gainscaling, offset=5, trim=True)
# sns.despine(ax=ax_gainscaling2, offset=5, trim=True)
for ax in [ax_gainscaling, ax_gainscaling2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ============================================================================
#                   FRAC DIFF
# ============================================================================

def get_net(task, seed):
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
        last_model = torch.load(SAVEDIR+f'/SavedModels/gsCIFAR10/ANRU/ANRU_seed{seed}_lr1e-05_p0.0_hs400_0.001/e_99.pth.tar', map_location='cpu')
    net.load_state_dict(last_model['state_dict'])

    torch.manual_seed(seed)
    np.random.seed(seed)
    return net

def artificial_forward(net, xis, n_iters=100):
    H, N, S = [], [], []
    for j in tqdm(range(n_iters), desc='Get hiddens'):
        cnt=0
        x = np.random.normal(loc=0.8, scale=0.1, size=[400])
        for xi in xis:
            # initial conditions
            g =  torch.randn(50)
            h = torch.zeros(400, dtype=torch.float32)
            parallel_net.rnn.init_states(1)
            inps=[]
            for t in range(150):
                eps = np.random.normal(loc=0, scale=0.1, size=[400])
                inp = torch.tensor(np.array([x+eps])).reshape(1,400).type(torch.FloatTensor)
                offset = xi if (t>50 and t<101) else 0.0

                h, (ct, _) = parallel_net.rnn(inp,h, 
                            return_ns=True,
                            external_drive=offset)
                H.append(h.detach().numpy()[0,:])
                N.append(np.mean(torch.tensor(np.array(ct)).squeeze().detach().numpy()[0,:]))
                S.append(np.mean(torch.tensor(np.array(ct)).squeeze().detach().numpy()[1,:]))
    H = np.array(H).reshape(n_iters, len(xis), 150, 400)
    N = np.mean(np.array(N).reshape(n_iters, len(xis), 150), axis=0)
    S = np.mean(np.array(S).reshape(n_iters, len(xis), 150), axis=0)
    return H, N, S

def adaptive_index(signal, res=1, window=15):
    # signal : (n_steps, )
    L_early, L_late = int(100/res), int(149/res)
    r_early = np.mean(signal[L_early:L_early+window])
    r_late = np.mean(signal[L_late-window:L_late])
    if (r_early+r_late)<1e-10:
        return 0.0
    else:
        return (r_early-r_late)/(r_early+r_late)

# def adaptive_index(signal, indices='even'):
#     try:
#         if indices=='even':
#             popt, pcov = curve_fit(figdefs.exp_func, np.arange(1,len(signal)+1)[::2], signal[::2])
#         elif indices=='odd':
#             popt, pcov = curve_fit(figdefs.exp_func, np.arange(1,len(signal)+1)[::2], signal[1::2])
#         if np.diag(pcov)[1] == np.inf:
#             raise RuntimeError
#         print(popt, np.diag(pcov))
#         return popt
#     except RuntimeError:
#         return [np.nan, np.nan, np.nan]


task = 'psMNIST'
seed = 400

x = np.arange(150)
resolution = 1
alphas = np.linspace(0,1.0,100)
offset = int(25/resolution)

xis= [0.0, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0]

# axs_ai = [ax_Ai, ax_Ai_isolated]
axs_ht = [ax_ht, ax_ht_isolated]
axs_ht_fracdiffd = [ax_ht_fracdiffd, ax_ht_isolated_fracdiffd]

for row, ISOLATED in enumerate([False, True]):
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

    # Compute MSEs
    MSEs = []
    for i, signal in  enumerate(tqdm(np.mean(np.mean(H, axis=0), axis=-1), desc='Frac. diff.')):
        if i==0: continue
        y2 = np.where(np.abs(x-75)<25, xis[i], 0.0)
        MSEs_sublevel = []
        for alpha in alphas:

            signal_fracdiffd = figdefs.frac_diff(signal, x, alpha=alpha)
            y1 = signal_fracdiffd.real- signal_fracdiffd.real[offset]
            MSEs_sublevel.append(MSE(y1[int(40/resolution):int(110/resolution)], y2[int(40/resolution):int(110/resolution)]))
        MSEs.append(MSEs_sublevel)

    if Adaptive_index:
        # Adaptive index 
        adaptive_indices = pd.DataFrame(columns=['Even','Odd'])
        adaptive_exp = pd.DataFrame(columns=['Even','Odd'])
        for signal in np.mean(H[:,-1,:,:], axis=0).T:
            adaptive_indices = adaptive_indices.append({
                # 'All':adaptive_index(signal), 
                'Even':adaptive_index(signal[::2], res=2*resolution, window=10), 
                'Odd':adaptive_index(signal[1::2], res=2*resolution, window=10)
                }, ignore_index=True)
            # adaptive_exp = adaptive_exp.append({
            #     # 'All':adaptive_index(signal), 
            #     'Even':adaptive_index(signal, indices='even')[1], 
            #     'Odd':adaptive_index(signal, indices='odd')[1]
            #     }, ignore_index=True)

        # if not ISOLATED:
        #     hp = sns.kdeplot(data=adaptive_indices, ax=axs_ai[row], fill=True, common_norm=True, palette=['tab:blue','tab:orange'])
        #     sns.move_legend(hp, "upper left", title='Indices')
        # else:
        #     hp = sns.kdeplot(data=adaptive_indices, ax=axs_ai[row], fill=True, legend=False, common_norm=True, palette=['tab:blue','tab:orange'])
        #     axs_ai[row].set_xlabel('Adaptive Index')

        # # axs_ai[row].scatter(adaptive_indices['Even'], adaptive_exp['Even'], s=7);
        # axs_ai[row].set_xlim([-1,1])
        # axs_ai[row].set_ylabel('')
        # axs_ai[row].set_yticks([])
        # axs_ai[row].set_yticklabels([])

    # Plot frac diffd signals 
    alpha = alphas[np.argmin(np.sum(MSEs, axis=0))]
    for i, signal in enumerate(np.mean(np.mean(H, axis=0), axis=-1)):
        if i%2==1: continue
        axs_ht[row].plot(x, signal,  c=col_dict[str(xis[i])]);

        signal_fracdiffd = figdefs.frac_diff(signal, x, alpha=alpha)
        axs_ht_fracdiffd[row].plot(x, signal_fracdiffd.real-signal_fracdiffd.real[offset],  c=col_dict[str(xis[i])]);# 
        
        label = None
        if i==len(xis)-1:
            label='drive'
        axs_ht_fracdiffd[row].plot(x, np.where(np.abs(x-75)<25, xis[i], np.nan),  c='lime', ls='--', label=label);

    # axs_ht[row].set_ylabel('Hidden states $h_t$')
    # axs_ht_fracdiffd[row].set_ylabel('Signal')
    axs_ht_fracdiffd[row].text(110, 28, r'$\alpha = $'+'${:1.2f}$'.format(alpha), fontsize=8);

# ax1.set_xlabel('Time steps $t$')
ax_ht_isolated.set_xlabel('Time steps $t$')
ax_ht_isolated_fracdiffd.set_xlabel('Time steps $t$')
# ax_ht_isolated_fracdiffd.set_ylabel('Signal')
ax_ht_fracdiffd.legend();

ax_ht.fill_between(np.arange(50,100), -15*np.ones(50), -7*np.ones(50), color='lime')
ax_ht_isolated.fill_between(np.arange(50,100), -11*np.ones(50), -7*np.ones(50), color='lime')

for ax in [ax_ht, ax_ht_fracdiffd]:
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)

for ax in [*axs_ht, *axs_ht_fracdiffd, ax_fracdiff]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ax_ht.set_title('ARU activity')
# ax_ht.set_title('ARU activity')

# fig.text(0.03, 0.25, 'Hidden states $h_t$', va='center', rotation='vertical')
# fig.text(0.33, 0.25, r'$D^{\alpha} h_t$', va='center', rotation='vertical')
# fig.text(0.64, 0.25, 'Density', va='center', rotation='vertical') 

fig.text(0.01, 0.15, 'Isolated', va='center', rotation='vertical', fontsize=11, color='tab:gray')
fig.text(0.01, 0.35, 'Connected', va='center', rotation='vertical', fontsize=11, color='tab:gray')

ax_ht.set_title("                                                                                            Fractional order differentiation\n\n")
ax_ht_isolated.set_ylabel('                              Hidden states $h_t$')
ax_ht_isolated_fracdiffd.set_ylabel(r'                                    $D^{\alpha} h_t$')

# ============================================================================
# finalise
# ============================================================================

# for ax in [ax_ht, ax_nt, ax_st, ax_ht_fracdiffd, ax_ht_isolated]:
#     ax.set_xticks(xticks)
# Big colorbar
# sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=plt.Normalize(vmin=xis[0], vmax=xis[-1]))
sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=colors.LogNorm(vmin=0.5, vmax=30))#norm=plt.Normalize(vmin=xis[0], vmax=xis[-1]))
cax = plt.axes([0.95, 0.05, 0.01, 0.82])
cbar = fig.colorbar(sm, cax=cax, shrink=1.0, aspect=30) #, pad=-0.3 
cbar.ax.set_title(r"Drive ($\xi$)", pad=10)

# Save & plot
# fig.subplots_adjust(hspace=2)
plt.savefig(HOMEPATH+'/Postprocessing/figures/fig3_biologicalmechanisms.png', format='png');
plt.show();
