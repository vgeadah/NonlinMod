import numpy as np 
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os 

import definitions as defs 


# ========= gamma ============
def sup_gamma_prime(n,s):
    if s > 1/(n+1): return (1+s*(n-1))**2 /(4*n*s)
    else: return (1-s)

# In Numpy
def npgam1(x,n): return (1 / n)*(np.log(1 + np.exp(n*x)))
def npgam2(x,n): return (np.exp(n*x))/(1 + np.exp(n*x))
def np_gamma(x, n, s): return (1-s)*npgam1(x,n) + s*npgam2(x,n)
def np_gamma_prime(x, n, s): return (1-s)*npgam2(x,n) + s*n*npgam2(x,n)*(1-npgam2(x,n))


task_params = {
    'sMNIST' : {'hidden_size':400, 'lr':0.0001, 'baseline':0.1, 'pmin':0, 'pmax':1, 'p':'Test\n accuracy', 'diff_grid':False},
    'psMNIST': {'hidden_size':400, 'lr':0.0001, 'baseline':0.1, 'pmin':0, 'pmax':1, 'p':'Test\n accuracy', 'diff_grid':True},
    'copy'   : {'hidden_size':128, 'lr':0.0001, 'baseline':0.095, 'pmin':0, 'pmax':0.15, 'p':'Train\n loss', 'diff_grid':False},
    'PTB'    : {'hidden_size':600, 'T':150, 'lr':0.0002, 'baseline':None, 'pmin':1.5, 'pmax':4, 'p':'Test\n BPC', 'diff_grid':True}
}

# ---------- Structure Parameters ----------
nonlins = [1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]+[3.75, 6.25, 8.75, 11.25, 13.75, 16.25, 18.75]
saturations = [0.0, 0.25, 0.5, 0.75, 1.0]


# ============ Figure
plot_heterogeneous=True

if plot_heterogeneous: nrows=4
else: nrows=3

fig, axs = plt.subplots(figsize=[13,12], nrows=nrows, ncols=3, constrained_layout=True)

for index in np.ndindex(nrows,3):
    if index[0]==2: continue
    axs[index].grid(True, zorder='below', which='both', color='lightgrey')
    axs[index].set_facecolor('w') # #f0f0f0
    # for spine in axs[index].spines.values():
    #     spine.set_visible(False)
    axs[index].xaxis.tick_bottom()
    axs[index].yaxis.tick_left()
    # if i!= 0 : axs[i].set_yticklabels([])

for i, label in enumerate(('A', 'B', 'C', 'D')):
    # ax = fig.add_subplot(2,2,i+1)
    axs[i,0].text(-0.2, 1.15, label, transform=axs[i,0].transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

# =============================================
# Col 1 : adaptation dynamics
# =============================================


print('-'*40+'\nadaptation dynamics\n'+'-'*40)

chosen_epoch = 99
run = 'run_1'

tasks = ['copy','psMNIST', 'PTB']

for i in range(3):
    task = tasks[i]
    cmap = 'viridis_r'
    if task == 'psMNIST': cmap = 'viridis'
    cmap2 = plt.cm.get_cmap('Greys')

    allparams, colors, maxs, mins = defs.param_animation(task, epoch=99, run=run, lr=1e-04, cmap=cmap)
    nonlins = [1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
    if task_params[task]['diff_grid']:
        nonlins += [3.75, 6.25, 8.75, 11.25, 13.75, 16.25, 18.75]
    saturations = [0.0, 0.25, 0.5, 0.75, 1.0]

    for index in np.ndindex(len(nonlins),len(saturations)):
        print(index)
        # for t in range(chosen_epoch):
        #     value = 0.2+0.8*(1-(chosen_epoch-t)/chosen_epoch)
        #     axs[i].scatter(allparams[index[0],index[1],t,0], allparams[index[0],index[1],t,1],
        #         s=1, c=[cmap2(value)], zorder=t)

        axs[0,i].plot(allparams[index[0],index[1],:chosen_epoch,0],
            allparams[index[0],index[1],:chosen_epoch,1],'-', c='lightgrey', zorder=2)
        axs[0,i].scatter(nonlins[index[0]], saturations[index[1]], s=5, c='white', alpha=1, zorder=0, edgecolor='k', linewidths=0.4)
        axs[0,i].scatter(allparams[index[0],index[1],chosen_epoch,0], allparams[index[0],index[1],chosen_epoch,1],
            s=45, c=[colors[index[0],index[1],chosen_epoch,:]], alpha=1, zorder=chosen_epoch+2, edgecolor='k', linewidths=0.3)
    
    axs[0,i].set_title(task+'\n', fontsize=15, fontweight='bold')
    axs[0,i].set_xlabel('Gain $(n)$')
    axs[0,i].set_xlim([0, 21])
    axs[0,i].set_ylim([-0.15,1.75])

    axs[0,i].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,1.75])
    # axs[0,i].set_yticklabels([0.0, ''  , 0.5, ''  , 1.0, ''  , 1.5,''])
    axs[0,0].set_ylabel('Saturation $(s)$')
    # if i!=0 : axs[0,i].set_yticklabels([])
    # axs[i].grid(True, zorder=-1);

    # fig.suptitle(task);
    # ===========================

    # colorbar :
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))

    pmin, pmax = task_params[task]['pmin'], task_params[task]['pmax']
    if task == 'copy' or task == 'PTB':
        ticks = list(map(str, [round(i,3) for i in np.linspace(pmin, pmax-(pmax-pmin)/5, 5)]))
        ticks.append('> {}'.format(pmax))
    else:
        ticks = list(map(str, [round(i,3) for i in np.linspace(pmin, pmax, 6)]))
    cbar = fig.colorbar(sm, ax=axs[0,i], location='right', shrink=0.9, aspect=20) #label=task_params[task]['p'],
    cbar.ax.set_yticklabels(ticks)
    cbar.ax.set_title(task_params[task]['p'], fontsize=8)


# ==========================================================================================
#  Col 2 : Lyapunov exponents
# ==========================================================================================


print('-'*40+'\nLyapunov exponents\n'+'-'*40)

def task_test(task, n, s, lp, nonlin):
    tresholds= {
        'copy':   {'top_50':[0.09,0.09, 0.09] ,'select':[0.075,0.075, 0.075], 'top_25':[0.079,0.075, 0.08], 'baseline':[0.09, 0.09, 0.09],'all':[1,1,1]},
        'psMNIST':{'top_50':[0.77,0.77,0.8],'top_25':[0.917,0.9005,0.907], 'top_10':[0.93,0.945,0.94],
                   'acc_50':0.50,'all':[0,0,0], 'select':[0.93,0.945,0.94], 'baseline':[0.12, 0.12, 0.12], 'all':[0.12, 0.12, 0.12]},
        'PTB':    {'top_50':[1.64,1.67,1.64],'top_25':[1.612,1.625,1.615],'top_10':[1.60,1.595,1.6],'acc_50':[2.5,2.5], 'all':[5,5,5], 'baseline':[5,5,5]} #1.67
    }
    # if task=='copy': treshold=tresholds[task]['select']
    # else: 
    treshold=tresholds[task]['top_25']
    accuracy = defs.get_accuracy(n, s, task, run=run, lr=lr, lp=lp, nonlin=nonlin)[0][-1]

    if lp==False:
        index=1
    else:
        if nonlin=='gamma':index=0
        else: index=2

    if task=='copy': return accuracy < treshold[index]
    elif task=='psMNIST': return accuracy > treshold[index]
    elif task=='PTB': return accuracy < treshold[index]


tasks=['copy','psMNIST','PTB']
col_titles = ['Static','Homogeneous', 'Heterogeneous']
scenarios = [[False, 'gamma'], [True, 'gamma'], [True, 'gamma2']]
LPs, colors = ['False', 'True', 'True'], ['tab:blue', 'tab:orange', 'tab:green']
ylims = [[-0.8,0.25],[-0.4, 0.3],[-1.5,0.5]]
titles=['Train loss:', 'Valid accuracy:', 'Valid bpc:']
labels=[['$\geq$ 0.075',' < 0.075 (top 10%)'],['$\leq$ 0.928',' > 0.928 (top 10%)'], ['$\geq$ 1.6',' < 1.6 (top 10%)']]

lr = 1e-04
run = 'run_1'
lp = False
epochs = [0,25,50,75,99]
cmap = plt.cm.get_cmap('jet')
SAVEDIR = '/Volumes/GEADAH/3_Research/data/LEs/finals/'

nonlins2 = [[1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0],
            [1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]+[1.25, 3.75, 6.25, 8.75, 11.25, 13.75, 16.25, 18.75]]
saturations2 = [0.0, 0.25, 0.5, 0.75, 1.0]


for i in range(3):
    task = tasks[i]
    goods, goods_stds = [], []
    for scenario_index in range(3):
        lp, nonlin = scenarios[scenario_index]
        try:
            if task=='psMNIST':
                if nonlin=='gamma2': LEs_array = np.load(SAVEDIR+f'{task}_LP_{lp}_NL_{nonlin}_nsteps_5000.npy')
                else: LEs_array = np.load(SAVEDIR+f'{task}_LP_{lp}_NL_{nonlin}_nsteps_10000.npy')
            else:
                LEs_array = np.load(SAVEDIR+f'{task}_LP_{lp}_NL_{nonlin}_nsteps_5000.npy')
        except Exception as e:
            print(e)
            LEs_array = np.nan*np.zeros([5,9,5,2])

        # if (task=='copy' and nonlin=='gamma2'): LEs_array = np.nan*np.zeros([5,9,5,2])

        good_trajs, bad_trajs  = [], []
        n_good, n_bad = 0, 0
        for index in np.ndindex(len(nonlins2[0][:]), len(saturations2)):
            if task_test(task, nonlins2[0][index[0]], saturations2[index[1]], lp=lp, nonlin=nonlin):
                good_trajs.append(LEs_array[:, index[0], index[1], 0])
#                 axs[row,col].plot(LEs_array[:, index[0], index[1], 0], color=colors[col], label=label);
                n_good +=1
            else:
                bad_trajs.append(LEs_array[:, index[0], index[1], 0])
#                 axs[row,col].plot(LEs_array[:, index[0], index[1], 0], color='grey', label=label);
                n_bad +=1

        print(f'LEs: {task}, {lp}, {nonlin}: {n_good}, {n_bad}')
        good, good_std = np.nanmean(np.array(good_trajs), axis=0), np.nanstd(np.array(good_trajs), axis=0)
        goods.append(good)
        goods_stds.append(good_std)

        bad, bad_std = np.nanmean(np.array(bad_trajs), axis=0), np.nanstd(np.array(bad_trajs), axis=0)
        

    for k in range(3):
        axs[1,i].plot(np.arange(5), goods[k], color=colors[k], label=col_titles[k], zorder=4);
        axs[1,i].fill_between(np.arange(5), goods[k]+goods_stds[k], goods[k]-goods_stds[k], color=colors[k], alpha = 0.15, zorder=4);

    # copy_gamma2 = np.load(SAVEDIR+f'copy_LP_True_NL_gamma2_nsteps_5000.npy')
    # print(copy_gamma2.shape)
    # axs[1,0].plot(np.arange(5), np.nanmean(copy_gamma2[:,:,0,0], axis=1), color=colors[2], zorder=4);

    # axs[1,0].set_title('Lyapunov exponents\n')
    axs[1,0].set_ylabel("Maximal Lyapunov exponent")
    axs[1,i].set_xlabel('Epochs')
    axs[1,i].set_xlim([0,4])
    axs[1,i].set_xticks(np.linspace(0,4,5))
    axs[1,i].set_xticklabels(['0','25','50','75','99'])
    if i==0 : axs[1,i].legend(loc='upper left');
    axs[1,i].axhline(y=0, linestyle='--', color='grey', zorder=2)
    
    # axs[1,1].set_ylim([-0.1,0.05])

    # axs[1,0].set_yticks([-0.08,-0.06, -0.04, -0.02, 0.0, 0.02, 0.04])
    # axs[1,0].set_yticklabels([-0.08,"", -0.04, "", 0.0, "", 0.04])

    # axs[1,1].set_yticks([-0.1,-0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.1])
    # axs[1,1].set_yticklabels([-0.1,"", -0.05, "", 0.0, "", 0.05, '', 0.1])

    # axs[1,1].set_yticks([-0.04,-0.02, 0.0, 0.02])
    # axs[1,2].set_yticks([-1.2, -0.8, -0.4, 0.0, 0.4])


# ==========================================================================================
#  Col 3 : Mutual information
# ==========================================================================================


print('-'*40+'\nMutual information\n'+'-'*40)
import mi_definitions as mi_defs

axs[2,0].set_ylabel('Saturation $(s)$')

for i in range(3):
    task = tasks[i]
    if task=='psMNIST':
        d, num_init= 2, 5
    else:
        d, num_init= 0, 10
    if os.path.exists(f'/Volumes/GEADAH/3_Research/data/MIs/{task}_all_mis.npy'):
        print(f'MI, {task}')
        mutual_informations = np.load(f'/Volumes/GEADAH/3_Research/data/MIs/{task}_all_mis.npy')
        ns_2,ss_2, plot_mis_2 = mi_defs.plot_mi_contour(mutual_informations,d,1,num_init)
        CF = axs[2,i].contourf(ns_2,ss_2,plot_mis_2, 30, cmap='viridis')
        cbar = fig.colorbar(CF, ax=axs[2,i], location='right', shrink=0.9, aspect=20)#, ticks=[0.2,0.6,1.0,1.4,1.8,2.2,2.6])

        # axs[2,i].set_title(r'$I(H_t,X_{\{t-1,t\}})$'))
    # axs[2,2].set_title('$d=0$, $n_i=2$')
    axs[2,i].set_xlabel('Gain $(n)$')

# cmap = plt.cm.get_cmap('viridis')
# sm2 = plt.cm.ScalarMappable(cmap='greys', norm=plt.Normalize(vmin=0, vmax=2.56)) # object to add the colors to, may not be needed
# cbar = fig.colorbar(sm2, ax=axs[:], label='Mutual information estimate ($nats$)', shrink=0.9, location='right', ticks=[0,0.5,1.0,1.5,2.0,2.5], aspect=65)

# plt.savefig('/home/stefan/code/data/gamma_mi/PTB/mi_exp.png')


# =============================================
#  Col 4 : Heterogeneous points
# =============================================


print('-'*40+'\nHeterogeneous points\n'+'-'*40)

if plot_heterogeneous:
    import scipy

    epoch = -1
    hid_sizes = [128,400,600]
    bests = [[12.5, 0.0], [10.0, 0.0], [2.5, 0.0]]
    homedir = f'/Volumes/GEADAH/3_Research/data/task_data/'

    for i in range(3):
        task = tasks[i]
        hid_size = hid_sizes[i]
        [n, s] = bests[i]
        if task=='PTB':
            file = f'{task}/T_150/LP_True_HS_{hid_size}_NL_gamma2_lr_0.0001_OP_adam/run_1/n_{n}_s_{s}/shapeparams_n_{n}_s_{s}.npy'
        elif task=='psMNIST':
            file = f'{task}/LP_True_HS_{hid_size}_NL_gamma2_lr_0.0001_OP_adam/run_1/n_{n}_s_{s}/shapeparams.npy'
        else:
            file = f'{task}/LP_True_HS_{hid_size}_NL_gamma2_lr_0.0001_OP_adam/run_1/n_{n}_s_{s}/shapeparams_n_{n}_s_{s}.npy'
        
        a = np.load(homedir+file)
        center = np.mean(a[epoch,:,:], axis=1)
        truth = defs.get_params(n, s, task)[epoch]
        mean_disp = center-[n,s]

        axs[3,i].scatter(a[epoch,0,:], a[epoch,1,:], s=10, c='k', zorder=5);
        axs[3,i].scatter([n], [s], s=30, c='red', label='Initialisation', zorder=6);
        axs[3,i].scatter(*center, s=30, c='tab:green', label='Center of mass', zorder=6)

        # axs[3,i].quiver([n],[s], *mean_disp, linewidth=8, angles='xy', scale_units='xy', scale=2, 
        #     zorder=5, color='blue', label='Mean variation');
        # axs[3,i].quiver([n],[s], *truth, linewidth=8, angles='xy', scale_units='xy', scale=2,
        #     zorder=5, color='red', label='Homogeneous');


        axs[3,i].set_xlabel('Gain $(n)$')
        axs[3,0].set_ylabel('Saturation $(s)$')
        # axs[i,3].set_xlim([n-0.2,n+0.2])
        # axs[i,3].set_ylim([-0.6,0.2])
        axs[3,0].legend(loc='lower left')

    # axs[3,0].set_title('Heterogeneous adaptatioxn\n')


# =================

SAVEDIR = '/Volumes/GEADAH/3_Research/data/figures/gamma'
FILE = os.path.join(SAVEDIR,f'big_fig_T')
plt.savefig(FILE, dpi=200)
# 
# plt.show()
