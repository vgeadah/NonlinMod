'''
Main performance curves for both tasks and learning settings (adapt, normal, etc)
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
from definitions import get_accuracy
import definitions as defs 
from defs_performance import plot_performance_hist
import os 

nonlins = np.sort([1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0])
saturations = [0.0, 0.25, 0.5, 0.75, 1.0]
crit = 'top_25'
labels = ['Static','Homogeneous','Heterogeneous']
colors = ['tab:blue', 'tab:orange', 'tab:green']
tasks = ['copy', 'psMNIST','PTB']

task_params = {
    'sMNIST' : {'hidden_size':400, 'lr':0.0001, 'baseline':0.1, 'pmin':0, 'pmax':1, 'p':'Test accuracy', 'diff_grid':False},
    'psMNIST': {'hidden_size':400, 'lr':0.0001, 'baseline':0.1, 'pmin':0, 'pmax':1, 'p':'Test accuracy', 'diff_grid':True},
    'copy'   : {'hidden_size':128, 'lr':0.0001, 'baseline':0.095, 'pmin':0, 'pmax':0.26, 'p':'Loss', 'diff_grid':False},
    'PTB'    : {'hidden_size':600, 'T':150, 'lr':0.0002, 'baseline':None, 'pmin':1.4, 'pmax':4.6, 'p':'Test Bit per Character (BPC)', 'diff_grid':True}
}

epoch = -1
lr = 1e-04
general_cmap = 'viridis'

# =========

# fig, axs = plt.subplots(figsize=[13,7],nrows=2, ncols=3, constrained_layout=True);
fig = plt.figure(figsize=[12,9], constrained_layout=True)
widths = [2,2,2]
heights = [1, 1.75+0.05, 1]
spec = fig.add_gridspec(ncols=3, nrows=3, width_ratios=widths, height_ratios=heights)


# =============================================
# Row 1 : Static performance
# =============================================


for i in range(3):
	ax = fig.add_subplot(spec[0,i])
	task = tasks[i]
	N, S = np.meshgrid(nonlins, saturations)
	Z = np.empty([len(nonlins), len(saturations)])
	for index in np.ndindex(9,5): 
		n, s = nonlins[index[0]], saturations[index[1]]
		try:
			Z[index] = get_accuracy(n, s, task, run='run_1', lr=1e-04, lp=False)[0][epoch]
		except Exception as e:
			print(e)

	if task=='PTB':
		Z = np.load(f'PTB_test-bpc_LP_False_NL_gamma_run_1.npy')[::2]

	if task == 'PTB' or task == 'copy': index = 0
	else: index = 1

	if task=='psMNIST':
		CS = ax.contourf(N, S, Z.transpose(), levels=np.linspace(0,1,16), cmap=general_cmap)
		cbar = fig.colorbar(CS, ax=ax, shrink = 0.9, aspect=20, label=task_params[task]['p'], location='bottom', ticks=np.linspace(0,1,5))
	else:
		CS = ax.contourf(N, S, Z.transpose(), 16, cmap=general_cmap)
		cbar = fig.colorbar(CS, ax=ax, shrink = 0.9, aspect=20, label=task_params[task]['p'], location='bottom')
	
	ax.set_title(task, fontsize=15, fontweight='bold')
	# ax.set_xticklabels([])
	if i==0:
		ax.set_ylabel('Saturation ($s$)')
	# else:
		# ax.set_yticklabels([])
	# cbar.ax.set_title(task_params[task]['p'], fontsize=9)
    # print('Min : {} at n={}, s={}'.format(np.amin(Z), nonlins[np.argmin(Z)%9], 
    #                                   saturations[(np.argmin(Z) - np.argmin(Z)%9)%5]))


# =============================================
# Row 2 : adaptation dynamics
# =============================================


print('-'*40+'\nadaptation dynamics\n'+'-'*40)

chosen_epoch = 99
run = 'run_1'

tasks = ['copy','psMNIST', 'PTB']

for i in range(3):
    task = tasks[i]
    pmax, pmin = task_params[task]['pmax'],task_params[task]['pmin']
    ax = fig.add_subplot(spec[1,i])
    cmap = plt.cm.get_cmap(general_cmap)
    # if task == 'psMNIST': cmap = 'viridis'
    cmap2 = plt.cm.get_cmap('Greys')

    allparams, colors, maxs, mins = defs.param_animation(task, epoch=99, run=run, lr=1e-04, cmap=cmap)
    nonlins = [1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
    if task_params[task]['diff_grid']:
        nonlins += [3.75, 6.25, 8.75, 11.25, 13.75, 16.25, 18.75]
    saturations = [0.0, 0.25, 0.5, 0.75, 1.0]

    for index in np.ndindex(len(nonlins),len(saturations)):
        print(index)
        # lines
        ax.plot(allparams[index[0],index[1],:chosen_epoch,0],
            allparams[index[0],index[1],:chosen_epoch,1],'-', c='grey', zorder=2)

        # grid
        ax.scatter(nonlins[index[0]], saturations[index[1]], s=5, c='white', alpha=1, zorder=0, edgecolor='k', linewidths=0.4)

        # adapted points
        try:
        	accuracy = defs.get_accuracy(nonlins[index[0]],saturations[index[1]],task)[0][-1]
        	normalised = (accuracy-pmin)/(pmax-pmin)
        	# print(normalised)
        	if task=='psMNIST': z=100*normalised+5
        	else: z=100* (1/normalised)+5
        	# print(accuracy)
        	# print(allparams[index[0],index[1],chosen_epoch,0])
        	ax.scatter(allparams[index[0],index[1],chosen_epoch,0], allparams[index[0],index[1],chosen_epoch,1],
                s=70, c=[cmap(normalised)], cmap='plasma', alpha=1, zorder=5, ec='k', linewidths=0.2)
        except Exception as e:
        	print(e)
        	None

    ax.set_xlabel('Gain $(n)$')
    ax.set_xlim([0, 21])
    ax.set_ylim([-0.15,1.75])

    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,1.75])

    ax.grid(True, zorder='below', which='both', color='lightgrey')
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    if i==0:
    	ax.set_ylabel('Saturation ($s$)')
    # else:
    	# ax.set_yticklabels([])
    ax.set_xlabel('Gain ($n$)')
    ax.set_xlim([1,20])
    # axs[1,i].set_yticklabels([0.0, ''  , 0.5, ''  , 1.0, ''  , 1.5,''])
    # axs[1,0].set_ylabel('Saturation $(s)$')
    # if i!=0 : axs[1,i].set_yticklabels([])
    # axs[i].grid(True, zorder=-1);

    # fig.suptitle(task);
    # ===========================

    # colorbar :
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))

    # pmin, pmax = task_params[task]['pmin'], task_params[task]['pmax']
    # if task == 'copy' or task == 'PTB':
    #     ticks = list(map(str, [round(i,3) for i in np.linspace(pmin, pmax-(pmax-pmin)/5, 5)]))
    #     ticks.append('> {}'.format(pmax))
    # else:
    #     ticks = list(map(str, [round(i,3) for i in np.linspace(pmin, pmax, 6)]))
    # cbar = fig.colorbar(sm, ax=axs[1,i], location='right', shrink=0.9, aspect=20) #label=task_params[task]['p'],
    # cbar.ax.set_yticklabels(ticks)
    # cbar.ax.set_title(task_params[task]['p'], fontsize=8)

# =============================================
#  row 3 : Heterogeneous points
# =============================================

import scipy

epoch = -1
hid_sizes = [128,400,600]
bests = [[12.5, 0.0], [10.0, 0.0], [2.5, 0.0]]
homedir = f'/Volumes/GEADAH/3_Research/data/task_data/'

for i in range(3):
    task = tasks[i]
    ax = fig.add_subplot(spec[2,i])
    hid_size = hid_sizes[i]
    [n, s] = bests[i]
    if task=='PTB':
        file = f'{task}/T_150/LP_True_HS_{hid_size}_NL_gamma2_lr_0.0001_OP_adam/run_1/n_{n}_s_{s}/shapeparams_n_{n}_s_{s}.npy'
    elif task=='psMNIST':
        file = f'{task}/LP_True_HS_{hid_size}_NL_gamma2_lr_0.0001_OP_adam/run_1/n_{n}_s_{s}/shapeparams.npy'
    else:
        file = f'{task}/LP_True_HS_{hid_size}_NL_gamma2_lr_0.0001_OP_adam/run_1/n_{n}_s_{s}/shapeparams_n_{n}_s_{s}.npy'
    
    a = np.load(homedir+file)
    origin = np.mean(a[epoch,:,:], axis=1)

    ax.scatter(a[epoch,0,:], a[epoch,1,:], s=10, c='k', zorder=5);
    ax.scatter([n], [s], s=30, c='red', label='Initialisation', zorder=6);
    ax.scatter(*origin, s=30, c='tab:green', label='Center of mass', zorder=6)
    ax.set_xlabel('Gain $(n)$')
    # axs[i,3].set_xlim([n-0.2,n+0.2])
    # axs[i,3].set_ylim([-0.6,0.2])
    if i==0: 
    	ax.set_ylabel('Saturation $(s)$')
    	ax.legend(loc='lower left')

    ax.grid(True, zorder='below', which='both', color='lightgrey')
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    # ax.set_xlabel('Epochs')
    # ax.set_xlim([1,20])

    # ax.set_title('Heterogeneous adaptation\n')


# =================

# plot_performance_hist('copy', 0, 'Q', 'baseline',  ax=axs[1,0])
# axs[1,0].set_ylabel('Loss')
# ===================
#     psMNIST
# ===================
# plot_performance_hist('psMNIST', 0, 'Q', 'baseline',  ax=axs[1,1])
# ===================
#     PTB
# ===================
# plot_performance_hist('PTB', -1.5, 'Q', 'baseline',  ax=axs[1,2])
# axs[1,2].set_ylim([1e-01,1e+00])

# print(fig.axes)
indices = [0,6,9]
for i, label in enumerate(('A', 'B', 'C')):
    # ax = fig.add_subplot(2,2,i+1)
    index = indices[i]
    fig.axes[index].text(-0.15, 1.0, label, transform=fig.axes[index].transAxes,
    	fontsize=16, fontweight='bold', va='top', ha='right')

SAVEDIR = '/Volumes/GEADAH/3_Research/data/figures/gamma'
FILE = os.path.join(SAVEDIR,f'main_performance_fig')
plt.savefig(FILE, dpi=300)


# plt.show();













# ===============================================================================================
#                       GARBAGE
# ===============================================================================================

# # ===================
# #      copy
# # ===================

# task= 'copy'
# print('='*40)
# print(task+'\n')
# res = 75
# steps = 5000

# # ===== get data =====
# accs_T = np.zeros([len(saturations),len(nonlins), int(steps)])
# accs_F = np.zeros([len(saturations),len(nonlins), int(steps)])

# for index in np.ndindex(len(saturations),len(nonlins)):
#     n, s = nonlins[index[1]], saturations[index[0]]
#     try: accs_T[index] = get_accuracy(n,s,task,lp=True)[0][:int(steps)]
#     except: accs_T[index] = np.zeros(int(steps))*np.nan
#     try: accs_F[index] = get_accuracy(n,s,task,lp=False)[0][:int(steps)]
#     except Exception as e:
#         print(e)
#         accs_F[index] = np.zeros(int(steps))*np.nan

# # ===== treshold =====
# tresholds= {
#     1000:{'select':[0.1,0.1],'top_50':[0.135,0.123],'acc_50':[0.06,0.06], 'top_25':[0.101,0.1002],'top_10':[0.096,0.09]},
#     2000:{'select':[0.01,0.01],'top_50':[0.1025,0.1],'acc_50':[0.06]*3, 'top_25':[0.097,0.0967],'top_10':[0.096,0.09]},
#     5000:{'select':[0.06,0.06],'top_50':[0.08,0.08,0.08],'acc_50':[0.06]*3, 'top_25':[0.0946,0.09461, 0.08],'top_10':[0.09,0.09]},
#     10000:{'select':[0.005,0.005],'top_50':[0.08,0.08,0.08],'acc_50':[0.06]*3, 'top_25':[0.0946,0.09461, 0.08],'top_10':[0.09,0.09]}
# }
# treshold= tresholds[steps]['acc_50']
# T_indices = accs_T[:,:,-1] < treshold[0]
# F_indices = accs_F[:,:,-1] < treshold[1]

# print('# Adapt  :', np.sum([int(i) for i in T_indices.flatten()]))
# print('# Normal :', np.sum([int(i) for i in F_indices.flatten()]))

# # ===== Plot =====

# mean_F = np.array([np.nanmean(accs_F[F_indices], axis=0)[i] for i in res*np.arange(int(steps/res))])
# std_F  = np.array([np.nanstd(accs_F[F_indices], axis=0)[i] for i in res*np.arange(int(steps/res))])
# axs[0].plot(mean_F, color=colors[0], label=labels[0]);
# axs[0].fill_between(np.arange(int(steps/res)),mean_F-std_F, mean_F+std_F, color=colors[0], alpha = 0.15);


# mean_T = np.array([np.nanmean(accs_T[T_indices], axis=0)[i] for i in res*np.arange(int(steps/res))])
# std_T  = np.array([np.nanstd(accs_T[T_indices], axis=0)[i] for i in res*np.arange(int(steps/res))])
# axs[0].plot(mean_T, color=colors[1], label=labels[1]);
# axs[0].fill_between(np.arange(int(steps/res)), mean_T-std_T, mean_T+std_T, color=colors[1], alpha = 0.15);

# # ===== Set axes
# axs[0].set_yscale('log');
# axs[0].set_ylim([1e-02,1e+00])
# # axs[0].set_ylim([0,1])
# axs[0].set_xlim([0,steps/res])
# # axs[0].grid(True, zorder='below', which='both')

# # axs[0].legend();
# axs[0].set_ylabel('Log Train Loss');
# axs[0].set_xlabel('Iterations');
# axs[0].set_title(task)
# axs[0].set_xticks(np.linspace(0,steps/res,5))
# axs[0].set_xticklabels([int(i) for i in np.linspace(0,steps,5)])


# # # =======================
# # #       psMNIST
# # # =======================


# task = 'psMNIST'
# ax_index = 0
# print('='*40)
# print(task+'\n')

# # treshold =====
# tresholds= {
# #     'copy':   {'top_50':[0.09,0.09] ,'acc_50':0.06},
#     'psMNIST':{'top_50':[0.77,0.77,0.8],'top_25':[0.91,0.895,0.9178], 'top_10':[0.928,0.945,0.94],'acc_50':0.50,'all':[0.2, 0.0]},
#     'PTB':    {'top_50':[1.64,1.67,1.64],'top_25':[1.612,1.625,1.615],'top_10':[1.60,1.595],'acc_50':[2.5,2.5], 'all':[5,5,5]}
# }
# shift = 0
    
# #containers
# accs_T = np.zeros([len(saturations),len(nonlins),100])
# accs_F = np.zeros([len(saturations),len(nonlins),100])
# accs_LT= np.zeros([len(saturations),len(nonlins),50 ])
# accs_g2= np.zeros([len(saturations),len(nonlins),100])

# for index in np.ndindex(len(saturations),len(nonlins)):
#     n, s = nonlins[index[1]], saturations[index[0]]
#     accs_T[index] = get_accuracy(n, s, task, lp=True)[0]
#     try:
#         accs_F[index] = get_accuracy(n, s, task, lp=False)[0]
#     except Exception as e:
#         # print(e)
#         accs_F[index] = np.zeros(100)*np.nan
# #     try:
# #         file = f'T_150/LP_True_HS_600_NL_gamma_lr_0.0001_OP_adam/late_adapt/start_50/n_{n}_s_{s}/RNN_Val_Losses'
# #         accs = np.array(pickle.load(open(savedir+file, 'rb' )))
# #         if len(accs)<50: accs = np.pad(accs, (0,50-len(accs)), 'constant', constant_values=(accs[-1]))
# #         accs_LT[index] = accs/np.log(2)
# #     except Exception as e:
# #         accs_LT[index] = np.zeros(50)*np.nan

#     # gamma2  
#     try:
#         accs_g2[index] = get_accuracy(n, s, task, lp=True, nonlin='gamma2')[0]
#     except:
#         accs_g2[index] = np.zeros(100)*np.nan
#     # try:
#     #     savedir = f'/Volumes/GEADAH/3_Research/data/task_data/{task}/'
#     #     if task=='psMNIST': file_accuracy = f'gamma2/LP_True_HS_400_NL_gamma2_n_{n}_s_{s}_lr_0.0001_OP_adam/RNN_Test_accuracy'
#     #     else: file_accuracy = f'T_150/gamma2/LP_True_HS_600_NL_gamma2_n_{n}_s_{s}_lr_0.0001_OP_adam/RNN_Val_Losses'
#     #     accs_g2[index] = np.array(pickle.load(open(savedir+file_accuracy,'rb')))
#     #     if task=='PTB': accs_g2[index] *= 1/np.log(2)
#     # except Exception as e:
#     #     # print(e)
#     #     accs_g2[index] = np.zeros(100)*np.nan
        
# treshold = tresholds[task][crit]
# if task == 'psMNIST':
#     T_indices = accs_T[:,:,-1]   > treshold[0]
#     F_indices = accs_F[:,:,-1]   > treshold[1]
#     LT_indices = accs_LT[:,:,-1] > treshold[0]
#     g2_indices = accs_g2[:,:,-1] > treshold[2]
# else:
#     T_indices = accs_T[:,:,-1] < treshold[0]
#     F_indices = accs_F[:,:,-1] < treshold[1]
#     LT_indices = accs_LT[:,:,-1] < treshold[0]
#     g2_indices = accs_g2[:,:,-1] < treshold[2]

# print('# Adapt  :', np.sum([int(i) for i in T_indices.flatten()]))
# print('# Normal :', np.sum([int(i) for i in F_indices.flatten()]))
# print('# Late a :', np.sum([int(i) for i in LT_indices.flatten()]))
# print('# gamma2 :', np.sum([int(i) for i in g2_indices.flatten()]))

# # ===== Plot =====

# mean_F, std_F = np.nanmean(accs_F[F_indices], axis=0)+shift, np.nanstd(accs_F[F_indices], axis=0)
# axs[ax_index].plot(mean_F, color=colors[0], label=labels[0]);
# # axs[ax_index].fill_between(np.arange(100),mean_F-std_F, mean_F+std_F, color=colors[0], alpha = 0.15);

# mean_T, std_T = np.nanmean(accs_T[T_indices], axis=0)+shift, np.nanstd(accs_T[T_indices], axis=0)
# axs[ax_index].plot(mean_T, color=colors[1], label=labels[1]);
# # axs[ax_index].fill_between(np.arange(100), mean_T-std_T, mean_T+std_T, color=colors[1], alpha = 0.15);

# # mean_LT, std_LT = np.nanmean(accs_LT[LT_indices], axis=0), np.nanstd(accs_LT[LT_indices], axis=0)
# # axs[i].plot(np.arange(50,100), mean_LT, color='tab:green', label='Late adapt');
# # axs[i].fill_between(np.arange(50,100),mean_LT-std_LT, mean_LT+std_LT, color='tab:green', alpha = 0.15);

# mean_g2, std_g2 = np.nanmean(accs_g2[g2_indices], axis=0)+shift, np.nanstd(accs_g2[g2_indices], axis=0)
# axs[ax_index].plot(np.arange(100), mean_g2, color=colors[2], label=labels[2]);
# # axs[ax_index].fill_between(np.arange(100),mean_g2-std_g2, mean_g2+std_g2, color=colors[2], alpha = 0.15);

# # ===== Set axes

# axs[ax_index].set_ylabel('valid accuracy');
# axs[ax_index].set_title(task)
# axs[ax_index].set_xlim([0,99])
# axs[ax_index].set_xlabel('Epochs');
# axs[ax_index].set_xticks([0,25,50,75,99]);

# from matplotlib.lines import Line2D
# lines = [Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in colors]
# axs[ax_index].legend(lines, labels, loc='lower right')

# # =======================
# #       PTB
# # =======================


# task= 'PTB'
# ax_index = 1
# print('='*40)
# print(task+'\n')
# shift = -1.5
# # ===== get data =====
# savedir = f'/Volumes/GEADAH/3_Research/data/task_data/{task}/'

# #containers
# accs_T = np.zeros([len(saturations),len(nonlins),100])
# accs_F = np.zeros([len(saturations),len(nonlins),100])
# accs_LT= np.zeros([len(saturations),len(nonlins),50 ])
# accs_g2= np.zeros([len(saturations),len(nonlins),100])

# for index in np.ndindex(len(saturations),len(nonlins)):
#     n, s = nonlins[index[1]], saturations[index[0]]
#     try:
#         accs_T[index] = get_accuracy(n, s, task, lp=True)[0]
#     except:
#         accs_T[index] = np.zeros(100)*np.nan
#     try:
#         accs_F[index] = get_accuracy(n, s, task, lp=False)[0]
#     except Exception as e:
# #         print(e)
#         accs_F[index] = np.zeros(100)*np.nan
# #     try:
# #         file = f'T_150/LP_True_HS_600_NL_gamma_lr_0.0001_OP_adam/late_adapt/start_50/n_{n}_s_{s}/RNN_Val_Losses'
# #         accs = np.array(pickle.load(open(savedir+file, 'rb' )))
# #         if len(accs)<50: accs = np.pad(accs, (0,50-len(accs)), 'constant', constant_values=(accs[-1]))
# #         accs_LT[index] = accs/np.log(2)
# #     except Exception as e:
# #         accs_LT[index] = np.zeros(50)*np.nan
    
#     # gamma2  
#     try:
#         if task=='psMNIST': file_accuracy = f'gamma2/LP_True_HS_400_NL_gamma2_n_{n}_s_{s}_lr_0.0001_OP_adam/RNN_Test_accuracy'
#         else: file_accuracy = f'T_150/gamma2/LP_True_HS_600_NL_gamma2_n_{n}_s_{s}_lr_0.0001_OP_adam/RNN_Val_Losses'
#         accs_g2[index] = np.array(pickle.load(open(savedir+file_accuracy,'rb')))
#         if task=='PTB': accs_g2[index] *= 1/np.log(2)
#     except Exception as e:
# #         print(e)
#         accs_g2[index] = np.zeros(100)*np.nan

# i=0
# data=[]

# for epoch in [0,50,99]:
#     filtered_F = accs_F[:,:,epoch][~np.isnan(accs_F[:,:,epoch])]
#     filtered_T = accs_T[:,:,epoch][~np.isnan(accs_T[:,:,epoch])]
#     filtered_g2 = accs_g2[:,:,epoch][~np.isnan(accs_g2[:,:,epoch])]
#     print(len(filtered_F),len(filtered_T),len(filtered_g2))
#     data.append([filtered_F+shift, filtered_T+shift, filtered_g2+shift])

#     med_c, symb = 'k', '+'
#     bp = axs[ax_index].boxplot(data[i], positions = 4*i+np.array([1,2,3]), widths = 0.6, 
#                     showfliers=True, patch_artist=True, medianprops=dict(color=med_c),sym=symb)
    
#     for patch, color in zip(bp['boxes'], colors):
#         patch.set_facecolor(color)
# #     for j in [0,1,2]:
# #         med = bp['medians'][j]
# #         ax.plot(np.average(med.get_xdata()), np.nanmean(data[i][j]),
# #                 color='w', marker='*', markeredgecolor='k', zorder=10)
#     i+=1
    

# # set axes limits and labels
# axs[ax_index].set_xticklabels([0, 50, 99])
# axs[ax_index].set_xticks(2+4*np.arange(3))
# # ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
# axs[ax_index].set_yscale('log')
# ticks=[1e-01,1]
# axs[ax_index].set_yticks(ticks)
# axs[ax_index].set_yticklabels([str(-shift)+'+'+str(ticks[i]) for i in range(len(ticks))])

# # configure background
# # axs[2].grid(True, zorder='below', which='both', color='w')
# # axs[2].set_facecolor('#f0f0f0')
# # for spine in axs[2].spines.values():
# #     spine.set_visible(False)
# # axs[2].xaxis.tick_bottom()
# # axs[2].yaxis.tick_left()

# axs[ax_index].set_xlabel('Epochs');
# axs[ax_index].set_ylabel('Log Test Bit per Character (BPC)');
# axs[ax_index].set_title('PTB character-level')

# # legend
# # import matplotlib.patches as mpatches
# # patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(3)] # ec = 'k'
# # axs[2].legend(handles=patches, loc='lower left');

# # SAVEDIR = '/Volumes/GEADAH/3_Research/data/figures/gamma'
# # FILE = os.path.join(SAVEDIR,f'psMNIST_PTB_performance')
# # plt.savefig(FILE, dpi=500)

# plt.show();