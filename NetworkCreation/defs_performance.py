import numpy as np 
from definitions import get_accuracy
 
def plot_performance_hist(task, shift, metric, crit, ax, criterion='final'):

    # ===== get data =====
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = ['Static','Homogeneous','Heterogeneous']
    savedir = f'/Volumes/GEADAH/3_Research/data/task_data/{task}/'
    nonlins = np.sort([1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0])#+[1.25, 3.75, 6.25, 8.75, 11.25, 13.75, 16.25, 18.75])
    saturations = [0.0, 0.25, 0.5, 0.75, 1.0]

    #containers
    accs_T = np.zeros([len(saturations),len(nonlins),100])
    accs_F = np.zeros([len(saturations),len(nonlins),100])
    accs_g2= np.zeros([len(saturations),len(nonlins),100])
    copy_accs_F_end, copy_accs_T_end, copy_accs_g2_end = [], [], []
    
    for index in np.ndindex(len(saturations),len(nonlins)):
        n, s = nonlins[index[1]], saturations[index[0]]
        if task=='copy':
            try:
                accs_T[index] = (get_accuracy(n, s, task, lp=True)[0][::500])[:100]
                copy_accs_T_end.append(get_accuracy(n, s, task, lp=True)[0][-1])
            except:
                accs_T[index] = np.zeros(100)*np.nan
            try:
                accs_F[index] = (get_accuracy(n, s, task, lp=False)[0][::500])[:100]
                copy_accs_F_end.append(get_accuracy(n, s, task, lp=False)[0][-1])
            except Exception as e:
        #         print(e)
                accs_F[index] = np.zeros(100)*np.nan
            try:
                accs_g2[index] = (get_accuracy(n, s, task, lp=True, nonlin='gamma2')[0][::500])[:100]
                copy_accs_g2_end.append(get_accuracy(n, s, task, lp=True, nonlin='gamma2')[0][-1])
            except Exception as e:
                print(e)
                accs_g2[index] = np.zeros(100)*np.nan
        else:
            try:
                accs_T[index] = get_accuracy(n, s, task, lp=True)[0]
            except:
                accs_T[index] = np.zeros(100)*np.nan
            try:
                accs_F[index] = get_accuracy(n, s, task, lp=False)[0]
            except Exception as e:
        #         print(e)
                accs_F[index] = np.zeros(100)*np.nan
            try:
                accs_g2[index] = get_accuracy(n, s, task, lp=True, nonlin='gamma2')[0]
            except:
                accs_g2[index] = np.zeros(100)*np.nan
#         try:
#             if task=='psMNIST': file_accuracy = f'gamma2/LP_True_HS_400_NL_gamma2_n_{n}_s_{s}_lr_0.0001_OP_adam/RNN_Test_accuracy'
#             else: file_accuracy = f'T_150/gamma2/LP_True_HS_600_NL_gamma2_n_{n}_s_{s}_lr_0.0001_OP_adam/RNN_Val_Losses'
#             accs_g2[index] = np.array(pickle.load(open(savedir+file_accuracy,'rb')))
#             if task=='PTB': accs_g2[index] *= 1/np.log(2)
#         except Exception as e:
#     #         print(e)
#             accs_g2[index] = np.zeros(100)*np.nan


#     print(copy_accs_F_end)
#     print(copy_accs_T_end)
#     print(copy_accs_g2_end)
    # fig, ax = plt.subplots(figsize=(6.5,5));

    # definitions for the axes
#     left, width = 0.1, 0.65
#     bottom, height = 0.1, 0.65
#     spacing = 0.005
#     rect_scatter = [left, bottom, width, height]
    # rect_histx = [left, bottom + height + spacing, width, 0.2]
#     rect_histy = [left + width + spacing, bottom, 0.2, height]
#     ax = fig.add_axes(rect_scatter)

    # ax_histx = fig.add_axes(rect_histx, sharex=ax)
#     ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # ===== treshold =====
    tresholds= {
        'copy':   {'top_50':[0.09,0.09, 0.09] ,'select':[0.07,0.06,0.06], 'top_25':[0.093,0.09461, 0.095],
                   'top_3':[0.01,0.01,0.01],'all':[2,2,2], 'baseline':[0.09,0.09,0.09]},
        'psMNIST':{'top_50':[0.78,0.77,0.75], 'top_25':[0.915,0.895,0.90], 'top_20':[0.9225,0.905,0.92], 'top_10':[0.928,0.945,0.92],#'top_25':[0.915,0.9006,0.913]
                   'acc_50':0.50,'all':[0,0,0], 'top_3':[0.93, 0.948,0.93], 'baseline':[0.12, 0.12, 0.12], 'all':[0.12, 0.12, 0.12],'top_5':[0.9242,0.945,0.9192]},
        'PTB':    {'top_50':[1.65,1.669,1.636],'top_25':[1.613,1.625,1.615], 'top_3':[1.594, 1.592, 1.59],
                   'top_10':[1.60,1.595, 1.6],'acc_50':[2.5,2.5], 'all':[5,5,5], 'baseline':[5,5,5]}# 'top_25':[1.611,1.622,1.614]
    }
    treshold = tresholds[task][crit]
    if criterion=='final':
        if task == 'psMNIST':
            T_indices = accs_T[:,:,-1]   > treshold[0]
            F_indices = accs_F[:,:,-1]   > treshold[1]
            g2_indices = accs_g2[:,:,-1] > treshold[2]
        else:
            T_indices = accs_T[:,:,-1] < treshold[0]
            F_indices = accs_F[:,:,-1] < treshold[1]
            g2_indices = accs_g2[:,:,-1] < treshold[2]
    
    elif criterion=='AUC':
        if task == 'psMNIST':
            print(np.nanmax(np.sum(accs_T,axis=2)))
            T_indices = np.sum(accs_T,axis=2)  > 88 # [88, 86.8, 85]
            F_indices = np.sum(accs_F,axis=2)  > 86.8
            g2_indices = np.sum(accs_g2,axis=2)  > 85
        elif task=='PTB':
            T_indices = np.nansum(accs_T,axis=2)< 182
            F_indices = np.nansum(accs_F,axis=2)< 186
            g2_indices =np.nansum(accs_g2,axis=2) < 181.8
        else:
            print(np.nanmax(np.sum(accs_T,axis=2)))
            T_indices = np.nansum(accs_T,axis=2)< 12
            F_indices = np.nansum(accs_F,axis=2)< 12
            g2_indices =np.nansum(accs_g2,axis=2) < 12
        
        
#     np.save(f'{task}_Static_indices', F_indices)
#     np.save(f'{task}_Homogeneous_indices', T_indices)
#     np.save(f'{task}_Heterogeneous_indices', g2_indices)
    print('# Static  :', np.sum([int(i) for i in F_indices.flatten()]))
    print('# Homogeneous :', np.sum([int(i) for i in T_indices.flatten()]))
    print('# Heterogeneous :', np.sum([int(i) for i in g2_indices.flatten()]))

    if task=='psMNIST':
        print('Static',np.nanmax(accs_F[:,:,-1]))
        print('Homo',np.nanmax(accs_T[:,:,-1]))
        print('Hetero',np.nanmax(accs_g2[:,:,-1]))
    else:
        print('Static',np.nanmin(accs_F[:,:,-1]))
        print('Homo',np.nanmin(accs_T[:,:,-1]))
        print('Hetero',np.nanmin(accs_g2[:,:,-1]))
        
    if metric == 'Q':
        q1_F, q2_F, q3_F = np.nanquantile(accs_F[F_indices], 0.25, axis=0)+shift, np.nanquantile(accs_F[F_indices], 0.5, axis=0)+shift, np.nanquantile(accs_F[F_indices], 0.75, axis=0)+shift
        q1_T, q2_T, q3_T = np.nanquantile(accs_T[T_indices], 0.25, axis=0)+shift, np.nanquantile(accs_T[T_indices], 0.5, axis=0)+shift, np.nanquantile(accs_T[T_indices], 0.75, axis=0)+shift
        q1_g2, q2_g2, q3_g2 = np.nanquantile(accs_g2[g2_indices], 0.25, axis=0)+shift, np.nanquantile(accs_g2[g2_indices], 0.5, axis=0)+shift, np.nanquantile(accs_g2[g2_indices], 0.75, axis=0)+shift
        
        if task=='psMNIST':
            modes = q3_F, q3_T, q3_g2
        elif task=='copy':
            modes = q2_F, q2_T, q2_g2
        else:
            modes = q1_F, q1_T, q1_g2
        
        for i in range(3):
            ax.plot(modes[i], color=colors[i], label=labels[i], zorder=5);
#             ax.fill_between(np.arange(100),q1_F, q3_F, color=colors[0], alpha=0.15, zorder=5)

#         ax.plot(q2_T, color=colors[1], label=labels[1], zorder=5);
        # ax.fill_between(np.arange(100),q1_T, q3_T, color=colors[1], alpha=0.15, zorder=5)

#         ax.plot(q2_g2, color=colors[2], label=labels[2]);
        # ax.fill_between(np.arange(100),q1_g2, q3_g2, color=colors[2], alpha=0.15)

    elif metric=='mean': 
        mean_F, std_F = np.nanmean(accs_F[F_indices], axis=0)+shift, np.nanstd(accs_F[F_indices], axis=0)
        mean_T, std_T = np.nanmean(accs_T[T_indices], axis=0)+shift, np.nanstd(accs_T[T_indices], axis=0)
        # mean_LT, std_LT = np.nanmean(accs_LT[LT_indices], axis=0), np.nanstd(accs_LT[LT_indices], axis=0)
        mean_g2, std_g2 = np.nanmean(accs_g2[g2_indices], axis=0)+shift, np.nanstd(accs_g2[g2_indices], axis=0)

        ax.plot(mean_F, color=colors[0], label=labels[0], zorder=5);
#         ax.fill_between(np.arange(100),mean_F-std_F, mean_F+std_F, color=colors[0], alpha = 0.15);

        ax.plot(mean_T, color=colors[1], label=labels[1], zorder=5);
#         ax.fill_between(np.arange(100), mean_T-std_T, mean_T+std_T, color=colors[1], alpha = 0.15);

        ax.plot(np.arange(100), mean_g2, color=colors[2], label=labels[2]);
#         ax.fill_between(np.arange(100),mean_g2-std_g2, mean_g2+std_g2, color=colors[2], alpha = 0.15);
# 
    if task=='psMNIST':
        ax.set_ylim([0.65,1])
        # ax.legend(loc='lower right');
        ax.set_ylabel('Test Accuracy');
        binwidth = 0.01
        bins = np.linspace(0.6, 1.0, 50)

    elif task=='PTB':
        if shift ==-1.5:
            ax.set_yscale('log')
            ticks=[1e-01,1]
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(-shift)+'+'+str(ticks[i]) for i in range(len(ticks))])
        else:
            ax.set_ylim([1.5,3])
        # ax.legend(loc='upper right');
        ax.set_ylabel('Log Valid BPC');
        binwidth = 0.05
        bins = np.linspace(1.5, 3.0, 50)
        
    else:
#         ax.axhline(y=0.095)
        ax.set_yscale('log')
        ax.legend(loc='upper right');
        ax.set_ylabel('Train loss');
        binwidth = 0.0
        bins = 50

    # for axs in fig.axes:
    #     # configure background
    #     axs.grid(True, zorder='below', which='both', color='w')
    #     axs.set_facecolor('#f0f0f0')
    #     for spine in axs.spines.values():
    #         spine.set_visible(False)
    #     axs.xaxis.tick_bottom()
    #     axs.yaxis.tick_left()

    ax.set_xlabel('Epochs');
    return accs_F, accs_T, accs_g2
    # ax.set_title(task)

    # ====== histograms
#     bin = np.arange(np.nanmin(accs_T)+shift, np.nanmax(accs_T)+shift, binwidth)
    
#     if task=='psMNIST':
#         ax_histy.tick_params(axis="y", labelleft=False, left=False, right=True)
#         ax_histy.tick_params(axis="x", labelbottom=False, bottom=False)
#         ax_histy.hist(accs_F[:,:,-1].flatten()+shift, bins=bins, orientation='horizontal', color=colors[0], alpha=0.8, zorder=5)
#         ax_histy.hist(accs_T[:,:,-1].flatten()+shift, bins=bins, orientation='horizontal', color=colors[1], alpha=0.8, zorder=3)
#         ax_histy.hist(accs_g2[:,:,-1].flatten()+shift, bins=bins, orientation='horizontal', color=colors[2], alpha=0.8,zorder=4)
        
#     elif task =='PTB':
#         accs_test_T = np.load(f'PTB_test-bpc_LP_True_NL_gamma_run_1.npy')
#         accs_test_F = np.load(f'PTB_test-bpc_LP_False_NL_gamma_run_1.npy')
#         accs_test_g2 = np.load(f'PTB_test-bpc_LP_True_NL_gamma2_run_1.npy')
        
#         ax_histy.tick_params(axis="y", labelleft=False, left=False, right=True)
#         ax_histy.tick_params(axis="x", labelbottom=False, bottom=False)
#         ax_histy.hist(accs_test_F.flatten()+shift, bins=bins, orientation='horizontal', color=colors[0], alpha=0.8, zorder=5)
#         ax_histy.hist(accs_test_T.flatten()+shift, bins=bins, orientation='horizontal', color=colors[1], alpha=0.8, zorder=3)
#         ax_histy.hist(accs_test_g2.flatten()+shift, bins=bins, orientation='horizontal', color=colors[2], alpha=0.8,zorder=4)
        
        
#     ax_histy.set_title('Final test\n performance', fontsize=9)
#     if task=='psMNIST':
#         ax_histy.axhline(y=np.nanquantile(accs_T[:,:,-1], 0.75), c='k', linestyle='--')
#     else:
#         ax_histy.axhline(y=np.nanquantile(accs_test_T.flatten(), 0.25), c='k', linestyle='--')
        
# #     plt.tight_layout();
#     SAVEDIR = '/Volumes/GEADAH/3_Research/data/figures/gamma'
#     FILE = os.path.join(SAVEDIR,f'{task}_performance+distribution.png')
#     plt.savefig(FILE, dpi=300)

# plt.show();