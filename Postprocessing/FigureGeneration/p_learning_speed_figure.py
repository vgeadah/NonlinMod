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

fig, axs = plt.subplots(figsize=[10,6],nrows=2, ncols=2, constrained_layout=True);


for ax in axs.flatten():
	ax.grid(True, zorder='below', which='both', color='lightgrey')
	ax.xaxis.tick_bottom()
	ax.yaxis.tick_left()

for i, label in enumerate(('A', 'B')):
    # ax = fig.add_subplot(2,2,i+1)
    axs[i,0].text(-0.12, 1.0, label, transform=axs[i,0].transAxes,
        fontsize=16, fontweight='bold', va='top', ha='right')



# ======= psMNIST
epoch = 40
accs_psmnist_F, accs_psmnist_T, accs_psmnist_g2 = plot_performance_hist('psMNIST', 0, 'Q', 'baseline',  ax=axs[0,0])
axs[0,0].set_title('psMNIST\n', fontsize=15, fontweight='bold')
axs[0,0].legend(loc='lower right')

bins = np.linspace(0.6, 1, 30)
axs[1,0].hist(accs_psmnist_F[:,:,epoch].flatten(), color=colors[0], label=labels[0], bins=bins, alpha = 0.8, zorder=5);
axs[1,0].hist(accs_psmnist_T[:,:,epoch].flatten(), color=colors[1], label=labels[1], bins=bins, alpha = 0.8, zorder=3);
axs[1,0].hist(accs_psmnist_g2[:,:,epoch].flatten(), color=colors[2], label=labels[2], bins=bins, alpha = 0.8, zorder=4);
axs[1,0].set_xlabel('Test accuracy')
axs[1,0].legend(loc='upper  left')


rect1 = plt.Rectangle((38, 0.66), 4, 0.33, facecolor="red", alpha=0.2, edgecolor='k')
axs[0,0].add_patch(rect1)


# ======= PTB
accs_PTB_F, accs_PTB_T, accs_PTB_g2 = plot_performance_hist('PTB', -1.5, 'Q', 'baseline',  ax=axs[0,1])
# axs[0,1].set_ylim([1e-01,1e+00])
axs[0,1].set_title('PTB\n', fontsize=15, fontweight='bold')


bins = np.linspace(1.5, 3.5, 30)
axs[1,1].hist(accs_PTB_F[:,:,epoch].flatten(), color=colors[0], label=labels[0], bins=bins, alpha = 0.8, zorder=5);
axs[1,1].hist(accs_PTB_T[:,:,epoch].flatten(), color=colors[1], label=labels[1], bins=bins, alpha = 0.8, zorder=3);
axs[1,1].hist(accs_PTB_g2[:,:,epoch].flatten(), color=colors[2], label=labels[2], bins=bins, alpha = 0.8, zorder=4);
axs[1,1].set_xlabel('Valid BPC')

rect2 = plt.Rectangle((38, 0.1), 4, 1.9, facecolor="red", alpha=0.2, edgecolor='k')
axs[0,1].add_patch(rect2)



# SAVEDIR = '/Volumes/GEADAH/3_Research/data/figures/gamma'
SAVEDIR = '/Users/Victor_Geadah_Mac/3_Research/gamma/figures'
FILE = os.path.join(SAVEDIR,f'psMNIST_PTB_performance_epochs')
plt.savefig(FILE, dpi=200)

plt.show();
