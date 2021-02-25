import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#from mpl_toolkits import mplot3d
#from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import requests
import pickle
import gzip
import math
import itertools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Function
import torchvision as T

import scipy
from scipy import linalg as la
from scipy import signal
from scipy.linalg import schur
from scipy.optimize import fsolve
from scipy.stats import unitary_group


task_params = {
    'sMNIST' : {'hidden_size':400, 'lr':0.0001, 'baseline':0.1, 'pmin':0, 'pmax':1, 'p':'Valid accuracy', 'diff_grid':False},
    'psMNIST': {'hidden_size':400, 'lr':0.0001, 'baseline':0.1, 'pmin':0, 'pmax':1, 'p':'Valid accuracy', 'diff_grid':True},
    'copy'   : {'hidden_size':128, 'lr':0.0001, 'baseline':0.095, 'pmin':0, 'pmax':0.15, 'p':'Train loss', 'diff_grid':False},
    'PTB'    : {'hidden_size':600, 'T':150, 'lr':0.0002, 'baseline':None, 'pmin':1.5, 'pmax':4, 'p':'Valid bpc', 'diff_grid':True}
}



fig, ax = plt.subplots();

# ========= gamma ============
def sup_gamma_prime(n,s):
    if s > 1/(n+1):
        return (1+s*(n-1))**2 /(4*n*s)
    else:
        return (1-s)

# In Numpy
def npgam1(x,n):
    return (1 / n)*(np.log(1 + np.exp(n*x)))
def npgam2(x,n):
    return (np.exp(n*x))/(1 + np.exp(n*x))
def np_gamma(x, n, s):
    return (1-s)*npgam1(x,n) + s*npgam2(x,n)
def np_gamma_prime(x, n, s):
    return (1-s)*npgam2(x,n) + s*n*npgam2(x,n)*(1-npgam2(x,n))



def grad_gamma(x,n,s):
    gamma_one = F.softplus(x, beta = n)
    gamma_two = torch.sigmoid(torch.mul(n,x))
    grad_n = ((1-s)/n * (x * gamma_two - gamma_one) + s*x*gamma_two*(1-gamma_two))
    grad_s = (gamma_two - gamma_one)
    return [grad_n, grad_s]

# ============ Definitions =============

def get_dir(n, s, task, run='run_1', lr=1e-04, OP='adam', lp=True,  nonlin='gamma'):
    HOMEDIR = '/Volumes/GEADAH/3_Research/data/task_data' # change that to yours
    if task == 'PTB':
        DIR = f'/PTB/T_150/LP_{lp}_HS_600_NL_{nonlin}_lr_{lr}_OP_{OP}/{run}/n_{n}_s_{s}'
    elif task == 'copy':
        DIR = f'/copy/LP_{lp}_HS_128_NL_{nonlin}_lr_{lr}_OP_adam/{run}/n_{n}_s_{s}'
    else:
        DIR = f'/{task}/LP_{lp}_HS_400_NL_{nonlin}_lr_{lr}_OP_adam/{run}/n_{n}_s_{s}'
    return HOMEDIR + DIR

def load_net(n, s, task, epoch=99, run='run_1', lr=1e-04, OP='adam', lp=True, nonlin='gamma'):
    DIR = get_dir(n, s, task, run, lr, OP, lp, nonlin)
    file = DIR + '/RNN_{}.pth.tar'.format(epoch)
    
    modeldict = torch.load(file, map_location=torch.device('cpu'))
    if nonlin=='gamma2':
        learned_n = modeldict['state_dict']['rnn.n']
        learned_s = modeldict['state_dict']['rnn.s']
    else:
        learned_n = modeldict['state_dict']['rnn.n'].item()
        learned_s = modeldict['state_dict']['rnn.s'].item()
    V = modeldict['state_dict']['rnn.V.weight']
    # if np.isnan(learned_n) or np.isnan(learned_s):
    #     return None, None, None
    # else:
    return learned_n, learned_s, V

def get_avail_nets(task, epoch = 99):  
    ns = []
    for n in nonlins:
        for s in saturations:
            if os.path.exists(get_dir(n, s, task) + '/RNN_{}.pth.tar'.format(epoch)): ns.append([n,s])
    ground = [[n,s] for n in nonlins for s in saturations]
    
    if ns == ground : return 'All'
    else: return ns


def remove_outliers(array, devs):
    mean = np.nanmean(array)
    std = np.nanstd(array)
    distance_from_mean = abs(array - mean)
    max_deviations = devs
    not_outlier = distance_from_mean < max_deviations * std
    no_outliers = array[not_outlier]
    return no_outliers


def get_accuracy(n, s, task, lr='0.0001', OP='adam', lp=True, run='run_1', nonlin='gamma'):
    l = 100 
    if run == 'late_adapt':
        l = 50
        DIR = get_dir(n, s, task, 'late_adapt/start_50', lr, OP, lp, nonlin)
        if task == 'PTB': file = DIR+'/RNN_Val_Losses'
        elif task == 'copy': file = DIR+'/RNN_Train_Accuracy'
        else: file = DIR+'/RNN_Test_Accuracy'
        
        RNN_accuracy = np.array(pickle.load(open(file, 'rb' )))
        if task == 'PTB':RNN_accuracy *= (1/math.log(2))
        if task=='copy': RNN_accuracy = np.array([RNN_accuracy[i] for i in 1000*np.arange(int(len(RNN_accuracy)/1000))])

        # Processing
        if len(RNN_accuracy)<l :
            RNN_accuracy = np.pad(RNN_accuracy, (0,l-len(RNN_accuracy)), 'constant', constant_values=(np.nan))

        return RNN_accuracy, np.nanmax(RNN_accuracy[10:]), np.nanmin(RNN_accuracy[10:])
    else:
        accuracies, a_maxs, a_mins = [], [], []
        for i in [1,2,3]:
            if (task=='copy' and i==2): continue
            DIR = get_dir(n, s, task, f'run_{i}', lr, OP, lp,  nonlin)
            if task == 'PTB': file = DIR+'/RNN_Val_Losses'
            elif task == 'copy': file = DIR+'/RNN_Train_Accuracy'
            else: file = DIR+'/RNN_Test_Accuracy'

            if os.path.exists(file):
                # print(file)
                try:
                    RNN_accuracy = np.array(pickle.load(open(file, 'rb' )))
                    if task == 'PTB':RNN_accuracy *= (1/math.log(2))
                    # if task=='copy': RNN_accuracy = np.array([RNN_accuracy[i] for i in 1000*np.arange(int(len(RNN_accuracy)/1000))])

                    # Processing
                    if len(RNN_accuracy)<100:
                        RNN_accuracy = np.pad(RNN_accuracy, (0,100-len(RNN_accuracy)), 'constant', constant_values=(np.nan))
                    # RNN_accuracy = remove_outliers(RNN_accuracy, 2)

                    accuracies.append(RNN_accuracy)
                    a_maxs.append(np.nanmax(RNN_accuracy[10:]))
                    a_mins.append(np.nanmin(RNN_accuracy[10:]))
                except Exception as e:
                    print('Error in get_accuracy : {}'.format(e))

        return np.nanmean(accuracies, axis=0), np.nanstd(accuracies, axis=0), np.nanmax(a_maxs), np.nanmin(a_mins)

# print(get_accuracy(1.0, 1.0, 'psMNIST'))

def normalised_accuracy(n, s, task, run='run_1', lr='0.0001', OP='adam', lp=True):
    DIR = get_dir(n, s, task, run, lr, OP,lp)
    
    if task == 'PTB': file = DIR+'/RNN_Val_Losses'
    elif task == 'copy': file = DIR+'/RNN_Train_Losses'
    else: file = DIR+'/RNN_Test_accuracy'
    try:
        RNN_Test_accuracy = np.array(pickle.load(open(file, 'rb' )))
        if task == 'PTB':
            RNN_Test_accuracy *= (1/math.log(2))

        RNN_Test_accuracy = remove_outliers(RNN_Test_accuracy)
        amax = np.nanmax(RNN_Test_accuracy[10:])
        amin = np.nanmin(RNN_Test_accuracy[10:])
        normalised_acc = (RNN_Test_accuracy-amin)/(amax-amin)
        # print(amax, amin)
        # if task=='copy':
            # return normalised_acc, amax, amin
        # else:
        return normalised_acc, amax, amin
    except Exception as e:
        print(e)
        return None


def sorted_x_train(digit):
    x_array = []
    for i in range(len(train_ds)):
        if train_ds[i][1].item() == digit:
            x_array.append(train_ds[i][0])
    return x_array

def sorted_y_train(digit):
    y_array = []
    for i in range(len(train_ds)):
        if train_ds[i][1].item() == digit:
            y_array.append(train_ds[i][1].item())
    return y_array

def pca_plot(dimension, init_state, num_steps, sample_steps, epoch=99, run='run_1', lr=1e-04, OP='adam'):
    learned_n, learned_s, V = load_net(n, s, task, epoch, run, lr)
    hidden_size = V.shape[0]
    
    # Forward pass
    h = torch.empty(num_steps, hidden_size)
    hidden = init_state

    for t in range(num_steps):
        hidden = Gamma.apply(torch.mv(V,hidden),learned_n, learned_s)
        h[t] = hidden
    
    
    pcau3d = decomposition.PCA(n_components= dimension)
    pcau3d.fit(h)

    pca_data = pcau3d.transform(h[-sample_steps:])
    pca_data = pca_data.reshape(sample_steps, dimension)

    if dimension == 2:
        plt.plot(pca_data[:,0],pca_data[:,1], ".", markersize=3);

    elif dimension == 3:
        ax.scatter3D(pca_data[:,0], pca_data[:,1], pca_data[:,2], s=0.4);
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        
def traj_plot(dimension, init_state, num_steps, sample_steps, epoch=99, run='run_1', lr=1e-04, OP='adam'):
    learned_n, learned_s, V = load_net(n, s, task, epoch, run, lr)
    hidden_size = V.shape[0]
    
    # Forward pass
    h = torch.empty(num_steps, hidden_size)
    hidden = init_state

    for t in range(num_steps):
        hidden = Gamma.apply(torch.mv(V,hidden), learned_n, learned_s)
        h[t] = hidden
    
    if dimension == 1:
        plt.plot(np.arange(sample_steps), h[-sample_steps:,1], '.', markersize=3)
    if dimension == 2:
        plt.plot(h[-sample_steps:,0], h[-sample_steps:,1], '.', markersize=3)
    else:
        ax.scatter3D(h[-sample_steps:,0], h[-sample_steps:,2],  h[-sample_steps:,2],s=0.4)

def task_accuracy(task, epoch=99, run='run_1', lr=1e-04, OP='adam', lp=True, axis=ax, figure=fig):
    nonlins = [1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
    saturations = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    titles = {
        'copy':'{} : Train loss'.format(task, epoch),
        'sMNIST':'{} : Valid accuracy'.format(task, epoch, lr),
        'psMNIST':'{} : Valid accuracy'.format(task, epoch, lr),
        'PTB':'{} : Valid bpc'.format(task, epoch, lr)
             }
    notes = ['(lower is better)', '(higher is better)']
    cmaps = ['viridis','viridis_r']
    linspaces = {
        'copy':np.linspace(0,0.3,16),
        'sMNIST':np.linspace(0,1,11),
        'psMNIST':np.linspace(0,1,11),
        'PTB':11
        }
    
    N, S = np.meshgrid(nonlins, saturations)
    Z = np.empty([len(nonlins), len(saturations),])
    for index in np.ndindex(9,5): 
        n = nonlins[index[0]]
        s = saturations[index[1]]
        try:
            # if get_accuracy(n, s, task, run='run_2', lr=lr, lp=lp) is not None:
            #     Z[index] = np.max([get_accuracy(n, s, task, run='run_1', lr=lr, lp=lp)[epoch], get_accuracy(n, s, task, run='run_2', lr=lr, lp=lp)[epoch]])
            # else:
            Z[index] = get_accuracy(n, s, task, run='run_1', lr=lr, lp=lp)[0][epoch]
        except Exception as e:
            print(e)
            None
     
    if task == 'PTB' or task == 'copy': index = 0
    else: index = 1
    
    CS = axis.contourf(N, S, Z.transpose(), levels=linspaces[task], cmap=cmaps[index])
    # CS2 = axis.contour(N, S, Z.transpose(), levels =[0.09], colors=('red'), linestyles=('--'))
    figure.colorbar(CS, ax=axis, shrink = 0.9)
    axis.set_title(titles[task]);
    print('Min : {} at n={}, s={}'.format(np.amin(Z), nonlins[np.argmin(Z)%9], 
                                      saturations[(np.argmin(Z) - np.argmin(Z)%9)%5]))
    return

# def getParams(n, s, task, epoch, run='run_1', lr=0.0001, returnparams=False):
#     try:
#         learned_n, learned_s, _ = load_net(n, s, task, epoch, run, lr)
#         if np.isnan(learned_n) or np.isnan(learned_s):
#             raise FileNotFoundError
#         else:
#             if returnparams:
#                 return learned_n, learned_s
#             else:
#                 return learned_n-n, learned_s-s
#     except: # to do : better exception handling
#         return None, None


def getParams2(n, s, task, epoch, run='run_1', lr=0.0001, OP='adam', returnparams=False):
    try:
        if run=='late_adapt': run='late_adapt/start_50'
        DIR = get_dir(n, s, task, run=run, lr=lr, OP='adam', lp=True)
        file = DIR + '/shapeparams_n_{}_s_{}.npy'.format(n,s,epoch)
        shapeparams = np.load(file)
        
        if returnparams:
            if task =='copy':
                return [shapeparams[i] for i in 1000*np.arange(100)]
            else:
                return shapeparams
        else:
            if task=='copy': 
                if epoch == 99: epoch = -1
                else: epoch*=1000

            if np.isnan(shapeparams[epoch]).any() == False:
                return shapeparams[epoch,0]-n, shapeparams[epoch,1]-s
            elif np.isnan(shapeparams[0]).any() == True:
                return None, None
            else:
                i=0
                while np.isnan(shapeparams[i]).any() == False and i < epoch:
                    i+=1
                return shapeparams[i-1,0]-n, shapeparams[i-1,1]-s
    except Exception as e:
        # print(e)
        if returnparams: return None
        else: return None, None

def get_params(n, s, task, run='run_1', lr=0.0001):
    if run=='no_grad':
        init = np.array([n,s]*100).reshape(100,2)
        params = getParams2(n, s, task, epoch=99, run=run, lr=lr, returnparams=True)

        return params-init
    else:
        params_array = []
        for i in [1,2,3]:
            init = np.array([n,s]*100).reshape(100,2)
            params = getParams2(n, s, task, epoch=99, run=f'run_{i}', lr=lr, returnparams=True)
            # print(params)
            if params is not None:
                params_array.append(params-init)
        params_array = np.array(params_array)
        means = np.mean(params_array, axis=0)
        # print(np.mean(params_array, axis=0).shape)
        # print(params_array[0,-1,:])
        return means


def task_streamplot(task, epoch=99, run='run_1', lr=1e-04, OP='adam', axis=ax, figure=fig):
    nonlins = [1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
    # if task_params[task]['diff_grid'] and run != 'no_grad':
    #     nonlins += [1.25, 3.75, 6.25, 8.75, 11.25, 13.75, 16.25, 18.75]
    #     nonlins = np.sort(nonlins)
    saturations = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    u = np.empty([len(nonlins), len(saturations)])
    v = np.empty([len(nonlins), len(saturations)])
    for i in range(len(nonlins)):
        for j in range(len(saturations)):
            n = nonlins[i]
            s = saturations[j]
            # if getParams2(n, s, task, epoch, run=2, lr=lr)[0] is not None:
            #     temp = np.mean((getParams2(n, s, task, epoch, run=1, lr=lr), getParams2(n, s, task, epoch, run=2, lr=lr)), axis=0)
            #     u[i,j] = temp[0]
            #     v[i,j] = temp[1]
            # else:
            u[i,j], v[i,j] = getParams2(n, s, task, epoch, run=run, lr=lr)
            # print(getParams2(n, s, task, epoch, run=run, lr=lr))

    N1, S1 = np.meshgrid(nonlins[:2], saturations)
    axis.streamplot(N1, S1, np.transpose(u[:2,:]), np.transpose(v[:2,:]),
                  density = 0.1, arrowsize=1.2, linewidth = 1, color='black') #0.3
    
    N2, S2 = np.meshgrid(nonlins[1:], saturations)
    axis.streamplot(N2, S2, np.transpose(u[1:,:]), np.transpose(v[1:,:]),
                  density = 0.5, arrowsize=1.2, linewidth = 1, color='black') #0.8
    axis.set_xbound(1-0.5,20+0.5)
    axis.set_ybound(-0.05,1+0.05)
    # axis.set_title(task)
    return
    
def task_streamplot2(task, epoch=99, lr=1e-04, OP='adam', axis=ax, figure=fig):
    nonlins = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
    saturations = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    N, S = np.meshgrid(nonlins, saturations)
    u = np.empty([len(nonlins), len(saturations)])
    v = np.empty([len(nonlins), len(saturations)])
    for i in range(len(nonlins)):
        for j in range(len(saturations)):
            n = nonlins[i]
            s = saturations[j]
            if getParams(n, s, task, epoch, run='run_2', lr=lr)[0] is not None:
                temp = np.mean((getParams(n, s, task, epoch, run='run_1', lr=lr), getParams(n, s, task, epoch, run='run_2', lr=lr)), axis=0)
                u[i,j] = temp[0]
                v[i,j] = temp[1]
            else:
                u[i,j], v[i,j] = getParams(n, s, task, epoch, run='run_1', lr=lr)
    
    axis.streamplot(N, S, np.transpose(u), np.transpose(v),
                  density = 0.8, arrowsize=1.3, linewidth = 1, color='black')
    axis.set_xbound(1-0.5,20+0.5)
    axis.set_ybound(-0.05,1+0.05)
    axis.set_title(task)
    return

def gamma_sup_contour(colormap='viridis', set_abs=False, contourf=False, ncontours=10, axis=ax, figure=fig):
    nonlinss = list(np.linspace(0.1,20,100))
    saturationss = list(np.linspace(-0.1,1,100))
    N, S = np.meshgrid(nonlinss, saturationss)
    Z = np.empty([len(saturationss), len(nonlinss)])
    for n in nonlinss:
        for s in saturationss:
            val = sup_gamma_prime(n,s)
            if set_abs:
                val = np.abs(val)
            Z[saturationss.index(s), nonlinss.index(n)] = val
    if contourf:
        CS = axis.contourf(N, S, Z, levels = 1+np.linspace(-3,3,ncontours), cmap = colormap)
    else:
        CS = axis.contour(N, S, Z, levels = [0.5, 1, 2, 3, 4], cmap = 'Reds')
        # CS2 = axis.contour(N, S, Z, levels = [1.0], colors=('white'))
        axis.clabel(CS, inline = 1, fontsize = 9)
        # axis.clabel(CS2, inline = 1, fontsize = 9)
    # axis.set_xlabel('Degree of nonlinearity $n$')
    # axis.set_ylabel('Degree of saturation $s$')

def gamma_sup_contour_task(task, epoch=99, run='run_1', lr=1e-04, OP='adam', colormap='viridis', set_abs=False, contourf=False, ncontours=10, lp=True, axis=ax, figure=fig):
    nonlins = [1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
    saturations = [0.0, 0.25, 0.5, 0.75, 1.0]
    N, S = np.meshgrid(nonlins, saturations)
    Z = np.empty([len(saturations), len(nonlins)])
    for n in nonlins:
        for s in saturations:
            if lp==True:
                learned_n, learned_s = getParams(n, s, task, epoch, run, lr, returnparams=True)
                try: val = sup_gamma_prime(learned_n, learned_s)
                except: val = None
            else:
                val = sup_gamma_prime(n,s)
            if set_abs:
                val = np.abs(val)
            Z[saturations.index(s), nonlins.index(n)] = val
    if contourf:
        if ncontours == 2: levels = [1-0.1, 1+0.1]
        else: levels = 1+np.linspace(-3,3,ncontours)
        CS = axis.contourf(N, S, Z, levels = levels, cmap = colormap)
    else:
        CS = axis.contour(N, S, Z, levels = 1+np.linspace(-3,3,ncontours), cmap = colormap)
        CS2 = axis.contour(N, S, Z, levels = [1.0], colors=('black'))
        axis.clabel(CS, inline = 1, fontsize = 12)
        axis.clabel(CS2, inline = 1, fontsize = 12)
    # axis.set_xlabel('Degree of nonlinearity $n$')
    # axis.set_ylabel('Degree of saturation $s$')
    return

def jacobians_contour(task, epoch=99, run='run_1', lr=1e-04, OP='adam', colormap='viridis'):
    nonlins = [1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
    saturations = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    N, S = np.meshgrid(nonlins, saturations)
    Z = np.empty([len(saturations), len(nonlins)])
    
    for n in nonlins:
        for s in saturations:
            try:
                jac = jacobian(n, s, task, epoch, run, lr, OP)
                value = np.linalg.norm(jac, 2)
            except:
                value = None
            Z[saturations.index(s), nonlins.index(n)] = value
    CS = ax.contourf(N, S, Z, 8, cmap = colormap)
    fig.colorbar(CS, ax=ax, shrink = 0.9)
    return

def param_animation(task, epoch=99, run='run_1', lr=1e-04, OP='adam', cmap='viridis_r', nonlin='gamma'):
    print('='*40+'\nGetting parameters, {}'.format(task))
    cmap = plt.cm.get_cmap(cmap)
    length = 100
    if run=='late_adapt': length=50

    nonlins = [1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
    if task_params[task]['diff_grid']:
        nonlins += [3.75, 6.25, 8.75, 11.25, 13.75, 16.25, 18.75]
    saturations = [0.0, 0.25, 0.5, 0.75, 1.0]

    if nonlin=='gamma':
        allparams = np.zeros([len(nonlins), len(saturations), length+1, 2])
        norms = np.zeros([len(nonlins), len(saturations)])
        colors = np.zeros([len(nonlins), len(saturations), length+1,4])
        maxs = []
        mins = []
        if task is not 'PTB': mins.append(0)

        for i in range(len(nonlins)):
            print('n:{}'.format(nonlins[i]))
            for j in range(len(saturations)):
                n, s = nonlins[i], saturations[j]
                allparams[i,j,0,:] = [n,s]
                colors[i,j,0,:] = [0,0,0,0.5]
                norms[i,j] = np.linalg.norm(np_gamma_prime(np.random.uniform(0,1,1000), n,s), np.inf)
                try:
                    allparams[i,j,1:,:] = [i+[n,s] for i in get_params(n,s, task=task, run=run, lr=lr)] #getParams2(nonlins[i], saturations[j], task, 99, run, lr, returnparams=True)
                    array, _, amax, amin = get_accuracy(nonlins[i], saturations[j], task, lr, run=run)
                    if task=='copy':
                        array = array[::1000]
                        amin, amax = np.nanmin(array[10:]), np.nanmax(array[10:])
                    array = (array-task_params[task]['pmin'])/(task_params[task]['pmax']-task_params[task]['pmin']) #1
                    # array = (array-amin)/(amax-amin) #2
                    
                    colors[i,j,1:,:] = np.array(list(cmap(array)))
                    maxs.append(amax)
                    mins.append(amin)
                except Exception as e:
                    print('Error in param_animation :',e)
                    allparams[i,j,1:,:] = np.zeros([length,2])*np.nan
                    colors[i,j,1:,:] = [[0.267004, 0.004874, 0.329415, 1.0] for i in np.arange(length)]

        # print('Maximum :',max(maxs))
        # print('Minimum :',min(mins))
        return allparams, colors, maxs, mins
    elif nonlin=='gamma2':
        hid_size = task_params[task]['hidden_size'] 
        allparams = np.zeros([len(nonlins), len(saturations), length+1, 2, hid_size])
        colors = np.zeros([len(nonlins), len(saturations), length+1, 4])
        # maxs = []
        # mins = []
        # if task is not 'PTB': mins.append(0)

        for i in range(len(nonlins)):
            print('n:{}'.format(nonlins[i]))
            for j in range(len(saturations)):
                n, s = nonlins[i], saturations[j]
                allparams[i,j,0,:] = [n*np.ones(hid_size), s*np.ones(hid_size)]
                colors[i,j,0,:] = [0,0,0,0.5]
                

                if task=='PTB': savedir = '/Volumes/GEADAH/3_Research/data/task_data/PTB/T_150/gamma2/'
                elif task=='psMNIST': savedir = '/Volumes/GEADAH/3_Research/data/task_data/psMNIST/gamma2/'

                directory = f'LP_True_HS_{hid_size}_NL_gamma2_n_{n}_s_{s}_lr_0.0001_OP_adam/'
                if task=='PTB':
                    file_params = savedir+directory+f'shapeparams_n_{n}_s_{s}.npy'
                    file_accuracy = savedir+directory+'RNN_Val_Losses'
                elif task=='psMNIST':
                    file_params = savedir+directory+'shapeparams.npy'
                    file_accuracy = savedir+directory+'RNN_Test_accuracy'

                # get files
                if os.path.exists(file_params):
                    params = np.load(file_params)
                    allparams[i,j,1:params.shape[0]+1,:,:] = params

                if os.path.exists(file_accuracy):
                    accuracies = np.array(pickle.load(open(file_accuracy, 'rb' )))/np.log(2)
                    accuracies = (accuracies-np.amin(accuracies))/(np.amax(accuracies)-np.amin(accuracies))
                    colors[i,j,1:accuracies.shape[0]+1,:] = np.array(list(cmap(accuracies)))
                # norms[i,j] = np.linalg.norm(np_gamma_prime(np.random.uniform(0,1,1000), nonlins[i], saturations[j]), np.inf)
                # try:
                #     array, amax, amin = get_accuracy(nonlins[i], saturations[j], task, lr, run=run)
                #     # normalize :
                #     # array = (array-task_params[task]['pmin'])/(task_params[task]['pmax']-task_params[task]['pmin']) #1
                #     array = (array-amin)/(amax-amin) #2
                    
                #     colors[i,j,1:,:] = np.array(list(cmap(array)))
                #     maxs.append(amax)
                #     mins.append(amin)
                # except Exception as e:
                #     print('Error in param_animation :',e)
                #     colors[i,j,1:,:] = [[0.267004, 0.004874, 0.329415, 1.0] for i in np.arange(length)]

        return allparams, colors, None, None


    else:
        print('no such nonlin')
        return None, None, None, None

def param_animation2(task, epoch=99, run='run_1', lr=1e-04, OP='adam', cmap='viridis_r'):
    print('Getting parameters, {}'.format(task))
    cmap = plt.cm.get_cmap(cmap)

    nonlins = [1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
    if task_params[task]['diff_grid']:
        nonlins += [1.25, 3.75, 6.25, 8.75, 11.25, 13.75, 16.25, 18.75]
    saturations = [0.0, 0.25, 0.5, 0.75, 1.0]

    allparams = np.zeros([len(nonlins), len(saturations), 101, 3])
    # norms = np.zeros([len(nonlins), len(saturations)])
    colors = np.zeros([len(nonlins), len(saturations), 101,4])
    # maxs = []
    # mins = []
    # if task is not 'PTB': mins.append(0)

    for i in range(len(nonlins)):
        print('n:{}'.format(nonlins[i]))
        for j in range(len(saturations)):
            allparams[i,j,0,:] = [nonlins[i], saturations[j], task_params[task]['pmin']]
            allparams[i,j,1:,:-1] = getParams2(nonlins[i], saturations[j], task, 99, run, lr, returnparams=True)
            
            accs, _, _, _ = get_accuracy(nonlins[i], saturations[j], task, run, lr)
            if len(accs)<100: accs = np.pad(accs, (0,100-len(accs)), 'constant', constant_values=(accs[-1]))# np.nan) #(array[-1])
            allparams[i,j,1:,-1] = np.array(accs)

            # norms[i,j] = np.linalg.norm(np_gamma_prime(np.random.uniform(0,1,1000), nonlins[i], saturations[j]), np.inf)
            # # for e in np.arange(1,101):
            # try:
            #     array, amax, amin = get_accuracy(nonlins[i], saturations[j], task, run, lr)
            #     if len(array)<100: array = np.pad(array, (0,100-len(array)), 'constant', constant_values=(array[-1]))# np.nan) #(array[-1])
            #     #normalize :
            #     array = (array-task_params[task]['pmin'])/(task_params[task]['pmax']-task_params[task]['pmin'])
            #     colors[i,j,1:,:] = np.array(list(cmap(array)))
            #     maxs.append(amax)
            #     mins.append(amin)
            # except Exception as e:
            #     print(e)
            #     colors[i,j,1:,:] = [[0.267004, 0.004874, 0.329415, 1.0] for i in np.arange(100)]

    return allparams#, colors, maxs, mins


from exp_numpy import expm

def W_hennaff(N):
    W_rec = torch.as_tensor(henaff_init(N))
    A = W_rec.triu(diagonal=1)
    A = A - A.t()
    W_rec = expm(A)
    return W_rec

def create_diag(s, n):
    diag = np.zeros(n-1)
    diag[::2] = s
    A_init = np.diag(diag, k=1)
    A_init = A_init - A_init.T
    return A_init.astype(np.float32)

def henaff_init(n):
    # Initialization of skew-symmetric matrix
    s = np.random.uniform(-np.pi, 0., size=int(np.floor(n / 2.)))
    return create_diag(s, n)

# for i in [1,2,3]:
#     print(getParams2(1.0, 0.0, 'psMNIST', run=f'run_{i}', epoch=-1))
# print(get_params(1.0, 0.0, 'psMNIST')[-1])


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
         'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
