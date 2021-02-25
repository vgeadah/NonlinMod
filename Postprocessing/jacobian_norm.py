#!/usr/local/misc/Python/Python-3.6.7/bin/python3
import numpy as np 
import torch
import torch.nn as nn
import argparse
import time
import os
import datetime
import pickle

import sys
sys.path.append('/Users/Victor_Geadah_Mac/3_Research/Adaptation/gamma')
#from NetworkCreation import definitions as defs

from NetworkCreation.Networks import RNN
from NetworkCreation.gamma_function import torch_dgamma

# In Numpy
def npgam1(x,n): return (1 / n)*(np.log(1 + np.exp(n*x)))
def npgam2(x,n): return (np.exp(n*x))/(1 + np.exp(n*x))
def np_gamma(x, n, s): return (1-s)*npgam1(x,n) + s*npgam2(x,n)
def np_gamma_prime(x, n, s): return (1-s)*npgam2(x,n) + s*n*npgam2(x,n)*(1-npgam2(x,n))


parser_JN = argparse.ArgumentParser(description='Jacobian norm (JN) simulation')
parser_JN.add_argument('-g', '--cuda', action='store_true', default=False,
                    help='Use CUDA')
parser_JN.add_argument('--seed', type=int, default=400, 
                    help='random seed for reproducibility')
parser_JN.add_argument('--batch', type=int, default=100,
                    help='batch size')
#parser_JN.add_argument('--task', type=str, default='psMNIST', choices=['copy', 'psMNIST', 'PTB'],
#                    help='task for JN simulation')
args = parser_JN.parse_args()
device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
#   Load data
###############################################################################

rng = np.random.RandomState(args.seed)
task = 'PTB'# args.task

if task=='psMNIST':
    from sMNIST_task_baseline import Model, testloader
    dataset = testloader
    order = rng.permutation(784)
    input_size, hidden_size = 1, 400

elif task=='PTB':
    from PTB_task_baseline import Model, corpus, test_data, get_batch 
    ntokens = len(corpus.dictionary)
    dataset = test_data
    input_size, hidden_size = 200, 600

else:
    print('Not implemented yet')

###############################################################################
#   Saving
###############################################################################

SAVEDIR = os.path.join('./Postprocessing/calculations',
                        'JN',
                        task,
                        str(args.seed),
                        str(datetime.date.today()))

if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)

with open(os.path.join(SAVEDIR, 'details.txt'), 'w') as fp:
    '''Details and hyperparameters, if desired'''
    fp.write(f'Jacobian norm calculations\n{task} task on {datetime.datetime.now()}\n')
    fp.write(f'\nSaving to: {SAVEDIR}')
    fp.write('\n\nHyperparameters:\n')
    for key, val in args.__dict__.items():
        fp.write(('\t{}: {} \n'.format(key, val)))

with open(os.path.join(SAVEDIR, 'log_file.txt'), 'w') as logfile:
    logfile.write('Logs:\n') 

def write_log(string):
    with open(os.path.join(SAVEDIR, 'log_file.txt'), 'a') as logfile:
        logfile.write('\n'+string)

######


class LocalModel(Model):
    def __init__(self, hidden_size, rnn):
        if task=='psMNIST':
            Model.__init__(self, hidden_size, rnn)

        elif task=='PTB':
            Model.__init__(self, rnn, ntoken=ntokens, ninp=200, nhid=hidden_size, tie_weights=False)

        if args.cuda:
            self.cuda()

    def mean_jacobian(self, inputs):
        h = None
        jacobian_norms = []

        V_rec = self.rnn.V.weight.data.detach()
        n, s  = self.rnn.n.item(), self.rnn.s.item()

        if task == 'psMNIST':
            inputs = inputs[:, order]
            for i, input in enumerate(torch.unbind(inputs, dim=1)):
                x = input.unsqueeze(1)
                h, a = self.rnn(x, h)
                js = np.linalg.norm((torch_dgamma(a, n, s) @ V_rec).cpu().numpy(), 
                    ord=np.inf, axis=1)
                jacobian_norms.append(js)

        elif task == 'PTB':
            emb = self.encoder(inputs)
            for i in range(emb.shape[0]):
                h, a = self.rnn(emb[i], h)
                js = np.linalg.norm((torch_dgamma(a, n, s) @ V_rec).cpu().numpy(), 
                    ord=np.inf, axis=1)
                jacobian_norms.append(js)



        jacobian_norms = np.array(jacobian_norms)
        return np.mean(jacobian_norms, axis=0)


def forward(self, input, hidden):

        emb = self.encoder(input)
        hs = []
        for i in range(emb.shape[0]):
            hidden, _ = self.rnn(emb[i], hidden, 1.0, i==0)
        output = torch.stack(hs)

        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, (e_s, alpha_s)

###############################################################################
#    Simulation
###############################################################################


nonlins = np.linspace(1,20,20)
saturations = np.linspace(0,1,10)

all_jacobians = []

temp_dict = {'nonlins':nonlins, 'saturations':saturations}
with open(os.path.join(SAVEDIR, 'grid'), 'wb') as f:
    pickle.dump(temp_dict, f)

for n in nonlins:
    for s in saturations:
        print('*'*10+'\nSetting up...')
        s_0 = time.time()

        rnn = RNN(input_size, hidden_size, bias=True, cuda=args.cuda, n_init=n, s_init=s,
                            r_initializer="henaff", i_initializer="kaiming", learn_params=False)
        net = LocalModel(400, rnn)

        write_log('Model built: ({:1.3f}, {:1.3f})'.format(n, s))
        
        if args.cuda:
            net = net.cuda()
            net.rnn = net.rnn.cuda()

        net.eval()
        jacobians = []
        with torch.no_grad():
            write_log('Set up done, time: {:2.5f}. Starting simulation:'.format(time.time()-s_0))

            if task=='psMNIST':
                for i, data in enumerate(dataset):

                    x, _ = data
                    x = x.view(-1, 784)
                    x = x.to(device)
                    if args.cuda:
                        x.cuda()

                    s_t = time.time()
                    batch_jacobians = net.mean_jacobian(x)
                    jacobians.append(np.mean(batch_jacobians))
                    if i%25==0:
                        write_log('\tSample {:2d}, Mean JN: {:3.5f}, time: {:2.5f}'.format(i, np.mean(batch_jacobians), time.time() - s_t))
            
            elif task=='PTB':
                for i in range(0, dataset.size(0) - 1, 150):
                    x, _ = get_batch(dataset, i, evaluation=True)
                    if args.cuda:
                        x.cuda() 
                    s_t = time.time()
                    batch_jacobians = net.mean_jacobian(x)
                    jacobians.append(np.mean(batch_jacobians))
                    if i%6000==0:
                        write_log('\tSample {:2d}, Mean JN: {:3.5f}, time: {:2.5f}'.format(i, np.mean(batch_jacobians), time.time() - s_t))
        
        all_jacobians.append(jacobians)
        np.save(os.path.join(SAVEDIR, 'JN_n_{:1.3f}_s_{:1.3f}.npy'.format(n,s)), jacobians)

temp_dict = {'nonlins':nonlins, 'saturations':saturations, 'LEs':all_jacobians}
with open(os.path.join(SAVEDIR, 'JN_final'), 'wb') as f:
    pickle.dump(temp_dict, f)

write_log('Finished.')
