import numpy as np
import torch
import time
import os
import datetime
import argparse
import pickle

#all_hidden_np=np.load('all_hidden_np.npy')
#test_data_np=np.concatenate(np.load('test_data_np.npy'))

# time_start = time.process_time()
# mi=mixed.Mixed_KSG(test_data_np[0:n], all_hidden_np[0:n])
# time_stop=time.process_time()

# # n=1000   ~ 1 sec       0.7681337719999997
# # n=10000  ~ 30 sec     34.513170852
# # n=50000  ~ 500 sec   467.136643217
# # n=100000 ~ 2000 sec 2154.880079986 (36min)

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute())) # append 'gamma' dir to sys.path

import mixed

from NetworkCreation.Networks import RNN
from NetworkCreation.gamma_function import torch_dgamma
from NetworkCreation.utils import write_log

#seed = 400
#task = 'psMNIST'
#CUDA = True

parser_MI = argparse.ArgumentParser(description='Mutual Information (JN) simulation')
parser_MI.add_argument('-g', '--cuda', action='store_true', default=False,
                    help='Use CUDA')
parser_MI.add_argument('--seed', type=int, default=400, 
                    help='random seed for reproducibility')
parser_MI.add_argument('--batch', type=int, default=100,
                    help='batch size')
parser_MI.add_argument('--note', type=str, default='',
                    help='Any details to be entered manually upon launch')
#parser_JN.add_argument('--task', type=str, default='psMNIST', choices=['copy', 'psMNIST', 'PTB'],
#                    help='task for JN simulation')
args = parser_MI.parse_args()
device = torch.device("cuda" if args.cuda else "cpu")

seed = 400
task = 'psMNIST'
CUDA = args.cuda

###############################################################################
#   Load data
###############################################################################

rng = np.random.RandomState(seed)

if task=='psMNIST':
    from sMNIST_task_baseline import Model, testloader, test_model
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

device = torch.device("cuda" if CUDA else "cpu")

###############################################################################
#   Saving
###############################################################################

BASEDIR = os.path.join('./Postprocessing/calculations',
                        'MI',
                        task,
                        str(seed),
                        str(datetime.date.today()))

# if not os.path.exists(BASEDIR): 
#     os.makedirs(BASEDIR)

def write_details(DIR, string=None):
    '''
    Write details and hyperparameters (optional)
    '''
    with open(os.path.join(DIR, 'details.txt'), 'w') as fp:
        fp.write(f'Mutual information calculations\n{task} task on {datetime.datetime.now()}\n')
        fp.write(f'\nSaving to: {DIR}')
        fp.write('\n\nHyperparameters:\n')
        if string is not None:
            fp.write(f'\nNote: {string}')
        try:
            for key, val in args.__dict__.items():
                fp.write(('\t{}: {} \n'.format(key, val)))
        except NameError:
            print('No args parser')
    return 

# with open(os.path.join(SAVEDIR, 'log_file.txt'), 'w') as logfile:
#     logfile.write('Logs:\n') 

###################################################################

class LocalModel(Model):
    def __init__(self, hidden_size, rnn):
        if task=='psMNIST':
            Model.__init__(self, hidden_size, rnn)
        elif task=='PTB':
            Model.__init__(self, rnn, ntoken=ntokens, ninp=200, nhid=hidden_size, tie_weights=False)

        if CUDA:
            self.cuda()

    def forward(self, inputs):
        h = None
        hiddens = []

        # V_rec = self.rnn.V.weight.data.detach()
        # n, s  = self.rnn.n.item(), self.rnn.s.item()

        if task == 'psMNIST':
            inputs = inputs[:, order]
            for i, input in enumerate(torch.unbind(inputs, dim=1)):
                x = input.unsqueeze(1)
                h, _ = self.rnn(x, h)
                hiddens.append(h.detach().cpu().numpy())

        # elif task == 'PTB':
        #     emb = self.encoder(inputs)
        #     for i in range(emb.shape[0]):
        #         h, a = self.rnn(emb[i], h)
        #         js = np.linalg.norm((torch_dgamma(a, n, s) @ V_rec).cpu().numpy(), 
        #             ord=np.inf, axis=1)
        #         jacobian_norms.append(js)

        # jacobian_norms = np.array(jacobian_norms)
        return np.array(hiddens)

    def evaluate(self, testloader):
        h, a = None, None

        inputs = inputs[:, order]
        for i, input in enumerate(torch.unbind(inputs, dim=1)):
            x = input.unsqueeze(1)
            h, _ = self.rnn(x, h)
            
            h.retain_grad()

        out = self.lin(h)

        loss = self.loss_func(out, y)
        preds = torch.argmax(out, dim=1)
        correct = torch.eq(preds, y).sum().item()
        return loss, correct #, shape_signals

###################################################################
        
# def get_mi(n, s, seed, N, d, ni):
#     inputs, hidden = None, None
#     inputs = np.load('/home/stefan/code/data/gamma_mi/psMNIST/MNISTpixels.npy')
#     hidden = np.load('/home/stefan/code/data/gamma_mi/psMNIST/psMNIST_hidden_n{}_s{}_seed{}.npy'.format(n,s,seed))
     
#     mi_inputs = np.zeros((N*100,ni))
#     for i in range(N):
#         for j in range(100):
#             #for k in range(ni):
#             mi_inputs[i*100+j,:] = inputs[i,(d+1)*5-ni:(d+1)*5,j]

#     mi_hidden = np.zeros((3,N*100,400))
#     for i in range(N):
#         for j in range(3):
#             for k in range(100):
#                     mi_hidden[j,i*100+k,:] = hidden[i,j,k,:]
        
#     return mixed.Mixed_KSG(mi_inputs, mi_hidden[d])

nonlins = [1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]#np.linspace(1,20,20)
saturations =[0.0, 0.25, 0.5, 0.75, 1.0] #np.linspace(0,1,10)
epochs = [0,25,50,75,99]

recovered_epoch = 99

def write_log(SAVEDIR, string):
   print(string)
   return 

all_MIs = []

for n in nonlins:
    for s in saturations:
        mutual_informations = []

        SAVEDIR = os.path.join(BASEDIR, 'n_{:1.2f}_s_{:1.2f}'.format(n,s))

        if not os.path.exists(SAVEDIR): 
            os.makedirs(SAVEDIR)

        with open(os.path.join(SAVEDIR, 'log_file.txt'), 'w') as logfile:
            logfile.write('Logfile for [n,s]=[{:1.2f}, {:1.2f}]:\n'.format(n,s)) 
        write_details(SAVEDIR)

        # Build model
        rnn = RNN(input_size, hidden_size, bias=True, cuda=CUDA, n_init=n, s_init=s,
                            r_initializer="henaff", i_initializer="kaiming", learn_params=False)
        net = LocalModel(400, rnn)

        start = time.time()
        
        for e in epochs:
            s_0 = time.time()
            
            try:
                # Recover pretrained model
                MODELDIR = f'/Volumes/GEADAH/3_Research/data/Adaptation/task_data/psMNIST/LP_False_HS_400_NL_gamma_lr_0.0001_OP_adam/run_1/'
                MODELDIR += f'n_{n}_s_{s}/'
                last_model = torch.load(os.path.join(MODELDIR, f'RNN_{e}.pth.tar'), map_location=torch.device(device))
                net.rnn.n, net.rnn.s = last_model['state_dict']['rnn.n'], last_model['state_dict']['rnn.s']
                del last_model['state_dict']['rnn.n']; del last_model['state_dict']['rnn.s']
                net.load_state_dict(last_model['state_dict'])
                
                if CUDA:
                    net = net.cuda()
                    net.rnn = net.rnn.cuda()

                net.eval()
                hiddens = []
                with torch.no_grad():
                    write_log(SAVEDIR,'\n'+'='*40+f'\nEpoch {e}:')

                    if task=='psMNIST':
                        for i, data in enumerate(dataset):

                            x, _ = data
                            x = x.view(-1, 784)
                            x = x.to(device)
                            if CUDA:
                                x.cuda()

                            loss, c = net.forward(x, y, order, transform=transform)

                            
                            s_t = time.time()
                            hiddens = net(x)
                            mutual_information = mixed.Mixed_KSG(x[0].cpu().numpy(), hiddens[:,0,:]) #np.mean([mixed.Mixed_KSG(x[i,-1].cpu().numpy(), hiddens[-1,i,:]) for i in range(100)])
                            mutual_informations.append(mutual_information)
                            if i%25==0:
                                write_log(SAVEDIR,'\tSample: {:2d}, MI: {:3.5f}, time: {:2.5f}'.format(i, mutual_information, time.time() - s_t))

                all_MIs.append(mutual_informations)
                np.save(os.path.join(SAVEDIR, f'MI_e_{e}.npy'), mutual_informations)
                write_log(SAVEDIR, 'Time for epoch: {}'.format(time.time()-s_0))
            except Exception as exception:
                print(f'Exception {exception} for {n}, {s} at epoch {e}')

        temp_dict = {'nonlins':nonlins, 'saturations':saturations, 'MIs':all_MIs}
        with open(os.path.join(SAVEDIR, 'MI_final'), 'wb') as f:
            pickle.dump(temp_dict, f)

        write_log(SAVEDIR, 'Finished in {}s'.format(time.time()-start))

        # mi = get_mi(n, s, seed, N, d, ni)
        # for ni in num_inputs:
        #     for d in delays:
        #         mi = None
        #         mi=get_mi(n,s,seed,N,d,ni)
        #         mutual_informations.append(np.array([n,s,seed,N,d,ni,mi]))
        #         all_mis.append(np.array([n,s,seed,N,d,ni,mi]))
                
        #         np.save('/home/stefan/code/data/gamma_mi/psMNIST/psMNIST_mi_N{}_seed{}.npy'.format(N,seed), mutual_informations)
        #         np.save('/home/stefan/code/data/gamma_mi/psMNIST/psMNIST_all_mis_N{}.npy'.format(N), all_mis)
                
        #         print(n,s,seed,N,d,ni,mi)
