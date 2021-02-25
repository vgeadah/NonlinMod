import argparse
from .Networks import RNN, AdaptationSupervisor
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def select_network(net_type, inp_size, hid_size, nonlin, n, s, rinit, iinit, cuda, lparams):
    if net_type == 'RNN':
        rnn = RNN(inp_size, hid_size, nonlin, n, s, bias=True, cuda=cuda, r_initializer=rinit, i_initializer=iinit, learn_params=lparams)
    if net_type == 'Adapt':
        rnn = AdaptationSupervisor(inp_size, hid_size, 'homogeneous')
    return rnn

def write_log(SAVEDIR, string):
    log_file = os.path.join(SAVEDIR, 'log_file.txt')
    
    if not os.path.exists(SAVEDIR): 
        os.makedirs(SAVEDIR)

    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('Logs:\n') 

    with open(log_file, 'a') as logfile:
        logfile.write('\n'+string)

