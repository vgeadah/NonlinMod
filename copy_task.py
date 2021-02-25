#!/usr/local/misc/Python/Python-3.6.7/bin/python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse
from tensorboardX import SummaryWriter
import time
import glob
import os
import sys

from common import henaff_init, cayley_init, random_orthogonal_init
from utils import str2bool, select_network

parser = argparse.ArgumentParser(description='auglang parameters')

parser.add_argument('--net-type', type=str, default='RNN', choices=['RNN', 'MemRNN'], help='options: RNN, MemRNN')
parser.add_argument('--nhid', type=int, default=128, help='hidden size of recurrent net')
parser.add_argument('--cuda', type=str2bool, default=True, help='use cuda')
parser.add_argument('--T', type=int, default=200, help='delay between sequence lengths')
parser.add_argument('--random-seed', type=int, default=400, help='random seed')
parser.add_argument('--labels', type=int, default=8, help='number of labels in the output and input')
parser.add_argument('--c-length', type=int, default=10, help='sequence length')
parser.add_argument('--vari', type=str2bool, default=False, help='variable length')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--rinit', type=str, default="henaff", help='recurrent weight matrix initialization')
parser.add_argument('--iinit', type=str, default="xavier", help='input weight matrix initialization')
parser.add_argument('--batch', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--onehot', type=str2bool, default=False)
parser.add_argument('--alpha', type=float, default=0.99)
parser.add_argument('--adam', action='store_true', default=True, help='Use adam')
parser.add_argument('--name', type=str, default='default', help='save name')
parser.add_argument('--log', action='store_true', default=False, help='Use tensorboardX')
parser.add_argument('--load', action='store_true', default=False, help='load, dont train')

parser.add_argument('--save-freq', type=int, default=25, help='frequency to save data')
parser.add_argument('--nonlin', type=str, default='gamma', help='non linearity none, modrelu, relu, tanh, sigmoid')
parser.add_argument('--n', type=float, default=1.0, help='degree of nonlinearity')
parser.add_argument('--s', type=float, default=1.0, help='degree of saturation')
parser.add_argument('--learn_params', type=str2bool, default=True, help='learn the shape parameters')

args = parser.parse_args()

def generate_copying_sequence(T, labels, c_length):
    items = [1, 2, 3, 4, 5, 6, 7, 8, 0, 9]
    x = []
    y = []

    ind = np.random.randint(labels, size=c_length)
    for i in range(c_length):
        x.append([items[ind[i]]])
    for i in range(T - 1):
        x.append([items[8]])
    x.append([items[9]])
    for i in range(c_length):
        x.append([items[8]])

    for i in range(T + c_length):
        y.append([items[8]])
    for i in range(c_length):
        y.append([items[ind[i]]])

    x = np.array(x)
    y = np.array(y)

    return torch.FloatTensor([x]), torch.LongTensor([y])


def create_dataset(size, T, c_length=10):
    d_x = []
    d_y = []
    for i in range(size):
        sq_x, sq_y = generate_copying_sequence(T, 8, c_length)
        sq_x, sq_y = sq_x[0], sq_y[0]
        d_x.append(sq_x)
        d_y.append(sq_y)  #

    d_x = torch.stack(d_x)
    d_y = torch.stack(d_y)
    return d_x, d_y


def onehot(inp):
    # print(inp.shape)
    onehot_x = inp.new_zeros(inp.shape[0], args.labels + 2)
    return onehot_x.scatter_(1, inp.long(), 1)


class Model(nn.Module):
    def __init__(self, hidden_size, rec_net):
        super(Model, self).__init__()
        self.rnn = rec_net

        self.lin = nn.Linear(hidden_size, args.labels + 1)
        self.hidden_size = hidden_size
        self.loss_func = nn.CrossEntropyLoss()

        nn.init.xavier_normal_(self.lin.weight)

    def forward(self, x, y):

        va = []
        hidden = None
        hiddens = []
        loss = 0
        accuracy = 0
        attn = 1.0
        for i in range(len(x)):
            #if i >= 110:
            #    attn = 0.0
            if args.onehot:
                inp = onehot(x[i])
                hidden, vals = self.rnn.forward(inp, hidden, attn)
            else:
                hidden, vals = self.rnn.forward(x[i], hidden, attn)
            va.append(vals)
            hidden.retain_grad()
            hiddens.append(hidden)
            out = self.lin(hidden)
            loss += self.loss_func(out, y[i].squeeze(1))

            if i >= T + args.c_length:
                preds = torch.argmax(out, dim=1)
                actual = y[i].squeeze(1)
                correct = preds == actual

                accuracy += correct.sum().item()

        accuracy /= (args.c_length * x.shape[1])
        loss /= (x.shape[0])
        return loss, accuracy, hiddens, va


def train_model(net, optimizer, batch_size, T, n_steps):
    with open(os.path.join(SAVEDIR, 'log_run.txt'), 'w') as fp:
        fp.write('Logs of runs : \n')

    save_norms = []
    accs = []
    losses = []
    shape_params = []
    lc = 0

    for i in range(n_steps):

        s_t = time.time()
        if args.vari:
            T = np.random.randint(1, args.T)
        x, y = create_dataset(batch_size, T, args.c_length)

        if CUDA:
            x = x.cuda()
            y = y.cuda()
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)

        optimizer.zero_grad()
        loss, accuracy, hidden_states, _ = net.forward(x, y)

        loss_act = loss
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 'inf')
        save_norms.append(norm)

        losses.append(loss_act.item())

        optimizer.step()
        accs.append(accuracy)
        shape_params.append([net.rnn.n.item(), net.rnn.s.item()])
        
        if args.log and len(accs) == 40:
            v1 = sum(accs) / len(accs)
            v2 = sum(losses) / len(losses)
            writer.add_scalar('Loss', v2, lc)
            writer.add_scalar('Accuracy', v1, lc)
            lc += 1
            accs, losses = [], []
            #writer.add_scalar('Grad Norms', norm, i)
        
        if i % 1000 == 0 or i == (n_steps-1):
            with open(os.path.join(SAVEDIR, 'log_run.txt'), 'a') as fp:
                fp.write('Update {}, Time : {}, n : {}, s : {}, Av. Loss: {}, Accuracy: {}\n'.format(
                    i, time.time() - s_t, round(net.rnn.n.item(),4), round(net.rnn.s.item(),4), loss_act.item(), accuracy))
    
        if i % (1000 * SAVEFREQ) == 0 or i == (n_steps-1):
            with open(SAVEDIR + '{}_Train_Losses'.format(NET_TYPE), 'wb') as fp:
                pickle.dump(losses, fp)

            with open(SAVEDIR + '{}_Train_Accuracy'.format(NET_TYPE), 'wb') as fp:
                pickle.dump(accs, fp)

            with open(SAVEDIR + '{}_Grad_Norms'.format(NET_TYPE), 'wb') as fp:
                pickle.dump(save_norms, fp)

            save_checkpoint({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': i
            },
                '{}_{}.pth.tar'.format(NET_TYPE,int(i/1000))
            )

    #torch.save(net.state_dict(), './copylogs/' + args.name + '.pt')

    np.save(os.path.join(SAVEDIR, 'shapeparams_n_{}_s_{}'.format(args.n, args.s)), shape_params)

    return


def load_model(net, optimizer, fname):
    if fname == 'l':
        print(SAVEDIR)
        list_of_files = glob.glob(SAVEDIR + '*')
        print(list_of_files)
        latest_file = max(list_of_files, key=os.path.getctime)
        print('Loading {}'.format(latest_file))

        check = torch.load(latest_file)
        net.load_state_dict(check['state_dict'])
        optimizer.load_state_dict(check['optimizer'])

    else:
        check = torch.load(fname)
        net.load_state_dict(check['state_dict'])
        optimizer.load_state_dict(check['optimizer'])
    epoch = check['epoch']
    return net, optimizer, epoch

def load_function():
    
    net.load_state_dict(torch.load('copylogs/' + args.name + '.pt'))
    x, y = create_dataset(1, T, args.c_length)
    if CUDA:
        x = x.cuda()
        y = y.cuda()
    x = x.transpose(0, 1)
    y = y.transpose(0, 1)

    a1 = []
    for i in range(11):
        a1.append([])
    _, acc, _, vals = net.forward(x, y)
    '''
    deltas = []
    for i in range(1, 120):
        diff = net.rnn.memory[i] - net.rnn.memory[i-1]
        val = torch.sum(diff ** 2).item()
        deltas.append(val)
    plt.plot(range(1, 120), deltas)
    #plt.scatter(np.array(av)[1:], np.zeros(10), c='orange')
    plt.title('Change in hidden state Copy task')
    plt.xlabel('t')
    plt.ylabel('delta h')
    plt.savefig('copylogs/delta_h.png')
    sys.exit(0)
    '''
    ctr = 1
    for (a, b) in vals:
        if a is None:
            continue
        print(ctr, torch.argmax(a.squeeze(1)).item())
        ctr += 1
        for i in range(min(a.size(0), 11)):
            a1[i].append(b[i][0].item())
        #a2.append(b[9][0].item())

    clrs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'brown', 'pink', 'orange', 'grey']
    legs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    for i in range(11):
        plt.plot(range(i+1, 120), a1[i], clrs[i], label=legs[i])
    plt.legend()
    #plt.plot(a1)
    #plt.plot(a2)
    plt.savefig('fig.png')


def save_checkpoint(state, fname):
    filename = SAVEDIR + fname
    torch.save(state, filename)


nonlins = ['relu', 'tanh', 'sigmoid', 'modrelu']
# nonlin = args.nonlin.lower()

random_seed = args.random_seed
NET_TYPE = args.net_type
CUDA = args.cuda
decay = args.weight_decay
hidden_size = args.nhid

torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)


SAVEFREQ = args.save_freq
inp_size = 1
T = args.T = 200
batch_size = args.batch
out_size = args.labels + 1
if args.onehot:
    inp_size = args.labels + 2
args.learn_params = True
args.lr = 1e-03

nonlins = [1.0, 2.5, 5.0, 7.5]#, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
saturations = [0.0, 0.25, 0.5, 0.75, 1.0]# 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Log ---------------
host_name = 'fox'
date = '09-04-20_2'
LOGFILE = os.path.join('logs','copy_{}_{}'.format(host_name, date))

with open(LOGFILE, 'w') as fp:
    fp.write('Copy log : \n' + '='*40 + '\n')
    fp.write('Nonlins : {}, saturations : {}\n'.format(nonlins, saturations, args.lr))
    fp.write('Args :\n')
    for key, val in args.__dict__.items():
        fp.write(('{}: {} | '.format(key, val)))
    fp.write('\n'+'='*40+'\n')
# -------------------

for n in nonlins:
    for s in saturations:
        nonlin, args.n, args.s = 'gamma', n, s

        with open(LOGFILE, 'a') as fp:
            fp.write("{}, n:{} s:{} process started.\n".format(nonlin, args.n, args.s))

        if args.adam:
            udir = 'LP_{}_HS_{}_NL_{}_n_{}_s_{}_lr_{}_OP_adam'.format(args.learn_params, hidden_size, nonlin, args.n, args.s, args.lr)
        else:
            udir = 'LP_{}_HS_{}_NL_{}_n_{}_s_{}_lr_{}_OP_rmsprop_alpha_{}'.format(args.learn_params, hidden_size, nonlin, args.n, args.s, args.lr, args.alpha)
        # udir = 'HS_{}_NL_{}_lr_{}_BS_{}_rinit_{}_iinit_{}_decay_{}_alpha_{}'.format(hidden_size, nonlin, args.lr, args.batch,
        #                                                                             args.rinit, args.iinit, decay, args.alpha)
        if args.onehot:
            udir = 'onehot/' + udir

        if not args.vari:
            n_steps = 100000
            SAVEDIR = './saves/copytask/{}/T_{}/{}/'.format(NET_TYPE, args.T, udir) #, random_seed)
            LOGDIR = SAVEDIR
        else:
            n_steps = 100000
            # LOGDIR = './logs/varicopytask/{}/{}/'.format(NET_TYPE, udir) #, random_seed)
            SAVEDIR = './saves/varicopytask/{}/T_{}/{}/'.format(NET_TYPE, args.T, udir) #, random_seed)
            LOGDIR = SAVEDIR

        if args.log:
            writer = SummaryWriter('./copylogs/' + args.name + '/')

        torch.cuda.manual_seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)


        rnn = select_network(NET_TYPE, inp_size, hidden_size, nonlin, args.n, args.s, args.rinit, args.iinit, CUDA, args.learn_params)
        net = Model(hidden_size, rnn)
        if CUDA:
            net = net.cuda()
            net.rnn = net.rnn.cuda()

        if not os.path.exists(SAVEDIR):
            os.makedirs(SAVEDIR)

        if not args.adam:
            optimizer = optim.RMSprop(net.parameters(), lr=args.lr, alpha=args.alpha, weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(net.parameters(), lr=args.lr)


        with open(SAVEDIR + 'hparams.txt', 'w') as fp:
            for key, val in args.__dict__.items():
                fp.write(('{}: {} \n'.format(key, val)))

        if args.load:
            load_function()
            sys.exit(0)

        train_model(net, optimizer, batch_size, T, n_steps)

        with open(LOGFILE, 'a') as fp:
            fp.write("{}, n:{} s:{} process finished.\n".format(nonlin, args.n, args.s))

with open(LOGFILE, 'a') as fp:
    fp.write("Done.")
