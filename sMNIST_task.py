import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as T
import numpy as np
import pickle
import argparse
import time
import os
import datetime

working_dir = os.path.dirname(os.path.realpath(__file__))

from NetworkCreation.common import henaff_init, cayley_init, random_orthogonal_init
from NetworkCreation.Networks import ANRU, RNN, LSTM, GRU
from Training.training_utils import LocalTransform, SinTransform, StepTransform
from torch._utils import _accumulate
from torch.utils.data import Subset

parser = argparse.ArgumentParser(description='(p)sMNIST task')
parser.add_argument('-g', '--cuda', action='store_true', default=False,
                    help='Use CUDA')
parser.add_argument('-p', '--permute', action='store_true', default=False, 
                    help='permute the order of sMNIST')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='Print details')

parser.add_argument('--gain', type=float, default=1.0, 
                    help='degree of nonlinearity at initialization')
parser.add_argument('--saturation', type=float, default=1.0, 
                    help='degree of saturation  at initialization')
parser.add_argument('--random', action='store_true', default=True, 
                    help='random shape parameters initialization')
parser.add_argument('--learn_params', action='store_true', default=True,
                    help='learn the shape parameters')
parser.add_argument('--nonlin', type=str, default='gamma2', 
                    choices=['gamma','gamma2','ReLU'],
                    help='Nonlinearity for RNN.')
parser.add_argument('--net-type', type=str, default='RNN',
                    choices=['ANRU', 'RNN', 'LSTM', 'GRU'],
                    help='Type of recurrent neural net.')
parser.add_argument('--nhid', type=int, default=400, 
                    help='hidden size of recurrent net')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')

parser.add_argument('--save-freq', type=int, default=25, 
                    help='frequency (in epochs) to save data')
parser.add_argument('--seed', type=int, default=400, 
                    help='random seed for reproducibility')
parser.add_argument('--rinit', type=str, default="henaff", 
                    help='recurrent weight matrix initialization')
parser.add_argument('--iinit', type=str, default="kaiming", 
                    help='input weight matrix initialization')
parser.add_argument('--batch', type=int, default=100,
                    help='batch size')
parser.add_argument('--note', type=str, default='',
                    help='Any details to be entered manually upon launch')
parser.add_argument('--test', action='store_true', default=False,
                    help='Test model, no training.')

parser.add_argument('--transform', type=str, default='sin',
                    help='Transform to be applied on test set')
parser.add_argument('--transform-ratio', type=float, default=0.0,
                    help='Ratio of dataset to apply sin transform on inputs.')
args = parser.parse_args()

CUDA = args.cuda
SAVEFREQ = args.save_freq
n, s = args.gain, args.saturation

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
np.random.seed(args.seed)

torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available():
    if not CUDA:
        print("WARNING: You have a CUDA device, so you should probably run with -g,--cuda. Enabling CUDA.")
        CUDA = True
device = torch.device("cuda" if CUDA else "cpu")


###############################################################################
#   Load MNIST data
###############################################################################

if __name__ == "__main__":
    rng = np.random.RandomState(args.seed)
    if args.permute:
        task = 'psMNIST'
        order = rng.permutation(784)
    else:
        task = 'sMNIST'
        order = np.arange(784)

    trainset = T.datasets.MNIST(root=working_dir+'/Training/training_data/mnist', train=True, download=True, transform=T.transforms.ToTensor())
    valset = T.datasets.MNIST(root=working_dir+'/Training/training_data/mnist', train=True, download=True, transform=T.transforms.ToTensor())
    offset = 10000

    R = rng.permutation(len(trainset))
    lengths = (len(trainset) - offset, offset)
    trainset, valset = [Subset(trainset, R[offset - length:offset]) for offset, length in
                        zip(_accumulate(lengths), lengths)]
    testset = T.datasets.MNIST(root=working_dir+'/Training/training_data/mnist', train=False, download=True, transform=T.transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, num_workers=2)


###############################################################################
#   Saving
###############################################################################

if __name__ == "__main__":
    if args.nonlin is not ('gamma' or 'gamma2'): 
        args.learn_params = False

    if args.net_type=='RNN':
        udir = f'{args.net_type}_lr{args.lr}_p{args.transform_ratio}_hs{args.nhid}_{args.nonlin}'
    else:
        udir = f'{args.net_type}_lr{args.lr}_p{args.transform_ratio}_hs{args.nhid}'
    exp_time = "{0:%Y-%m-%d}_{0:%H-%M-%S}".format(datetime.datetime.now())
    SAVEDIR = os.path.join('./Training/SavedModels',
                            task,
                            str(args.seed),
                            udir,
                            str(datetime.date.today()))
            # Savedir is of type: ./Training/SavedModels/psMNIST/400/ANRU_lr0.0001_p0.0_hs400/2020-01-02/

    if not args.test:
        # Considering args.verbose runs as just debugging, so allowing overwritting. 
        if args.verbose: SAVEDIR+='--X'
        # Else
        if not os.path.exists(SAVEDIR):
            os.makedirs(SAVEDIR)
        elif not args.verbose:
            SAVEDIR += '--0'
            try:
                os.makedirs(SAVEDIR)
            except FileExistsError:
                SAVEDIR = SAVEDIR[:-1]+str(int(SAVEDIR[-1])+1)
                os.makedirs(SAVEDIR)

        LOGFILE = os.path.join(SAVEDIR, 'logfile.txt')
        LOGTEST = os.path.join(SAVEDIR, 'test_results.txt')
        with open(LOGFILE, 'w') as fp:
            '''Details and hyperparameters, if desired'''
            fp.write(f'{task} task on {datetime.datetime.now()}\n')
            fp.write(f'\nSaving to: {SAVEDIR}')
            if args.note!='':
                fp.write(f'\nNote     : {args.note}')
        
            fp.write('\n\nHyperparameters: ')
            for key, val in args.__dict__.items():
                fp.write(('{}: {}, '.format(key, val)))

        with open(LOGTEST, 'w') as fp:
            fp.write('Testing data.')
            fp.write('\n\nTest    => accuracy on original testing set\nTest T. => accuracy on testing set with SinTransform(freq=1, phase=0, amplitude=0.5)')
            fp.write('\n\n'+'='*50+'\n  Epoch   Test      Test T.  \n'+'='*50)


###############################################################################
# Define the model
###############################################################################


class Model(nn.Module):
    def __init__(self, hidden_size, rnn):
        super(Model, self).__init__()
        self.hidden_size = hidden_size

        # Modules
        self.lin = nn.Linear(hidden_size, 10)
        self.rnn = rnn

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs, y, order, 
                transform=None, return_ns=False, external_drive=None):
        h_net0, h_net1, a = None, None, None
        hiddens, shape_signals, pre_activations = [], [], []

        inputs = inputs[:, order] # permute inputs

        for i, input in enumerate(torch.unbind(inputs, dim=1)):
            x = input.unsqueeze(1)
            if transform is not None:
                x = transform(i, x)

            if external_drive is None:
                shift = 0.0
            else:
                shift = external_drive.get_factor(i)-1
            
            h_net0 = self.rnn(x, h_net0, external_drive=shift)

            for temp in [h_net0, h_net1]:
                if temp is not None and temp.requires_grad: 
                    temp.retain_grad()

            # if return_ns:
            #     if args.net_type=='ANRU': 
            #         shape_signals.append(shape_parameters)
            #         pre_activations.append(pre_activs.cpu().detach().numpy())
            #     hiddens.append(h_net0.cpu().detach().numpy())

        out = self.lin(h_net0)  # decode
        loss = self.loss_func(out, y)
        preds = torch.argmax(out, dim=1)
        correct = torch.eq(preds, y).sum().item()
        return loss, correct, hiddens

        # if transform is not None: suffix = '_T'
        # elif external_drive is not None: suffix='_D'
        # else: suffix+''
            
        # shape_signals_label = 'shapesignals'+suffix+'.npy'
        # hiddens_label = 'net0_hiddenstates'+suffix+'.npy'
        # preactivs_label = 'net0_preactivations'+suffix+'.npy'

        # if return_ns: 
        #     if args.net_type=='ANRU': 
        #         np.save(os.path.join(MODELDIR, shape_signals_label), shape_signals)
        #         np.save(os.path.join(MODELDIR, preactivs_label), pre_activations)
        #     np.save(os.path.join(MODELDIR, hiddens_label), hiddens)


def test_model(net, dataloader, transform=None, return_parameters=False, external_drive=None):
    accuracy = 0
    loss = 0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):

            x, y = data
            x = x.view(-1, 784)
            if CUDA:
                x = x.cuda()
                y = y.cuda()
            if args.net_type == 'LSTM' or args.net_type == 'ANRU':
                net.rnn.init_states(x.shape[0])
            loss, c, _ = net.forward(x, y, order, transform=transform, return_ns=False, external_drive=external_drive)
            if args.verbose:
                if i%10==0: print(f'Step {i}, Loss: {loss.item()}')
            accuracy += c

    accuracy /= len(testset)
    return loss, accuracy


def save_checkpoint(state, fname):
    filename = os.path.join(SAVEDIR, fname)
    torch.save(state, filename)

def train_model(net, optimizer, scheduler, num_epochs):
    with open(LOGFILE, 'a') as fp:
        fp.write('\n\n'+'-'*70+'\nBeginning of training.')
        fp.write('\n\nTrain  => accuracy on training set \nVal    => accuracy on original validation set.\nVal T. => accuracy on validation set with sinuisoidal transform')
        fp.write('\n\n'+'='*45+'\n  Epoch  Time     Train    Val      Val T. \n'+'='*45)

    with open(os.path.join(SAVEDIR, 'training_loss_details.txt'), 'w') as fp:
        fp.write('Loss details:\n')

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    val_T_losses, val_T_accuracies = [], []
    save_norms = []
    shape_params = []

    best_val_acc = 0
    ta = 0
    for epoch in range(recover, recover+num_epochs):
        s_t = time.time()
        accs = []
        losses = []
        norms = []
        processed = 0
        net.train()
        correct = 0

        ts = np.random.binomial(1, args.transform_ratio, size=[500])
        for i, data in enumerate(trainloader, 0):
            inp_x, inp_y = data
            inp_x = inp_x.view(-1, 784)
            if bool(ts[i]):
                trans1 = SinTransform(freq=np.random.uniform(0,2), phase=0, amplitude=0.2)
            else:
                trans1 = None

            if CUDA:
                inp_x = inp_x.cuda()
                inp_y = inp_y.cuda()
            if args.net_type == 'LSTM' or args.net_type=='ANRU':
                net.rnn.init_states(inp_x.shape[0])

            optimizer.zero_grad()

            loss, c, _ = net.forward(inp_x, inp_y, order, transform=trans1)
            correct += c
            processed += inp_x.shape[0]

            accs.append(correct / float(processed))

            loss.backward()
            if args.verbose:
                if i%10==0: print(f'Step {i}, Loss: {loss.item()}')
            losses.append(loss.item())

            if np.isnan(loss.item()):
                raise ValueError

            optimizer.step()
            if i%50==0:
                with open(os.path.join(SAVEDIR, 'training_loss_details.txt'), 'a') as fp:
                    fp.write('{:3.0f} Loss = {:5.8f}\n'.format(i, loss.item()))

            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 'inf')
            norms.append(norm)

        # Validation
        val_loss, val_acc = test_model(net, valloader, transform=None)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss) # lr scheduler based on validation loss

        # Validation on transformed set
        trans2 = SinTransform(freq=np.random.uniform(0,2), phase=0, amplitude=0.5)
        val_loss_T, val_acc_T = test_model(net, valloader, transform=trans2)
        val_accuracies_T.append(val_acc_T)
        val_losses_T.append(val_loss_T)

        with open(LOGFILE, 'a') as fp:
            fp.write('\n  {:3.0f}    {}  {:2.5f}  {:2.5f}  {:2.5f}'.
                format(epoch + 1, str(datetime.timedelta(seconds=int(time.time() - s_t))), 
                np.mean(accs), val_acc, val_acc_T))

        train_losses.append(np.mean(losses))
        train_accuracies.append(np.mean(accs))
        save_norms.append(np.mean(norms))

        # Save data
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            with open(os.path.join(SAVEDIR, 'Train_Losses'), 'wb') as fp:
                pickle.dump(train_losses, fp)
            with open(os.path.join(SAVEDIR, 'Val_Losses'), 'wb') as fp:
                pickle.dump(val_losses, fp)

            with open(os.path.join(SAVEDIR, 'Train_Accuracy'), 'wb') as fp:
                pickle.dump(train_accuracies, fp)
            with open(os.path.join(SAVEDIR, 'Val_Accuracy'), 'wb') as fp:
                pickle.dump(val_accuracies, fp)

            with open(os.path.join(SAVEDIR, 'Grad_Norms'), 'wb') as fp:
                pickle.dump(save_norms, fp)

            with open(os.path.join(SAVEDIR, 'val_Accuracy_T'), 'wb') as fp:
                pickle.dump(val_accuracies_T, fp)
            with open(os.path.join(SAVEDIR, 'val_Losses_T'), 'wb') as fp:
                pickle.dump(val_losses_T, fp)


        if epoch % SAVEFREQ == 0 or epoch == num_epochs - 1:
            save_checkpoint({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            },
                'e_{}.pth.tar'.format(epoch)
            )
    
            _, test_acc = test_model(net, testloader, transform=None)
            _, test_T_acc = test_model(net, testloader, transform=SinTransform(freq=1, phase=0, amplitude=0.5))
            with open(LOGTEST, 'a') as fp:
                fp.write('\n  {:3.0f}     {:2.6f}  {:2.6f}'.format(epoch,  test_acc, test_T_acc))

    return


###############################################################################
#   Training
###############################################################################

inp_size = 1

if __name__ == "__main__":
    if args.net_type == 'ANRU':
        rnn = ANRU(inp_size, main_hidden_size=args.nhid, supervisor_hidden_size=50, cuda=CUDA,
                            r_initializer=args.rinit, i_initializer=args.iinit, adaptation_type='heterogeneous', verbose=args.verbose)
    elif args.net_type == 'RNN':
        if args.random:
            n0 = 5+2*torch.rand(1); s0 = 0.0
        else:
            n0, s0 = args.gain, args.saturation
        rnn = RNN(inp_size, args.nhid, bias=True, nonlin=args.nonlin, cuda=CUDA, n_init=n0, s_init=s0,
                        r_initializer=args.rinit, i_initializer=args.iinit, learn_params=args.learn_params)
    elif args.net_type == 'LSTM':
        rnn = LSTM(inp_size, args.nhid, cuda=CUDA)
    elif args.net_type == 'GRU':
        rnn = GRU(inp_size, args.nhid, cuda=CUDA)
    else:
        print('Net-type unrecognised. Using default: RNN')
    
    # set training modules
    net = Model(args.nhid, rnn)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') 

    if args.verbose:
        print('\nNumber of trainable parameters:')
        total_params = 0
        for name, parameter in net.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params+=param
            print(name, param)
        print('-> Total: {:6.0f} ({:3.1f}K)\n'.format(total_params, total_params/1000))

    if CUDA:
        net = net.cuda()

    epoch = 0
    recover = 0
    #recover_date = '2021-02-12'
    epoch = recover
    num_epochs = 100 - recover 

    if recover > 0: 
        # Recover a pretrained model + continue training
        MODELDIR = os.path.join('./Training/SavedModels', task, str(args.seed), udir, recover_date)
        last_model = torch.load(os.path.join(MODELDIR, f'e_{recover}.pth.tar'))
        net.load_state_dict(last_model['state_dict'])
        print(f'Recovered: NET={args.net_type}, lr={args.lr}, p={args.transform_ratio}, epoch={recover}')

    if not args.test:
        try:
            train_model(net, optimizer, scheduler, num_epochs)
        except ValueError:
            with open(LOGFILE, 'a') as fp:
                fp.write("\n"+"*"*70+"\nNan loss encountered, program terminated.")
        except KeyboardInterrupt:
            with open(LOGFILE, 'a') as fp:
                fp.write("\n"+"*"*70+"\nExited from training early")
    else:
        '''
        Equivalent to: if args.test. Recover a pre-trained model.
        '''
        # set the following yourself
        recover = 50 
        recover_date = '2021-02-07--1'

        MODELDIR = os.path.join('./Training/SavedModels', task, str(args.seed), udir, recover_date)

        last_model = torch.load(os.path.join(MODELDIR, f'e_{recover}.pth.tar'))
        net.load_state_dict(last_model['state_dict'])
        print(f'Recovered: NET={args.net_type}, lr={args.lr}, p={args.transform_ratio}, epoch={recover}')

        start_time = time.time()
        test_loss, test_acc = test_model(net, testloader, transform=None, return_parameters=True)
        print('\nOriginal:\n\tTest loss: {}\n\tTest accuracy: {}'.format(test_loss, test_acc))

        test_T_loss, test_T_acc = test_model(net, testloader, transform=SinTransform(freq=1, phase=0, amplitude=0.5), return_parameters=True)
        print('\nTransformed:\n\tTest loss: {}\n\tTest accuracy: {}'.format(test_T_loss, test_T_acc))

        test_D_loss, test_D_acc = test_model(net, testloader, external_drive=StepTransform(step_size=1.0, step_length=200, step_position=200), return_parameters=True)
        print('\nExternal drive:\n\tTest loss: {}\n\tTest accuracy: {}'.format(test_D_loss, test_D_acc))    
        print('\nTime:', time.time() - start_time)
