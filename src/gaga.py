import numpy as np
import torch
import torch.nn as nn
from gaga_helpers import *
from sinkhorn import *
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import datetime
import copy
from tqdm import tqdm
import random
import phsp
from matplotlib import pyplot as plt

# from torch.utils.data.sampler import Sampler
# from matplotlib import pyplot as plt
# import helpers
# from scipy.stats import entropy   ## this is the KL divergence

#  Initial code from :
#  https://github.com/znxlwm/pytorch-generative-model-collections.git
#  https://github.com/InsuJeon/Hello-Generative-Model/blob/master/Day04/GAN/gan.ipynb
#  All mistakes and bullshits are mine


''' ---------------------------------------------------------------------------------- '''
class Discriminator(nn.Module):
    '''
    Discriminator: D(x, θD) -> probability that x is real data
    or with Wasserstein GAN :
    Discriminator is the Critic D(x, θD) -> Wasserstein distance
    
    The discriminator takes in both real and fake input data and returns
    probabilities, a number between 0 and 1, with 1 representing a prediction
    of authenticity and 0 representing fake.
    
    At Nash equilibrium, half of input will be real, half fake: D(x) = 1/2
    '''

    # def __init__(self, x_dim,
    #              d_hidden_dim,
    #              d_layers,
    #              wasserstein=False):
    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.params = params
        x_dim = params['x_dim']
        d_dim = params['d_dim']
        self.d_layers = params['d_layers']
        self.wasserstein = (params['type'] == 'wasserstein') or (params['type'] == 'gradient_penalty') 
        
        self.map1 = nn.Linear(x_dim, d_dim)
        self.maps = nn.ModuleList()

        for i in range(self.d_layers):
            self.maps.append(nn.Linear(d_dim,d_dim))
            
        self.map3 = nn.Linear(d_dim, 1)

        # FIXME --> initialisation
        # for p in self.parameters():
        #     if p.ndimension()>1:
        #         #nn.init.xavier_normal_(p)
        #         nn.init.kaiming_normal_(p)

    def forward(self, x):
        x = F.relu(self.map1(x))
        for i in range(self.d_layers):
            x = F.relu(self.maps[i](x))
        if (self.wasserstein):
            # NO SIGMOID with Wasserstein
            # https://paper.dropbox.com/doc/Wasserstein-GAN--AZxqBJuXjF5jf3zyCdJAVqEMAg-GvU0p2V9ThzdwY3BbhoP7
            x = self.map3(x)
        else:
            x = torch.sigmoid(self.map3(x))  # sigmoid needed to output probabilities 0-1
        return x



''' --------------------------------------------------------------------------------- '''
class Generator(nn.Module):
    '''
    Generator: G(z, θG) -> x fake samples
    
    Create samples that are intended to come from the same distrib than the
    training dataset. May have several z input at different layers.
    '''
    
    #def __init__(self, z_dim, x_dim, g_hidden_dim, g_layers):
    def __init__(self, params, cmin, cmax):
        super(Generator, self).__init__()

        self.params = params
        z_dim = self.params['z_dim']
        self.x_dim = self.params['x_dim']
        g_dim = self.params['g_dim']
        self.g_layers = self.params['g_layers']
        self.cmin = cmin
        self.cmax = cmax
        
        self.map1 = nn.Linear(z_dim, g_dim)
        self.maps = nn.ModuleList()

        for i in range(self.g_layers):
            self.maps.append(nn.Linear(g_dim, g_dim))
            
        self.map3 = nn.Linear(g_dim, self.x_dim)

        # FIXME --> initialisation
        for p in self.parameters():
            if p.ndimension()>1:
                #nn.init.kaiming_normal_(p)
                #nn.init.xavier_normal_(p)
                nn.init.kaiming_uniform_(p, nonlinearity='sigmoid')

    def forward(self, x):
        x = F.relu(self.map1(x))
        for i in range(self.g_layers-1):
            x = F.relu(self.maps[i](x))
            
        x = self.maps[self.g_layers-1](x)  # last one
        x = torch.sigmoid(x) # to output probability within [0-1]
        x = self.map3(x)

        # clamp values
        x = torch.max(x, self.cmin)
        x = torch.min(x, self.cmax)
        
        return x



''' --------------------------------------------------------------------------------- '''
class Gan(object):
    '''
    Main GAN object
    - Input params = dict with all parameters
    - Input x      = input dataset

    '''
    def __init__(self, params, x):
        '''
        Create a Gan object from params and samples x
        '''

        # parameters from the dataset
        self.params = params

        # init gpu
        self.dtypef, self.device = init_pytorch_cuda(self.params['gpu_mode'], True)

        # main dataset
        self.x = x;

        # init G and D parameters
        x_dim = params['x_dim']
        g_dim = params['g_dim']
        d_dim = params['d_dim']
        z_dim = params['z_dim']
        g_layers = params['g_layers']
        d_layers = params['d_layers']
        x_std = params['x_std']
        x_mean = params['x_mean']
        self.wasserstein_loss = (params['type'] == 'wasserstein') or (params['type'] == 'gradient_penalty')

        # clamp take normalisation into account
        cmin, cmax = gaga.get_min_max_constraints(params)
        cmin = torch.from_numpy(cmin).type(self.dtypef)
        cmax = torch.from_numpy(cmax).type(self.dtypef)

        # init G and D        
        self.D = Discriminator(params)
        self.G = Generator(params, cmin, cmax)

        # init optimizer
        d_learning_rate = params['d_learning_rate']
        g_learning_rate = params['g_learning_rate']
        if (params['optimiser'] == 'adam'):
            g_weight_decay = float(params["g_weight_decay"])
            d_weight_decay = float(params["d_weight_decay"])
            print('Optimizer regularisation L2 G weight:', g_weight_decay)
            print('Optimizer regularisation L2 D weight:', d_weight_decay)
            self.d_optimizer = torch.optim.Adam(self.D.parameters(),
                                                weight_decay=d_weight_decay,
                                                lr=d_learning_rate)
            self.g_optimizer = torch.optim.Adam(self.G.parameters(),
                                                weight_decay=g_weight_decay,
                                                lr=g_learning_rate)
            
        if (params['optimiser'] == 'RMSprop'):
            self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=d_learning_rate)
            self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=g_learning_rate)

        # auto decreasing learning_rate
        # self.g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer,
        #                                                          'min', verbose=True, patience=200)
        # self.d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.d_optimizer,
        #                                                          'min', verbose=True, patience=200)

        # criterion init
        if (str(self.device) != 'cpu'):
            self.G.cuda()
            self.D.cuda()
            self.criterion = nn.BCELoss().cuda()
        else:
            self.criterion = nn.BCELoss()

        # nb of weights
        d_param = filter(lambda p: p.requires_grad, self.D.parameters())
        params['d_nb_weights'] = sum([np.prod(p.size()) for p in d_param])
        g_param = filter(lambda p: p.requires_grad, self.G.parameters())
        params['g_nb_weights'] = sum([np.prod(p.size()) for p in g_param])
        print('Number of parameters for D :', params['d_nb_weights'])
        print('Number of parameters for G :', params['g_nb_weights'])


    ''' ----------------------------------------------------------------------------- '''
    def train(self):
        '''
        Train the GAN
        '''

        # get mean/std of input data for normalisation
        # self.x_mean = np.mean(self.x, 0, keepdims=True)
        # self.x_std = np.std(self.x, 0, keepdims=True)
        # self.params['x_mean'] = self.x_mean
        # self.params['x_std'] = self.x_std
        # self.x = (self.x-self.x_mean)/self.x_std
        self.x_mean = self.params['x_mean']
        self.x_std = self.params['x_std']

        # save optim epoch values
        optim = {}
        optim['g_loss'] = []
        optim['d_loss_real'] = []
        optim['d_loss_fake'] = []
        optim['g_model_state'] = []
        optim['current_epoch'] = []
        si = 0 # nb of stored epoch

        # Real/Fake labels (1/0)
        self.batch_size = self.params['batch_size']
        batch_size = self.batch_size
        real_labels = Variable(torch.ones(batch_size, 1)).type(self.dtypef)
        fake_labels = Variable(torch.zeros(batch_size, 1)).type(self.dtypef)
        # One-sided label smoothing
        if ('label_smoothing' in self.params):
            s = self.params['label_smoothing']
            real_labels = Variable((1.0-s)+s*torch.rand(batch_size, 1)).type(self.dtypef)
            fake_labels = Variable(s*torch.rand(batch_size, 1)).type(self.dtypef)
            
        # Sampler
        loader = DataLoader(self.x,
                            batch_size=batch_size,
                            num_workers=1,
                            # pin_memory=True,
                            # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702/4
                            shuffle=self.params['shuffle'],  ## if false ~20% faster, seems identical
                            #shuffle=False,  ## if false ~20% faster, seems identical
                            drop_last=True)

        # Start training
        epoch = 0
        start = datetime.datetime.now()
        pbar = tqdm(total=self.params['epoch'], disable=not self.params['progress_bar'])
        z_dim = self.params['z_dim']
        while (epoch < self.params['epoch']):
            for batch_idx, data in enumerate(loader):
                
                # Clamp D if wasserstein mode (not in gradient_penalty mode)
                if (self.params['type'] == 'wasserstein'):
                    clamp_lower = self.params['clamp_lower']
                    clamp_upper = self.params['clamp_upper']
                    for p in self.D.parameters():
                        p.data.clamp_(clamp_lower, clamp_upper)

                # PART 1 : D
                for _ in range(self.params['d_nb_update']):
                    # the input data
                    x = Variable(data).type(self.dtypef)
                    
                    # get decision from the discriminator
                    d_real_decision = self.D(x)

                    # compute loss BCELoss between decision and vector of ones (y_real_)
                    if (self.wasserstein_loss):
                        d_real_loss = -torch.mean(d_real_decision)
                    else:
                        d_real_loss = self.criterion(d_real_decision, real_labels)

                    # generate z noise (latent)
                    z = Variable(torch.randn(batch_size, z_dim)).type(self.dtypef)

                    # generate fake data
                    # (detach to avoid training G on these labels (?))
                    d_fake_data = self.G(z).detach()

                    # get the fake decision on the fake data
                    d_fake_decision = self.D(d_fake_data)
                    
                    # compute loss between fake decision and vector of zeros
                    if (self.wasserstein_loss):
                        d_fake_loss = torch.mean(d_fake_decision)
                    else:
                        d_fake_loss = self.criterion(d_fake_decision, fake_labels)

                    # FIXME NOT OK for non-saturating version ? -> BCE is negative

                    # sum of loss
                    if (self.params['type'] == 'gradient_penalty'):
                        gradient_penalty = self.compute_gradient_penalty(x, d_fake_data)
                        d_loss = d_real_loss + d_fake_loss + self.params['gp_weight']*gradient_penalty
                    else:
                        d_loss = d_real_loss + d_fake_loss

                    # backprop + optimize
                    self.D.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                # PART 2 : G
                for _ in range(self.params['g_nb_update']):

                    # generate z noise (latent)
                    z = Variable(torch.randn(batch_size, z_dim)).type(self.dtypef)

                    # generate the fake data
                    g_fake_data = self.G(z)#.detach()

                    # get the fake decision
                    g_fake_decision = self.D(g_fake_data)

                    # compute loss
                    if (self.wasserstein_loss):
                        g_loss = -torch.mean(g_fake_decision)
                    else:
                        # I think this is the non-saturated version (see Fedus2018)
                        # loss is  BCE(D(G(z), 1)) instead of
                        # non-saturated : BCE(D(G(z), 1)) = -1/2 E_z[log(D(G(z)))]
                        # minmax : -BCE(D(G(z)), 0) = E_z[log(1-D(G(z)))]
                        g_loss = self.criterion(g_fake_decision, real_labels)

                    # Backprop + Optimize
                    g_loss.backward()
                    self.g_optimizer.step()

                # Housekeeping
                self.D.zero_grad() # FIXME not needed ? 
                self.G.zero_grad() # FIXME to put before g backward ? 

                # print info sometimes
                if (epoch) % 100 == 0:
                    tqdm.write('Epoch %d d_loss: %.5f   g_loss: %.5f     d_real_loss: %.5f  d_fake_loss: %.5f'
                               %(epoch,
                                 d_loss.data.item(),
                                 g_loss.data.item(),
                                 d_real_loss.data.item(),
                                 d_fake_loss.data.item()))

                # plot sometimes
                if (self.params['plot']):
                    if (epoch) % int(self.params['plot_every_epoch']) == 0:
                        self.plot_epoch(self.params['keys'], epoch)
                    
                # save loss value 
                optim['d_loss_real'].append(d_real_loss.data.item())
                optim['d_loss_fake'].append(d_fake_loss.data.item())
                optim['g_loss'].append(g_loss.data.item())
                
                # dump sometimes
                if (epoch>self.params['dump_epoch_start']):
                    should_dump1 = (epoch-self.params['dump_epoch_start']) % self.params['dump_epoch_every']
                    should_dump1 = (should_dump1 == 0)
                    should_dump2 = self.params['epoch']-epoch < self.params['dump_last_n_epoch']
                    if should_dump1 or should_dump2:
                        state = copy.deepcopy(self.G.state_dict())
                        optim['g_model_state'].append(state)
                        optim['current_epoch'].append(epoch)
                        si = si+1

                # update loop
                pbar.update(1)
                epoch += 1

                # should we stop ?
                if (epoch > self.params['epoch']):
                    break

        # end of training
        pbar.close()
        stop = datetime.datetime.now()
        print('Training completed epoch = ', epoch)
        print('Start time    = ', start.strftime(gaga.date_format))
        print('End time      = ', stop.strftime(gaga.date_format))
        print('Duration time = ', (stop-start))
        return optim
    

    ''' ----------------------------------------------------------------------------- '''
    def compute_gradient_penalty(self, real_data, fake_data):
        #https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
        #https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
        gpu = (str(self.device) != 'cpu')
        
        # alpha
        alpha = torch.rand(self.batch_size, 1)#, 1, 1)
        alpha = alpha.expand_as(real_data)
        if gpu:
            alpha = alpha.cuda()

        # interpolated
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated = Variable(interpolated, requires_grad=True)
        if gpu:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # gradient
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if gpu else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]

        # norm
        #LAMBDA = .1  # Smaller lambda seems to help for toy tasks specifically
        #gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()        
        return gradient_penalty

    
    ''' ----------------------------------------------------------------------------- '''
    def plot_epoch(self, keys, epoch):
        '''
        Plot data during training (slow)
        '''

        n = int(1e5)
        nb_bins = 200
        
        # create fig
        nrow, ncol = phsp.fig_get_nb_row_col(len(keys))
        f, ax = plt.subplots(nrow, ncol, figsize=(25,10))

        # get random true data ; un-normalize
        x = self.x
        start = random.randint(0,len(x)-n)
        real = x[start:start+n,:]
        real = (real*self.x_std)+self.x_mean
        
        # generate z noise (latent) and fake real
        # z = Variable(torch.randn(n, self.params['z_dim'])).type(self.dtypef)
        # fake = self.G(z)
        # fake = fake.cpu().data.numpy()
        # fake = (fake*self.x_std)+self.x_mean
        fake = generate_samples(self.params, self.G, self.dtypef, n)
        
        # plot all keys for real data
        i = 0
        for k in keys:
            gaga.fig_plot_marginal(real, k, keys, ax, i, nb_bins, 'g')
            i = i+1

        # plot all keys for fake data
        i = 0
        for k in keys:
            gaga.fig_plot_marginal(fake, k, keys, ax, i, nb_bins, 'r')
            i = i+1
            
        # remove empty plot
        phsp.fig_rm_empty_plot(len(keys), ax)
        #plt.show()
        output_filename = 'aa.png'
        plt.suptitle('Epoch '+str(epoch))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(output_filename)
        plt.close()
