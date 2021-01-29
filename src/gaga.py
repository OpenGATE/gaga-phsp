import numpy as np
import torch
import torch.nn as nn
from gaga.gaga_helpers import *
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import datetime
import copy
from tqdm import tqdm
import random
import gatetools.phsp as phsp
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# from torch.utils.data.sampler import Sampler
# from matplotlib import pyplot as plt
# import helpers
# from scipy.stats import entropy   ## this is the KL divergence

#  Initial code from :
#  https://github.com/znxlwm/pytorch-generative-model-collections.git
#  https://github.com/InsuJeon/Hello-Generative-Model/blob/master/Day04/GAN/gan.ipynb
#  All mistakes and bullshits are mine


# -----------------------------------------------------------------------------
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

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.params = params
        x_dim = params['x_dim']
        d_dim = params['d_dim']
        self.d_layers = params['d_layers']
        self.wasserstein = (params['type'] == 'wasserstein') or (params['type'] == 'gradient_penalty')

        self.map1 = nn.Linear(x_dim, d_dim)
        self.maps = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.activ = F.relu
        if 'leaky_relu' in params:
            self.activ = F.leaky_relu

        for i in range(self.d_layers):
            self.maps.append(nn.Linear(d_dim,d_dim))
            self.norms.append(nn.LayerNorm(d_dim))

        self.map3 = nn.Linear(d_dim, 1)

    def forward(self, x):
        x = self.activ(self.map1(x))

        if self.params['layer_norm'] == True:
            for i in range(self.d_layers):
                x = self.activ(self.norms[i](self.maps[i](x)))
        else:
            for i in range(self.d_layers):
                x = self.activ(self.maps[i](x))

        if (self.wasserstein):
            # NO SIGMOID with Wasserstein
            # https://paper.dropbox.com/doc/Wasserstein-GAN--AZxqBJuXjF5jf3zyCdJAVqEMAg-GvU0p2V9ThzdwY3BbhoP7
            x = self.map3(x)
        else:
            x = torch.sigmoid(self.map3(x))  # sigmoid needed to output probabilities 0-1
        return x



# -----------------------------------------------------------------------------
class Generator(nn.Module):
    '''
    Generator: G(z, θG) -> x fake samples

    Create samples that are intended to come from the same distrib than the
    training dataset. May have several z input at different layers.
    '''

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

        self.activ = F.relu
        if 'leaky_relu' in params:
            self.activ = F.leaky_relu

        # initialisation
        for p in self.parameters():
            if p.ndimension()>1:
                nn.init.kaiming_normal_(p) ## seems better ???
                #nn.init.xavier_normal_(p)
                #nn.init.kaiming_uniform_(p, nonlinearity='sigmoid')

    def forward(self, x):
        x = self.activ(self.map1(x))
        for i in range(self.g_layers-1):
            x = self.activ(self.maps[i](x))

        x = self.maps[self.g_layers-1](x)  # last one
        x = torch.sigmoid(x) # to output probability within [0-1]
        #x = self.activ(x)
        x = self.map3(x)

        # clamp values
        x = torch.max(x, self.cmin)
        x = torch.min(x, self.cmax)

        return x



# -----------------------------------------------------------------------------
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
        if 'dump_wasserstein_every' not in self.params:
            self.params['dump_wasserstein_every'] = -1
        if 'w_n' not in self.params:
            self.params['w_n'] = int(1e4)
        if 'w_l' not in self.params:
            self.params['w_l'] = int(1e2)
        if 'w_p' not in self.params:
            self.params['w_p'] = 1
        print('w', self.params['dump_wasserstein_every'], self.params['w_n'], self.params['w_l'], self.params['w_p'])

        # clamp take normalisation into account
        cmin, cmax = gaga.get_min_max_constraints(params)
        cmin = torch.from_numpy(cmin).type(self.dtypef)
        cmax = torch.from_numpy(cmax).type(self.dtypef)

        # init G and D
        if ('start_pth' not in self.params) or (params['start_pth'] == None):
            self.D = Discriminator(params)
            self.G = Generator(params, cmin, cmax)
        else:
            f = params['start_pth']
            print('Loading ', f)
            start_params, start_G, start_D, start_optim, start_dtypef = gaga.load(f)
            self.D = start_D
            self.G = start_G

        # init optimizer
        d_learning_rate = params['d_learning_rate']
        g_learning_rate = params['g_learning_rate']
        if (params['optimiser'] == 'adam'):
            g_weight_decay = float(params["g_weight_decay"])
            d_weight_decay = float(params["d_weight_decay"])
            print('Optimizer regularisation L2 G weight:', g_weight_decay)
            print('Optimizer regularisation L2 D weight:', d_weight_decay)
            if "beta1" in params["beta_1"]:
                beta1 = float(params["beta_1"])
                beta2 = float(params["beta_2"])
            else:
                beta1 = 0.9
                beta2 = 0.999
            self.d_optimizer = torch.optim.Adam(self.D.parameters(),
                                                weight_decay=d_weight_decay,
                                                betas=[beta1,beta2],
                                                lr=d_learning_rate)
            self.g_optimizer = torch.optim.Adam(self.G.parameters(),
                                                weight_decay=g_weight_decay,
                                                betas=[beta1,beta2],
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
        optim['d_loss'] = []
        optim['d_loss_real'] = []
        optim['d_loss_fake'] = []
        optim['g_model_state'] = []
        optim['current_epoch'] = []
        optim['w_value'] = []
        optim['w_epoch'] = []
        optim['validation_d_loss_real'] = []
        optim['validation_d_loss_fake'] = []
        optim['validation_d_loss'] = []
        optim['validation_g_loss'] = []
        optim['validation_epoch'] = []
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
                            pin_memory=True,
                            # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702/4
                            shuffle=self.params['shuffle'],  ## if false ~20% faster, seems identical
                            #shuffle=False,  ## if false ~20% faster, seems identical
                            drop_last=True)

        # Sampler for wasserstein distance
        loader_w = DataLoader(self.x,
                              batch_size=int(self.params['w_n']),
                              num_workers=3,
                              # pin_memory=True,
                              # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702/4
                              shuffle=False,
                              drop_last=True)

        # Sampler for validation
        vdfn = self.params['validation_filename']
        if vdfn != None:
            print('Validation file:', vdfn)
            self.validation_x, validation_read_keys, m = phsp.load(vdfn)
            print('Validation read keys', validation_read_keys, len(self.validation_x), len(self.validation_x[0]))
            self.validation_x = phsp.select_keys(self.validation_x, validation_read_keys, self.params['keys'])
            print('Validation selected keys', self.params['keys'], len(self.validation_x), len(self.validation_x[0]))

            # normalisation
            x_mean = self.params['x_mean']
            x_std = self.params['x_std']
            self.validation_x = (self.validation_x-x_mean)/x_std
            print('Validation normalisation', x_mean)

            loader_validation = DataLoader(self.validation_x,
                                           batch_size=batch_size,
                                           num_workers=2, # faster if diff from main loader num_workers=1
                                           pin_memory=True,
                                           # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702/4
                                           shuffle=False,
                                           drop_last=True)
            print('Validation loader done')


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
                        # this is the non-saturated version (see Fedus2018)
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
                if (epoch) % 500 == 0:
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
                optim['d_loss'].append(d_loss.data.item())
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

                # compute wasserstein distance sometimes
                dwe = self.params['dump_wasserstein_every']
                if (dwe >0) and (epoch % dwe == 0):
                    n = int(self.params['w_n'])
                    l = int(self.params['w_l'])
                    p = int(self.params['w_p'])
                    real = next(iter(loader_w))
                    real = Variable(real).type(self.dtypef)
                    z = Variable(torch.randn(n, z_dim)).type(self.dtypef)
                    fake = self.G(z)
                    d = gaga.sliced_wasserstein(real, fake, l, p)
                    optim['w_value'].append(d)
                    optim['w_epoch'].append(epoch)
                    tqdm.write('Epoch {} wasserstein: {:5f}'.format(epoch, d))

                # compute loss on validation dataset sometimes
                vdfn = self.params['validation_filename']
                vde = self.params['validation_every_epoch']
                if (vdfn != None) and (epoch % vde == 0):

                    with torch.set_grad_enabled(False):
                        data_v = next(iter(loader_validation)) ## FIXME SLOW ??? better if num_workers=2

                        xx = Variable(data_v).type(self.dtypef)
                        dv_real_decision = self.D(xx).detach()                    
                        if (self.wasserstein_loss):
                            dv_real_loss = -torch.mean(dv_real_decision)
                        else:
                            dv_real_loss = self.criterion(dv_real_decision, real_labels)

                        zz = Variable(torch.randn(batch_size, z_dim)).type(self.dtypef)
                        dv_fake_data = self.G(zz).detach()
                        dv_fake_decision = self.D(dv_fake_data).detach()
                        if (self.wasserstein_loss):
                            dv_fake_loss = torch.mean(dv_fake_decision)
                        else:
                            dv_fake_loss = self.criterion(dv_fake_decision, fake_labels)
                        if (self.params['type'] == 'gradient_penalty'):
                            gradient_penalty = self.compute_gradient_penalty(xx, dv_fake_data)
                            dv_loss = dv_real_loss + dv_fake_loss + self.params['gp_weight']*gradient_penalty
                        else:
                            dv_loss = dv_real_loss + dv_fake_loss
                            
                        # G
                        zz = Variable(torch.randn(batch_size, z_dim)).type(self.dtypef)
                        gv_fake_data = self.G(zz).detach()
                        gv_fake_decision = self.D(gv_fake_data).detach()
                        if (self.wasserstein_loss):
                            gv_loss = -torch.mean(gv_fake_decision)
                        else:
                            gv_loss = self.criterion(gv_fake_decision, real_labels)
                        
                        optim['validation_d_loss_real'].append(dv_real_loss.data.item())
                        optim['validation_d_loss_fake'].append(dv_fake_loss.data.item())
                        optim['validation_d_loss'].append(dv_loss.data.item())
                        optim['validation_g_loss'].append(gv_loss.data.item())
                        optim['validation_epoch'].append(epoch)

                        tqdm.write('Epoch {} validation: G {:5f} vs {:5f} '
                                   .format(epoch, g_loss.data.item(), gv_loss.data.item()))
                        #break


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

        # Two sides penalty
        #gradient_penalty = ((gradients_norm - 1) ** 2).mean()

        # one side penalty
        a = torch.max(gradients_norm - 1, torch.zeros_like(gradients_norm))
        gradient_penalty = (a** 2).mean()

        # one side gradient penalty
        # replace
        # E((|∇f(αx_real −(1−α)x_fake)|−1)²)
        # by
        # (max(|∇f|−1,0))²
        #

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
        #fake = generate_samples(self.params, self.G, self.dtypef, n)
        z_dim = self.params['z_dim']
        x_mean = self.params['x_mean']
        x_std = self.params['x_std']
        z = Variable(torch.randn(n, z_dim)).type(self.dtypef)
        fake = self.G(z)
        fake = fake.cpu().data.numpy()
        fake = (fake*x_std)+x_mean

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
        #output_filename = 'aa_{:06d}.png'.format(epoch)
        output_filename = 'aa.png'

        plt.suptitle('Epoch '+str(epoch))
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(output_filename)
        plt.close()
