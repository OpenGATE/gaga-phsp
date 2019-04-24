import numpy as np
import torch
import torch.nn as nn
from gaga_helpers import *
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.data.sampler import Sampler
# from matplotlib import pyplot as plt
# import helpers
# import time
# import copy
# from scipy.stats import entropy   ## this is the KL divergence
from tqdm import tqdm

#  Initial code from :
#  https://github.com/znxlwm/pytorch-generative-model-collections.git
#  https://github.com/InsuJeon/Hello-Generative-Model/blob/master/Day04/GAN/gan.ipynb
#  All mistakes and bullshits are mine


''' ----------------------------------------------------------------------------
Discriminator: D(x, θD) -> probability that x is real data
or with Wasserstein GAN : Discriminator is the Critic D(x, θD) -> Wasserstein distance

The discriminator takes in both real and fake input data and returns
probabilities, a number between 0 and 1, with 1 representing a prediction
of authenticity and 0 representing fake.

At Nash equilibrium, half of input will be real, half fake: D(x) = 1/2
---------------------------------------------------------------------------- '''
class Discriminator(nn.Module):

    def __init__(self, x_dim,
                 d_hidden_dim,
                 d_layers,
                 wasserstein=False):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(x_dim, d_hidden_dim)
        self.maps = nn.ModuleList()
        self.d_layers = d_layers
        self.wasserstein = wasserstein
        for i in range(d_layers):
            self.maps.append(nn.Linear(d_hidden_dim,d_hidden_dim))
            
        self.map3 = nn.Linear(d_hidden_dim, 1)

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



''' ----------------------------------------------------------------------------
Generator: G(z, θG) -> x fake samples

Create samples that are intended to come from the same distrib than the
training dataset. May have several z input at different layers.
---------------------------------------------------------------------------- '''
class Generator(nn.Module):
    
    def __init__(self, z_dim, x_dim, g_hidden_dim, g_layers):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(z_dim, g_hidden_dim)
        self.maps = nn.ModuleList()
        self.g_layers = g_layers
        for i in range(g_layers):
            self.maps.append(nn.Linear(g_hidden_dim, g_hidden_dim))
            
        self.map3 = nn.Linear(g_hidden_dim, x_dim)

        # FIXME --> initialisation
        # for p in self.parameters():
        #     if p.ndimension()>1:
        #         #nn.init.kaiming_normal_(p)
        #         #nn.init.xavier_normal_(p)
        #         nn.init.kaiming_uniform_(p, nonlinearity='sigmoid')

    def forward(self, x):
        x = F.relu(self.map1(x))
        for i in range(self.g_layers-1):
            x = F.relu(self.maps[i](x))
            
        x = self.maps[self.g_layers-1](x)  # last one
        x = torch.sigmoid(x) # to output probability within [0-1]
        x = self.map3(x)
        return x



''' ----------------------------------------------------------------------------
Main GAN object
---------------------------------------------------------------------------- '''
class Gan(object):
    def __init__(self, params, x):

        # parameters from the dataset
        self.params = params

        # init gpu
        self.dtypef, self.device = init_pytorch_cuda(self.params['gpu_mode'], True)

        # main dataset
        self.x = x;

        # init G and D
        x_dim = params['x_dim']
        g_dim = params['g_dim']
        d_dim = params['d_dim']
        z_dim = params['z_dim']
        g_layers = params['g_layers']
        d_layers = params['d_layers']
        wasserstein = (params['type'] == 'wasserstein')
        self.D = Discriminator(x_dim, d_dim, d_layers, wasserstein)
        self.G = Generator(z_dim, x_dim, g_dim, g_layers)

        # init optimizer
        d_learning_rate = params['d_learning_rate']
        g_learning_rate = params['g_learning_rate']
        if (params['optimiser'] == 'adam'):
            g_weight_decay = float(params["g_weight_decay"])
            d_weight_decay = float(params["d_weight_decay"])
            print('Optimizer regularisation L2 G weight:', g_weight_decay)
            print('Optimizer regularisation L2 D weight:', d_weight_decay)
            self.D_optimizer = torch.optim.Adam(self.D.parameters(),
                                                weight_decay=dweight_decay,
                                                lr=d_learning_rate)
            self.G_optimizer = torch.optim.Adam(self.G.parameters(),
                                                weight_decay=g_weight_decay,
                                                lr=g_learning_rate)
            
        if (params['optimiser'] == 'RMSprop'):
            self.D_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=d_learning_rate)
            self.G_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=g_learning_rate)

        # auto decreasing learning_rate
        # self.G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.G_optimizer,
        #                                                          'min', verbose=True, patience=200)
        # self.D_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.D_optimizer,
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


    def train(self):

        # normalize

        # FIXME
        self.x_mean = np.mean(self.x, 0, keepdims=True)
        self.x_std = np.std(self.x, 0, keepdims=True)
        print('Normalization ', self.x_mean, self.x_std)
        self.x_mean = Variable(torch.from_numpy(self.x_mean)).type(self.dtypef)
        self.x_std = Variable(torch.from_numpy(self.x_std)).type(self.dtypef)
        self.params['x_mean'] = self.x_mean
        self.params['x_std'] = self.x_std

        # print
        print(self.params['encoding_policy'])
        policy = self.params['encoding_policy']

        # angles index for clip FIXME removed
        self.angles_index = []
        # if (policy["psi"] == "sincos"):
        #     if (policy['E'] != 'ignore'):
        #         self.angles_index.append(1)
        #         self.angles_index.append(2)
        #     else:
        #         self.angles_index.append(0)
        #         self.angles_index.append(1)
        print('angles_index', self.angles_index)

        # save optim epoch values
        optim = {}
        optim['G_loss'] = []
        optim['D_loss_real'] = []
        optim['D_loss_fake'] = []
        #optim['D_loss_equi'] = []
        optim['E_D_x'] = []
        optim['E_D_G_z'] = []
        optim['G_model_state'] = []
        optim['current_epoch'] = []
        si = 0 # nb of stored epoch

        # Real/Fake labels (1/0)
        real_labels = Variable(torch.ones(self.batch_size, 1)).type(self.dtypef)
        fake_labels = Variable(torch.zeros(self.batch_size, 1)).type(self.dtypef)

        # Sampler
        #batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, self.batch_size, False)
        # Pytorch Dataset encapsulation
        #train_data = TensorDataset(torch.from_numpy(x_train).type(dtypef))
        #sampler = Sampler(self.x, 1, batch_size)
        #train_loader = DataLoader(train_data, batch_sampler=sampler)
        loader = DataLoader(self.x,
                            batch_size=self.batch_size,
                            num_workers=1,
                            # pin_memory=True,
                            # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702/4
                            shuffle=self.params['shuffle'],  ## if false ~20% faster, seems identical
                            #shuffle=False,  ## if false ~20% faster, seems identical
                            drop_last=True)

        # Start training
        epoch = 0
        start = time.strftime("%c")
        pbar = tqdm(total=self.epoch, disable=not self.params['progress_bar'])
        while (epoch < self.epoch):
        #for epoch in range(0,self.epoch):
            #print(epoch)
            for batch_idx, data in enumerate(loader):
                #print('batch_idx', batch_idx)

                # Clamp if wasserstein mode
                clamp_lower = self.params['clamp_lower']
                clamp_upper = self.params['clamp_upper']
                if (self.wasserstein_mode):
                    for p in self.D.parameters():
                        #print('param before',p.data)
                        p.data.clamp_(clamp_lower, clamp_upper)
                        #print('param after',p.data)

                # FIXME: no clamp for G ?

                # One-sided label smoothing
                if ('label_smoothing' in self.params):
                    s = self.params['label_smoothing']
                    real_labels = Variable((1.0-s)+s*torch.rand(self.batch_size, 1)).type(self.dtypef)
                    fake_labels = Variable(s*torch.rand(self.batch_size, 1)).type(self.dtypef)

                # DD(real_labels)
                # DD(fake_labels)

                #============= PART (1): Train the discriminator =============#
                if 'D_nb_loop' in self.params:
                    n_D = self.params['D_nb_loop']
                else:
                    n_D = 1
                # if (self.wasserstein_mode):
                #     n_D = 5

                for _ in range(n_D):
                    # the input data
                    x = Variable(data).type(self.dtypef)
                    # ----> WARNING
                    x = (x-self.x_mean)/self.x_std

                    # print('x sh', x.shape, x.size)
                    # #x = x.view(self.x_dim, self.batch_size)
                    if (self.x_dim == 1):
                        x = x.view(self.batch_size, self.x_dim)
                    # print('x sh', x.shape, x.size)

                    # get decision from the discriminator
                    d_real_decision = self.D(x)

                    # compute loss BCELoss between decision and vector of ones (y_real_)
                    if (self.wasserstein_mode):
                        d_real_loss = -torch.mean(d_real_decision)
                    else:
                        d_real_loss = self.criterion(d_real_decision, real_labels)

                    # E_D_x = torch.mean(d_real_decision)

                    # generate z noise (latent)
                    z = Variable(torch.randn(self.batch_size, self.z_dim)).type(self.dtypef)

                    # generate fake data
                    # (detach to avoid training G on these labels (?))
                    d_fake_data = self.G(z).detach()

                    # get the fake decision on the fake data
                    d_fake_decision = self.D(d_fake_data)

                    # E_D_G_z = torch.mean(d_fake_decision)

                    # compute loss between fake decision and vector of zeros
                    if (self.wasserstein_mode):
                        d_fake_loss = torch.mean(d_fake_decision)
                    else:
                        d_fake_loss = self.criterion(d_fake_decision, fake_labels)


                    # FIXME NOT OK for non-saturating version ? -> BCE is negative


                    #d_equi_loss = torch.abs(torch.mean(d_real_decision)-0.5) + torch.abs(torch.mean(d_fake_decision)-0.5)

                    # sum of loss
                    d_loss = d_real_loss + d_fake_loss
                    #+ d_equi_loss

                    # regularization FIXME
                    # reg = 1e-6
                    # W = self.G.weight
                    # for name, param in self.G.named_parameters():
                    #     print(name)
                    #     if 'bias' not in name:
                    #         d_loss = d_loss + (0.5 * reg * torch.sum(torch.pow(W, 2)))


                    # Backprop + Optimize
                    self.D.zero_grad()
                    d_loss.backward()
                    self.D_optimizer.step()

                #============= PART (2): Train the generator =============#

                if 'G_nb_loop' in self.params:
                    n_G = self.params['G_nb_loop']
                else:
                    n_G = 1
                # if (self.wasserstein_mode):
                #     n_D = 5

                for _ in range(n_G):

                    # generate z noise (latent)
                    z = Variable(torch.randn(self.batch_size, self.z_dim)).type(self.dtypef)

                    # generate the fake data
                    g_fake_data = self.G(z)#.detach()

                    # get the fake decision
                    g_fake_decision = self.D(g_fake_data)

                    # compute loss
                    if (self.wasserstein_mode):
                        g_loss = -torch.mean(g_fake_decision)
                    else:
                        # I think this is the non-saturated version (see Fedus2018)
                        # loss is  BCE(D(G(z), 1)) instead of
                        # non-saturated : BCE(D(G(z), 1)) = -1/2 E_z[log(D(G(z)))]
                        # minmax : -BCE(D(G(z)), 0) = E_z[log(1-D(G(z)))]
                        g_loss = self.criterion(g_fake_decision, real_labels)

                    # Backprop + Optimize
                    g_loss.backward()
                    self.G_optimizer.step()

                # Housekeeping
                self.D.zero_grad()
                self.G.zero_grad()

                if (epoch+1) % 100 == 0:
                    tqdm.write('Epoch %d, d_loss: %.4f, d_real_loss: %.4f, d_fake_loss: %.4f, g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f'
                          %(epoch,
                            d_loss.data.item(),
                            d_real_loss.data.item(),
                            d_fake_loss.data.item(),
                            g_loss.data.item(),
                            d_real_decision.data.mean(),
                            d_fake_decision.data.mean()))

                # slow !!
                if (self.params['plot']):
                    if (epoch+1) % self.params['plot_every_epoch'] == 0:
                        # tqdm.write('Epoch %d, plot in aa.png and bb.png'%(epoch))
                        self.plot_decoded_fig(data,
                                              self.params['param_names'],
                                              self.params['param_encoded_index'], epoch)
                        self.plot_encoded_fig(data, self.params['param_encoded_index'], epoch)

                # update loop
                pbar.update(1)
                epoch += 1

                # scheduler for learning rate
                # self.G_scheduler.step(g_loss.data.item())
                # self.D_scheduler.step(d_loss.data.item())

                # for loss_name, loss in return_dict['losses'].items():
                #     return_dict['losses'][loss_name] = loss.unsqueeze(0)
                # return_dict['metrics']['accuracy_cls'] = return_dict['metrics']['accuracy_cls'].unsqueeze(0)
                # print('dic', self.G.state_dict().keys)

                # save loss value 
                optim['D_loss_real'].append(d_real_loss.data.item())
                optim['D_loss_fake'].append(d_fake_loss.data.item())
                #optim['D_loss_equi'].append(d_equi_loss.data.item())
                # optim['E_D_x'].append(E_D_x.data.item())
                # optim['E_D_G_z'].append(E_D_G_z.data.item())
                optim['G_loss'].append(g_loss.data.item())
                
                # save values
                # optim['D_loss'].append(d_loss.data.item())
                if (epoch>self.params['dump_epoch_start']):
                    # if (si<self.params['dump_epoch_nb']):
                    should_dump1 = (epoch-self.params['dump_epoch_start']) % self.params['dump_epoch_every']
                    should_dump1 = (should_dump1 == 0)
                    should_dump2 = self.epoch-epoch < self.params['dump_last_n_epoch']
                    if should_dump1 or should_dump2:
                        #if (epoch+1) % self.params['dump_every_epoch'] == 0:
                        # print('store ', epoch)
                        state = copy.deepcopy(self.G.state_dict())
                        optim['G_model_state'].append(state)
                        optim['current_epoch'].append(epoch)
                        si = si+1

                # optim['loss'].append(g_loss.data.item())

                if (epoch > self.epoch):
                    break

        pbar.close()
        now = time.strftime("%c")
        print('end epoch start/stop ', start, now)
        return optim
