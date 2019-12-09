from gaga_helpers import *
from gaga_functions import *
from gaga_model import Discriminator, Generator
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import grad as torch_grad
import copy

# import numpy as np
# import torch
# from torch.autograd import Variable
# import torch.nn.functional as F
# import datetime
# import copy
# from tqdm import tqdm
# import random
# import gatetools.phsp as phsp
# from matplotlib import pyplot as plt
# import matplotlib.animation as animation

'''

Initial code from :
https://github.com/znxlwm/pytorch-generative-model-collections.git
https://github.com/InsuJeon/Hello-Generative-Model/blob/master/Day04/GAN/gan.ipynb
https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
and others

Disclaimer: experimental work. All mistakes and bullshits are mine.

'''


# -----------------------------------------------------------------------------
class Gan(object):
    '''
    Main GAN object
    - Input params = dict with all parameters and options
    '''

    def __init__(self, params):

        # store parameters
        self.params = params

        # initialisations

        # init_gpu
        # init model, from scratch or start
        # init optimizer, scheduler
        # init set loss fct
        # init set penalty fct
        # init real/fake labels + smoothing
        # init instance noise
        # init data loader
        # init validation data loader

        # init gpu
        self.dtypef, self.device = init_pytorch_cuda(self.params['gpu_mode'], True)

        # init model
        self.init_model()

        # init optimiser
        self.init_optimiser()

        # init loss functions
        self.init_loss_functions()

        # init penality function        
        self.init_penalty_functions()

        # init G and D parameters
        x_dim = params['x_dim']
        g_dim = params['g_dim']
        d_dim = params['d_dim']
        z_dim = params['z_dim']
        g_layers = params['g_layers']
        d_layers = params['d_layers']
        # x_std = params['x_std']
        # x_mean = params['x_mean']

        # nb of weights
        d_param = filter(lambda p: p.requires_grad, self.D.parameters())
        params['d_nb_weights'] = sum([np.prod(p.size()) for p in d_param])
        g_param = filter(lambda p: p.requires_grad, self.G.parameters())
        params['g_nb_weights'] = sum([np.prod(p.size()) for p in g_param])
        print('Number of parameters for D :', params['d_nb_weights'])
        print('Number of parameters for G :', params['g_nb_weights'])


    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Below : initialisation functions
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    def init_model(self):
        '''
        Initialise the GAN model, G and D
        '''

        p = self.params
        if ('start_pth' not in p) or (p['start_pth'] == None):
            self.D = Discriminator(p)
            self.G = Generator(p)
        else:
            f = p['start_pth']
            print('Loading ', f)
            start_params, start_G, start_D, start_optim, start_dtypef = gaga.load(f)
            self.D = start_D
            self.G = start_G

        if (str(self.device) != 'cpu'):
            print('Set model to GPU')
            self.G.cuda()
            self.D.cuda()


    # --------------------------------------------------------------------------
    def init_optimiser(self):
        '''
        Initialise the optimiser and scheduler
        '''

        p = self.params
        d_learning_rate = p['d_learning_rate']
        g_learning_rate = p['g_learning_rate']

        if (p['optimiser'] == 'adam'):
            g_weight_decay = float(p["g_weight_decay"])
            d_weight_decay = float(p["d_weight_decay"])
            print('Optimizer regularisation L2 G weight:', g_weight_decay)
            print('Optimizer regularisation L2 D weight:', d_weight_decay)
            if "beta1" in p["beta_1"]:
                beta1 = float(p["beta_1"])
                beta2 = float(p["beta_2"])
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

        if p['optimiser'] == 'RMSprop':
            self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=d_learning_rate)
            self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=g_learning_rate)

        # auto decreasing learning_rate
        if 'scheduler_patience' in p:
            p = p['scheduler_patience']
            print('Scheduler Patience ', p)
            self.g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer,
                                                                          'min', verbose=True, patience=p)
            self.d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.d_optimizer,
                                                                          'min', verbose=True, patience=p)


    # --------------------------------------------------------------------------
    def init_loss_functions(self):
        '''
        Initialise the loss
        '''
        loss = self.params['loss_type']
        print(f'Loss is {loss}')

        if loss == 'wasserstein':
            self.criterion_r = gaga.WassersteinNegLoss()
            self.criterion_f = gaga.WassersteinLoss()
            return

        if loss == 'non-saturating-bce':
            if (str(self.device) != 'cpu'):
                self.criterion_r = nn.BCELoss().cuda()
                self.criterion_f = nn.BCELoss().cuda()
            else:
                self.criterion_r = nn.BCELoss()
                self.criterion_f = nn.BCELoss()
            return

        print(f'Error, cannot set loss {loss}')
        exit(0)

                
    # --------------------------------------------------------------------------
    def init_penalty_functions(self):
        '''
        Initialise the penalty
        '''
        t = self.params['penalty_type']
        self.penalty_fct = gaga.zero_penalty

        if t == 'gradient_penalty':
            print('Penalty is: gradient_penalty')
            self.penalty_fct = gaga.gradient_penalty

        if t == 'gradient_penalty_max':
            print('Penalty is: gradient_penalty_max')
            self.penalty_fct = gaga.gradient_penalty_max

        if self.penalty_fct == gaga.zero_penalty:
            print('Penalty is: None')

        self.penalty_weight = self.params['gp_weight']
        print(f'Penalty weight {self.penalty_weight}')


    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Below : main train function
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    def train(self, x):
        '''
        Train the GAN
        '''

        # normalisation
        x, x_mean, x_std = gaga.normalize_data(x)
        # x_mean = np.mean(x, 0, keepdims=True)
        # x_std = np.std(x, 0, keepdims=True)
        self.params['x_mean'] = x_mean
        self.params['x_std'] = x_std
        # x = (x-x_mean)/x_std

        # main dataset
        self.x = x;

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

        #penalty_fct, penalty_weight = self.set_penalty(self.params)

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


        # Allocate noise if needed
        if self.params['r_instance_noise_sigma']>0:
            print('Allocate noise')
            #self.noise = Variable(torch.zeros(shape,shape).type(self.dtypef))
            print(self.device)
            self.noise = torch.tensor(0).to(self.device)
            print('self.noise', self.noise);

        # Sampler
        loader = DataLoader(self.x,
                            batch_size=batch_size,
                            num_workers=1,
                            pin_memory=True,
                            # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702/4
                            shuffle=self.params['shuffle'],  ## if false ~20% faster, seems identical
                            #shuffle=False,  ## if false ~20% faster, seems identical
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
                if (self.params['penalty_type'] == 'clamp_penalty'):
                    clamp_lower = self.params['clamp_lower']
                    clamp_upper = self.params['clamp_upper']
                    for p in self.D.parameters():
                        p.data.clamp_(clamp_lower, clamp_upper)

                # PART 1 : D
                for _ in range(self.params['d_nb_update']):
                    # the input data
                    x = Variable(data).type(self.dtypef)

                    # FIXME -------------------------------------------------------> ADD INSTANCE NOISE real HERE
                    x = self.add_Gaussian_noise(x, self.params['r_instance_noise_sigma'])

                    # get decision from the discriminator
                    d_real_decision = self.D(x)

                    # compute loss Loss between decision and vector of ones (y_real_)
                    d_real_loss = self.criterion_r(d_real_decision, real_labels)

                    # generate z noise (latent)
                    z = Variable(torch.randn(batch_size, z_dim)).type(self.dtypef)

                    # generate fake data
                    # (detach to avoid training G on these labels (?))
                    d_fake_data = self.G(z).detach()

                    # FIXME -------------------------------------------------------> ADD INSTANCE NOISE fake HERE
                    d_fake_data = self.add_Gaussian_noise(d_fake_data, self.params['f_instance_noise_sigma'])

                    # get the fake decision on the fake data
                    d_fake_decision = self.D(d_fake_data)

                    # compute loss between fake decision and vector of zeros
                    d_fake_loss = self.criterion_f(d_fake_decision, fake_labels)

                    # FIXME NOT OK for non-saturating version ? -> BCE is negative

                    # sum of loss
                    penalty = self.penalty_fct(self, x, d_fake_data)
                    d_loss = d_real_loss + d_fake_loss + self.penalty_weight * penalty

                    # backprop + optimize
                    self.D.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    # scheduler
                    if 'scheduler_patience' in self.params:
                        self.d_scheduler.step(d_loss)


                # PART 2 : G
                for _ in range(self.params['g_nb_update']):

                    # generate z noise (latent)
                    z = Variable(torch.randn(batch_size, z_dim)).type(self.dtypef)

                    # generate the fake data
                    g_fake_data = self.G(z)#.detach()

                    # FIXME -------------------------------------------------------> ADD INSTANCE NOISE fake HERE
                    g_fake_data = self.add_Gaussian_noise(g_fake_data, self.params['f_instance_noise_sigma'])

                    # get the fake decision
                    g_fake_decision = self.D(g_fake_data)

                    # compute loss
                    g_loss = self.criterion_r(g_fake_decision, real_labels)

                    # Backprop + Optimize
                    g_loss.backward()
                    self.g_optimizer.step()

                    # scheduler
                    if 'scheduler_patience' in self.params:
                        self.g_scheduler.step(g_loss)

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


    # --------------------------------------------------------------------------
    def save(self, optim, filename):
        '''
        Save the model
        '''
        output = dict()
        output['params'] = self.params
        output['optim'] = optim
        state = copy.deepcopy(self.G.state_dict())
        output['g_model_state'] = state
        state = copy.deepcopy(self.D.state_dict())
        output['d_model_state'] = state
        torch.save(output, filename)



    ''' ----------------------------------------------------------------------------- '''
    def add_Gaussian_noise(self, x, sigma):
        '''
        Add Gaussian noise to x. Do nothing is sigma<0
        https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/4
        '''

        if sigma<0:
            return x

        print(x.shape, sigma)

        scale = sigma * x
        #self.noise = Variable(torch.zeros(shape,shape).cuda())
        sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
        x = x + sampled_noise
        return x


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
