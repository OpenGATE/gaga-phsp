import copy
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from .gaga_functions import *
from .gaga_helpers import *
import gaga_phsp

'''
Initial code from :
https://github.com/znxlwm/pytorch-generative-model-collections.git
https://github.com/InsuJeon/Hello-Generative-Model/blob/master/Day04/GAN/gan.ipynb
https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
and others

Disclaimer: experimental work. All mistakes and bullshits are mine.
'''


class Gan(object):
    """
    Main GAN object
    Input params = dict with all parameters and options
    """

    def __init__(self, params):

        # store parameters
        self.params = params

        # init gpu
        self.dtypef, self.device = init_pytorch_cuda(self.params['gpu_mode'], True)

        # init model
        self.init_model()

        # init optimiser
        self.init_optimiser()

        # init loss functions
        self.init_loss_functions()

        # init penalty function
        self.init_penalty_functions()

        # init labels
        self.init_labels()

        # compute and store nb of weights
        d_param = filter(lambda p: p.requires_grad, self.D.parameters())
        params['d_nb_weights'] = sum([np.prod(p.size()) for p in d_param])
        g_param = filter(lambda p: p.requires_grad, self.G.parameters())
        params['g_nb_weights'] = sum([np.prod(p.size()) for p in g_param])
        print('Number of parameters for D :', params['d_nb_weights'])
        print('Number of parameters for G :', params['g_nb_weights'])

    def init_model(self):
        """
        Initialise the GAN model, G and D
        """

        p = self.params
        if 'start_pth' in p and p['start_pth'] is not None:
            f = p['start_pth']
            print('Loading previous pth ', f)
            start_params, start_G, start_D, start_optim, start_dtypef = gaga_phsp.load(f)
            self.D = start_D
            self.G = start_G
            try:
                self.params['start_epoch'] = start_optim['last_epoch']
            except:
                self.params['start_epoch'] = start_optim['current_epoch'][-1]
        else:
            self.G, self.D = create_G_and_D_model(p)
            self.params['start_epoch'] = 0

        self.z_rand = get_z_rand(self.params)

    def init_optimiser(self):
        """
        Initialise the optimiser and scheduler
        """

        p = self.params
        d_learning_rate = p['d_learning_rate']
        g_learning_rate = p['g_learning_rate']

        if p['optimiser'] == 'adam':
            g_weight_decay = float(p["g_weight_decay"])
            d_weight_decay = float(p["d_weight_decay"])
            print('Optimizer regularisation L2 G weight:', g_weight_decay)
            print('Optimizer regularisation L2 D weight:', d_weight_decay)
            beta1 = float(p["beta_1"])
            beta2 = float(p["beta_2"])
            print('Adam beta:', beta1, beta2)
            self.d_optimizer = torch.optim.Adam(self.D.parameters(),
                                                weight_decay=d_weight_decay,
                                                lr=d_learning_rate,
                                                betas=(beta1, beta2))
            self.g_optimizer = torch.optim.Adam(self.G.parameters(),
                                                weight_decay=g_weight_decay,
                                                lr=g_learning_rate,
                                                betas=(beta1, beta2))

        if p['optimiser'] == 'RMSprop':
            # self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=d_learning_rate)
            # self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=g_learning_rate)

            # FIXME
            """
            momentum (float, optional) – momentum factor (default: 0)
            alpha (float, optional) – smoothing constant (default: 0.99)
            centered (bool, optional) – if True, compute the centered RMSProp, 
                         the gradient is normalized by an estimation of its variance
            weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
            """
            self.d_optimizer, self.g_optimizer = get_RMSProp_optimisers(self, p)

        if p['optimiser'] == 'SGD':
            self.d_optimizer = torch.optim.SGD(self.D.parameters(), lr=d_learning_rate)
            self.g_optimizer = torch.optim.SGD(self.G.parameters(), lr=g_learning_rate)

        if p['optimiser'] == 'gdtuo-RMSprop':
            from gradient_descent_the_ultimate_optimizer import gdtuo
            self.d_optimizer_gdtuo = gdtuo.RMSProp(optimizer=gdtuo.SGD(1e-5))
            self.g_optimizer_gdtuo = gdtuo.RMSProp(optimizer=gdtuo.SGD(1e-5))
            self.d_optimizer = gdtuo.ModuleWrapper(self.D, optimizer=self.d_optimizer_gdtuo)
            self.g_optimizer = gdtuo.ModuleWrapper(self.G, optimizer=self.g_optimizer_gdtuo)
            self.d_optimizer.initialize()
            self.g_optimizer.initialize()
            print('NOT YET')
            exit(0)


        # auto decreasing learning_rate
        self.is_scheduler_enabled = False
        try:
            step_size = p['schedule_learning_rate_step']
            gamma = p['schedule_learning_rate_gamma']
            print('Scheduler is enabled ', step_size, gamma)
            # WARNING step_size is not nb of epoch but nb of optimiser.step (nb of D update per epoch)
            d_ss = step_size * self.params['d_nb_update']
            g_ss = step_size * self.params['g_nb_update']
            print('scheduler: step_size, gamma: ', step_size, gamma)
            print('scheduler d_ss and g_ss:', d_ss, g_ss)
            self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.d_optimizer, step_size=d_ss, gamma=gamma)
            self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.g_optimizer, step_size=g_ss, gamma=gamma)
            self.is_scheduler_enabled = True
        except:
            print('Scheduler is disabled')

    def init_loss_functions(self):
        """
        Initialise the loss
        """
        loss = self.params['loss']
        print(f'Loss is {loss}')

        # https://github.com/AlexiaJM/MaximumMarginGANs/blob/master/Code/GAN.py

        if loss == 'wasserstein':
            self.criterion_dr = gaga_phsp.WassersteinNegLoss()
            self.criterion_df = gaga_phsp.WassersteinLoss()
            self.criterion_g = self.criterion_dr
            return

        if loss == 'hinge':
            self.criterion_dr = gaga_phsp.HingeNegLoss()
            self.criterion_df = gaga_phsp.HingeLoss()
            self.criterion_g = gaga_phsp.WassersteinNegLoss()
            return

        if loss == 'non_saturating_bce':
            self.criterion_dr = nn.BCELoss()
            self.criterion_df = nn.BCELoss()
            return

        print(f'Error, cannot set loss {loss}')
        exit(0)

    def init_penalty_functions(self):
        """
        Initialise the penalty
        """
        t = self.params['penalty']
        self.penalty_fct = gaga_phsp.zero_penalty
        self.penalty_weight = self.params['penalty_weight']
        print(f'Penalty weight {self.penalty_weight}')
        print(f'Penalty is: {t}')

        # Change names: 8 different gradient penalties
        # L1_LS L1_Hinge
        # L2_LS L2_Hinge
        # Linf_LS Linf_Hinge
        # GP0 SqHinge
        penalties = {
            'GP_L1_LS': gaga_phsp.GP_L1_LS,
            'GP_L2_LS': gaga_phsp.GP_L2_LS,
            'GP_Linf_LS': gaga_phsp.GP_Linf_LS,
            'GP_L1_Hinge': gaga_phsp.GP_L1_Hinge,
            'GP_L2_Hinge': gaga_phsp.GP_L2_Hinge,
            'GP_Linf_Hinge': gaga_phsp.GP_Linf_Hinge,
            'GP_0GP': gaga_phsp.GP_0GP,
            'GP_SquareHinge': gaga_phsp.GP_SquareHinge,
        }
        # print(t, penalties)
        for p in penalties:
            if t == p:
                self.penalty_fct = penalties[p]
                return

        if t == 'clamp':
            self.penalty_fct = gaga_phsp.zero_penalty
            self.clamp_lower = self.params['clamp_lower']
            self.clamp_upper = self.params['clamp_upper']
            return

        if t == 'no_penalty':
            self.penalty_fct = gaga_phsp.zero_penalty
            return

        print(f'Error, cannot set penalty {t}')
        exit(0)

    def init_labels(self):
        """
        Helper to init the Real=1.0 and Fake=0.0 labels. May be smoothed.
        """

        # Real/Fake labels (1/0)
        batch_size = self.params['batch_size']
        self.real_labels = Variable(torch.ones(batch_size, 1)).type(self.dtypef)
        self.fake_labels = Variable(torch.zeros(batch_size, 1)).type(self.dtypef)
        # One-sided label smoothing
        if 'label_smoothing' in self.params:
            s = self.params['label_smoothing']
            self.real_labels = Variable((1.0 - s) + s * torch.rand(batch_size, 1)).type(self.dtypef)
            self.fake_labels = Variable(s * torch.rand(batch_size, 1)).type(self.dtypef)

    def init_optim_data(self):
        """
        Allocate the optim data structure that store information of the training process
        """

        optim = {}
        optim['g_loss'] = []
        optim['d_loss'] = []
        optim['d_loss_real'] = []
        optim['d_loss_fake'] = []
        optim['g_model_state'] = []
        optim['current_epoch'] = []
        optim['w_value'] = []
        optim['w_epoch'] = []
        optim['validation_d_loss'] = []
        optim['d_best_loss'] = 1e9
        optim['d_best_epoch'] = 0

        return optim

    def init_cuda(self):
        if str(self.device) == 'cpu':
            return
        print('Set model to GPU')
        self.G.cuda()
        self.D.cuda()

        # print('Set data to GPU')
        # real_labels and fake_labels are set to cuda before

        print('Set optim to GPU')
        self.criterion_dr.cuda()
        self.criterion_df.cuda()
        self.criterion_g.cuda()

    def add_Gaussian_noise(self, x, sigma):
        """
        Add Gaussian noise to x. Do nothing is sigma<0
        https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/4
        """

        if sigma <= 0:
            return x

        if str(self.device) != 'cpu':
            s = torch.std(x, 0).cuda()
            sampled_noise = torch.randn(*x.size(), requires_grad=False).cuda() * sigma * s
        else:
            s = torch.std(x, 0)
            sampled_noise = torch.randn(*x.size(), requires_grad=False) * sigma * s

        x = x + sampled_noise * sigma
        return x

    # Below : main train function

    def train(self, x):
        """
        Train the GAN
        """

        # normalisation
        print('Normalization')
        x, x_mean, x_std = gaga_phsp.normalize_data(x)
        self.params['x_mean'] = x_mean
        self.params['x_std'] = x_std

        # main dataset
        self.x = x
        self.batch_size = self.params.batch_size

        # init cuda
        self.init_cuda()

        # initialise the data structure that will store info during training
        optim = self.init_optim_data()
        self.optim = optim

        # init conditional
        condn = len(self.params['cond_keys'])
        nx = self.params['x_dim']
        conditional = condn > 0
        if conditional:
            print(f'Conditional : {self.params["cond_keys"]} ' + str(condn))

        # Sampler
        print('Dataloader')
        batch_size = self.params['batch_size']
        loader = DataLoader(self.x,
                            batch_size=batch_size,
                            num_workers=2,  # no gain if larger than 2 (?)
                            pin_memory=True,
                            # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702/4
                            shuffle=self.params['shuffle'],
                            # shuffle=False,  ## if false ~20% faster, seems identical
                            drop_last=True)

        # Start training
        epoch = self.params['start_epoch']
        self.params['end_epoch'] = epoch + self.params['epoch']
        print(f"Epoch from {self.params['start_epoch']} to {self.params['end_epoch']} (total = {self.params['epoch']})")
        real_labels = self.real_labels
        fake_labels = self.fake_labels
        start = datetime.datetime.now()
        pbar = tqdm(total=self.params['epoch'], disable=not self.params['progress_bar'])
        z_dim = self.params['z_dim']

        it = iter(loader)

        for batch_idx in range(self.params['epoch']):

            # Clamp D if needed
            if self.params['penalty'] == 'clamp':
                gaga_phsp.clamp_parameters(self)

            # FIXME V3
            for p in self.D.parameters():
                p.requires_grad = True

            # PART 1 : D -------------------------------------------------------
            for _ in range(self.params['d_nb_update']):

                #
                if self.params['optimiser'] == 'gdtuo-RMSprop':
                    self.d_optimizer.begin()
                self.D.zero_grad()

                # the input data
                # https://github.com/pytorch/pytorch/issues/1917
                try:
                    data = next(it)
                except StopIteration:
                    print('dataset empty, restart from zero', epoch)  # restart from zero
                    it = iter(loader)
                    data = next(it)
                x = Variable(data).type(self.dtypef)

                ## torch.cuda.empty_cache() ??

                # add instance noise
                x = self.add_Gaussian_noise(x, self.params['r_instance_noise_sigma'])

                # get decision from the discriminator
                d_real_decision = self.D(x)

                # generate z noise (latent)
                z = Variable(self.z_rand(batch_size, z_dim)).type(self.dtypef)

                # concat conditional vector (if any)
                if conditional:
                    condx = x[:, nx - condn:nx]
                    z = torch.cat((z.float(), condx.float()), dim=1)

                # generate fake data
                # (detach to avoid training G on these labels)
                d_fake_data = self.G(z).detach()  # FIXME detach ?

                # concat conditional vector (if any)
                if conditional:
                    d_fake_data = torch.cat((d_fake_data.float(), condx.float()), dim=1)

                # add instance noise
                d_fake_data = self.add_Gaussian_noise(d_fake_data, self.params['f_instance_noise_sigma'])

                # get the fake decision on the fake data
                d_fake_decision = self.D(d_fake_data)

                # set penalty
                penalty = self.penalty_weight * self.penalty_fct(self, x, d_fake_data)

                # compute loss between decision on real and vector of ones (real_labels)
                d_real_loss = self.criterion_dr(d_real_decision, real_labels)

                # compute loss between decision on fake and vector of zeros (fake_labels)
                d_fake_loss = self.criterion_df(d_fake_decision, fake_labels)

                # backward
                d_real_loss.backward()
                d_fake_loss.backward()
                if self.penalty_fct != gaga_phsp.zero_penalty:
                    penalty.backward()

                # sum of loss
                d_loss = d_real_loss + d_fake_loss + penalty

                # optimizer
                self.d_optimizer.step()

                # scheduler
                if self.is_scheduler_enabled:
                    self.d_scheduler.step()

            # PART 2 : G -------------------------------------------------------
            for p in self.D.parameters():
                p.requires_grad = False

            for _ in range(self.params['g_nb_update']):

                #
                if self.params['optimiser'] == 'gdtuo-RMSprop':
                    self.g_optimizer.begin()

                # required
                self.G.zero_grad()

                # generate z noise (latent)
                z = Variable(self.z_rand(batch_size, z_dim)).type(self.dtypef)

                # conditional
                if conditional:
                    z = torch.cat((z.float(), condx.float()), dim=1)

                # generate the fake data
                g_fake_data = self.G(z)

                # concat conditional vector (if any)
                if conditional:
                    g_fake_data = torch.cat((g_fake_data.float(), condx.float()), dim=1)

                # add instance noise
                g_fake_data = self.add_Gaussian_noise(g_fake_data, self.params['f_instance_noise_sigma'])

                # get the fake decision
                g_fake_decision = self.D(g_fake_data)

                g_loss = self.criterion_g(g_fake_decision, real_labels)

                # Backprop + Optimize
                g_loss.backward(retain_graph=True)
                self.g_optimizer.step()

                # scheduler
                if self.is_scheduler_enabled:
                    self.g_scheduler.step()

            # housekeeping (to not accumulate gradient)
            # zero_grad clears old gradients from the last step
            # (otherwise you’d just accumulate the gradients from all loss.backward() calls).
            # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            self.D.zero_grad()
            self.G.zero_grad()

            # keep some data (for epoch_dump)
            self.d_loss = d_loss
            self.g_loss = g_loss
            self.d_real_loss = d_real_loss
            self.d_fake_loss = d_fake_loss

            # save loss value
            optim['d_loss_real'].append(d_real_loss.data.item())
            optim['d_loss_fake'].append(d_fake_loss.data.item())
            optim['d_loss'].append(d_loss.data.item())
            optim['g_loss'].append(g_loss.data.item())

            # sometimes: dump, plot, store
            self.epoch_dump(epoch)
            self.epoch_store(epoch)

            # update loop
            pbar.update(1)
            epoch += 1

            # should we stop ?
            if epoch > self.params['end_epoch']:
                break

        # end of training
        pbar.close()
        stop = datetime.datetime.now()
        optim['last_epoch'] = epoch
        print('Training completed epoch = ', epoch)
        print('Start time    = ', start.strftime(gaga_phsp.date_format))
        print('End time      = ', stop.strftime(gaga_phsp.date_format))
        print('Duration time = ', (stop - start))
        self.params.Duration = str(stop - start)
        self.params.duration = str(stop - start)
        return optim

    def save(self, optim, filename):
        """
        Save the model
        """
        output = dict()
        output['params'] = self.params
        output['optim'] = optim
        state_g = copy.deepcopy(self.G.state_dict())
        state_d = copy.deepcopy(self.D.state_dict())
        output['g_model_state'] = state_g
        output['d_model_state'] = state_d
        torch.save(output, filename)

    def epoch_dump(self, epoch):
        """
        Dump during training
        """
        try:
            n = self.params['epoch_dump']
        except:
            n = 500

        if epoch % n != 0:
            return
        tqdm.write('Epoch %d d_loss: %.5f   g_loss: %.5f     d_real_loss: %.5f  d_fake_loss: %.5f'
                   % (epoch,
                      self.d_loss.data.item(),
                      self.g_loss.data.item(),
                      self.d_real_loss.data.item(),
                      self.d_fake_loss.data.item()))

    def epoch_store(self, epoch):
        """
        Store during training
        """

        try:
            n = self.params['epoch_store_model_every']
        except:
            n = -1

        if epoch % n != 0:
            return

        state = copy.deepcopy(self.G.state_dict())
        self.optim['g_model_state'].append(state)
        self.optim['current_epoch'].append(epoch)
