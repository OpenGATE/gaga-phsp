import torch.nn as nn
import torch
import gaga_phsp as gaga
from torch.autograd import Variable
from types import MethodType


class MyLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(MyLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.clamp(x, min=0.0) + torch.clamp(x, max=0.0) * self.negative_slope


def get_activation(params):
    activation = None
    # set activation
    if params['activation'] == 'relu':
        activation = nn.ReLU()
    if params['activation'] == 'leaky_relu':
        activation = nn.LeakyReLU()
    if params['activation'] == 'my_leaky_relu':
        activation = MyLeakyReLU()
    if not activation:
        print('Error, activation unknown: ', params['activation'])
        exit(0)
    return activation


class Discriminator(nn.Module):
    """
    Discriminator: D(x, θD)
    """

    def __init__(self, params):
        super(Discriminator, self).__init__()
        x_dim = params['x_dim']
        d_dim = params['d_dim']
        d_l = params['d_layers']
        sn = False
        if 'spectral_norm' in params:
            sn = params['spectral_norm']
        # activation function
        activation = get_activation(params)
        # create the net
        self.net = nn.Sequential()
        # first layer
        if sn:
            self.net.add_module('1st_layer', nn.utils.spectral_norm(nn.Linear(x_dim, d_dim)))
        else:
            self.net.add_module('1st_layer', nn.Linear(x_dim, d_dim))

        # hidden layers
        for i in range(d_l):
            self.net.add_module(f'activation_{i}', activation)
            if sn:
                self.net.add_module(f'layer_{i}', nn.utils.spectral_norm(nn.Linear(d_dim, d_dim)))
            else:
                self.net.add_module(f'layer_{i}', nn.Linear(d_dim, d_dim))
        # latest layer
        if params['loss'] == 'non-saturating-bce':
            self.net.add_module('sigmoid', nn.Sigmoid())
        else:
            self.net.add_module('last_activation', activation)
            self.net.add_module('last_layer', nn.Linear(d_dim, 1))

        # for p in self.parameters():
        #    if p.ndimension() > 1:
        #        nn.init.kaiming_uniform_(p, nonlinearity='sigmoid')

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    """
    Generator: G(z, θG) -> x fake samples
    """

    def __init__(self, params):
        super(Generator, self).__init__()
        self.params = params
        # the total input dim for the G is z_dim + conditional_keys (if any)
        z_dim = params['z_dim'] + len(params['cond_keys'])
        x_dim = params['x_dim'] - len(params['cond_keys'])
        g_dim = params['g_dim']
        g_l = params['g_layers']
        # activation function
        activation = get_activation(params)
        # create the net
        self.net = nn.Sequential()
        # first layer
        self.net.add_module('first_layer', nn.Linear(z_dim, g_dim))
        # hidden layers
        for i in range(g_l):
            self.net.add_module(f'activation_{i}', activation)
            self.net.add_module(f'layer_{i}', nn.Linear(g_dim, g_dim))
        # last layer
        self.net.add_module(f'last_activation_{i}', activation)
        self.net.add_module(f'last_layer', nn.Linear(g_dim, x_dim))

        # initialisation (not sure better than default init). Keep default.
        for p in self.parameters():
            if p.ndimension() > 1:
                nn.init.kaiming_normal_(p)  ## seems better ???
                # nn.init.xavier_normal_(p)
                # nn.init.kaiming_uniform_(p, nonlinearity='sigmoid')

    def forward(self, x):
        return self.net(x)

    def forward_with_post_processing(self, x):
        # generate data
        y = self.net(x)
        # denormalize
        y = (y * self.x_std) + self.x_mean
        # apply post_process
        y = self.post_process(y, self.params)
        return y

    def init_forward_with_post_processing(self, f, gpu):
        self.post_process = f
        self.init_forward_with_denorm(gpu)
        self.forward = self.forward_with_post_processing

    def init_forward_with_denorm(self, gpu):
        # init the std/mean in torch variable
        dtypef, device = gaga.init_pytorch_cuda(gpu, False)
        self.x_mean = Variable(torch.from_numpy(self.params['x_mean']).type(dtypef))
        self.x_std = Variable(torch.from_numpy(self.params['x_std']).type(dtypef))
        # by default, bypass the test in  denormalization
        # (if forward_with_post_processing is used, the test is kept)
        self.forward = self.forward_with_norm

    def forward_with_norm(self, x):
        y = self.net(x)
        y = (y * self.x_std) + self.x_mean
        return y
