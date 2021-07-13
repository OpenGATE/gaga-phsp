import torch.nn as nn
import torch


class MyLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(MyLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.clamp(x, min=0.0) + torch.clamp(x, max=0.0) * self.negative_slope


def get_activation(params):
    activation = None
    if params.activation == 'relu':
        activation = nn.ReLU()
    if params.activation == 'leaky_relu':
        activation = nn.LeakyReLU()
    if params.activation == 'my_leaky_relu':
        activation = MyLeakyReLU()
    if not activation:
        print('Error, activation unknown: ', params.activation)
        exit(0)
    return activation


class Discriminator3(nn.Module):
    """
    Discriminator: D(x, θD)
    """

    def __init__(self, params):
        super(Discriminator3, self).__init__()
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


class Generator3(nn.Module):
    """
    Generator: G(z, θG) -> x fake samples
    """

    def __init__(self, params):
        super(Generator3, self).__init__()
        z_dim = params['z_dim']
        x_dim = params['x_dim']
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
