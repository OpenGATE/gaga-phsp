import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator2(nn.Module):
    """
    Discriminator: D(x, θD) -> probability that x is real data
    Or with Wasserstein GAN :
    Discriminator is called Critic D(x, θD) -> Wasserstein distance

    The discriminator takes in both real and fake input data and returns
    probabilities, a number between 0 and 1, with 1 representing a prediction
    of authenticity and 0 representing fake.

    At Nash equilibrium, half of input will be real, half fake: D(x) = 1/2 (?)

    Input:
        - x_dim       : input dimension
        - d_dim       : nb of neurons per layer
        - d_layers    : nb of hidden layers (+ first and last layer)
        - leaky_relu  : boolean (relu if False, leaky_relu if True)
        - layer_norm  : boolean
        - loss_type   : special case for 'non-saturating-bce', use sigmoid at the end
    """

    def __init__(self, params):
        super(Discriminator2, self).__init__()
        x_dim = params['x_dim']
        d_dim = params['d_dim']
        d_l = params['d_layers']
        # activation function
        activation = nn.ReLU()
        if 'leaky_relu' in params:
            activation = nn.LeakyReLU()
        # create the net
        self.net = nn.Sequential()
        # first layer
        self.net.add_module('1st_layer', nn.Linear(x_dim, d_dim))
        if params['layer_norm']:
            self.net.add_module('1st_layer_norm', nn.LayerNorm(d_dim))
        # hidden layers
        for i in range(d_l):
            self.net.add_module(f'activation_{i}', activation)
            self.net.add_module(f'layer_{i}', nn.Linear(d_dim, d_dim))
            if params['layer_norm']:
                self.net.add_module(f'layer_norm_{i}', nn.LayerNorm(d_dim))
        # latest layer
        if params['loss_type'] == 'non-saturating-bce':
            self.net.add_module('sigmoid', nn.Sigmoid())
        else:
            self.net.add_module('last_activation', activation)
            self.net.add_module('last_layer', nn.Linear(d_dim, 1))

    def forward(self, x):
        return self.net(x)


class Generator2(nn.Module):
    """
    Generator: G(z, θG) -> x fake samples

    Create samples that are intended to come from the same distrib than the
    training dataset. May have several z input at different layers.

    Input:
        - z_dim
        - x_dim
        - d_layers
        - leaky_relu  : boolean (relu if False, leaky_relu if True)
        - layer_norm  : boolean
        - loss_type   : special case for 'wasserstein'

    """

    def __init__(self, params):
        super(Generator2, self).__init__()
        z_dim = params['z_dim']
        x_dim = params['x_dim']
        g_dim = params['g_dim']
        g_l = params['g_layers']
        # activation function
        activation = nn.ReLU()
        if 'leaky_relu' in params:
            activation = nn.LeakyReLU()
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

        # initialisation
        for p in self.parameters():
            if p.ndimension() > 1:
                nn.init.kaiming_normal_(p)  ## seems better ???
                # nn.init.xavier_normal_(p)
                # nn.init.kaiming_uniform_(p, nonlinearity='sigmoid')

    def forward(self, x):
        return self.net(x)
