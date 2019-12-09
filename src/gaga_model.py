import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
class Discriminator(nn.Module):
    '''
    Discriminator: D(x, θD) -> probability that x is real data

    Or with Wasserstein GAN :
    Discriminator is called Critic D(x, θD) -> Wasserstein distance

    The discriminator takes in both real and fake input data and returns
    probabilities, a number between 0 and 1, with 1 representing a prediction
    of authenticity and 0 representing fake.

    At Nash equilibrium, half of input will be real, half fake: D(x) = 1/2 (?)

    Input:
        - x_dim
        - d_dim
        - d_layers
        - leaky_relu  : boolean (relu if False, leaky_relu if True)
        - layer_norm  : boolean
        - loss_type   : special case for 'wasserstein'
    
    '''

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.params = params
        x_dim = params['x_dim']
        d_dim = params['d_dim']
        self.d_layers = params['d_layers']

        self.map1 = nn.Linear(x_dim, d_dim)
        self.maps = nn.ModuleList()
        self.norms = nn.ModuleList()

        activ = F.relu
        if 'leaky_relu' in params:
            activ = F.leaky_relu

        for i in range(self.d_layers):
            self.maps.append(nn.Linear(d_dim, d_dim))
            self.norms.append(nn.LayerNorm(d_dim))

        self.map3 = nn.Linear(d_dim, 1)
        self.activation_fct = activ

    def forward(self, x):
        activ = self.activation_fct
        x = activ(self.map1(x))

        if self.params['layer_norm'] == True:
            for i in range(self.d_layers):
                x = activ(self.norms[i](self.maps[i](x)))
        else:
            for i in range(self.d_layers):
                x = activ(self.maps[i](x))

        if self.params['loss_type'] == 'wasserstein':
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

    Input:
        - z_dim
        - x_dim
        - d_layers
        - leaky_relu  : boolean (relu if False, leaky_relu if True)
        - layer_norm  : boolean
        - loss_type   : special case for 'wasserstein'
    
    '''

    def __init__(self, params):
        super(Generator, self).__init__()

        self.params = params
        z_dim = self.params['z_dim']
        self.x_dim = self.params['x_dim']
        g_dim = self.params['g_dim']
        self.g_layers = self.params['g_layers']

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

        return x



