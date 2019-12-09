import torch
import torch.nn as nn
from torch.autograd import grad as torch_grad
from torch.autograd import Variable

'''
Set of helpers functions for losses an penalties
'''


# -----------------------------------------------------------------------------
class WassersteinLoss(torch.nn.Module):
    '''
    Wasserstein Loss
    '''
    
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, x, y):        
        return torch.mean(x)


# -----------------------------------------------------------------------------
class WassersteinNegLoss(torch.nn.Module):
    '''
    Wasserstein Loss
    '''
    
    def __init__(self):
        super(WassersteinNegLoss, self).__init__()

    def forward(self, x, y):        
        return -torch.mean(x)


# -----------------------------------------------------------------------------
def zero_penalty(self, real_data, fake_data):
    '''
    No penalty
    '''    
    return 0


# -----------------------------------------------------------------------------
def clamp_parameters(self):
    '''
    Clamp
    '''
    for p in self.D.parameters():
        p.data.clamp_(self.clamp_lower, self.clamp_upper)


# -----------------------------------------------------------------------------
def get_interpolated_gradient(self, real_data, fake_data):
    '''
    Common function to gradient penalty functions
    '''
    alpha = torch.rand(self.batch_size, 1)#, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()
    
    # interpolated
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = self.D(interpolated)
    
    # gradient
    gradients = torch_grad(outputs=prob_interpolated,
                           inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, # needed ?
                           retain_graph=True, # needed ?
                           only_inputs=True)[0]
    return gradients



# -----------------------------------------------------------------------------
def gradient_penalty(self, real_data, fake_data):
    '''
    Gulrajani2017 $(||\nabla_a D(a)||_2 - 1)^2$
    with a = interpolated samples.
    Called two-sided penalty.
    '''

    # gradient
    gradients = get_interpolated_gradient(self, real_data, fake_data)

    # norm
    #gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradients_norm = gradients.norm(2,dim=1)

    # Two sides penalty
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()

    return gradient_penalty


# -----------------------------------------------------------------------------
def gradient_penalty_max(self, real_data, fake_data):
    '''
    Petzka2018 $(max||\nabla_a D(a)||_2 - 1)^2$
    with a = interpolated samples.
    '''

    # gradient
    gradients = get_interpolated_gradient(self, real_data, fake_data)

    # norm
    #gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradients_norm = gradients.norm(2,dim=1)

    # max
    gradient_penalty = (torch.max(gradients_norm - 1, torch.zeros_like(gradients_norm))**2).mean()

    return gradient_penalty


