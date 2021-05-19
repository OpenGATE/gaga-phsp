import torch
import torch.nn as nn
from torch.autograd import grad as torch_grad
from torch.autograd import Variable

'''
Set of helpers functions for losses an penalties
'''


class WassersteinLoss(torch.nn.Module):
    """
    Wasserstein Loss
    """

    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, x, y):
        return torch.mean(x)


class WassersteinNegLoss(torch.nn.Module):
    """
    Wasserstein Loss
    """

    def __init__(self):
        super(WassersteinNegLoss, self).__init__()

    def forward(self, x, y):
        return -torch.mean(x)


class HingeLoss(torch.nn.Module):
    """
    Hinge Loss
    """

    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, x, y):
        return torch.mean(torch.nn.ReLU()(1.0 + x))


class HingeNegLoss(torch.nn.Module):
    """
    Hinge Loss
    """

    def __init__(self):
        super(HingeNegLoss, self).__init__()

    def forward(self, x, y):
        return torch.mean(torch.nn.ReLU()(1.0 - x))


def zero_penalty(self, real_data, fake_data):
    """
    No penalty
    """
    return 0


def clamp_parameters(self):
    """
    Clamp
    """
    for p in self.D.parameters():
        p.data.clamp_(self.clamp_lower, self.clamp_upper)


def get_interpolated_gradient(self, real_data, fake_data):
    """
    Common function to gradient penalty functions
    """
    alpha = torch.rand(self.batch_size, 1)  # , 1, 1)
    alpha = alpha.expand_as(real_data)
    if str(self.device) != 'cpu':
        alpha = alpha.cuda()

    # interpolated
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated = Variable(interpolated, requires_grad=True)
    if str(self.device) != 'cpu':
        interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = self.D(interpolated)

    # gradient
    if str(self.device) != 'cpu':
        ones = torch.ones(prob_interpolated.size()).cuda()
    else:
        ones = torch.ones(prob_interpolated.size())

    gradients = torch_grad(outputs=prob_interpolated,
                           inputs=interpolated,
                           grad_outputs=ones,
                           create_graph=True,  # needed ?
                           retain_graph=True,  # needed ?
                           only_inputs=True)[0]

    # gradients.requires_grad_(False) # FIXME
    return gradients


def gradient_penalty(self, real_data, fake_data):
    """
    Gulrajani2017 $(||\nabla_a D(a)||_2 - 1)^2$
    with a = interpolated samples.
    Called two-sided penalty.
    1-GP = one centered GP.
    """

    # gradient
    gradients = get_interpolated_gradient(self, real_data, fake_data)

    # norm
    # gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradients_norm = gradients.norm(2, dim=1)

    # Two sides penalty
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()

    return gradient_penalty


def gradient_penalty_centered(self, real_data, fake_data):
    """
    Thanh-Tung2019 $(||\nabla_a D(a)||_2)^2$
    with a = interpolated samples.
    0-GP = zero centered GP.
    """

    # gradient
    gradients = get_interpolated_gradient(self, real_data, fake_data)

    # norm
    # gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradients_norm = gradients.norm(2, dim=1)

    # Two sides penalty
    gradient_penalty = ((gradients_norm) ** 2).mean()

    return gradient_penalty


def gradient_penalty_max(self, real_data, fake_data):
    """
    Petzka2018 $(max||\nabla_a D(a)||_2 - 1)^2$
    with a = interpolated samples.
    """

    # gradient
    gradients = get_interpolated_gradient(self, real_data, fake_data)

    # norm
    # gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradients_norm = gradients.norm(2, dim=1)

    # max FIXME why  **2 ????
    gradient_penalty = (torch.max(gradients_norm - 1, torch.zeros_like(gradients_norm)) ** 2).mean()
    # ? gradient_penalty = torch.nn.ReLU()(gradients_norm - 1, torch.zeros_like(gradients_norm))**2).mean()

    return gradient_penalty


def gradient_penalty_linf_hinge(self, real_data, fake_data):
    """
    Jolicoeur2019 (TEST)
    """

    # gradient
    gradients = get_interpolated_gradient(self, real_data, fake_data)

    # L_inf margin
    gradients_norm = gradients.norm(1, dim=1)

    # hinge
    gradient_penalty = torch.nn.ReLU()(gradients_norm - 1).mean()

    return gradient_penalty


def gradient_penalty_linf_hinge_abs(self, real_data, fake_data):
    """
    Jolicoeur2019 (TEST), idem with abs
    """

    # gradient
    gradients = get_interpolated_gradient(self, real_data, fake_data)

    # abs gradients
    gradients = torch.abs(gradients)

    # L_inf margin
    gradients_norm = gradients.norm(1, dim=1)

    # hinge
    gradient_penalty = torch.nn.ReLU()(gradients_norm - 1).mean()

    return gradient_penalty


def GP_L1_LS(self, real_data, fake_data):
    # gradient
    gradients = get_interpolated_gradient(self, real_data, fake_data)
    # norm
    gradients_norm = gradients.norm(1, dim=1)
    # penalty
    gradient_penalty = (gradients_norm - 1).pow(2)
    return gradient_penalty.mean()


def GP_L2_LS(self, real_data, fake_data):
    # Gulrajani2017 $(||\nabla_a D(a)||_2 - 1)^2$
    # gradient
    gradients = get_interpolated_gradient(self, real_data, fake_data)
    # norm
    gradients_norm = gradients.norm(2, dim=1)
    # penalty
    gradient_penalty = (gradients_norm - 1).pow(2)
    return gradient_penalty.mean()


def GP_Linf_LS(self, real_data, fake_data):
    # gradient
    gradients = get_interpolated_gradient(self, real_data, fake_data)
    # norm
    gradients_norm = gradients.norm(p=float("inf"), dim=1)
    # inf norm is defined as max(sum(abs(x), axis=1)) for matrix and max(abs(x)) for vector
    # https://github.com/pytorch/pytorch/issues/24802
    # grad_abs = torch.abs(grad)  # Absolute value of gradient
    # grad_norm, _ = torch.max(grad_abs, 1)
    # penalty
    gradient_penalty = (gradients_norm - 1).pow(2)
    return gradient_penalty.mean()


def GP_L1_Hinge(self, real_data, fake_data):
    # gradient
    gradients = get_interpolated_gradient(self, real_data, fake_data)
    # norm
    gradients_norm = gradients.norm(1, dim=1)
    # penalty
    gradient_penalty = torch.nn.ReLU()(gradients_norm - 1)
    return gradient_penalty.mean()


def GP_L2_Hinge(self, real_data, fake_data):
    # Gulrajani2017 $(||\nabla_a D(a)||_2 - 1)^2$
    # gradient
    gradients = get_interpolated_gradient(self, real_data, fake_data)
    # norm
    gradients_norm = gradients.norm(2, dim=1)
    # penalty
    gradient_penalty = torch.nn.ReLU()(gradients_norm - 1)
    return gradient_penalty.mean()


def GP_Linf_Hinge(self, real_data, fake_data):
    # gradient
    gradients = get_interpolated_gradient(self, real_data, fake_data)
    # norm
    gradients_norm = gradients.norm(p=float("inf"), dim=1)
    # inf norm is defined as max(sum(abs(x), axis=1)) for matrix and max(abs(x)) for vector
    # https://github.com/pytorch/pytorch/issues/24802
    # grad_abs = torch.abs(grad)  # Absolute value of gradient
    # grad_norm, _ = torch.max(grad_abs, 1)
    # penalty
    gradient_penalty = torch.nn.ReLU()(gradients_norm - 1)
    return gradient_penalty.mean()


def GP_0GP(self, real_data, fake_data):
    # thanh-tung2019, zero-centered
    # gradient
    gradients = get_interpolated_gradient(self, real_data, fake_data)
    # norm
    gradients_norm = gradients.norm(2, dim=1)
    # penalty
    gradient_penalty = (gradients_norm).pow(2)
    return gradient_penalty.mean()


def GP_SquareHinge(self, real_data, fake_data):
    # petzka2018, square hinge
    # gradient
    gradients = get_interpolated_gradient(self, real_data, fake_data)
    # norm
    gradients_norm = gradients.norm(2, dim=1)
    # penalty
    gradient_penalty = (torch.nn.ReLU()(gradients_norm - 1)).pow(2)
    return gradient_penalty.mean()
