import torch
import torch.nn as nn
from torch.autograd import grad as torch_grad
from torch.autograd import Variable
import numpy as np

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


def langevin_latent_sampling(G, D, params, z):
    print('generate with langevin')
    n = len(z)
    print('size', n)

    """
    line 85 run_synthetic.py
    line 49 langevin.py
    https://github.com/clear-nus/DGflow/blob/main/dgflow.py
    """
    # params
    n_steps = params['lgs_nb_steps']
    eta = params['lgs_lr']
    gamma = params['lgs_gamma']
    f = params['lgs_f']

    # parameters (will be in params)
    # n_steps = 20  # 50000 in che2020 ? ; 25 in ansari2020
    # alpha = 1.0  # for rejection
    # step_lr = 1e-4  ## see eta
    # eps_std = 1e-2
    # eta = 0.1  # step_lr = step_size
    # f = 'KL'  # KL logD JS ; used in "velocity"
    # gamma = 0.02  # diffusion term

    fn = None
    if f == 'KL':
        fn = lambda d_score: torch.ones_like(d_score.detach())
    if f == 'logD':
        fn = lambda d_score: 1 / (1 + d_score.detach().exp())
    if f == 'JS':
        fn = lambda d_score: 1 / (1 + 1 / d_score.detach().exp())
    if not fn:
        print('ERROR lgs_f must be KL, LogD or JS')
        return z

    noise_factor = np.sqrt(gamma)
    print('langevin nb steps ', n_steps)
    print('langevin lr       ', eta)
    print('langevin gamma    ', gamma)
    print('langevin f        ', f)

    """
    KL-divergence by setting f = r log r 
    ==> equivalent to DDLS Che2020 ? NO 
    s = torch.ones_like(d_score.detach())
    
    f == 'logD' => s = 1 / (1 + d_score.detach().exp())
    Discriminator Optimal Transport (DOT) (Tanaka, 2019) ; with λ = 1/2 ? unsure 
    NO it is different 
    
    f == 'JS' => s = 1 / (1 + 1 / d_score.detach().exp())
    proposed by ansari2020 ?
    
    DGflow performs well even without the diffusion term (i.e., with γ = 0)
    """

    def velocity(z):
        z_t = z.clone()
        z_t.requires_grad_(True)
        if z_t.grad is not None:
            z_t.grad.zero_()
        x_t = G(z_t)
        d_score = D(x_t)
        # s = torch.ones_like(d_score.detach())
        # s = 1 / (1 + d_score.detach().exp())
        # s = 1 / (1 + 1 / d_score.detach().exp())  # FIXME use f
        s = fn(d_score)
        s.expand_as(z_t)
        d_score.backward(torch.ones_like(d_score).to(z.device))
        grad = z_t.grad
        return s.data * grad.data

    # loop on number of steps
    for i in range(n_steps):
        print(i, len(z))
        v = velocity(z)
        # FIXME warning randn !
        z = z.data + eta * v + np.sqrt(2 * eta) * noise_factor * torch.randn_like(z)
        """
            v = velocity(z)  <--- grad ?
            z = z.data + eta * v + np.sqrt(2*eta) * noise_factor * torch.randn_like(z)
        """
        """
            eps = eps_std * xp.random.randn(batch_size, z_dim).astype(xp.float32)
            E, grad = e_grad(z, P, gen, dis, alpha, ret_e=True)
            z = z - step_lr * grad[0] + eps
        """

    return z
