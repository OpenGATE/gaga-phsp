import numpy as np
import torch
import gaga
import datetime
import os

''' ---------------------------------------------------------------------------------- '''
'''
date format
'''
date_format = "%Y-%m-%d %H:%M:%S"


''' ---------------------------------------------------------------------------------- '''
def init_pytorch_cuda(gpu_mode, verbose=False):
    '''
    Test if pytorch use CUDA. Return type and device
    '''
    
    if (verbose):
        print('pytorch version', torch.__version__)
    dtypef = torch.FloatTensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if (gpu_mode == 'auto'):
        if (torch.cuda.is_available()):
            dtypef = torch.cuda.FloatTensor
    elif (gpu_mode == 'true'):
        if (torch.cuda.is_available()):
            dtypef = torch.cuda.FloatTensor
        else:
            print('Error GPU mode not available')
            exit(0)
    else:
        device = torch.device('cpu');

    if (verbose):
        if (str(device) != 'cpu'):
            print('GPU is enabled')
            # print('CUDA version: ', torch.version.cuda)
            # print('CUDA device name:', torch.cuda.get_device_name(0))
            # print('CUDA current device: ', torch.cuda.current_device())
            # print('CUDA device:', torch.cuda.device(0))
            # print('CUDA device counts: ', torch.cuda.device_count())
        else:
            print('CPU only (no GPU)')

    return dtypef, device



''' ---------------------------------------------------------------------------------- '''
def print_network(net):
    '''
    Print info about a network
    '''

    num_params = get_network_nb_parameters(net)
    print(net)
    print('Total number of parameters: %d' % num_params)


''' ---------------------------------------------------------------------------------- '''
def get_network_nb_parameters(net):
    '''
    Compute total nb of parameters
    '''
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params


''' ---------------------------------------------------------------------------------- '''
def print_info(params, optim):
    '''
    Print info about a trained GAN-PHSP
    '''
    # print parameters
    for e in params:
        if (e[0] != '#') and (e != 'x_mean') and (e != 'x_std'):
            print('   {:20s} {}'.format(e, str(params[e])))

    # additional info
    try:
        start = datetime.datetime.strptime(params['start date'], gaga.date_format)
        end = datetime.datetime.strptime(params['end date'], gaga.date_format)
    except:
        start = 0
        end = 0
    delta = end-start
    print('   {:20s} {}'.format('Duration', delta))

    d_loss_real = np.asarray(optim['d_loss_real'][-1])
    d_loss_fake = np.asarray(optim['d_loss_fake'][-1])
    d_loss = d_loss_real + d_loss_fake
    g_loss = np.asarray(optim['g_loss'][-1])
    print('   {:20s} {}'.format('Final d_loss', d_loss))
    print('   {:20s} {}'.format('Final g_loss', g_loss))


''' ---------------------------------------------------------------------------------- '''
def load(filename):
    '''
    Load a GAN-PHSP
    Output params   = dict with all parameters
    Output G        = Generator network
    Output optim    = dict with information of the training process
    '''
    
    dtypef, device = init_pytorch_cuda('auto', True)
    if (str(device) == 'cpu'):
        nn = torch.load(filename, map_location=lambda storage, loc: storage)
    else:
        nn = torch.load(filename)

    # get elements
    params = nn['params']
    if not 'optim' in nn:
        optim = nn['model'] ## FIXME compatibility --> to remove
    else:
        optim =  nn['optim']
    G_state = nn['g_model_state']

    # create the Generator
    z_dim = params['z_dim']
    x_dim = params['x_dim']
    g_dim = params['g_dim']
    g_layers = params['g_layers']

    G = gaga.Generator(z_dim, x_dim, g_dim, g_layers)
    if (str(device) != 'cpu'):
        G.cuda()
    G.load_state_dict(G_state)

    return params, G, optim



    
