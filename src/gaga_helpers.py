import numpy as np
import torch


''' ----------------------------------------------------------------------------
 Test if pytorch use CUDA. Return type and device
---------------------------------------------------------------------------- '''
def init_pytorch_cuda(gpu_mode, verbose=False):
    if (verbose):
        print('pytorch version: ', torch.__version__)
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



''' ------------------------------------------------------------------------
Print info about a network
'''
def print_network(net):
    num_params = get_network_nb_parameters(net)
    print(net)
    print('Total number of parameters: %d' % num_params)


''' ------------------------------------------------------------------------
Compute total nb of parameters
'''
def get_network_nb_parameters(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params
