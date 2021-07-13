import os
import torch
import gatetools.phsp as phsp


def update_params(params, user_param):
    for up in user_param:
        params[up[0]] = up[1]


def select_keys2(x, params, read_keys):
    """
    Consider the keys chosen by the user in params.
    Select among the read_keys of x
    Add params.keys_list
    """
    if 'keys' not in params:
        params.keys_list = read_keys
        return read_keys, x

    keys_list = phsp.str_keys_to_array_keys(params['keys'])

    if 'skeys' in params:
        skeys = params['skeys']
        print('special case skeys', skeys)
        s = skeys.replace('_', ' ')
        keys_list = phsp.str_keys_to_array_keys(s)
        print(keys_list)

    x = phsp.select_keys(x, read_keys, keys_list)
    params.keys_list = keys_list

    return x


def auto_output_filename(params, output, output_folder):
    if output != 'auto':
        params.output_filename = output
        return
    # create an output name with some params
    b, extension = os.path.splitext(os.path.basename(params.params_filename))
    output = f'{b}_{params.penalty}_{params.penalty_weight}_{params.end_epoch}.pth'
    params.output_filename = os.path.join(output_folder, output)


def get_RMSProp_optimisers(self, p):
    """
        momentum (float, optional) – momentum factor (default: 0)
        alpha (float, optional) – smoothing constant (default: 0.99)
        centered (bool, optional) – if True, compute the centered RMSProp,
                 the gradient is normalized by an estimation of its variance
        weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
    """

    d_learning_rate = p['d_learning_rate']
    g_learning_rate = p['g_learning_rate']

    if 'RMSprop_d_momentum' not in p:
        p['RMSprop_d_momentum'] = 0
    if 'RMSprop_g_momentum' not in p:
        p['RMSprop_g_momentum'] = 0

    if 'RMSProp_d_alpha' not in p:
        p['RMSProp_d_alpha'] = 0.99
    if 'RMSProp_g_alpha' not in p:
        p['RMSProp_g_alpha'] = 0.99

    if 'RMSProp_d_weight_decay' not in p:
        p['RMSProp_d_weight_decay'] = 0
    if 'RMSProp_g_weight_decay' not in p:
        p['RMSProp_g_weight_decay'] = 0

    if 'RMSProp_d_centered' not in p:
        p['RMSProp_d_centered'] = False
    if 'RMSProp_g_centered' not in p:
        p['RMSProp_g_centered'] = False

    RMSprop_d_momentum = p['RMSprop_d_momentum']
    RMSprop_g_momentum = p['RMSprop_g_momentum']
    RMSProp_d_alpha = p['RMSProp_d_alpha']
    RMSProp_g_alpha = p['RMSProp_g_alpha']
    RMSProp_d_weight_decay = p['RMSProp_d_weight_decay']
    RMSProp_g_weight_decay = p['RMSProp_g_weight_decay']
    RMSProp_d_centered = p['RMSProp_d_centered']
    RMSProp_g_centered = p['RMSProp_g_centered']

    d_optimizer = torch.optim.RMSprop(self.D.parameters(),
                                      lr=d_learning_rate,
                                      momentum=RMSprop_d_momentum,
                                      alpha=RMSProp_d_alpha,
                                      weight_decay=RMSProp_d_weight_decay,
                                      centered=RMSProp_d_centered)
    g_optimizer = torch.optim.RMSprop(self.G.parameters(),
                                      lr=g_learning_rate,
                                      momentum=RMSprop_g_momentum,
                                      alpha=RMSProp_g_alpha,
                                      weight_decay=RMSProp_g_weight_decay,
                                      centered=RMSProp_g_centered)

    return d_optimizer, g_optimizer
