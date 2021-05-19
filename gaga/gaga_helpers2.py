import os

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
    x = phsp.select_keys(x, read_keys, keys_list)
    params.keys_list = keys_list

    return x


def auto_output_filename(params, output, output_folder):
    if output != 'auto':
        params.output_filename = output
        return
    # create an output name with some params
    b, extension = os.path.splitext(os.path.basename(params.params_filename))
    output = f'{b}_{params.penalty}_{params.penalty_weight}_{params.epoch}.pth'
    params.output_filename = os.path.join(output_folder, output)
