#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
import gatetools.phsp as phsp
import numpy as np
# from test_mmd import maximum_mean_discrepancy
from tqdm import tqdm
import gaga_phsp as gaga
from torch.autograd import Variable
# from helpers import *
# from helpers_reconstruction import *
from types import MethodType
import torch
import json
import os
from box import Box

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('pth_filename', nargs=1)
@click.option('--output', '-o', default='AUTO', help='output pth')
@click.option('--gpu/--no-gpu', default=False)
@click.option('-method', '-m', default=3, help='Param method to test (1,2,3,4)')
@click.option('--radius1', '-r', default=170, help='Cylindrical phsp radius in mm')
@click.option('--radius2', '-d', default=300, help='Detector radius in mm')
@click.option('--ignore_direction', default=False, is_flag=True)
def go(pth_filename, output, method, radius1, radius2, gpu, ignore_direction):
    """

    Recreate a pth file with additional re-parametrisation after G

    """

    if gpu:
        gpu = 'true'
    else:
        gpu = 'false'

    # open both GAN
    params, G, D, optim, dtypef = gaga.load(pth_filename, gpu, False)

    input_keys = params['keys_list']
    print('input_keys ', input_keys)

    # normalization
    x_mean = params['x_mean']
    x_std = params['x_std']

    params['tlor'] = True
    G.input_keys = input_keys.copy()
    G.ignore_direction = ignore_direction

    params['keys_list'] = ['t1', 't2', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2',
                           'dX1', 'dY1', 'dZ1', 'dX2', 'dY2', 'dZ2',
                           'E1', 'E2']
    output_keys = params['keys_list']

    # modifs to params
    tlor_params = Box()
    tlor_params.input_keys = input_keys.copy()
    tlor_params.output_keys = output_keys.copy()
    tlor_params.radius1 = radius1
    tlor_params.method = method
    tlor_params.ignore_direction = ignore_direction
    params['tlor'] = tlor_params
    params['is_pairs'] = True
    params['apply_normalization'] = False

    def forward_transform(self, x):
        print('new forward here')
        y = self.net(x)
        y, keys = gaga.convert_tlor_to_pairs_torch(y, self.params)
        print(type(y))
        return y

    print(' ')

    # output filename
    if output == 'auto' or output == 'AUTO':
        b, extension = os.path.splitext(pth_filename)
        output_pt = b + '.pt'
        output_json = b + '.json'
    else:
        output_pt = output + '.pt'
        output_json = output + '.json'

    # convert to pt
    b = 10
    z_dim = params['z_dim']
    z = Variable(torch.randn(b, z_dim)).type(dtypef)

    # index
    print('initial keys', input_keys)
    print('output keys', output_keys)
    index_in_E1 = input_keys.index('E1')
    index_in_t1 = input_keys.index('t1')
    index_out_E1 = output_keys.index('E1')
    index_out_t1 = output_keys.index('t1')
    print('Index in : ', index_in_E1, index_in_t1)
    print('Index out : ', index_out_E1, index_out_t1)

    # first
    print('-' * 50)
    y1 = G(z)
    y1 = y1.data.cpu().numpy()
    y1 = (y1 * x_std) + x_mean
    print('y1', y1.shape)
    print('before y1 E1', y1[:, index_in_E1])
    print('before y1 t1', y1[:, index_in_t1])
    y1, keys = gaga.from_phsp_lor_to_pairs(y1, input_keys, radius1, method, ignore_direction)
    print('y1', y1.shape)
    print('y1 E1', y1[:, index_out_E1])
    print('y1 t1', y1[:, index_out_t1])

    # second
    print('-' * 50)
    G.forward = MethodType(forward_transform, G)  # FIXME
    y2 = G(z)
    print(type(y2))
    y2 = y2.data.cpu().numpy()
    print('y2', y2.shape)
    print('y2 E1', y2[:, index_out_E1])
    print('y2 t1', y2[:, index_out_t1])

    # generate trace
    print('-' * 50)
    traced_script_module = torch.jit.trace(G, z)
    traced_script_module.save(output_pt)

    # Save dict nn into json
    p = {}
    params['x_dim'] = len(keys)
    params["x_mean"] = params["x_mean"][0].tolist()
    params["x_std"] = params["x_std"][0].tolist()
    params['d_nb_weights'] = int(params['d_nb_weights'])
    params['g_nb_weights'] = int(params['g_nb_weights'])

    for k in keys:
        params[k[0]] = k[1]
    print(params)

    print('Writing ', output_json)
    outfile = open(output_json, 'w')
    json.dump(params, outfile, indent=4)

    print('done')


# --------------------------------------------------------------------------
if __name__ == '__main__':
    go()
