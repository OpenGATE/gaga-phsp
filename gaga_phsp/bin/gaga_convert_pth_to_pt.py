#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
import gaga_phsp as gaga
import torch
import os
import json
from torch.autograd import Variable
from types import MethodType

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('pth_filename')
@click.option('--output', '-o', default='auto', help="If 'auto', use filename.pt and filename.json")
@click.option('--gpu/--no-gpu', default=False)
@click.option('--verbose/--no-verbose', '-v', default=False)
@click.option('--keys', '-k', type=(str, float), multiple=True,
              help='Default value for a given key, e.g. -k Z 1000 (for fixed Z plane)')
@click.option('--denorm/--no-denorm', default=True, help='If "true", embed the denormalisation in the Generator.')
@click.option('--epoch', '-e', default=-1, help='Load the G net at the given epoch (-1 for last stored epoch)')
@click.option('--cyl_radius', '-r', default=-1, help='Cylindrical radius in mm (if pth is pairs_tlor)')
@click.option('--cyl_height', default=-1, help='Cylindrical cyl_height in mm (if pth is pairs_tlor)')
def gaga_convert_pth_to_pt(pth_filename, output, gpu, denorm, verbose, keys, epoch, cyl_radius, cyl_height):
    """
    Convert a .pth file from gaga to a .pt and a .json files (used by libtorch in Gate)

    <PTH_FILENAME> : input GAN PTH file (.pth)
    """

    # FIXME harmonize with create_pt for GARF

    # output filename
    if output == 'auto':
        b, extension = os.path.splitext(pth_filename)
        output_pt = b + '.pt'
        output_json = b + '.json'
    else:
        output_pt = output + '.pt'
        output_json = output + '.json'

    # gpu or not ? (default = no gpu in Gate)
    if gpu:
        gpu = 'true'
    else:
        gpu = 'false'

    # load pth
    params, G, D, optim, dtypef = gaga.load(pth_filename, gpu, verbose, epoch=epoch, fatal_on_unknown_keys=False)

    # de-normalization ?
    # if true, the denormalization is performed in G, not in Gate (recommended)
    # if false, the denormalization is performed by Gate
    if denorm:
        params['gate_apply_denormalization'] = 0
    else:
        params['gate_apply_denormalization'] = 1
    if denorm:
        print('Generator: initialize the denormalization during forward. gpu =', gpu)
        G.init_forward_with_denorm(gpu)

    # If pairs, convert tlor to pairs
    keys_list = params['keys_list']
    params['is_pairs'] = 0
    if 'Cx' in keys_list:
        params['is_pairs'] = 1
        if cyl_radius == -1 or cyl_height == -1:
            print('Requires : --cyl_radius and --cyl_height')
            exit()
        params['cyl_radius'] = float(cyl_radius)
        params['cyl_height'] = float(cyl_height)
        params['ignore_directions'] = False
        # The following line means that GATE will not denormalize (it is done here)
        params['gate_apply_denormalization'] = 0
        print('Generator: add a post-process during forward.')
        G.init_forward_with_post_processing(gaga.from_tlor_to_pairs, gpu)
    if 'X1' in keys_list:
        params['is_pairs'] = 1
        print('pth generates pairs of particles')

    print(params)

    b = 100
    z_dim = params['z_dim']
    z = Variable(torch.randn(b, z_dim)).type(dtypef)

    # generate trace
    traced_script_module = torch.jit.trace(G, z)

    try:
        keys_out = params['keys_output']
    except:
        keys_out = params['keys_list']
    print(keys_out)
    params['keys_input'] = params['keys_list']
    params['keys_list'] = keys_out
    traced_script_module.save(output_pt)

    # Save dict nn into json
    params["x_mean"] = params["x_mean"][0].tolist()
    params["x_std"] = params["x_std"][0].tolist()
    params['d_nb_weights'] = int(params['d_nb_weights'])
    params['g_nb_weights'] = int(params['g_nb_weights'])

    # add the default key values in the param
    for k in keys:
        print('Default value', k)
        params[k[0]] = k[1]

    outfile = open(output_json, 'w')
    if verbose:
        print('Writing model', output_pt)
        print('Writing json ', output_json)
    json.dump(params, outfile, indent=4)


# --------------------------------------------------------------------------
if __name__ == '__main__':
    gaga_convert_pth_to_pt()
