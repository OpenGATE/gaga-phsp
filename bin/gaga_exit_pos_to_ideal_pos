#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
import gatetools as gt
from gatetools import phsp
import gaga_phsp as gaga

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('dataset_filename', nargs=1)
@click.option('-n', default='-1', help='Number of samples')
@click.option('--shuffle', '-s', is_flag=True, default=False, help='Shuffle the n samples (slow if file is large)')
@click.option('--output', '-o', required=True, help='output filename (npy)')
def go(dataset_filename, n, output, shuffle):
    """
        input : consider input key exit position X,Y,Z
        output: parametrize with ideal position = p - c x t x dir

        Input: PrePosition_X PreDirection_X TimeFromBeginOfEvent
        Output: IdealPosition_X (+idem)
    """

    # read data
    n = int(float(n))
    phsp, keys, m = gt.phsp.load(dataset_filename, nmax=n, shuffle=shuffle)
    print('Input', dataset_filename, n)
    print('Input keys', keys)

    # convert param
    params = {'keys_list': keys}
    x, keys_out = gaga.from_exit_pos_to_ideal_pos(phsp, params)

    # save
    n = len(x)
    print('Output', output, n)
    print('Output keys', keys_out)
    gt.phsp.save_npy(output, x, keys_out)


# --------------------------------------------------------------------------
if __name__ == '__main__':
    go()
