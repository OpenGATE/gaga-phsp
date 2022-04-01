#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
import gatetools as gt
from gatetools import phsp
import gaga_phsp as gaga
from matplotlib import pyplot as plt

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('pairs_filename', nargs=1)
@click.option('-n', default='-1', help='Number of samples')
@click.option('--shuffle', '-s', is_flag=True, default=False, help='Shuffle the n samples (slow if file is large)')
@click.option('--plot', is_flag=True, default=False, help='Plot for debug')
@click.option('--radius', '-r', required=False, default=210.0, help='ONLY FOR PLOT: phsp sphere radius in mm')
@click.option('--output', '-o', required=True, help='output filename (npy)')
def go(pairs_filename, n, output, shuffle, plot, radius):
    """
        input : consider A (X1, Y1, Z1) and B (X2, Y2, Z2) points
        output: parametrize with timed-LOR

        Input: t1 t2 X1 Y1 Z1 X2 Y2 Z2 dX1 dY1 dZ1 dX2 dY2 dZ2 E1 E2
        Output:  Cx Cy Cz Vx Vy Vz dAx dAy dAz dBx dBy dBz tt E1 E2

        C is on the A to B segment, with the exact position according to t1/t2
        tt is t1+t2

    """

    # read data
    n = int(float(n))
    phsp, keys, m = gt.phsp.load(pairs_filename, nmax=n, shuffle=shuffle)
    print('Input ', pairs_filename, n, keys)

    # convert pairs to lor
    params = {'keys_list': keys}
    x, keys_out = gaga.from_pairs_to_tlor(phsp, params)

    # save
    n = len(x)
    print('Output', n, keys_out)
    gt.phsp.save_npy(output, x, keys_out)

    # debug
    if plot:
        if n > 100:
            print(f'Too much samples={n} to plot. Considering the 10 first only')
            n = 10
            phsp = phsp[:n, :]
            x = x[:n, :]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        gaga.plot_sphere_LOR(ax, x, keys_out, phsp, keys, radius=radius)
        plt.show()


# --------------------------------------------------------------------------
if __name__ == '__main__':
    go()
