#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
from matplotlib import pyplot as plt
from gatetools import phsp
import numpy as np
import gaga_phsp as gaga

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('output_filename', nargs=1)
@click.option('--n', '-n', default=1e4, help='Number of samples to generate')
@click.option('-m', default=5, help='Number of xy positions')
@click.option('--plot/--no-plot', default=False)
def gaga_test_gauss(output_filename, plot, n, m):
    """
    \b
    Generate test dataset with several 2D mixture of Gaussian

    \b
    <OUTPUT> : output npy filename
    """

    # nb of values
    n = int(n)

    # init data (must be None)
    data = None

    # print('Gauss type', gtype)
    maxx = m
    for x in range(maxx):
        mean = [x, x / 3.0]
        covar = [[0.005, 0], [0, 0.2]]
        vx = np.ones(n) * x
        vy = np.ones(n) * x / 3.0
        data = gaga.append_gaussian(data, mean, covar, n, vx, vy)
    print(data.shape)

    print('Shuffle ...')
    np.random.shuffle(data)
    print(f'Save in {output_filename}')
    phsp.save_npy(output_filename, data, ['X', 'Y', 'cx', 'cy'])

    # plot
    if plot:
        x = data[:, 0]
        y = data[:, 1]
        plt.scatter(x, y, marker='.', s=0.1)
        plt.axis('equal')
        plt.show()


# --------------------------------------------------------------------------
if __name__ == '__main__':
    gaga_test_gauss()
