#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from shutil import copyfile
import click
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from gatetools import phsp
import numpy as np
import gaga_phsp as gaga

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('output_filename', nargs=1)
@click.option('--n', '-n', default=1e4, help='Number of samples to generate')
@click.option('--plot/--no-plot', default=False)
@click.option('--gtype', '-t', default='v1', help='Type : v1, v2 etc')
def gaga_test_gauss(output_filename, plot, n, gtype):
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
    if gtype == 'v2':
        # Gauss1 (half of n)
        data = gaga.append_gaussian(data, [5, 0], [[5, 0], [0, 10]], int(n * 0.5))
        # Gauss 2
        data = gaga.append_gaussian(data, [2, 2], [[2, 0], [0, 1]], int(n * 0.35))
        # Gauss 3
        data = gaga.append_gaussian(data, [-4, -2], [[0.01, 0], [0, 1]], int(n * 0.15))
        # Gauss 4
        # data = gaga.append_gaussian(data, [0, -3], [[0.1, 0], [0, 0.2]], int(n * 0.2))

    if gtype == 'v1':
        data = gaga.append_gaussian(data, [-2, 3], [[5, 0], [0, 2]], n)

    np.random.shuffle(data)
    x = data[:, 0]
    y = data[:, 1]
    phsp.save_npy(output_filename, data, ['X', 'Y'])

    # plot
    if plot:
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        plt.scatter(x, y, c=z, marker='.')
        plt.axis('equal')
        axes = plt.gca()
        axes.set_xlim([-15, 15])
        axes.set_ylim([-15, 15])
        plt.show()


# --------------------------------------------------------------------------
if __name__ == '__main__':
    gaga_test_gauss()
