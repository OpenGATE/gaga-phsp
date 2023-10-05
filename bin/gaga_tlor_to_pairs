#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
import gatetools as gt
from gatetools import phsp
import gaga_phsp as gaga
from matplotlib import pyplot as plt
import torch
from torch import Tensor
import numpy as np
from garf.helpers import get_gpu_device

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("tlor_filename", nargs=1)
@click.option(
    "--radius", "-r", required=True, type=float, help="Sphere phsp radius in mm"
)
@click.option("-n", default="-1", help="Number of samples")
@click.option(
    "--shuffle",
    "-s",
    is_flag=True,
    default=False,
    help="Shuffle the n samples (slow if file is large)",
)
@click.option("--plot", is_flag=True, default=False, help="Plot for debug")
@click.option("--output", "-o", required=True, help="output filename (npy)")
def go(tlor_filename, radius, n, output, shuffle, plot):
    """
    Convert a phsp file that contains pairs of particles parameterized with tlor
    into phsp with pairs of particles. It is assumed that all particles are on a sphere of the
    given radius

    tlor : Cx Cy Cz Vx Vy Vz Dx Dy Dz Wx Wy Wz t1 t2 t3 E1 E2
    pairs: t1 t2 Ax Ay Az Bx By Bz dAx dAy dAz dBx dBy dBz E1 E2
    """

    # read data
    n = int(float(n))
    phsp, keys, m = gt.phsp.load(tlor_filename, nmax=n, shuffle=shuffle)
    if n == -1:
        n = m
    print("Input ", tlor_filename, n, keys)

    # parameters for conversion
    params = {"radius": float(radius), "keys_list": keys, "ignore_directions": False}

    # convert numpy to torch --> this function may be embedded in the Generator for Gate
    # (I need to copy, otherwise : 'warning NumPy array is not writeable')
    current_gpu_mode, current_gpu_device = get_gpu_device("auto")
    phsp = Tensor(torch.from_numpy(phsp.copy()).to(current_gpu_device))

    # convert tlor to pairs
    x = gaga.from_tlor_to_pairs(phsp, params)
    keys_out = params["keys_output"]
    print(f"Output Keys {keys_out}")
    print(f"Output conversion: {len(x)}/{n}")

    # convert back to numpy and save
    x = x.cpu().data.numpy()
    gt.phsp.save_npy(output, x, keys_out)

    # torch to numpy
    phsp = phsp.cpu().data.numpy()

    # plot
    if plot:
        if n > 100:
            print(f"Too much samples={n} to plot. Considering the 10 first only")
            n = 10
            phsp = phsp[:n, :]
            x = x[:n, :]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        gaga.plot_sphere_LOR(ax, phsp, keys, x, keys_out, radius)
        plt.show()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    go()
