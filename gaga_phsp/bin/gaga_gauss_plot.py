#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
import gatetools.phsp as phsp
import numpy as np
from matplotlib import pyplot as plt
import gaga_phsp as gaga
from scipy.stats import gaussian_kde

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("phsp_filename")
@click.argument("pth_filename")
@click.option("--n", "-n", default=1e4, help="Number of samples to get from the phsp")
@click.option(
    "--m", "-m", default=1e4, help="Number of samples to generate from the GAN"
)
@click.option("-x", default=float(1), help="Condition x")
@click.option("-y", default=float(1), help="Condition y")
@click.option("--output", "-o", default="cond.png", help="Output image")
@click.option(
    "--epoch",
    "-e",
    default=-1,
    help="Load the G net at the given epoch (-1 for last stored epoch)",
)
def gaga_gauss_plot(phsp_filename, pth_filename, n, m, epoch, x, y, output):
    """
    \b
    Plot 2D mixture of Gaussian ref in phsp, gan in pth

    \b
    <PHSP_FILENAME>   : input reference phase space file PHSP file (.npy)
    <PTH_FILENAME>    : input GAN PTH file (.pth)
    """

    # nb of values
    n = int(n)
    m = int(m)

    # load phsp
    real, r_keys, mm = phsp.load(phsp_filename, nmax=n, shuffle=True)

    # load pth
    params, G, D, optim = gaga.load(pth_filename, epoch=epoch)

    # generate samples with condition
    cond = None
    if len(params["cond_keys"]) > 0:
        condx = np.ones(m) * x
        condy = np.ones(m) * y
        print(condx.shape, condy.shape)
        cond = np.column_stack((condx, condy))
        print(cond.shape)
        fake = gaga.generate_samples3(params, G, m, cond=cond)
    else:
        fake = gaga.generate_samples_non_cond(params, G, m, m, False, True)

    # get 2D points
    x_ref = real[:, 0]
    y_ref = real[:, 1]
    x = fake[:, 0]
    y = fake[:, 1]
    print("ref shape", x_ref.shape, y_ref.shape)
    print("gan shape", x.shape, y.shape)

    print("ref y min max", y_ref.min(), y_ref.max())
    print("ref x min max", x_ref.min(), x_ref.max())

    print("gan y min max", y.min(), y.max())
    print("gan x min max", x.min(), x.max())

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    a = ax
    a.scatter(x_ref, y_ref, marker=".", s=0.1)
    a.scatter(x, y, marker=".", s=0.1)
    a.axis("equal")

    plt.title(pth_filename)
    print(output)
    plt.savefig(output)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    gaga_gauss_plot()
