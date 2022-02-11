import numpy as np
import torch
from torch.autograd import Variable
import gaga_phsp
import datetime
import time
import garf
import gatetools.phsp as phsp
from scipy.stats import kde
from matplotlib import pyplot as plt
from scipy.stats import entropy
from scipy.spatial.transform import Rotation
import SimpleITK as sitk
import logging
import sys


def plot_epoch(ax, params, optim, filename):
    """
    Plot D loss wrt to epoch
    3 panels : all epoch / first 20% / last 1%
    """

    x1 = np.asarray(optim['d_loss_real'])
    x2 = np.asarray(optim['d_loss_fake'])
    # x = -np.add(x1,x2)
    x = -np.asarray(optim['d_loss'])  # with grad penalty

    epoch = np.arange(params['start_epoch'], params['end_epoch'], 1)
    a = ax  # [0]
    l = filename
    a.plot(epoch, x, '-', label='D_loss (GP) ' + l)
    z = np.zeros_like(x)
    a.set_xlabel('epoch')
    a.plot(epoch, z, '--')
    a.legend()

    # print(params['validation_dataset'])
    if not 'validation_dataset' in params or params['validation_dataset'] == None:
        return
    print('validation')
    x = -np.asarray(optim['validation_d_loss'])
    a.plot(epoch, x, '-', label='Valid')
    a.legend()

    return
    # debug
    a = ax[1]
    n = int(len(x) * 0.2)  # first 20%
    xc = x[0:n]
    a.plot(epoch, x, '-', label='D_loss ' + l)
    z = np.zeros_like(xc)
    a.set_xlabel('epoch')
    a.set_xlim((0, n))
    ymin = np.amin(xc)
    ymax = np.amax(xc)
    a.set_ylim((ymin, ymax))
    a.plot(z, '--')
    a.legend()

    a = ax[2]
    n = max(10, int(len(x) * 0.01))  # last 1%
    xc = x
    a.plot(xc, '.-', label='D_loss ' + l)
    z = np.zeros_like(xc)
    a.set_xlabel('epoch')
    a.set_xlim((len(xc) - n, len(xc)))
    xc = x[len(x) - n:len(x)]
    ymin = np.amin(xc)
    ymax = np.amax(xc)
    a.set_ylim((ymin, ymax))
    a.plot(epoch, z, '--')
    a.legend()


def plot_epoch2(ax, params, optim, filename):
    """
    Plot D loss wrt to epoch
    """

    dlr = np.asarray(optim['d_loss_real'])
    dlf = np.asarray(optim['d_loss_fake'])
    dl = np.asarray(optim['d_loss'])  # with grad penalty
    gl = np.asarray(optim['g_loss'])
    epoch = np.arange(params['start_epoch'], params['end_epoch'], 1)

    # one epoch is when all the training dataset is seen
    step = 1  # int(params['training_size'] / params['batch_size'])
    print(step)
    dlr = dlr[::step]
    dlf = dlf[::step]
    dl = dl[::step]
    gl = gl[::step]
    epoch = epoch[::step] / step

    a = ax  # [0]
    l = filename
    # a.plot(epoch, dlr, '-', label='D_loss_real' + l, alpha=0.5)
    # a.plot(epoch, dlf, '-', label='D_loss_fake' + l, alpha=0.5)
    a.plot(epoch, dl, '-', label='D_loss (GP) ' + l)
    a.plot(epoch, gl, '-', label='G_loss ' + l, alpha=0.5)
    z = np.zeros_like(dl)
    a.set_xlabel('epoch')
    a.plot(epoch, z, '--')
    a.legend()

    # print(params['validation_dataset'])
    if not 'validation_dataset' in params or params['validation_dataset'] is None:
        return
    print('Plot with validation dataset')
    x = np.asarray(optim['validation_d_loss'])
    x = x[::step]
    a.plot(epoch, x, '-', label='Valid ' + l)
    a.legend()

    return
    # debug
    a = ax[1]
    n = int(len(x) * 0.2)  # first 20%
    xc = x[0:n]
    a.plot(epoch, x, '-', label='D_loss ' + l)
    z = np.zeros_like(xc)
    a.set_xlabel('epoch')
    a.set_xlim((0, n))
    ymin = np.amin(xc)
    ymax = np.amax(xc)
    a.set_ylim((ymin, ymax))
    a.plot(z, '--')
    a.legend()

    a = ax[2]
    n = max(10, int(len(x) * 0.01))  # last 1%
    xc = x
    a.plot(xc, '.-', label='D_loss ' + l)
    z = np.zeros_like(xc)
    a.set_xlabel('epoch')
    a.set_xlim((len(xc) - n, len(xc)))
    xc = x[len(x) - n:len(x)]
    ymin = np.amin(xc)
    ymax = np.amax(xc)
    a.set_ylim((ymin, ymax))
    a.plot(epoch, z, '--')
    a.legend()


def plot_epoch_wasserstein(ax, optim, filename):
    """
    Plot wasserstein 
    """

    y = np.asarray(optim['w_value'])
    x = np.asarray(optim['w_epoch'])
    if len(x) < 1:
        return

    a = ax[0].twinx()
    a.plot(x, y, '-', color='r', label='W')
    a.legend()

    a = ax[1].twinx()
    a.plot(x, y, '-', color='r', label='W')
    a.legend()

    a = ax[2].twinx()
    a.plot(x, y, '.-', color='r', label='W')
    a.legend()


def fig_plot_marginal(x, k, keys, ax, i, nb_bins, color, r='', lab=''):
    a = phsp.fig_get_sub_fig(ax, i)
    index = keys.index(k)
    if len(x[0]) > 1:
        d = x[:, index]
    else:
        d = x
    # label = ' {} $\mu$={:.2f} $\sigma$={:.2f}'.format(k, np.mean(d), np.std(d))
    label = f'{lab} {k} $\mu$={np.mean(d):.2f} $\sigma$={np.std(d):.2f}'
    if r != '':
        a.hist(d, nb_bins,
               # density=True,
               histtype='stepfilled',
               facecolor=color,
               alpha=0.5,
               range=r,
               label=label)
    else:
        a.hist(d, nb_bins,
               # density=True,
               histtype='stepfilled',
               facecolor=color,
               alpha=0.5,
               label=label)
    a.set_ylabel('Counts')
    a.legend()


def fig_plot_marginal_2d(x, k1, k2, keys, ax, i, nbins, color):
    a = phsp.fig_get_sub_fig(ax, i)
    index1 = keys.index(k1)
    d1 = x[:, index1]
    index2 = keys.index(k2)
    d2 = x[:, index2]
    label = '{} {}'.format(k1, k2)

    ptype = 'scatter'
    # ptype = 'hist'

    if ptype == 'scatter':
        a.scatter(d1, d2, color=color,
                  alpha=0.25,
                  edgecolor='none',
                  s=1)

    if ptype == 'hist':
        cmap = plt.cm.Greens
        if color == 'r':
            cmap = plt.cm.Reds
        a.hist2d(d1, d2,
                 bins=(nbins, nbins),
                 alpha=0.7,
                 cmap=cmap)

    if ptype == 'density':
        x = d1
        y = d2
        print('kde')
        k = kde.gaussian_kde([x, y])
        xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        a.pcolormesh(xi, yi,
                     zi.reshape(xi.shape),
                     alpha=0.5)

    a.set_xlabel(k1)
    a.set_ylabel(k2)


def fig_plot_diff_2d(x, y, keys, kk, ax, fig, nb_bins):
    k1 = kk[0]
    k2 = kk[1]
    index1 = keys.index(k1)
    index2 = keys.index(k2)
    x1 = x[:, index1]
    x2 = x[:, index2]
    y1 = y[:, index1]
    y2 = y[:, index2]
    label = '{} {}'.format(k1, k2)

    # compute histo
    H_x, xedges_x, yedges_x = np.histogram2d(x1, x2, bins=nb_bins)
    H_y, xedges_y, yedges_y = np.histogram2d(y1, y2, bins=(xedges_x, yedges_x))

    # make diff
    # H = (H_y - H_x)/H_y
    np.seterr(divide='ignore', invalid='ignore')
    H = np.divide(H_y - H_x, H_y)

    # plot
    X, Y = np.meshgrid(xedges_x, yedges_x)
    im = ax.pcolormesh(X, Y, H.T)  # , norm=LogNorm())
    im.set_clim(vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(k1)
    ax.set_ylabel(k2)


def fig_plot_projected(data):
    """
    Debug plot
    """

    b = 100
    x = data[:, 0]
    y = data[:, 1]
    theta = data[:, 2]
    phi = data[:, 3]
    E = data[:, 4]

    f, ax = plt.subplots(2, 2, figsize=(10, 10))

    n, bins, patches = ax[0, 0].hist(theta, b, density=True, facecolor='g', alpha=0.35)
    n, bins, patches = ax[0, 1].hist(phi, b, density=True, facecolor='g', alpha=0.35)
    n, bins, patches = ax[1, 0].hist(E * 1000, b, density=True, facecolor='b', alpha=0.35)
    ax[1, 1].scatter(x, y, color='r', alpha=0.35, s=1)

    ax[0, 0].set_xlabel('Theta angle (deg)')
    ax[0, 1].set_xlabel('Phi angle (deg)')
    ax[1, 0].set_xlabel('Energy (keV)')
    ax[1, 1].set_xlabel('X')
    ax[1, 1].set_ylabel('Y')

    plt.tight_layout()
    plt.show()
