#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import uproot
import click
import gatetools as gt
import gatetools.phsp as phsp
from matplotlib import pyplot as plt
import numpy as np
import gaga_phsp as gaga
import os
import scipy

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('phsp_filename', nargs=1)
@click.option('--output', '-o', default='auto', help='output filename (npy)')
@click.option('--n', '-n', default='-1', help='Number of samples to read (-1 for all)')
def go(phsp_filename, tree, output, unpaired, n):
    # read the phsp file
    n = int(float(n))
    data, keys, m = phsp.load(phsp_filename, treename=tree, nmax=n)
    if n == -1:
        n = m
    print(f'PHSP read with {n}/{m}')
    print(f'{keys}')

    # get some branches
    event_id = data[:, keys.index('EventID')]
    ve = data[:, keys.index('KineticEnergy')]
    vpos = data[:, keys.index('PrePosition_X'):keys.index('PrePosition_Z') + 1]
    vdir = data[:, keys.index('PreDirection_X'):keys.index('PreDirection_Z') + 1]
    vt = data[:, keys.index('TimeFromBeginOfEvent')]
    vp0 = data[:, keys.index('EventPosition_X'):keys.index('EventPosition_Z') + 1]
    vd0 = data[:, keys.index('TrackVertexMomentumDirection_X'):keys.index('TrackVertexMomentumDirection_Z') + 1]

    # count
    print('Count the nb of unpaired, pairs etc ...')
    unique, counts = np.unique(event_id, return_counts=True)
    u, c = np.unique(counts, return_counts=True)
    print(u)
    max_u = u[len(u) - 1]
    print('Count max: ', c, max_u)
    info = [f'{c[i] * (i + 1) / n * 100}%' for i in range(len(c))]
    print('Count max: ', info)
    ign = [c[i] * (i + 1) for i in range(2, len(c))]
    ign = np.array(ign).sum()

    # store pairs, unpaired independently
    absorbed = []
    singles = []
    pairs1 = []
    pairs2 = []
    nb_ignored = 0

    # main loop, 'i' is the index of the current hit
    for i in range(len(event_id)):
        # skip negative event id (a negative index means: already considered)
        eid = event_id[i]
        if eid < 0:
            continue
        # look for same event ID within the max_u next values
        r = event_id[i:i + max_u]
        idx = np.where(r == eid)[0]
        # print
        if i % 1e5 == 0:
            print(f' {i}/{n} {i / n * 100:1f}%: event id {eid} ; idx={idx}')
        ne = len(idx)
        # switch
        idx1 = i
        if ne == 2:
            idx2 = i + idx[1]
            pairs1.append(data[idx1, :])
            pairs2.append(data[idx2, :])
            event_id[idx2] = -1
        if ne == 1:
            e1 = ve[idx1]
            if e1 == 0:
                absorbed.append(data[idx1, :])
            else:
                singles.append(data[idx1, :])
        if ne > 2:
            for ii in idx:
                event_id[i + ii] = -1
            nb_ignored = nb_ignored + len(idx)
            continue

    print(f'Number of pairs    {len(pairs1)}')
    print(f'Number of singles  {len(singles)}')
    print(f'Number of absorbed {len(absorbed)} -> {len(singles) + len(absorbed)} ')
    print(f'Number of ignored  {nb_ignored} / {ign}')
    total = len(absorbed) + len(singles) + 2 * len(pairs1) + nb_ignored
    print(f'Total              {total}/{n}')

    pairs1 = np.reshape(pairs1, newshape=(len(pairs1), len(keys)))
    pairs2 = np.reshape(pairs2, newshape=(len(pairs2), len(keys)))
    singles = np.reshape(singles, newshape=(len(singles), len(keys)))
    absorbed = np.reshape(absorbed, newshape=(len(absorbed), len(keys)))

    phsp.save_npy(output.replace('.npy', '_pairs1.npy'), pairs1, keys)
    phsp.save_npy(output.replace('.npy', '_pairs2.npy'), pairs2, keys)
    phsp.save_npy(output.replace('.npy', '_singles.npy'), singles, keys)
    phsp.save_npy(output.replace('.npy', '_absorbed.npy'), absorbed, keys)


# --------------------------------------------------------------------------
if __name__ == '__main__':
    go()
