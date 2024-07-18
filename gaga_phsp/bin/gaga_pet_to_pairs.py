#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
import gatetools.phsp as phsp
from matplotlib import pyplot as plt
import numpy as np
import gaga_phsp as gaga
from box import Box

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('phsp_filename', nargs=1)
@click.option('--output', '-o', default='auto', help='output filename (npy)')
@click.option('--tree', '-t', default='phase_space', help='Name of the tree to analyse')
@click.option('--n', '-n', default='-1', help='Number of samples to read (-1 for all)')
@click.option('--nstart', default=0.0, help='Starting samples to read')
@click.option('--radius', '-r', required=False, default=210.0, help='phsp sphere radius in mm')
@click.option('--plot', is_flag=True, default=False, help='Plot for debug')
def go(phsp_filename, tree, output, n, nstart, radius, plot):
    # read the phsp file
    n = int(float(n))
    nstart = int(float(nstart))
    data, keys, m = phsp.load(phsp_filename, treename=tree, nmax=n, nstart=nstart)
    if n == -1:
        n = m
    print(f'PHSP read with {n}/{m}')
    print(f'{keys}')
    print(f'Shape {data.shape}')

    # get some branches
    event_id = data[:, keys.index('EventID')]
    ve = data[:, keys.index('KineticEnergy')]
    vpos = data[:, keys.index('PrePosition_X'):keys.index('PrePosition_Z') + 1]
    vdir = data[:, keys.index('PreDirection_X'):keys.index('PreDirection_Z') + 1]
    vt = data[:, keys.index('TimeFromBeginOfEvent')]
    vp0 = data[:, keys.index('EventPosition_X'):keys.index('EventPosition_Z') + 1]

    # vd0 = data[:, keys.index('TrackVertexMomentumDirection_X'):keys.index('TrackVertexMomentumDirection_Z') + 1]
    # FIXME vd0 is not really used. Only for absorbed or single events, to compute fake position/direction
    # (this may (?) help the GAN to learn instead of using a zero value)
    vd0 = data[:, keys.index('EventDirection_X'):keys.index('EventDirection_Z') + 1]

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
    out = []
    nbs = Box()
    nbs.absorbed = 0
    nbs.singles = 0
    nbs.pairs = 0
    nbs.ignored = 0
    nbs.radius = radius

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
        # nb of events with the same eventID
        ne = len(idx)
        # pairs them
        #gaga.pet_pairing_v1(ne, i, idx, ve, vpos, vdir, vt, vp0, vd0, out, nbs)
        gaga.pet_pairing_v2(ne, i, idx, ve, vpos, vdir, vt, vp0, vd0, out, nbs)
        # store they were considered
        for ii in idx:
            event_id[i + ii] = -1

    keys_out = ['E1', 'E2',  # output Energy
                'X1', 'Y1', 'Z1',  # output position
                'X2', 'Y2', 'Z2',  # output position
                'dX1', 'dY1', 'dZ1',  # output direction
                'dX2', 'dY2', 'dZ2',  # output direction
                't1', 't2',  # output time
                'eX', 'eY', 'eZ'  # input condition
                ]
    out = np.reshape(out, newshape=(len(out), len(keys_out)))
    phsp.save_npy(output, out, keys_out)

    print(f'Number of pairs    {nbs.pairs}')
    print(f'Number of singles  {nbs.singles}')
    print(f'Number of absorbed {nbs.absorbed} -> {nbs.singles + nbs.absorbed} ')
    print(f'Number of ignored  {nbs.ignored} / {ign}')
    total = nbs.absorbed + nbs.singles + 2 * nbs.pairs + nbs.ignored
    print(f'Total              {total}/{n}')

    # plot ?
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        gaga.plot_sphere_pairing(ax, out, keys_out, radius=radius, type='pairs')
        gaga.plot_sphere_pairing(ax, out, keys_out, radius=radius, type='singles')
        gaga.plot_sphere_pairing(ax, out, keys_out, radius=radius, type='absorbed')
        plt.show()


# --------------------------------------------------------------------------
if __name__ == '__main__':
    go()
