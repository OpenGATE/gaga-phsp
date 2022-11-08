#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import uproot
import click
import gatetools as gt
import gatetools.phsp as phsp
from matplotlib import pyplot as plt
import numpy as np
import os
import scipy

# import awkward1 as aw

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('root_filename', nargs=1)
@click.option('--output', '-o', default='auto', help='output filename (npy)')
@click.option('--n', '-n', default='-1', help='Number of samples to read (-1 for all)')
@click.option('--unpaired', default=False, is_flag=True, help='Also insert unpaired events')
@click.option('--time_col', '-t', default='TimeFromBeginOfEvent', help='Time column ? TimeFromBeginOfEvent or Time')
def go(root_filename, output, time_col, unpaired, n):
    # read the root file
    try:
        f = uproot.open(root_filename)
    except Exception:
        print(f'Cannot open the file {root_filename}. Is this a root file ?')
        exit(-1)

    if output == 'auto':
        output = os.path.splitext(root_filename)[0] + '_pairs.npy'
    print(f'Output file will be {output}')

    print(f'Names of the branches: {f.keys()}')
    print(f'(we consider the first branch only)')
    phspf = f[f.keys()[0]]  # first branch
    phspf.show()

    # names of the branches
    sekine = 'Ekine'
    if sekine not in phspf:
        sekine = 'KineticEnergy'
    sx = 'X'
    sy = 'Y'
    sz = 'Z'
    if sx not in phspf:
        sx = 'PrePosition_X'
        sy = 'PrePosition_Y'
        sz = 'PrePosition_Z'
    sdx = 'dX'
    sdy = 'dY'
    sdz = 'dZ'
    if sdx not in phspf:
        sdx = 'PreDirection_X'
        sdy = 'PreDirection_Y'
        sdz = 'PreDirection_Z'
    st = time_col  # 'TimeFromBeginOfEvent'
    if st not in phspf:
        st = 'LocalTime'
    print('Timing column is ', st)

    n = int(float(n))
    if n == -1:
        n = phspf['EventID'].num_entries
    print(f'reading event id {n} / {phspf["EventID"].num_entries}')

    event_id = phspf['EventID'].array(library='numpy', entry_stop=n)
    print('reading ekine')
    ekine = phspf[sekine].array(library='numpy', entry_stop=n)
    print('reading pos')
    posx = phspf[sx].array(library='numpy', entry_stop=n)
    posy = phspf[sy].array(library='numpy', entry_stop=n)
    posz = phspf[sz].array(library='numpy', entry_stop=n)
    print('reading dir')
    dx = phspf[sdx].array(library='numpy', entry_stop=n)
    dy = phspf[sdy].array(library='numpy', entry_stop=n)
    dz = phspf[sdz].array(library='numpy', entry_stop=n)
    print('reading time')
    time = phspf[st].array(library='numpy', entry_stop=n)

    weight_enabled = False
    if 'Weight' in phspf:
        weights = phspf['Weight'].array(library='numpy', entry_stop=n)
        weight_enabled = True
        print('reading weights', len(weights))

    vdir_enabled = False
    if 'TrackVertexMomentumDirection_X' in phspf:
        vdirx = phspf['TrackVertexMomentumDirection_X'].array(library='numpy', entry_stop=n)
        vdiry = phspf['TrackVertexMomentumDirection_Y'].array(library='numpy', entry_stop=n)
        vdirz = phspf['TrackVertexMomentumDirection_Z'].array(library='numpy', entry_stop=n)
        vdir_enabled = True
        print('reading TrackVertexMomentumDirection', len(vdirx))

    vpos_enabled = False
    if 'TrackVertexPosition_X' in phspf:
        vposx = phspf['TrackVertexPosition_X'].array(library='numpy', entry_stop=n)
        vposy = phspf['TrackVertexPosition_Y'].array(library='numpy', entry_stop=n)
        vposz = phspf['TrackVertexPosition_Z'].array(library='numpy', entry_stop=n)
        vpos_enabled = True
        print('reading TrackVertexPosition', len(vposx))

    epos_enabled = False
    if 'EventPosition_X' in phspf:
        eposx = phspf['EventPosition_X'].array(library='numpy', entry_stop=n)
        eposy = phspf['EventPosition_Y'].array(library='numpy', entry_stop=n)
        eposz = phspf['EventPosition_Z'].array(library='numpy', entry_stop=n)
        epos_enabled = True
        print('reading EventPosition', len(eposx))

    print('Count the nb of unpaired, pairs etc ...')
    unique, counts = np.unique(event_id, return_counts=True)
    # print(unique, counts)
    n = len(event_id)  # .num_entries
    print('len', n)

    u, c = np.unique(counts, return_counts=True)
    print(u)
    max_u = u[len(u) - 1]
    print('Count max (+1)', c, max_u)

    # debug print first values
    # for eid, en in zip(event_id[:30], ekine[:30]):
    #    print(eid, en)

    # x will be the total list of pairs (and unpairs)
    x = []
    # number of ignored events and unpaired events
    nb_ignored = 0
    nb_unpaired = 0
    nb_pairs = 0
    nb_absorbed = 0
    # mm.ns-1
    c = scipy.constants.speed_of_light * 1000 / 1e9

    # main loop, 'i' is the index of the current hit
    for i in range(len(event_id)):
        # skip negative event id (a negative index means: already considered)
        eid = event_id[i]
        # print(i, eid)
        if eid < 0:
            continue
        # look for same event ID within the max_u next values
        r = event_id[i:i + max_u]
        idx = np.where(r == eid)[0]
        # print ?
        if i % 1e5 == 0:
            print(f' {i}/{n} {i / n * 100:1f}%: event id {eid} ; idx={idx}')
        # we can have 1 (unpaired), 2 (paired) or more matches
        # we only keep the pairs except if the flag 'unpaired' is on
        if unpaired:
            if len(idx) != 2 and len(idx) != 1:
                nb_ignored = nb_ignored + len(idx)
                # prevent the other events to be later considered as valid event
                # print('triple', idx)
                for ii in idx:
                    event_id[i + ii] = -1
                continue
        else:
            if len(idx) != 2:
                # prevent the other events to be later considered as valid event
                for ii in idx:
                    event_id[i + ii] = -1
                nb_ignored = nb_ignored + len(idx)
                continue

        # prevent the matched events to be later considered
        for ii in idx:
            event_id[i + ii] = -1

        # if unpaired event -> duplicate it
        is_unpaired = False
        if len(idx) == 2:
            nb_pairs = nb_pairs + 1
        if len(idx) == 1:
            is_unpaired = True
            nb_unpaired = nb_unpaired + 1
            idx = [0, 0]
            # print('unpaired')

        # Get the indexes of the two events
        idx1 = i + idx[0]
        idx2 = i + idx[1]

        # Get the energies of the two events
        e1 = ekine[idx1]
        e2 = ekine[idx2]
        # print(f' {eid} {i} --> {event_id[i + idx[0]]} {event_id[i + idx[1]]} => {e1} {e2}')

        if e1 == 0:
            nb_absorbed += 1
            nb_unpaired = nb_unpaired - 1

        if is_unpaired:
            # set energy to 0 (will be ignored)
            e2 = 0
            # computed distance from time
            d = time[idx1] * c
            # set to a previous position to store fake value
            idx2 = idx1 - 1
            # fake position : opposite to the current pos
            posx[idx2] = posx[idx1] - d * dx[idx1]
            posy[idx2] = posy[idx1] - d * dy[idx1]
            posz[idx2] = posz[idx1] - d * dz[idx1]
            # fake direction, opposite
            dx[idx2] = -dx[idx1]
            dy[idx2] = -dy[idx1]
            dz[idx2] = -dz[idx1]

        z = [e1, e2,
             posx[idx1], posy[idx1], posz[idx1],
             posx[idx2], posy[idx2], posz[idx2],
             dx[idx1], dy[idx1], dz[idx1],
             dx[idx2], dy[idx2], dz[idx2],
             time[idx1], time[idx2]]

        if weight_enabled:
            z.extend([weights[idx1], weights[idx2]])

        if vdir_enabled:
            z.extend([vdirx[idx1], vdiry[idx1], vdirz[idx1]])

        if vpos_enabled:
            z.extend([vposx[idx1], vposy[idx1], vposz[idx1]])

        if epos_enabled:
            z.extend([eposx[idx1], eposy[idx1], eposz[idx1]])

        x.append(z)

    # nb_pairs = len(x) - nb_unpaired
    print('Pairs:            ', nb_pairs)
    print('Unpaired:         ', nb_unpaired)
    print('Absorbed:         ', nb_absorbed)
    print(f'All:               {len(x)}/{nb_pairs + nb_unpaired + nb_absorbed}')
    print(f'Ignored:           {nb_ignored} (triplet x3, quadruplet x4 etc)')
    print(f'Total:             {nb_pairs * 2 + nb_unpaired + nb_absorbed + nb_ignored} / {n}')

    keys = ['E1', 'E2',
            'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2',
            'dX1', 'dY1', 'dZ1', 'dX2', 'dY2', 'dZ2',
            't1', 't2']
    if weight_enabled:
        keys.extend(['w1', 'w2'])
    if vdir_enabled:
        keys.extend(['vdX', 'vdY', 'vdZ'])
    if vpos_enabled:
        keys.extend(['vX', 'vY', 'vZ'])
    if epos_enabled:
        keys.extend(['eX', 'eY', 'eZ'])

    print('Keys', keys)
    x = np.reshape(x, newshape=(len(x), len(keys)))
    print(x.shape)
    phsp.save_npy(output, x, keys)


# --------------------------------------------------------------------------
if __name__ == '__main__':
    go()
