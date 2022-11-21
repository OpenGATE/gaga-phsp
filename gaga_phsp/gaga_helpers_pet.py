#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy
import numpy as np
import torch
import gaga_phsp as gaga

speed_of_light = scipy.constants.speed_of_light * 1000 / 1e9


def from_tlor_to_pairs(x, params, verbose=False):
    """
        WARNING: the input 'x' is considered to be a torch Variable (not numpy)
        Expected options in params: keys_lists, cyl_radius, ignore_directions

        Input:  Cx Cy Cz Vx Vy Vz dAx dAy dAz dBx dBy dBz t1 E1 E2
        Output: t1 t2 X1 Y1 Z1 X2 Y2 Z2 dX1 dY1 dZ1 dX2 dY2 dZ2 E1 E2

    """

    keys = params['keys_list']
    params['keys_output'] = ['t1', 't2', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2',
                             'dX1', 'dY1', 'dZ1', 'dX2', 'dY2', 'dZ2', 'E1', 'E2']
    # FIXME add weights

    # Step1: name the columns according to key
    C, V, = get_key_3d(x, keys, ['Cx', 'Vx'])
    if params['ignore_directions']:
        params['keys_output'].remove('dX1')
        params['keys_output'].remove('dY1')
        params['keys_output'].remove('dZ1')
        params['keys_output'].remove('dX2')
        params['keys_output'].remove('dY2')
        params['keys_output'].remove('dZ2')
    else:
        dA, dB = get_key_3d(x, keys, ['dAx', 'dBx'])
    tt, E1, E2 = get_key(x, keys, ['t1', 'E1', 'E2'])

    # FIXME weights optional
    w = False
    if 'w1' in keys:
        w = get_key(x, keys, ['w1'])[0]
        params['keys_output'].append('w')

    # Step1: find intersection between line C V and sphere
    A, B, non_valid_index = line_sphere_intersection_torch(params['radius'], C, V)
    if verbose:
        nb_to_remove = torch.unique(non_valid_index, return_counts=True)[1][1]
        print(f'Remove non valid (out of sphere): {nb_to_remove}/{len(A)}')

    # Step2: retrieve time weighted position
    tA, tB = compute_times_wrt_weighted_position(C, A, B, tt)

    # Detect non valid data (negative energy). Keep energy == 0
    non_valid_index = torch.logical_or((E1 <= 0).squeeze(), non_valid_index)
    non_valid_index = torch.logical_or((E2 <= 0).squeeze(), non_valid_index)
    non_valid_index = torch.logical_or((tA <= 0).squeeze(), non_valid_index)
    non_valid_index = torch.logical_or((tB <= 0).squeeze(), non_valid_index)
    if verbose:
        nb_to_remove = torch.unique(non_valid_index, return_counts=True)[1][1]
        print(f'Remove non valid (E<=0): {nb_to_remove}/{len(A)}')

    # Step3: stack
    x = torch.stack((tA, tB), dim=0).T
    x = torch.hstack([x, A])
    x = torch.hstack([x, B])
    if not params['ignore_directions']:
        x = torch.hstack([x, dA])
        x = torch.hstack([x, dB])
    x = torch.hstack([x, E1])
    x = torch.hstack([x, E2])

    # weights
    if 'w1' in keys:
        x = torch.hstack([x, w])

    # mask non valid samples
    x = x[~non_valid_index]

    # end
    return x


def from_pairs_to_tlor(x, params):
    """
        x is numpy array
        Convert pairs into tlor parametrisation
        Input:  t1 t2 X1 Y1 Z1 X2 Y2 Z2 dX1 dY1 dZ1 dX2 dY2 dZ2 E1 E2
        Output: Cx Cy Cz Vx Vy Vz dAx dAy dAz dBx dBy dBz t1 E1 E2 + others
    """

    keys = params['keys_list']
    keys_output = ['Cx', 'Cy', 'Cz',
                   'Vx', 'Vy', 'Vz',
                   'dAx', 'dAy', 'dAz',
                   'dBx', 'dBy', 'dBz',
                   't1', 'E1', 'E2', 'w1']

    # Step1: name the columns according to key
    A, B, dA, dB = get_key_3d(x, keys, ['X1', 'X2', 'dX1', 'dX2'])
    tA, tB, E1, E2 = get_key(x, keys, ['t1', 't2', 'E1', 'E2'])

    # special case for weight that can be ignored
    if 'w1' in keys:
        w1 = get_key(x, keys, ['w1'])[0]
    else:
        w1 = np.ones_like(tA)

    # Step2: compute time weighted position for AB (relative to time at Ap not A)
    # t1 is sum of tA+tB
    C, V, t1 = compute_time_weighted_position(A, B, tA, tB)

    # Step3: stack
    y = np.column_stack([C, V, dA, dB, t1, E1, E2, w1])
    done_keys = ['X1', 'Y1', 'Z1', 'Ax', 'Ay', 'Az',
                 'X2', 'Y2', 'Z2', 'Bx', 'By', 'Bz',
                 'dX1', 'dY1', 'dZ1',
                 'dX2', 'dY2', 'dZ2',
                 't1', 't2', 'E1', 'E2', 'w', 'w1', 'w2']

    # Step4: additional keys
    for k in keys:
        if k not in done_keys:
            z = x[:, keys.index(k)]
            y = np.column_stack([y, z])
            keys_output += [k]

    return y, keys_output


def compute_time_weighted_position(A, B, tA, tB):
    # vector from A to B
    V = (B - A)
    # norm of |AB|
    n = np.linalg.norm(V, axis=1)[:, np.newaxis]
    # relative timing
    tt = tA + tB

    # special case for t = 0 (to avoid divide by zero)
    mask = tt == 0
    tt[mask] = 1.0
    mask = n == 0
    n[mask] = 1
    print(f'Number of total time is 0: {mask.sum()}/{len(tt)}')

    tAr = tA / tt
    tBr = tB / tt
    # relative time weighted position
    CA = A + tAr * V
    CB = B - tBr * V
    # CA should be equal to CB, consider the mean
    C = (CA + CB) / 2
    # normalize direction vector
    V = V / n
    return C, V, tt


def compute_time_at_detector(A, B, Ap, Bp, tA, tB):
    # convert to mm.ns-1
    # c = scipy.constants.speed_of_light * 1000 / 1e9
    # distance from A to Ap
    distance_AAp = np.linalg.norm(Ap - A, axis=1)[:, np.newaxis]
    distance_BBp = np.linalg.norm(Bp - B, axis=1)[:, np.newaxis]
    # convert in time
    tAp = tA + distance_AAp / speed_of_light
    tBp = tB + distance_BBp / speed_of_light
    return tAp, tBp


def line_sphere_intersection_np_old(radius, P, dir):
    # print('line sphere intersection', radius, P.shape, dir.shape)

    print(f'TODO')
    exit()

    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    # nabla ∇
    nabla = np.einsum('ij, ij->i', P, dir)
    nabla = np.square(nabla)
    nabla = nabla - (np.linalg.norm(P, axis=1) ** 2 - radius ** 2)

    # check >0 -> ok
    # print('nabla', nabla)
    # mask = nabla <= 0
    # print('nabla<0', np.count_nonzero(mask))
    non_valid = nabla <= 0

    # distances
    d = -np.einsum('ij, ij->i', P, dir) + np.sqrt(nabla)
    # compute points
    x = P + d[:, np.newaxis] * dir
    return x, non_valid


def line_sphere_intersection_np(radius, P, dir):
    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    a = np.linalg.norm(dir, axis=1, ord=None) ** 2
    b = 2 * np.sum(P * dir, dim=-1)
    # b = 2 * np.sum(P * dir)
    d = np.linalg.norm(P, axis=1, ord=None) ** 2
    c = (d - radius ** 2)
    # delta
    delta = b ** 2 - 4 * a * c
    # consider non valid even if tangent
    non_valid = delta <= 0
    # d
    d1 = (-b - np.sqrt(delta)) / (2 * a)
    d2 = (-b + np.sqrt(delta)) / (2 * a)
    x = P + d1[:, np.newaxis] * dir
    y = P + d2[:, np.newaxis] * dir

    return x, y, non_valid


def line_sphere_intersection_one(radius, P, dir):
    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    a = np.linalg.norm(dir) ** 2
    b = 2 * np.sum(P * dir)
    d = np.linalg.norm(P) ** 2
    c = (d - radius ** 2)
    # delta
    delta = b ** 2 - 4 * a * c
    # consider non valid even if tangent
    non_valid = delta <= 0
    if non_valid:
        return 0, 0, non_valid
    # d
    d1 = (-b - np.sqrt(delta)) / (2 * a)
    d2 = (-b + np.sqrt(delta)) / (2 * a)
    x = P + d1 * dir
    y = P + d2 * dir

    return x, y, non_valid


def line_sphere_intersection_torch(radius, P, dir):
    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    a = torch.linalg.norm(dir, axis=1, ord=None) ** 2
    b = 2 * torch.sum(P * dir, dim=-1)
    d = torch.linalg.norm(P, axis=1, ord=None) ** 2
    c = (d - radius ** 2)
    # delta
    delta = b ** 2 - 4 * a * c
    # consider non valid even if tangent
    non_valid_index = delta <= 0
    # d
    d1 = (-b - torch.sqrt(delta)) / (2 * a)
    d2 = (-b + torch.sqrt(delta)) / (2 * a)
    x = P + d1[:, np.newaxis] * dir
    y = P + d2[:, np.newaxis] * dir

    ## alternative (idem)
    '''dotp = torch.einsum('ij, ij->i', P, dir)
    nabla = torch.square(dotp)
    nabla = nabla - (torch.linalg.norm(P, axis=1) ** 2 - radius ** 2)
    d1 = -dotp - torch.sqrt(nabla)
    d2 = -dotp + torch.sqrt(nabla)
    x = P + d1[:, np.newaxis] * dir
    y = P + d2[:, np.newaxis] * dir'''

    return x, y, non_valid_index


def get_key(x, keys_list, keys):
    v = []
    for k in keys:
        i = keys_list.index(k)
        v.append(x[:, i:i + 1])
    return v


def get_key_3d(x, keys_list, keys):
    # assume Kx Ky kz successive, or X1 Y1 Y1
    v = []
    for k in keys:
        i = keys_list.index(k)
        v.append(x[:, i:i + 3])
    return v


def compute_directions(D, W, t2, t3, A, B):
    """
    Consider the point D and direction W.
        t2 and t3 are the distance from D in W direction
    """
    # retrieve Ap and Bp at virtual spherical detector
    Ap = D - t2 * W
    Bp = D + t3 * W

    # get back the initial A to Ap direction (normalized)
    dA = Ap - A
    n = torch.linalg.norm(dA, axis=1)[:, np.newaxis]
    # FIXME change to torch.nn.functional.normalize ?
    dA = dA / n

    dB = Bp - B
    n = torch.linalg.norm(dB, axis=1)[:, np.newaxis]
    dB = dB / n

    #
    """tAp, tBp = compute_time_at_detector(A, B, Ap, Bp, tA, tB)
    distance_AAp = np.linalg.norm(Ap - A, axis=1)
    distance_BBp = np.linalg.norm(Bp - B, axis=1)
    # convert in time
    tAp = tA + distance_AAp / c
    tBp = tB + distance_BBp / c

    # correct to real timing for A and B: tA and tB
    distance_AAp = np.linalg.norm(Ap - A, axis=1)
    distance_BBp = np.linalg.norm(Bp - B, axis=1)
    # convert in time
    tA = tAp - distance_AAp / c
    tB = tBp - distance_BBp / c
    """

    return dA, dB


def compute_times_wrt_weighted_position(C, A, B, t1):
    """
    According to a point C between AB segment and the time t1,
    compute the times at A and B (tA and tB)
    """
    distance = torch.linalg.norm(B - A, axis=1)
    # timing (t1 is the sum tA+tB)
    t1 = torch.squeeze(t1)
    f = t1 / distance
    # prevent nan -> set both time to zero
    mask = distance == 0
    f[mask] = 0
    # print(f'Number of distance==0 {mask.sum()}/{len(distance)}')
    # distances
    distA = torch.linalg.norm(C - A, axis=1)
    distB = torch.linalg.norm(C - B, axis=1)
    tA = distA * f
    tB = distB * f
    return tA, tB


def line_cylinder_intersections(radius, point, direction):
    """
    Compute intersection between the lines and a cylinder.
    Lines : determine by position 'point', and direction 'direction'
    Cylinder: radius in parameter, height assumed to be Z axis
    """
    # consider xy part
    cx = point[:, 0]
    cy = point[:, 1]
    vx = direction[:, 0]
    vy = direction[:, 1]

    # solve intersection with cylinder
    a = vx ** 2 + vy ** 2
    b = 2 * cx * vx + 2 * cy * vy
    c = cx ** 2 + cy ** 2 - radius ** 2
    delta = b ** 2 - 4 * a * c
    non_valid = delta < 0

    # debug
    print(f'Nb values delta < 0 {torch.count_nonzero(non_valid)}')

    s2 = (-b + torch.sqrt(delta)) / (2 * a)
    s1 = (-b - torch.sqrt(delta)) / (2 * a)
    # print('s1', s1)
    # print('s2', s2)

    # find A and B
    A = point + s1[:, np.newaxis] * direction
    B = point + s2[:, np.newaxis] * direction
    # print('A', A.shape)
    # print('B', B.shape)

    return A, B, non_valid


def plot_sphere_LOR(ax, phsp, keys, x, keys_out, radius):
    # A and B points
    A, B, dA, dB = gaga.get_key_3d(x, keys_out, ['X1', 'X2', 'dX1', 'dX2'])
    ax.plot(A[:, 0], A[:, 1], A[:, 2], '.')
    ax.plot(B[:, 0], B[:, 1], B[:, 2], '.')
    for i in range(len(x)):
        print('A ', A[i])
        print('B', B[i])

    # vectors
    d = B - A
    ax.quiver(A[:, 0], A[:, 1], A[:, 2], d[:, 0], d[:, 1], d[:, 2], arrow_length_ratio=0.1, alpha=0.5)

    # C and V
    C, V, = gaga.get_key_3d(phsp, keys, ['Cx', 'Vx'])
    ax.plot(C[:, 0], C[:, 1], C[:, 2], '.')
    d = 20 * V
    ax.quiver(C[:, 0], C[:, 1], C[:, 2], d[:, 0], d[:, 1], d[:, 2], arrow_length_ratio=0.1, color='b')
    for i in range(len(x)):
        print('C ', C[i])
        print('V', V[i])

    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v) * radius
    y = np.sin(u) * np.sin(v) * radius
    z = np.cos(v) * radius
    ax.plot_wireframe(x, y, z, color="r", linewidth=0.1, alpha=0.8)
    # ax.set_aspect("auto")
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))


def plot_sphere_pairing(ax, x, keys, radius, type):
    if (type == 'pairs'):
        e1 = x[:, keys.index('E1')]
        mask = e1 != 0
        x = x[mask, :]
        e2 = x[:, keys.index('E2')]
        mask = e2 != 0
        x = x[mask, :]
        plot_sphere_pairing_color(ax, x, keys, radius, 'r')
    if (type == 'singles'):
        e1 = x[:, keys.index('E1')]
        mask = e1 != 0
        x = x[mask, :]
        e2 = x[:, keys.index('E2')]
        mask = e2 == 0
        x = x[mask, :]
        plot_sphere_pairing_color(ax, x, keys, radius, 'g')
    if (type == 'absorbed'):
        e1 = x[:, keys.index('E1')]
        mask = e1 == 0
        x = x[mask, :]
        e2 = x[:, keys.index('E2')]
        mask = e2 == 0
        x = x[mask, :]
        plot_sphere_pairing_color(ax, x, keys, radius, 'b')


def plot_sphere_pairing_color(ax, x, keys_out, radius, color):
    # energy
    e1 = x[:, keys_out.index('E1')]
    e2 = x[:, keys_out.index('E2')]
    t1 = x[:, keys_out.index('t1')]
    t2 = x[:, keys_out.index('t2')]

    # P0
    P0 = gaga.get_key_3d(x, keys_out, ['eX'])[0]
    ax.plot(P0[:, 0], P0[:, 1], P0[:, 2], '.', color=color)
    for i in range(len(P0)):
        ax.text(P0[i, 0], P0[i, 1], P0[i, 2], '%s' % (str(i)), size=8, zorder=1, color='k', alpha=0.7)

    # A and B points
    A, B, dA, dB = gaga.get_key_3d(x, keys_out, ['X1', 'X2', 'dX1', 'dX2'])
    ax.plot(A[:, 0], A[:, 1], A[:, 2], '.')
    ax.plot(B[:, 0], B[:, 1], B[:, 2], '.')
    for i in range(len(A)):
        ax.text(A[i, 0] + 5, A[i, 1] + 5, A[i, 2] + 5, '%s' % (str(i)), size=8, zorder=1, color='k', alpha=0.6)
        ax.text(A[i, 0], A[i, 1], A[i, 2], f'{e1[i]:.2f}', size=8, zorder=1, color='g', alpha=0.6)
        ax.text(B[i, 0], B[i, 1], B[i, 2], f'{e2[i]:.2f}', size=8, zorder=1, color='g', alpha=0.6)
        ax.text(A[i, 0] - 10, A[i, 1] - 10, A[i, 2] - 10, f'{t1[i]:.2f}', size=8, zorder=1, color='b', alpha=0.6)
        ax.text(B[i, 0] - 10, B[i, 1] - 10, B[i, 2] - 10, f'{t2[i]:.2f}', size=8, zorder=1, color='b', alpha=0.6)

    # vectors
    d = B - A
    ax.quiver(A[:, 0], A[:, 1], A[:, 2], d[:, 0], d[:, 1], d[:, 2], arrow_length_ratio=0.01, alpha=0.5)

    # vectors outgoing
    d = dA * 30
    ax.quiver(A[:, 0], A[:, 1], A[:, 2], d[:, 0], d[:, 1], d[:, 2], arrow_length_ratio=0.1, alpha=0.5, color='b')
    d = dB * 30
    ax.quiver(B[:, 0], B[:, 1], B[:, 2], d[:, 0], d[:, 1], d[:, 2], arrow_length_ratio=0.1, alpha=0.5, color='b')

    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v) * radius
    y = np.sin(u) * np.sin(v) * radius
    z = np.cos(v) * radius
    ax.plot_wireframe(x, y, z, color="k", linewidth=0.1, alpha=0.8)
    # ax.set_aspect("auto")
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))


def pet_pairing_v1(n, i, idx, energy, pos, dir, time, p0, d0, out, nbs):
    if n == 2:
        return pet_pairing_pairs(i, i + idx[1], energy, pos, dir, time, p0, out, nbs)
    if n == 1:
        e1 = energy[i]
        if e1 == 0:
            return pet_pairing_absorbed_v1(i, pos, dir, time, p0, d0, out, nbs)
        else:
            return pet_pairing_singles_v1(i, e1, pos, dir, time, p0, d0, out, nbs)
    nbs.ignored += n


def pet_pairing_v2(n, i, idx, energy, pos, dir, time, p0, d0, out, nbs):
    if n == 2:
        return pet_pairing_pairs(i, i + idx[1], energy, pos, dir, time, p0, out, nbs)
    if n == 1:
        e1 = energy[i]
        if e1 == 0:
            return pet_pairing_absorbed_v2(i, pos, dir, time, p0, d0, out, nbs)
        else:
            return pet_pairing_singles_v2(i, e1, pos, dir, time, p0, d0, out, nbs)
    nbs.ignored += n


def pet_pairing_pairs(idx1, idx2, energy, pos, dir, time, p0, out, nbs):
    out.append([energy[idx1], energy[idx2]] +
               list(pos[idx1, :]) + list(pos[idx2, :]) +
               list(dir[idx1, :]) + list(dir[idx2, :]) +
               [time[idx1], time[idx2]] +
               list(p0[idx1]))
    nbs.pairs += 1


def pet_pairing_absorbed_v1(i, pos, dir, time, p0, d0, out, nbs):
    p0 = p0[i]
    a, b, non_valid = gaga.line_sphere_intersection_one(nbs.radius, p0, d0[i])
    if non_valid:
        return
    t1 = np.linalg.norm(a - p0) / speed_of_light
    t2 = np.linalg.norm(b - p0) / speed_of_light
    out.append([0, 0] +  # energy
               list(a) + list(b) +  # position (unused)
               list(-d0[i]) + list(d0[i]) +  # direction (unused)
               [t1, t2] +  # time (unused)
               list(p0))  # event position
    nbs.absorbed += 1


def pet_pairing_absorbed_v2(i, pos, dir, time, p0, d0, out, nbs):
    p0 = p0[i]
    z = np.zeros_like(p0)
    out.append([0, 0] +  # energy
               list(z) + list(z) +  # position
               list(d0[i]) + list(-d0[i]) +  # direction
               [0, 0] +  # time (unused)
               list(p0))  # event position
    nbs.absorbed += 1


def pet_pairing_singles_v2(i, e1, pos, dir, time, p0, d0, out, nbs):
    p0 = p0[i]
    p1 = pos[i]
    d1 = dir[i]
    t1 = time[i]
    p2 = np.zeros_like(p0)
    out.append([e1, 0] +
               list(p1) + list(p2) +  # p2 position is unused
               list(d1) + list(-d1) +  # d2 direction is unused
               [t1, 0] +  # t2 time is unused
               list(p0))
    nbs.singles += 1


def pet_pairing_singles_v1(i, e1, pos, dir, time, p0, d0, out, nbs):
    p0 = p0[i]
    p1 = pos[i]
    d1 = dir[i]
    t1 = time[i]
    a, b, nv = gaga.line_sphere_intersection_one(nbs.radius, p0, d0[i])
    if nv:
        nbs.ignored += 1
        return
    dia = np.linalg.norm(p1 - a)
    dib = np.linalg.norm(p1 - b)
    # consider the more far
    if dia > dib:
        p2 = a
    else:
        p2 = b
    d2 = p2 - p0
    n2 = np.linalg.norm(d2)
    t2 = n2 / speed_of_light
    d2 = d2 / n2
    out.append([e1, 0] +
               list(p1) + list(p2) +  # p2 position is unused
               list(d1) + list(d2) +  # d2 direction is unused
               [t1, t2] +  # t2 time is unused
               list(p0))
    nbs.singles += 1
