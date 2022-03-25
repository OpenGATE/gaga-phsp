#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy
import numpy as np
import torch

speed_of_light = scipy.constants.speed_of_light * 1000 / 1e9


def from_tlor_to_pairs(x, params):
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
    t1, E1, E2 = get_key(x, keys, ['t1', 'E1', 'E2'])

    # FIXME weights optional
    w = False
    if 'w1' in keys:
        w = get_key(x, keys, ['w1'])[0]
        params['keys_output'].append('w')

    # Step2: find intersections with cylinder
    A, B, non_valid = line_cylinder_intersections(params['cyl_radius'], C, V)
    h = params['cyl_height']
    # print('cyl inter', torch.unique(non_valid, return_counts=True))
    non_valid = torch.logical_or(A[:, 2] > h, non_valid)
    # print('A h', torch.unique(non_valid, return_counts=True))
    non_valid = torch.logical_or(A[:, 2] < -h, non_valid)
    # print('A -h', torch.unique(non_valid, return_counts=True))
    non_valid = torch.logical_or(B[:, 2] > h, non_valid)
    # print('B h', torch.unique(non_valid, return_counts=True))
    non_valid = torch.logical_or(B[:, 2] < -h, non_valid)
    # print('B -h', torch.unique(non_valid, return_counts=True))
    removed = torch.unique(non_valid, return_counts=True)[1][0]
    print(f'Remove non valid (out of cylinder): {removed}/{len(A)}')

    # Step3: retrieve time weighted position
    tA, tB = compute_times_wrt_weighted_position(C, A, B, t1)

    # alternative to step4 -> intersection with virtual spherical detector ? (no need t2, t3 ??)
    """dtypef, device = gaga.init_pytorch_cuda('auto', verbose=False)
    D = D.cpu().data.numpy()
    W = W.cpu().data.numpy()
    Ap = line_sphere_intersection(500, D, -W)
    Bp = line_sphere_intersection(500, D, W)
    Ap = Variable(torch.from_numpy(Ap).type(dtypef))
    Bp = Variable(torch.from_numpy(Bp).type(dtypef))

    dA = Ap - A
    dA = F.normalize(dA, p=2, dim=1)
    dB = Bp - B
    dB = F.normalize(dB, p=2, dim=1)"""

    # Step4: direction
    # dA, dB = compute_directions(D, W, t2, t3, A, B)

    # clean non valid data (negative energy)
    non_valid = torch.logical_or((E1 <= 0).squeeze(), non_valid)
    # print('E1', torch.unique(non_valid, return_counts=True))
    non_valid = torch.logical_or((E2 <= 0).squeeze(), non_valid)
    # print('E2', torch.unique(non_valid, return_counts=True))
    removed = torch.unique(non_valid, return_counts=True)[1][0]
    print(f'Remove non valid (E<=0):            {removed} {len(A)}')

    # Step4: stack
    x = torch.stack((tA, tB), dim=0).T
    x = torch.hstack([x, A])
    x = torch.hstack([x, B])
    if not params['ignore_directions']:
        x = torch.hstack([x, dA])
        x = torch.hstack([x, dB])
    x = torch.hstack([x, E1])
    x = torch.hstack([x, E2])

    # FIXME weights
    if 'w1' in keys:
        x = torch.hstack([x, w])

    # mask non valid samples
    x = x[~non_valid]

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
    keys_output = ['Cx', 'Cy', 'Cz', 'Vx', 'Vy', 'Vz', 'dAx', 'dAy', 'dAz', 'dBx', 'dBy', 'dBz',
                   't1', 'E1', 'E2', 'w1']

    # Step1: name the columns according to key
    A, B, dA, dB = get_key_3d(x, keys, ['X1', 'X2', 'dX1', 'dX2'])
    tA, tB, E1, E2 = get_key(x, keys, ['t1', 't2', 'E1', 'E2'])

    # special case for weight that can be ignored
    if 'w1' in keys:
        w1 = get_key(x, keys, ['w1'])[0]
    else:
        w1 = np.ones_like(tA)

    # Step2: compute intersection and times with a virtual spherical detector
    # radius = params['det_radius']  # FIXME arbitrary !
    # Ap = line_sphere_intersection(radius, A, dA)
    # Bp = line_sphere_intersection(radius, B, dB)

    # Step3: compute time weighted position for AB (relative to time at Ap not A)
    # t1 is sum of tA+tB
    C, V, t1 = compute_time_weighted_position(A, B, tA, tB)

    # compute time weighted position for A'B'
    # tAp, tBp = compute_time_at_detector(A, B, Ap, Bp, tA, tB)
    # D, W, tt = compute_time_weighted_position(Ap, Bp, tAp, tBp)
    # t2 and t3 are distances (not time!) # FIXME
    # t2 = np.linalg.norm(Ap - D, axis=1)
    # t3 = np.linalg.norm(Bp - D, axis=1)

    # Step4: stack
    y = np.column_stack([C, V, dA, dB, t1, E1, E2, w1])
    done_keys = ['X1', 'Y1', 'Z1', 'Ax', 'Ay', 'Az',
                 'X2', 'Y2', 'Z2', 'Bx', 'By', 'Bz',
                 'dX1', 'dY1', 'dZ1',
                 'dX2', 'dY2', 'dZ2',
                 't1', 't2', 'E1', 'E2', 'w', 'w1', 'w2']

    # optional vertex info
    c = scipy.constants.speed_of_light * 1000  # in mm/s
    if 'use_vertex_distance' in params:
        print('add distance from vertex to point (convert in time ns)')
        vertex_pos = get_key_3d(x, keys, ['vX'])[0]
        dv1 = np.linalg.norm(vertex_pos - A, axis=1) / c * 1e9
        dv2 = np.linalg.norm(vertex_pos - B, axis=1) / c * 1e9
        y = np.column_stack([y, dv1, dv2])
        keys_output += ['dv1', 'dv2']
        done_keys += ['vX', 'vY', 'vZ']

    # optional vertex info
    if 'use_vertex_angle' in params:
        print('add distance from vertex to point')
        vertex_dir = get_key_3d(x, keys, ['vdX'])[0]
        vangle1 = (vertex_dir * dA).sum(1)
        vangle2 = (vertex_dir * dB).sum(1)
        y = np.column_stack([y, vangle1, vangle2])
        keys_output += ['av1', 'av2']
        done_keys += ['vdX', 'vdY', 'vdZ']

    # Step5: additional keys
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
    tAr = tA / tt
    tBr = tB / tt
    # relative time weighted position
    CA = A + tAr * V
    CB = B - tBr * V
    # CA should be equal to CB, consider the mean
    C = (CA + CB) / 2
    # normalize vector
    V = V / n
    # t1 factor
    # tt = tt[:, np.newaxis]
    # print('shape tt and n', tt.shape, n.shape)
    # t1 = tt / n  # FIXME total time divided by segment length
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


def line_sphere_intersection(radius, P, dir):
    # print('line sphere intersection', radius, P.shape, dir.shape)

    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    # nabla ∇
    nabla = np.einsum('ij, ij->i', P, dir)
    nabla = np.square(nabla)
    nabla = nabla - (np.linalg.norm(P, axis=1, ord=2) - radius ** 2)

    # check >0 -> ok
    # print('nabla', nabla)
    # mask = nabla <= 0
    # print('nabla<0', np.count_nonzero(mask))

    # distances
    d = -np.einsum('ij, ij->i', P, dir) + np.sqrt(nabla)
    # compute points
    x = P + d[:, np.newaxis] * dir
    return x


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
    t1 = torch.squeeze(t1)
    f = t1 / distance
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
