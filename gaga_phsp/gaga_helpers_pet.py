#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy
import numpy as np
from torch.autograd import Variable
import torch
import gaga_phsp

c = scipy.constants.speed_of_light * 1000 / 1e9


def from_tlor_to_pairs(x, params):
    """
        WARNING: the input 'x' is considered to be a torch Variable (not numpy)
        Expected options in params: keys_lists, cyl_radius,

        Input:  Cx Cy Cz Vx Vy Vz Dx Dy Dz Wx Wy Wz t1 t2 t3 E1 E2
        Temp:   t1 t2 Ax Ay Az Bx By Bz dAx dAy dAz dBx dBy dBz E1 E2
        Output: t1 t2 X1 Y1 Z1 X2 Y2 Z2 dX1 dY1 dZ1 dX2 dY2 dZ2 E1 E2

    """

    keys = params['keys_list']
    params['keys_output'] = ['t1', 't2', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2',
                             'dX1', 'dY1', 'dZ1', 'dX2', 'dY2', 'dZ2', 'E1', 'E2']
    # FIXME add weights

    # Step1: name the columns according to key
    C, V, D, W = get_key_3d(x, keys, ['Cx', 'Vx', 'Dx', 'Wx'])
    t1, t2, t3, E1, E2 = get_key(x, keys, ['t1', 't2', 't3', 'E1', 'E2'])
    # FIXME weights optional

    # Step2: find intersections with cylinder
    A, B, non_valid = line_cylinder_intersections(params['cyl_radius'], C, V)
    h = params['cyl_height'] / 2
    # print(torch.unique(non_valid, return_counts=True))
    non_valid = torch.logical_or(A[:, 2] > h, non_valid)
    non_valid = torch.logical_or(A[:, 2] < -h, non_valid)
    non_valid = torch.logical_or(B[:, 2] > h, non_valid)
    non_valid = torch.logical_or(B[:, 2] < -h, non_valid)
    # print(torch.unique(non_valid, return_counts=True))

    # Step3: retrieve time weighted position
    tA, tB = compute_times_wrt_weighted_position(C, A, B, t1)

    # Step4: direction
    dA, dB = compute_directions(D, W, t2, t3, A, B)

    # clean non valid data (negative energy)
    non_valid = torch.logical_or((E1 <= 0).squeeze(), non_valid)
    # print(torch.unique(non_valid, return_counts=True))
    non_valid = torch.logical_or((E2 <= 0).squeeze(), non_valid)
    # print(torch.unique(non_valid, return_counts=True))

    # Step4: stack
    x = torch.stack((tA, tB), dim=0).T
    x = torch.hstack([x, A])
    x = torch.hstack([x, B])
    x = torch.hstack([x, dA])
    x = torch.hstack([x, dB])
    x = torch.hstack([x, E1])
    x = torch.hstack([x, E2])
    # FIXME weights

    # mask non valid samples
    x = x[~non_valid]

    # end
    return x


def from_pairs_to_tlor(x, params):
    """
        x is numpy array
        Convert pairs into tlor parametrisation
        Input:  t1 t2 X1 Y1 Z1 X2 Y2 Z2 dX1 dY1 dZ1 dX2 dY2 dZ2 E1 E2
        Temp:   t1 t2 Ax Ay Az Bx By Bz dAx dAy dAz dBx dBy dBz E1 E2
        Output: Cx Cy Cz Vx Vy Vz Dx Dy Dz Wx Wy Wz t1 t2 t3 E1 E2
    """

    keys = params['keys_list']
    keys_output = ['Cx', 'Cy', 'Cz', 'Vx', 'Vy', 'Vz', 'Dx', 'Dy', 'Dz', 'Wx', 'Wy', 'Wz', 't1', 't2', 't3', 'E1', 'E2']

    # Step1: name the columns according to key
    A, B, dA, dB = get_key_3d(x, keys, ['X1', 'X2', 'dX1', 'dX2'])
    tA, tB, E1, E2 = get_key(x, keys, ['t1', 't2', 'E1', 'E2'])

    # Step2: compute intersection and times with the detector
    radius = params['det_radius']
    Ap = line_sphere_intersection(radius, A, dA)
    Bp = line_sphere_intersection(radius, B, dB)
    tAp, tBp = compute_time_at_detector(A, B, Ap, Bp, tA, tB)

    # Step3: compute time weighted position for AB (relative to time at Ap not A)
    C, V, tt, n = compute_time_weighted_position(A, B, tA, tB)
    t1 = tt

    # compute time weighted position for A'B'
    D, W, tt, n = compute_time_weighted_position(Ap, Bp, tAp, tBp)
    t2 = np.linalg.norm(Ap - D, axis=1)
    t3 = np.linalg.norm(Bp - D, axis=1)

    # Step4: stack
    x = np.column_stack([C, V, D, W, t1, t2, t3, E1, E2])

    return x, keys_output


def compute_time_weighted_position(A, B, tA, tB):
    # norm of |AB|
    n = np.linalg.norm(B - A, axis=1)[:, np.newaxis]
    # vector from A to B
    V = (B - A)
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
    return C, V, tt, n


def compute_time_at_detector(A, B, Ap, Bp, tA, tB):
    # convert to mm.ns-1
    # c = scipy.constants.speed_of_light * 1000 / 1e9
    # distance from A to Ap
    distance_AAp = np.linalg.norm(Ap - A, axis=1)[:, np.newaxis]
    distance_BBp = np.linalg.norm(Bp - B, axis=1)[:, np.newaxis]
    # convert in time
    tAp = tA + distance_AAp / c
    tBp = tB + distance_BBp / c
    return tAp, tBp


def line_sphere_intersection(radius, P, dir):
    # print('line sphere intersection', radius, P.shape, dir.shape)

    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    # nabla ∇
    nabla = np.einsum('ij, ij->i', P, dir)
    nabla = np.square(nabla)
    nabla = nabla - (np.linalg.norm(P, axis=1, ord=2) - radius ** 2)
    # FIXME check >0
    # print('nabla', nabla)
    mask = nabla <= 0
    n = nabla[mask]
    # print('<0', np.min(nabla), len(n))
    # distances
    d = -np.einsum('ij, ij->i', P, dir) + np.sqrt(nabla)
    # print('d', d)
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

    """
    # retrieve Ap and Bp
    Ap = D - t2 * W
    Bp = D + t3 * W

    # directions (normalized) at A and B
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


def line_cylinder_intersections(radius, P, dir):
    """
    Compute intersection between the lines and a cylinder.
    Lines : determine by position 'P', and direction 'dir'
    Cylinder: radius in parameter, height assumed to be Z axis
    """
    # consider xy part
    Cx = P[:, 0]
    Cy = P[:, 1]
    vx = dir[:, 0]
    vy = dir[:, 1]

    # solve intersection with cylinder
    # a = np.power(vx, 2) + np.power(vy, 2)
    a = vx ** 2 + vy ** 2
    b = 2 * Cx * vx + 2 * Cy * vy
    # c = np.power(Cx, 2) + np.power(Cy, 2) - radius * radius
    c = Cx ** 2 + Cy ** 2 - radius * radius
    # delta = np.power(b, 2) - 4 * a * c
    delta = b ** 2 - 4 * a * c
    non_valid = delta < 0
    # print(f'Nb values delta {len(delta)}')
    # print('delta', delta)

    s2 = (-b + torch.sqrt(delta)) / (2 * a)
    s1 = (-b - torch.sqrt(delta)) / (2 * a)
    # print('s1', s1)
    # print('s2', s2)

    # find A and B
    A = P + s1[:, np.newaxis] * dir
    B = P + s2[:, np.newaxis] * dir
    # print('A', A.shape)
    # print('B', B.shape)

    return A, B, non_valid
