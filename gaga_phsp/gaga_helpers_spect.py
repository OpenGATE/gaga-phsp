#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy
import torch
import numpy as np
import gaga_phsp as gaga

speed_of_light = scipy.constants.speed_of_light * 1000 / 1e9


def from_exit_pos_to_ideal_pos(x, params):
    """
    input : consider input key exit position X,Y,Z
    output: parametrize with ideal position = p - c x t x dir

    Input: PrePosition_X PreDirection_X TimeFromBeginOfEvent
    Output: IdealPosition_X (+idem)
    """

    # input and output keys
    keys = params["keys_list"]
    keys_out = [
        "KineticEnergy",
        "IdealPosition_X",
        "IdealPosition_Y",
        "IdealPosition_Z",
        "PreDirection_X",
        "PreDirection_Y",
        "PreDirection_Z",
        "TimeFromBeginOfEvent",
        "EventPosition_X",
        "EventPosition_Y",
        "EventPosition_Z",
        "EventDirection_X",
        "EventDirection_Y",
        "EventDirection_Z",
    ]

    # Step1: name the columns according to key
    p_ideal, d_exit, e_pos, e_dir = gaga.get_key_3d(
        x,
        keys,
        ["PrePosition_X", "PreDirection_X", "EventPosition_X", "EventDirection_X"],
    )
    t_exit = gaga.get_key(x, keys, ["TimeFromBeginOfEvent"])[0]
    ene = gaga.get_key(x, keys, ["KineticEnergy"])[0]

    # Step2: compute P_ideal
    P_ideal = p_ideal - speed_of_light * t_exit * d_exit

    # Step3: stack
    # x = torch.stack((ene, P_ideal, d_exit, t_exit, e_pos), dim=0).T # FIXME torch or numpy ?
    x = np.concatenate((ene, P_ideal, d_exit, t_exit, e_pos, e_dir), axis=1)

    # end
    return x, keys_out


def from_ideal_pos_to_exit_pos(x, params):

    # input and output keys
    keys = params["keys_list"]
    keys_out = [
        "KineticEnergy",
        "PrePosition_X",
        "PrePosition_Y",
        "PrePosition_Z",
        "PreDirection_X",
        "PreDirection_Y",
        "PreDirection_Z",
        "TimeFromBeginOfEvent",
        "EventPosition_X",
        "EventPosition_Y",
        "EventPosition_Z",
        "EventDirection_X",
        "EventDirection_Y",
        "EventDirection_Z",
    ]

    # Step1: name the columns according to key
    p_ideal, d_exit, e_pos, e_dir = gaga.get_key_3d(
        x,
        keys,
        ["IdealPosition_X", "PreDirection_X", "EventPosition_X", "EventDirection_X"],
    )
    t_exit = gaga.get_key(x, keys, ["TimeFromBeginOfEvent"])[0]
    ene = gaga.get_key(x, keys, ["KineticEnergy"])[0]

    # Step2: compute P_ideal
    P_exit = p_ideal + speed_of_light * t_exit * d_exit

    # Step3: stack
    # x = torch.stack((ene, P_ideal, d_exit, t_exit, e_pos), dim=0).T # FIXME torch or numpy ?
    x = np.concatenate((ene, P_exit, d_exit, t_exit, e_pos, e_dir), axis=1)

    # end
    return x, keys_out
