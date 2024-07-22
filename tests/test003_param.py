#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gaga_phsp as gaga
from gatetools.phsp import load
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    """
    Convert ideal pos to exit pos
    """

    # input
    dataset_filename = Path("data") / "test003" / "spect_training_dataset.root"
    npy1_filename = Path("output") / "test003_exit_pos.npy"
    npy2_filename = Path("output") / "test003_ideal_pos.npy"
    plot_filename = Path("output") / "test003.png"

    # step 1
    cmd = f"gaga_exit_pos_to_ideal_pos {dataset_filename} -n 1e4 -o {npy1_filename}"
    gaga.run_and_check(cmd)

    # step 2
    cmd = f"gaga_ideal_pos_to_exit_pos {npy1_filename} -n 1e4 -o {npy2_filename}"
    gaga.run_and_check(cmd)

    # step 3
    cmd = f"gt_phsp_plot {npy2_filename} {dataset_filename} -n 1e4 -o {plot_filename}"
    gaga.run_and_check(cmd)
    print(f"Results in {plot_filename}")

    # compare
    data1, read_keys1, m1 = load(dataset_filename, "phase_space")
    data2, read_keys2, m2 = load(npy2_filename)

    keys = ['KineticEnergy',
            'PrePosition_X', 'PrePosition_Y', 'PrePosition_Z',
            'PreDirection_X', 'PreDirection_Y', 'PreDirection_Z',
            'TimeFromBeginOfEvent',
            'EventPosition_X', 'EventPosition_Y', 'EventPosition_Z',
            'EventDirection_X', 'EventDirection_Y', 'EventDirection_Z']
    tol = 0.01
    is_ok = True
    for k in keys:
        index = read_keys1.index(k)
        x1 = data1[:, index]
        index = read_keys2.index(k)
        x2 = data2[:, index]
        m1 = np.mean(x1)
        m2 = np.mean(x2)
        d = abs(m1 - m2) / m1
        b = d < tol
        print(f"{k} = {m1:.3f} +/- {d:.3f} < {tol:.3f} : {b}")
        is_ok = is_ok and b

    # end
    gaga.test_ok(True)
