#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gaga_phsp as gaga
from gatetools.phsp import load
import numpy as np
from gaga_phsp.gaga_helpers_tests import get_tests_folder

if __name__ == "__main__":
    """
    Convert ideal pos to exit pos
    """

    # input
    output_folder = get_tests_folder() / "output"
    data_folder = get_tests_folder() / "data"
    root_filename = data_folder / "test005_pet_iec.root"
    pairs_filename = output_folder / "test005_pairs.npy"
    tlor_filename = output_folder / "test005_tlor.npy"
    pairs2_filename = output_folder / "test005_pairs2.npy"

    # step 1
    cmd = f"gaga_pet_to_pairs {root_filename} -o {pairs_filename} -n 1e4"
    gaga.run_and_check(cmd)

    # step 2
    cmd = f"gaga_pairs_to_tlor {pairs_filename} -o {tlor_filename}"
    gaga.run_and_check(cmd)

    # step 3
    cmd = f"gaga_tlor_to_pairs {tlor_filename} -o {pairs2_filename} -r 210"
    gaga.run_and_check(cmd)

    # compare
    data1, read_keys1, m1 = load(pairs_filename, "phase_space")
    data2, read_keys2, m2 = load(pairs2_filename)
    print('k1', pairs_filename, read_keys1)
    print('k2', pairs2_filename, read_keys2)

    keys = read_keys2
    tol = 0.3
    is_ok = True
    for k in keys:
        if k == 'w' or 'd' in k:
            continue
        index = read_keys1.index(k)
        x1 = data1[:, index]
        index = read_keys2.index(k)
        x2 = data2[:, index]
        m1 = np.mean(x1)
        m2 = np.mean(x2)
        d = abs(m1 - m2) / m1
        b = d < tol
        print(f"{k} = {m1:.3f} vs {m2:.3f}        +/- {d:.3f} < {tol:.3f} : {b}")
        is_ok = is_ok and b

    # end
    gaga.test_ok(is_ok)
