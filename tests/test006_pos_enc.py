#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gaga_phsp as gaga
from gaga_phsp.gaga_helpers_tests import get_tests_folder
from test006_helpers import (
    generate_continuous_character_points_rejection_sampling,
    evaluate,
)
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # input
    output_folder = get_tests_folder() / "output"
    phsp_filename = output_folder / "test006_input.npy"
    pth_filename = output_folder / "test006.pth"
    png = output_folder / "test006.png"
    json_file = get_tests_folder() / "json" / "test006_pos_enc.json"

    # step 1: generate training data
    print("generate training data")
    c = "%"
    fs = 150
    points = generate_continuous_character_points_rejection_sampling(
        character=c, num_points=10000, font_size=fs
    )
    # Create a structured array with named fields for x and y coordinates
    structured_points = np.zeros(
        points.shape[0],
        dtype=[("bidon", "f4"), ("x", "f4"), ("y", "f4"), ("color", "f4")],
        # dtype=[("x", "f4"), ("y", "f4")],
    )
    structured_points["x"] = points[:, 0]
    structured_points["y"] = points[:, 1]
    structured_points["bidon"] = np.random.uniform(0, 1, points.shape[0])
    structured_points["color"] = np.random.uniform(0, 1, points.shape[0])

    # Save the structured array with named fields to a .npy file
    np.save(phsp_filename, structured_points)
    loaded_points = np.load(phsp_filename)
    print("Field names in saved file:", loaded_points.dtype.names)

    # step 2
    cmd = f"gaga_train {phsp_filename} {json_file} -o {pth_filename} -pi epoch 100"
    # gaga.run_and_check(cmd)

    # load training data
    n = 1e3
    points = generate_continuous_character_points_rejection_sampling(
        character=c, num_points=int(n), font_size=fs
    )
    plt.scatter(points[:, 0], points[:, 1], s=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    # load gaga
    params, G, D, optim = gaga.load(pth_filename)
    keys_list = params["keys_list"]
    print(f"Keys : {params['keys']}")
    print(f"Keys_list : {params['keys_list']}")

    # generate (non cond)
    fake = gaga.generate_samples3(params, G, n, cond=None)
    print(f"fake shape {fake.shape}")

    # compute
    inside = evaluate(points, character=c, font_size=fs)
    print(f"Number of points inside ref  = {inside/n*100} %")

    fake = fake[:, keys_list.index("x") : keys_list.index("y") + 1]
    inside = evaluate(fake, character=c, font_size=fs)
    print(f"Number of points inside fake = {inside/n*100} %")

    # plot
    plt.scatter(fake[:, 0], fake[:, 1], s=2)
    plt.show()

    # compare fake and real
    print()
    # is_ok = gaga.compare_sampled_points(r_keys[0:2], real, fake, wtol=0.21, tol=0.03)

    # end
    # gaga.test_ok(is_ok)
