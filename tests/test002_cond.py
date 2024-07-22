#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gaga_phsp as gaga
import gatetools.phsp as phsp
from pathlib import Path

if __name__ == "__main__":
    """
    Training : about 2 min (gpu)
    """

    # input
    phsp_filename = Path("output") / "test002_cond.npy"
    pth_filename = Path("output") / "test002_cond.pth"
    png = Path("output") / "test002_cond.png"

    # step 1
    cmd = f"gaga_gauss_cond_test {phsp_filename} -n 4e4 -m 10"
    gaga.run_and_check(cmd)

    # step 2
    cmd = f"gaga_train {phsp_filename} json/cg1.json -o {pth_filename} -pi epoch 30"
    gaga.run_and_check(cmd)

    # step 3
    cmd = f"gaga_gauss_plot {phsp_filename} {pth_filename} -n 1e4 -o {png}"
    gaga.run_and_check(cmd)
    print(f"Results in {png}")

    plt = str(pth_filename).replace(".pth", ".png")
    cmd = f"gaga_plot  {phsp_filename} {pth_filename} --cond_phsp {phsp_filename} -o {plt}"
    gaga.run_and_check(cmd)
    print(f"Results in {plt}")

    # load phsp
    n = 1e5
    print(f"Load phsp : {phsp_filename}")
    real, r_keys, m = phsp.load(phsp_filename, nmax=n, shuffle=True)
    print(f"real shape {real.shape}  {r_keys}")
    cond = real[:, 2:4]
    print(f"cond shape {cond.shape}")

    # load gaga
    params, G, D, optim = gaga.load(pth_filename)
    print(f"Keys : {params['keys']}")
    print(f"Keys_list : {params['keys_list']}")

    # generate (non cond)
    fake = gaga.generate_samples3(params, G, n, cond)
    print(f"fake shape {fake.shape}")

    # compare fake and real
    print()
    gaga.compare_sampled_points(r_keys[0:2], real, fake, wtol=0.21, tol=0.03)

    # end
    gaga.test_ok(True)
