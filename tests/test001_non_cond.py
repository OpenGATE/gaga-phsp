#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gaga_phsp as gaga
import gatetools.phsp as phsp

if __name__ == "__main__":
    """
    Training : about 20-30 sec (gpu)
    """

    # input
    phsp_filename = f"npy/gauss_v1.npy"
    pth_filename = f"pth/test001_non_cond.pth"

    # step 1
    cmd = f"gaga_gauss_test {phsp_filename} -n 4e5 -t v1"
    gaga.run_and_check(cmd)

    # step 2
    cmd = f"gaga_train {phsp_filename} json/g1.json -o {pth_filename} -pi epoch 20"
    gaga.run_and_check(cmd)

    # step 3
    cmd = f"gaga_gauss_plot {phsp_filename} {pth_filename} -n 1e4"
    gaga.run_and_check(cmd)
    print("Results in cond.png")

    plt = pth_filename.replace(".pth", ".png")
    cmd = f"gaga_plot  {phsp_filename} {pth_filename} -o {plt}"
    gaga.run_and_check(cmd)
    print(f"Results in {plt}")

    # load phsp
    n = 1e5
    print(f"Load phsp : {phsp_filename}")
    real, r_keys, m = phsp.load(phsp_filename, nmax=n)
    print(f"real shape {real.shape}  {r_keys}")

    # load gaga
    params, G, D, optim = gaga.load(pth_filename)
    print(f"Keys : {params['keys']}")
    print(f"Keys_list : {params['keys_list']}")

    # generate (non cond)
    batch_size = 1e5
    fake = gaga.generate_samples_non_cond(
        params, G, n, batch_size, normalize=False, to_numpy=True
    )
    print(f"fake shape {fake.shape}")

    # compare fake and real
    print()
    gaga.compare_sampled_points(r_keys, real, fake, wtol=0.3, tol=0.08)

    # end
    gaga.test_ok(True)
