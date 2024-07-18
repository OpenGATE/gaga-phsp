#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
import gaga_phsp as gaga
import gatetools.phsp as phsp
import os
import time

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("pth_filename")
@click.option("--n", "-n", default="1e4", help="Number of samples to generate")
@click.option("--output", "-o", default="AUTO", help="If AUTO, use pth_filename.npy")
@click.option("--output_folder", "-f", default=None, help="Output folder")
@click.option("--toggle/--no-toggle", default=False, help="Convert XY to angle")
@click.option("--epoch", default=-1, help="Use G at this epoch")
@click.option(
    "--radius", default=350, help="When convert angle, need the radius (in mm)"
)
@click.option("--cond_phsp", "-c", default=None, help="Conditional phsp")
def gaga_generate(
    pth_filename, n, output, output_folder, toggle, radius, epoch, cond_phsp
):
    """
    Generate a PHSP from a (trained) GAN

    \b
    <PTH_FILENAME>    : input GAN PTH file (.pth)
    """

    init_n = str(n)
    n = int(float(n))

    # load pth
    params, G, D, optim, dtypef = gaga.load(
        pth_filename, "auto", verbose=False, epoch=epoch
    )
    f_keys = list(params["keys_list"])

    # cond ?
    cond_data = None
    if cond_phsp is not None:
        cond_keys = params["cond_keys"]
        print(f"Conditional keys {cond_keys}")
        cond_data, cond_read_keys, m = phsp.load(cond_phsp, nmax=n)
        cond_keys = phsp.str_keys_to_array_keys(cond_keys)
        cond_data = phsp.select_keys(cond_data, cond_read_keys, cond_keys)
        print(f"Conditional keys {cond_keys} {cond_data.shape}")

    # generate samples (b is batch size)
    b = 1e5
    start = time.time()
    if cond_phsp is not None:
        fake = gaga.generate_samples3(params, G, n, cond=cond_data)
    else:
        fake = gaga.generate_samples_non_cond(params, G, n, b, False, True)
    end = time.time()
    elapsed = end - start
    pps = n / elapsed
    print(f"Timing: {end - start:0.1f} s   PPS = {pps:0.0f}")

    # Keep X,Y or convert to toggle
    if toggle:
        keys = phsp.keys_toggle_angle(f_keys)
        fake, f_keys = phsp.add_missing_angle(fake, f_keys, keys, radius)
        fake = phsp.select_keys(fake, f_keys, keys)
    else:
        keys = f_keys

    if cond_phsp is not None:
        print(cond_keys)
        print(keys)
        for k in cond_keys:
            keys.remove(k)

    # write
    if output == "AUTO":
        gp = params["penalty"]
        gpw = params["penalty_weight"]
        full_path = os.path.split(pth_filename)
        b, extension = os.path.splitext(full_path[1])
        if not output_folder:
            output_folder = "."
        output = f"{b}_{gp}_{gpw}_{init_n}.npy"
        output = os.path.join(output_folder, output)
        print(output)
    phsp.save_npy(output, fake, keys)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    gaga_generate()
